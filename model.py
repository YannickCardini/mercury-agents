"""
Réseau de politique et valeur pour le bot RL Mercury.

ENCODAGE COURANT (143 dims) — repris de models/model_v3.py. Point clé : la main est
encodée CARTE PAR CARTE (5 slots × 13 rangs), ce que l'ancien encodage 54-dim ne faisait
pas (il ne donnait qu'un bitmask des rangs présents, sans ordre). Or l'espace d'action
est indexé PAR SLOT de carte (cf. mercury_legal_moves.py) : sans connaître la carte de
chaque slot, le réseau ne pouvait pas choisir QUELLE carte dépenser quand plusieurs
cartes jouent le même pion. C'est la correction de fond.

Le repère de position est désormais l'index d'ANNEAU PARTAGÉ (0..55 dans MAIN_PATH,
normalisé) au lieu de pos/68 (les cases brutes vont jusqu'à 223 → valeurs > 3, non
comparables entre couleurs).

Architecture INCHANGÉE vs la version 54-dim qui convergeait : 2 couches ReLU, têtes
directes, softmax masqué, hidden=384. On ne change QU'UN seul facteur — l'encodage
d'entrée (54 → 143). La tentative v3 d'origine échouait parce qu'elle changeait AUSSI
l'architecture (3 couches + LayerNorm) en même temps : impossible d'isoler la cause.

ENCODAGE LEGACY (54 dims, encode_state_legacy) + MercuryNetLegacy 2×256 : conservés
UNIQUEMENT pour charger/jouer les anciens modèles (ex. model_ref_082.pt, l'agent 0.82
en prod) comme adversaire de duel. encode_state_for() route chaque réseau vers le bon
encodeur selon la largeur de son entrée.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from mercury_legal_moves import (
    HOME_POSITIONS, START_POSITIONS, ARRIVAL_POSITIONS,
    _MAIN_PATH_IDX, _MAIN_PATH_LEN, ALL_COLORS,
)
# Progrès d'une bille : source unique dans reward.py (aussi utilisée par plan_hand,
# heuristic_bot). On la réutilise ici plutôt que d'en redéfinir une copie identique.
from reward import marble_progress

# ── Constantes d'espace ───────────────────────────────────────────────────────

STATE_DIM        = 143   # encodage courant (main par slot 5×14 + anneau partagé + flags ;
                         # 14e rang = Joker, cf. mise à jour du jeu → 138→143)
STATE_DIM_LEGACY = 54    # ancien encodage (bitmask de rangs) — anciens modèles seulement
ACTION_DIM       = 501   # nombre d'actions (cartes × cibles ou split-7)

# Ordre des rangs pour l'encodage de la main (one-hot par slot). 14e rang = Joker.
_CARD_VALUES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'Joker']
_CARD_IDX    = {v: i for i, v in enumerate(_CARD_VALUES)}

_DANGER_RANGE = 12   # bille adverse à ≤ 12 cases derrière (≙ Q) = menace de capture


# ── Helpers d'encodage par bille ──────────────────────────────────────────────

def _marble_abs_enc(pos: int, color: str) -> float:
    """Index absolu dans MAIN_PATH normalisé (repère d'ANNEAU PARTAGÉ par toutes les
    couleurs) : HOME=0, chemin 1..56, arrivée 57..60 — le tout /61. Permet au réseau de
    comparer physiquement les positions (qui peut capturer qui)."""
    if pos in HOME_POSITIONS[color]:
        return 0.0
    arrival = ARRIVAL_POSITIONS[color]
    if pos in arrival:
        return (57 + arrival.index(pos)) / 61.0
    if pos in _MAIN_PATH_IDX:
        return (_MAIN_PATH_IDX[pos] + 1) / 61.0
    return 0.0


def _marble_safe(pos: int, color: str) -> float:
    """1.0 si la bille est sur sa propre case de départ (invulnérable)."""
    return 1.0 if pos == START_POSITIONS[color] else 0.0


def _marble_danger(pos: int, color: str, marbles_by_color: dict) -> float:
    """1.0 si une bille adverse est à ≤ _DANGER_RANGE cases derrière, sur le main path
    (menace de capture — signal utile pour le Valet défensif et le recul au 4)."""
    if pos not in _MAIN_PATH_IDX or pos == START_POSITIONS[color]:
        return 0.0
    my_idx = _MAIN_PATH_IDX[pos]
    for opp_color, positions in marbles_by_color.items():
        if opp_color == color:
            continue
        for opp_pos in positions:
            if opp_pos in _MAIN_PATH_IDX:
                dist_behind = (my_idx - _MAIN_PATH_IDX[opp_pos]) % _MAIN_PATH_LEN
                if 1 <= dist_behind <= _DANGER_RANGE:
                    return 1.0
    return 0.0


# ── Encodage d'état COURANT (143 dims) ─────────────────────────────────────────

def encode_state(game_state: dict, my_color: str) -> torch.Tensor:
    """Encode l'état du jeu en vecteur de STATE_DIM (143) dimensions.

    Structure :
      [0:16]    position absolue (anneau partagé) des 16 billes — ma couleur d'abord
      [16:32]   progrès relatif (0→1) des 16 billes vers l'arrivée de leur camp
      [32:48]   flag protégé (sur sa case start) des 16 billes
      [48:118]  main : 5 slots × 14 rangs one-hot (14e = Joker)   ← LA correction clé
      [118]     canDiscard
      [119:135] flag danger des 16 billes (adversaire ≤ 12 cases derrière)
      [135:139] fraction de billes en jeu (hors home) par couleur (ma couleur d'abord)
      [139:143] fraction de billes en zone d'arrivée par couleur
    """
    vec: list[float] = []
    color_order      = [my_color] + [c for c in ALL_COLORS if c != my_color]
    marbles_by_color = {p['color']: p['marblePositions'] for p in game_state['players']}

    def marbles(color: str) -> list[int]:
        return marbles_by_color.get(color, [0, 0, 0, 0])

    # [0:16] position absolue (anneau partagé)
    for color in color_order:
        for pos in marbles(color):
            vec.append(_marble_abs_enc(pos, color))

    # [16:32] progrès relatif au camp
    for color in color_order:
        for pos in marbles(color):
            vec.append(marble_progress(pos, color))

    # [32:48] flag protégé (case start)
    for color in color_order:
        for pos in marbles(color):
            vec.append(_marble_safe(pos, color))

    # [48:118] main : 5 slots × 14 rangs one-hot (14e = Joker ; padding zéros si < 5 cartes)
    hand = game_state.get('hand', [])
    for i in range(5):
        slot = [0.0] * 14
        if i < len(hand):
            card  = hand[i]
            value = card.get('value') if isinstance(card, dict) else str(card)
            ci    = _CARD_IDX.get(value)
            if ci is not None:
                slot[ci] = 1.0
        vec.extend(slot)

    # [118] canDiscard
    vec.append(1.0 if game_state.get('canDiscard', False) else 0.0)

    # [119:135] flag danger
    for color in color_order:
        for pos in marbles(color):
            vec.append(_marble_danger(pos, color, marbles_by_color))

    # [135:139] fraction de billes en jeu (hors home) par couleur
    for color in color_order:
        ms = marbles(color)
        vec.append(sum(1 for p in ms if p not in HOME_POSITIONS[color]) / 4.0)

    # [139:143] fraction de billes en zone d'arrivée par couleur
    for color in color_order:
        ms = marbles(color)
        vec.append(sum(1 for p in ms if p in ARRIVAL_POSITIONS[color]) / 4.0)

    return torch.tensor(vec, dtype=torch.float32)


# ── Encodage d'état LEGACY (54 dims) ──────────────────────────────────────────
# Conservé pour les modèles entraînés sur l'ancien encodage (ex. model_ref_082.pt joué
# comme adversaire de duel). NE PAS utiliser pour entraîner le réseau courant.

def encode_state_legacy(game_state: dict, my_color: str) -> torch.Tensor:
    """ANCIEN encodage 54-dim (bitmask de rangs, positions /68, sans info par slot)."""
    feat = torch.zeros(STATE_DIM_LEGACY, dtype=torch.float32)

    marbles_by_color = {p['color']: p['marblePositions'] for p in game_state['players']}

    my_marbles = marbles_by_color.get(my_color, [0, 0, 0, 0])
    for i, pos in enumerate(my_marbles):
        feat[i] = pos / 68.0

    idx = 4
    for color in ALL_COLORS:
        if color != my_color:
            for pos in marbles_by_color.get(color, [0, 0, 0, 0]):
                feat[idx] = pos / 68.0
                idx += 1

    for i, pos in enumerate(my_marbles):
        feat[16 + i] = marble_progress(pos, my_color)

    idx = 20
    for color in ALL_COLORS:
        if color != my_color:
            for pos in marbles_by_color.get(color, [0, 0, 0, 0]):
                feat[idx] = marble_progress(pos, color)
                idx += 1

    for i, color in enumerate(ALL_COLORS):
        arrival = ARRIVAL_POSITIONS[color]
        ms      = marbles_by_color.get(color, [])
        feat[32 + i] = sum(1 for p in ms if p in arrival) / 4.0

    current_turn = game_state.get('currentTurn', my_color)
    feat[36 + ALL_COLORS.index(current_turn)] = 1.0

    hand        = game_state.get('hand', [])
    hand_values = set()
    for card in hand:
        hand_values.add(card.get('value', '') if isinstance(card, dict) else str(card))
    card_names = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    for ci, name in enumerate(card_names):
        feat[40 + ci] = 1.0 if name in hand_values else 0.0

    feat[53] = 1.0 if game_state.get('canDiscard', False) else 0.0
    return feat


def encode_state_for(net: nn.Module, game_state: dict, my_color: str) -> torch.Tensor:
    """Route vers le bon encodeur selon la largeur d'entrée du réseau :
      - entrée 54  → encode_state_legacy (anciens modèles, ex. ref 0.82)
      - sinon (143) → encode_state (encodage courant)."""
    if net.fc1.in_features == STATE_DIM_LEGACY:
        return encode_state_legacy(game_state, my_color)
    return encode_state(game_state, my_color)


# ── Réseau de politique-valeur ────────────────────────────────────────────────

def _apply_mask_dist(logits: torch.Tensor, legal_mask: torch.Tensor) -> Categorical:
    """Masque les actions illégales (-inf) puis construit la Categorical.
    Factorisé pour être partagé par MercuryNet et MercuryNetLegacy."""
    logits = logits.masked_fill(~legal_mask, float('-inf'))
    return Categorical(F.softmax(logits, dim=-1))


class _PolicyValueNet(nn.Module):
    """Tronc commun politique + valeur : 2 couches ReLU nu, têtes directes, softmax masqué.
    Les sous-classes ne fixent QUE la largeur d'entrée (`in_features`) et la largeur cachée
    par défaut — l'architecture est volontairement identique. Les modules sont nommés
    `fc1/fc2/policy_head/value_head` (clés du state_dict) : les checkpoints existants
    restent chargeables et `load_net`/`encode_state_for` peuvent lire `fc1.in_features`."""

    def __init__(self, in_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, ACTION_DIM)
        self.value_head  = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, legal_mask: torch.Tensor) -> tuple:
        is_batch = state.dim() == 2
        if not is_batch:
            state = state.unsqueeze(0)
            legal_mask = legal_mask.unsqueeze(0)
        h = F.relu(self.fc1(state))
        h = F.relu(self.fc2(h))
        dist  = _apply_mask_dist(self.policy_head(h), legal_mask)
        value = self.value_head(h).squeeze(-1)
        if not is_batch:
            value = value.squeeze(0)
        return dist, value


class MercuryNet(_PolicyValueNet):
    """Réseau PPO courant. Entrée = encodage COURANT (STATE_DIM=143). Architecture
    identique à la version 54-dim qui convergeait — SEUL l'encodage d'entrée change."""

    def __init__(self, hidden_dim: int = 384):
        super().__init__(STATE_DIM, hidden_dim)


class MercuryNetLegacy(_PolicyValueNet):
    """Architecture HISTORIQUE (2×256) + entrée LEGACY (STATE_DIM_LEGACY=54). Conservée
    UNIQUEMENT pour charger les anciens checkpoints (ex. model_ref_082.pt, l'agent 0.82
    en prod) comme adversaire de duel. Ne sert plus à l'entraînement."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__(STATE_DIM_LEGACY, hidden_dim)


# ── Chargement auto-détectant l'architecture ──────────────────────────────────

def load_net(path, map_location="cpu", eval_mode: bool = True) -> nn.Module:
    """Charge un checkpoint (dict {net,...} ou state_dict brut) dans la BONNE
    architecture, auto-détectée par la LARGEUR D'ENTRÉE (`fc1.weight` → in_features) :
      - in_features == 54 → MercuryNetLegacy (anciens modèles, encodage legacy)
      - sinon (143)        → MercuryNet (encodage courant)
    La largeur cachée (256/384) est lue depuis le checkpoint. Renvoie le réseau prêt."""
    ckpt   = torch.load(path, weights_only=True, map_location=map_location)
    state  = ckpt['net'] if isinstance(ckpt, dict) and 'net' in ckpt else ckpt
    hidden, in_features = state['fc1.weight'].shape
    net = MercuryNetLegacy(hidden) if in_features == STATE_DIM_LEGACY else MercuryNet(hidden)
    net.load_state_dict(state)
    if eval_mode:
        net.eval()
    return net
