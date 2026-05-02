"""
Réseau de politique et valeur pour le bot RL Mercury.
Architecture : 2 couches de 256 unités, sortie pour 501 actions + valeur scalaire.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from mercury_legal_moves import (
    MAIN_PATH, HOME_POSITIONS, START_POSITIONS, ARRIVAL_POSITIONS,
    _MAIN_PATH_IDX, _MAIN_PATH_LEN, _ALL_STARTS, ALL_COLORS,
)

# ── Constantes d'espace ───────────────────────────────────────────────────────

STATE_DIM  = 92   # encodage de l'état du jeu
ACTION_DIM = 501  # nombre d'actions (cartes × cibles ou split-7)


# ── Encodage d'état ───────────────────────────────────────────────────────────

def encode_state(game_state: dict, my_color: str) -> torch.Tensor:
    """
    Encode l'état du jeu en vecteur de STATE_DIM dimensions.

    Structure (92 dims) :
      [0:4]       → positions de mes 4 billes (0–67 ou OOB)
      [4:20]      → positions des 16 billes adverses (4 couleurs × 4 billes)
      [20:40]     → positions normalisées (progrès vers l'arrivée) pour mes billes
      [40:56]     → progrès des 16 billes adverses
      [56:60]     → billes de chaque couleur arrivées en zone d'arrivée
      [60:64]     → ordre de tour (OHE sur 4 couleurs)
      [64:84]     → main du joueur courant (carte 2..A, 1 si en main, 0 sinon)
      [84]        → canDiscard (float 0/1)
      [85:92]     → nombre de billes arrivées par couleur adverse (4) + padding
    """
    feat = torch.zeros(STATE_DIM, dtype=torch.float32)

    # Récupérer les positions des billes
    marbles_by_color = {}
    for player in game_state['players']:
        marbles_by_color[player['color']] = player['marblePositions']

    # [0:4] Mes billes
    my_marbles = marbles_by_color.get(my_color, [0, 0, 0, 0])
    for i, pos in enumerate(my_marbles):
        feat[i] = pos / 68.0  # normaliser par nombre max de positions (~68)

    # [4:20] Billes adverses (4 couleurs × 4)
    idx = 4
    for color in ALL_COLORS:
        if color != my_color:
            opp_marbles = marbles_by_color.get(color, [0, 0, 0, 0])
            for pos in opp_marbles:
                feat[idx] = pos / 68.0
                idx += 1

    # [20:40] Progrès normalisé de mes billes
    for i, pos in enumerate(my_marbles):
        feat[20 + i] = _marble_progress_norm(pos, my_color)

    # [40:56] Progrès des billes adverses
    idx = 40
    for color in ALL_COLORS:
        if color != my_color:
            opp_marbles = marbles_by_color.get(color, [0, 0, 0, 0])
            for pos in opp_marbles:
                feat[idx] = _marble_progress_norm(pos, color)
                idx += 1

    # [56:60] Billes de chaque couleur en zone d'arrivée
    for i, color in enumerate(ALL_COLORS):
        arrival = ARRIVAL_POSITIONS[color]
        marbles = marbles_by_color.get(color, [])
        n_arrived = sum(1 for p in marbles if p in arrival)
        feat[56 + i] = n_arrived / 4.0

    # [60:64] Ordre de tour (OHE)
    current_turn = game_state.get('currentTurn', my_color)
    turn_idx = ALL_COLORS.index(current_turn)
    feat[60 + turn_idx] = 1.0

    # [64:84] Main du joueur courant (4 copies × 5 cartes)
    hand = game_state.get('hand', [])
    # Extraire les rangs (hand peut être une liste de strings ou de dicts)
    hand_ranks = set()
    for card in hand:
        if isinstance(card, dict):
            hand_ranks.add(card.get('rank', ''))
        else:
            hand_ranks.add(str(card))

    card_names = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    for card_idx, card_name in enumerate(card_names):
        feat[64 + card_idx] = 1.0 if card_name in hand_ranks else 0.0
    # Padding pour atteindre 20

    # [84] canDiscard
    feat[84] = 1.0 if game_state.get('canDiscard', False) else 0.0

    # [85:89] Billes arrivées par couleur adverse
    idx = 85
    for color in ALL_COLORS:
        if color != my_color:
            arrival = ARRIVAL_POSITIONS[color]
            marbles = marbles_by_color.get(color, [])
            n_arrived = sum(1 for p in marbles if p in arrival)
            feat[idx] = n_arrived / 4.0
            idx += 1

    return feat


def _marble_progress_norm(pos: int, color: str) -> float:
    """Progrès normalisé (0–1) d'une bille vers la victoire."""
    if pos in HOME_POSITIONS[color]:
        return 0.0
    arrival = ARRIVAL_POSITIONS[color]
    if pos in arrival:
        i = arrival.index(pos)
        return (0.775 + i * 0.075) / 1.0  # 0.775, 0.85, 0.925, 1.0
    if pos in _MAIN_PATH_IDX:
        start_idx = _MAIN_PATH_IDX[START_POSITIONS[color]]
        pos_idx   = _MAIN_PATH_IDX[pos]
        steps     = (pos_idx - start_idx) % _MAIN_PATH_LEN
        return (0.05 + steps / _MAIN_PATH_LEN * 0.65) / 1.0  # 0.05 → 0.70
    return 0.0


# ── Réseau de politique-valeur ────────────────────────────────────────────────

class MercuryNet(nn.Module):
    """
    Réseau PPO : politique + valeur.
    Entrée : état (STATE_DIM), masque légal (ACTION_DIM).
    Sortie : distribution catégorique (actions légales), valeur scalaire.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Backbone partagé
        self.fc1 = nn.Linear(STATE_DIM, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Tête de politique
        self.policy_head = nn.Linear(hidden_dim, ACTION_DIM)

        # Tête de valeur
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, legal_mask: torch.Tensor
                ) -> tuple:
        """
        Args:
            state : shape (STATE_DIM,) ou (batch, STATE_DIM)
            legal_mask : shape (ACTION_DIM,) ou (batch, ACTION_DIM), bool

        Returns:
            (dist, value) où dist est une Categorical et value un float/tensor
        """
        is_batch = state.dim() == 2
        if not is_batch:
            state = state.unsqueeze(0)
            legal_mask = legal_mask.unsqueeze(0)

        # Backbone
        h = F.relu(self.fc1(state))
        h = F.relu(self.fc2(h))

        # Politique
        logits = self.policy_head(h)

        # Masquer les actions illégales
        logits = logits.masked_fill(~legal_mask, float('-inf'))
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        # Valeur
        value = self.value_head(h)

        if not is_batch:
            value = value.squeeze(0)

        return dist, value
