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

STATE_DIM  = 54   # encodage compact de l'état du jeu (sans zéros parasites)
ACTION_DIM = 501  # nombre d'actions (cartes × cibles ou split-7)


# ── Encodage d'état ───────────────────────────────────────────────────────────

def encode_state(game_state: dict, my_color: str) -> torch.Tensor:
    """
    Encode l'état du jeu en vecteur de STATE_DIM dimensions.

    Structure (54 dims) :
      [0:4]   → positions de mes 4 billes (normalisé 0–68)
      [4:16]  → positions des 12 billes adverses (3 couleurs × 4)
      [16:20] → progression normalisée de mes 4 billes vers l'arrivée (0→1)
      [20:32] → progression des 12 billes adverses (3 couleurs × 4)
      [32:36] → nombre de billes arrivées par couleur (4 couleurs, normalisé /4)
      [36:40] → tour actuel (one-hot sur 4 couleurs)
      [40:53] → main : présence de chaque rang (2..A), one-hot 13 dims
      [53]    → canDiscard (float 0/1)
    """
    feat = torch.zeros(STATE_DIM, dtype=torch.float32)

    # Récupérer les positions des billes
    marbles_by_color = {}
    for player in game_state['players']:
        marbles_by_color[player['color']] = player['marblePositions']

    # [0:4] Mes billes
    my_marbles = marbles_by_color.get(my_color, [0, 0, 0, 0])
    for i, pos in enumerate(my_marbles):
        feat[i] = pos / 68.0

    # [4:16] Billes adverses (3 couleurs × 4 = 12)
    idx = 4
    for color in ALL_COLORS:
        if color != my_color:
            opp_marbles = marbles_by_color.get(color, [0, 0, 0, 0])
            for pos in opp_marbles:
                feat[idx] = pos / 68.0
                idx += 1

    # [16:20] Progression normalisée de mes 4 billes
    for i, pos in enumerate(my_marbles):
        feat[16 + i] = _marble_progress_norm(pos, my_color)

    # [20:32] Progression des billes adverses (3 couleurs × 4 = 12)
    idx = 20
    for color in ALL_COLORS:
        if color != my_color:
            opp_marbles = marbles_by_color.get(color, [0, 0, 0, 0])
            for pos in opp_marbles:
                feat[idx] = _marble_progress_norm(pos, color)
                idx += 1

    # [32:36] Billes arrivées par couleur (4 couleurs)
    for i, color in enumerate(ALL_COLORS):
        arrival = ARRIVAL_POSITIONS[color]
        marbles = marbles_by_color.get(color, [])
        n_arrived = sum(1 for p in marbles if p in arrival)
        feat[32 + i] = n_arrived / 4.0

    # [36:40] Tour actuel (one-hot)
    current_turn = game_state.get('currentTurn', my_color)
    turn_idx = ALL_COLORS.index(current_turn)
    feat[36 + turn_idx] = 1.0

    # [40:53] Main du joueur (13 rangs 2..A, one-hot)
    hand = game_state.get('hand', [])
    hand_values = set()
    for card in hand:
        if isinstance(card, dict):
            hand_values.add(card.get('value', ''))
        else:
            hand_values.add(str(card))

    card_names = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    for card_idx, card_name in enumerate(card_names):
        feat[40 + card_idx] = 1.0 if card_name in hand_values else 0.0

    # [53] canDiscard
    feat[53] = 1.0 if game_state.get('canDiscard', False) else 0.0

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

        # Valeur — squeeze la dim de sortie (…, 1) → (…,) pour matcher returns
        value = self.value_head(h).squeeze(-1)

        if not is_batch:
            value = value.squeeze(0)

        return dist, value
