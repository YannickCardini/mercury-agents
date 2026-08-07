"""
Progrès normalisé d'une bille — utilisé par plan_hand.py pour scorer ses plans.
"""

from mercury_legal_moves import (
    HOME_POSITIONS, START_POSITIONS, ARRIVAL_POSITIONS,
    _MAIN_PATH_IDX, _MAIN_PATH_LEN,
)


def marble_progress(pos: int, color: str) -> float:
    """
    Progrès normalisé d'une bille vers la victoire.
      0.0  = en HOME (n'a pas encore joué)
      0.05 = vient d'entrer en jeu (case START)
      0.70 = a fait un tour complet, prête à entrer en zone d'arrivée
      0.775..1.0 = en zone d'arrivée (slot 0→3)
    """
    if pos in HOME_POSITIONS[color]:
        return 0.0
    arrival = ARRIVAL_POSITIONS[color]
    if pos in arrival:
        i = arrival.index(pos)
        return 0.775 + i * 0.075            # 0.775, 0.85, 0.925, 1.0
    if pos in _MAIN_PATH_IDX:
        start_idx = _MAIN_PATH_IDX[START_POSITIONS[color]]
        pos_idx   = _MAIN_PATH_IDX[pos]
        steps     = (pos_idx - start_idx) % _MAIN_PATH_LEN
        return 0.05 + steps / _MAIN_PATH_LEN * 0.65   # 0.05 → 0.70
    return 0.0
