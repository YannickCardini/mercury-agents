"""
Bot heuristique scripté pour Mercury — adversaire COHÉRENT (≠ random) servant de
benchmark (et plus tard d'adversaire d'entraînement, pour la robustesse).

Principe : il ne réimplémente AUCUNE règle. Il score les coups DÉJÀ validés par
mercury_legal_moves.get_legal_mask (chaque entrée légale est (card, action_dict) avec
type/from/to, + splitFrom/splitTo/splitType pour le 7), et joue l'argmax.

Jeu ÉQUILIBRÉ (humain, qui FINIT — pas du capture-spam) :
  finir (promote) ≫ faire avancer la bille de tête vers l'arrivée (bonus de course) ;
  on capture surtout les billes AVANCÉES (bonus de déni ∝ progrès de la cible), pas les
  billes sans valeur ; pénalité si la destination laisse la bille à portée d'un adversaire.
"""

import random

from mercury_legal_moves import (
    START_POSITIONS, HOME_POSITIONS,
    _MAIN_PATH_IDX, _MAIN_PATH_LEN, DISCARD_IDX,
)
from reward import marble_progress

_THREAT_RANGE = 7   # une bille adverse à ≤ 7 cases derrière menace de capture

# Malus "économie de cartes" (brise les égalités) : éviter de cramer une carte précieuse
# sur un coup banal. Petit devant le score de base d'un move (~100).
_CARD_WASTE = {
    'Joker': 5.0, 'Q': 4.0, 'J': 4.0, '7': 4.0, 'K': 2.0, 'A': 2.0, '10': 1.5,
    '9': 1.0, '8': 1.0, '6': 0.6, '5': 0.5, '4': 1.0, '3': 0.3, '2': 0.2,
}


def _owner_at(pos: int, marbles_by_color: dict) -> str | None:
    for c, positions in marbles_by_color.items():
        if pos in positions:
            return c
    return None


def _threatened(pos: int, color: str, marbles_by_color: dict) -> bool:
    """True si une bille adverse est à ≤ _THREAT_RANGE cases derrière `pos` (main path).
    Une case start propre (invincible) n'est jamais menacée."""
    if pos not in _MAIN_PATH_IDX or pos == START_POSITIONS[color]:
        return False
    idx = _MAIN_PATH_IDX[pos]
    for c, positions in marbles_by_color.items():
        if c == color:
            continue
        for op in positions:
            if op in _MAIN_PATH_IDX:
                d = (idx - _MAIN_PATH_IDX[op]) % _MAIN_PATH_LEN
                if 1 <= d <= _THREAT_RANGE:
                    return True
    return False


def _move_value(frm: int, to: int, color: str) -> float:
    """Valeur d'un déplacement pour MOI. Le terme de COURSE croît en p² → pousser le
    LEADER vers l'arrivée sûre DOMINE tout le reste, pour que les billes atteignent le
    safe et que les parties SE TERMINENT (au lieu de se faire sniper en boucle)."""
    gain = marble_progress(to, color) - marble_progress(frm, color)
    p    = marble_progress(to, color)
    return 100.0 + 150.0 * gain + 250.0 * p * p


def _effect_value(frm: int, to: int, etype: str, color: str, mbc: dict) -> float:
    """Valeur d'un déplacement de bille (sert aussi pour chaque demi-coup d'un 7-split)."""
    if etype == 'promote':
        return 2000.0                      # finir : toujours, écrase tout le reste
    base = _move_value(frm, to, color)
    if etype == 'capture':
        # Déni OPPORTUNISTE seulement (réduit 400→120) : on capture si c'est sur la route,
        # jamais au prix de sa propre course → casse le gridlock de snipe mutuel.
        owner = _owner_at(to, mbc)
        base += 120.0 * (marble_progress(to, owner) if owner else 0.0)
    return base


def _danger(to: int, color: str, mbc: dict) -> float:
    return 150.0 * marble_progress(to, color) if _threatened(to, color, mbc) else 0.0


def _score(card: dict, a: dict, color: str, mbc: dict, n_in_play: int) -> float:
    etype = a['type']
    if etype == 'enter':
        return 200.0 + 40.0 * (4 - n_in_play)          # start = sûr, plus utile si bloqué
    if etype == 'swap':                                 # Valet : ma bille ↔ bille adverse
        gain = (marble_progress(a['to'], _owner_at(a['to'], mbc))
                - marble_progress(a['from'], color))
        base = 300.0 if gain > 0 else 0.0
        return base + 150.0 * gain - _danger(a['to'], color, mbc)
    # move / capture / promote (+ 7-split : on somme les deux demi-coups)
    s = _effect_value(a['from'], a['to'], etype, color, mbc) - _danger(a['to'], color, mbc)
    if 'splitTo' in a:
        s += _effect_value(a['splitFrom'], a['splitTo'], a['splitType'], color, mbc)
        s -= _danger(a['splitTo'], color, mbc)
    if etype == 'move':
        s -= _CARD_WASTE.get(card['value'], 0.0)
    return s


def heuristic_pick(game_state: dict, color: str,
                   mask: list[bool], actions: list) -> int:
    """Renvoie l'index d'action de meilleur score (greedy + tie-break aléatoire léger)."""
    mbc = {p['color']: p['marblePositions'] for p in game_state['players']}
    n_in_play = sum(1 for p in mbc.get(color, []) if p not in HOME_POSITIONS[color])
    best_i, best_s = None, float('-inf')
    for i, ok in enumerate(mask):
        if not ok:
            continue
        entry = actions[i]
        if entry is None or entry == 'discard':
            s = -1000.0
        else:
            card, a = entry
            s = _score(card, a, color, mbc, n_in_play)
        s += random.uniform(0.0, 1e-3)   # un peu de variété, sans casser le greedy
        if s > best_s:
            best_s, best_i = s, i
    return best_i if best_i is not None else DISCARD_IDX
