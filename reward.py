"""
Récompenses denses pour le bot RL Mercury.
La récompense est calculée entre deux états successifs du même joueur
(i.e. entre deux tours du même bot, couvrant aussi les tours adverses).
"""

from mercury_legal_moves import (
    MAIN_PATH, HOME_POSITIONS, START_POSITIONS, ARRIVAL_POSITIONS,
    _MAIN_PATH_IDX, _MAIN_PATH_LEN, _ALL_STARTS, ALL_COLORS,
)

_ALL_HOME = {p for positions in HOME_POSITIONS.values() for p in positions}

# ── Coefficients du shaping ───────────────────────────────────────────────────
# Placés en haut pour faciliter le tuning ultérieur.
OWN_PROGRESS_COEF    = 0.6   # gain/perte de progrès sur mes billes
                              # (réduit de 1.5 : évite que l'avancement dense domine la victoire)
OPP_PROGRESS_COEF    = 1.0   # progrès perdu par un adversaire (capture + swap + 4)
OPP_ARRIVAL_PENALTY  = 0.15  # malus fixe par bille adverse entrant en zone d'arrivée
OPP_ARRIVAL_SCALE    = 0.10  # malus additionnel par bille déjà arrivée chez l'adversaire
THREAT_PENALTY       = 0.05  # malus si une bille à moi est à portée d'un adversaire
THREAT_RANGE         = 6     # distance (en cases) considérée comme menaçante
# Bonus d'entrée explicite : doit largement dominer "advance-1" pour que l'agent
# apprenne à faire entrer ses billes plutôt qu'à avancer d'une case avec l'As.
# Avec OWN_PROGRESS_COEF=0.6, le delta d'entrée vaut 0.05×0.6=0.03 — sans ce
# bonus, "advance-1" (0.017) et "enter" (0.03) sont quasiment équivalents.
ENTRY_BONUS          = 1.0
# Bonus d'arrivée : faire rentrer une bille en zone d'arrivée est l'une des
# étapes les plus importantes du jeu (arriver 4 billes = victoire). C'est donc
# la 2ème plus grosse récompense après la victoire (+10), nettement au-dessus
# de l'entrée en jeu (+1) et du recul adverse (+1). Déclenché une seule fois,
# au passage main path → zone d'arrivée (pas par slot).
ARRIVAL_BONUS        = 3.0
WIN_REWARD           = 10.0  # doit dominer la somme des rewards denses (~4.0 avec coef 0.6)
LOSS_REWARD          = -4.0  # symétrie renforcée : perdre doit être aussi significatif que gagner


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


def _get_marbles(game_state: dict, color: str) -> list[int]:
    for player in game_state['players']:
        if player['color'] == color:
            return player['marblePositions']
    return []


def _is_threatened(my_pos: int, curr_gs: dict, my_color: str) -> bool:
    """True si une bille adverse est à ≤ THREAT_RANGE cases derrière ma bille (sur le main path)."""
    if my_pos not in _MAIN_PATH_IDX or my_pos in _ALL_STARTS:
        return False
    my_idx = _MAIN_PATH_IDX[my_pos]
    for color in ALL_COLORS:
        if color == my_color:
            continue
        for opp_p in _get_marbles(curr_gs, color):
            if opp_p in _MAIN_PATH_IDX:
                dist_behind = (my_idx - _MAIN_PATH_IDX[opp_p]) % _MAIN_PATH_LEN
                if 1 <= dist_behind <= THREAT_RANGE:
                    return True
    return False


def compute_reward(prev_gs: dict, curr_gs: dict, my_color: str) -> float:
    """
    Récompense façonnée dense entre deux états successifs du même bot.

    Termes :
      - Progrès de mes billes         : delta_progress × OWN_PROGRESS_COEF
      - Entrée d'une bille (home→jeu) : + ENTRY_BONUS  (valeur actualisée de la bille)
      - Arrivée d'une bille (plateau→zone d'arrivée) : + ARRIVAL_BONUS
      - Progrès perdu par un adversaire (capture / swap / 4 adverse)
                                      : |delta_progress_adverse| × OPP_PROGRESS_COEF
      - Adversaire entrant en arrivée : − OPP_ARRIVAL_PENALTY − n_déjà × OPP_ARRIVAL_SCALE
      - Bille à moi menacée           : − THREAT_PENALTY (signal défensif pour J et 4)
    """
    reward = 0.0

    prev_mine = _get_marbles(prev_gs, my_color)
    curr_mine = _get_marbles(curr_gs, my_color)
    my_arrival = ARRIVAL_POSITIONS[my_color]

    # ── Progrès de mes billes (positif = avance, négatif = capturé/swap arrière)
    for prev_p, curr_p in zip(prev_mine, curr_mine):
        delta = marble_progress(curr_p, my_color) - marble_progress(prev_p, my_color)
        reward += delta * OWN_PROGRESS_COEF
        # Bonus d'entrée : sans cela le signal immédiat pour "enter" (≈0.075)
        # ne domine pas "advance-1" (≈0.017) malgré la valeur stratégique réelle.
        if prev_p in HOME_POSITIONS[my_color] and curr_p not in HOME_POSITIONS[my_color]:
            reward += ENTRY_BONUS
        # Bonus d'arrivée : la bille passe du plateau à ma zone d'arrivée.
        # Étape stratégique majeure → 2ème plus grosse récompense après la victoire.
        if prev_p not in my_arrival and curr_p in my_arrival:
            reward += ARRIVAL_BONUS

    # ── Progrès perdu par les adversaires (capture + swap + card-4 adverse) ──
    # En sémantique semi-MDP (du tour T au tour T+1 du même bot), ce terme
    # capture explicitement le bénéfice d'un swap au Valet (la bille adverse
    # part vers ma case de départ, souvent un gros recul).
    for color in ALL_COLORS:
        if color == my_color:
            continue
        prev_opp = _get_marbles(prev_gs, color)
        curr_opp = _get_marbles(curr_gs, color)
        for prev_p, curr_p in zip(prev_opp, curr_opp):
            delta = marble_progress(curr_p, color) - marble_progress(prev_p, color)
            if delta < 0:
                reward += abs(delta) * OPP_PROGRESS_COEF

    # ── Adversaires entrant en zone d'arrivée ────────────────────────────────
    for color in ALL_COLORS:
        if color == my_color:
            continue
        arrival   = ARRIVAL_POSITIONS[color]
        prev_opp  = _get_marbles(prev_gs, color)
        curr_opp  = _get_marbles(curr_gs, color)
        n_already = sum(1 for p in prev_opp if p in arrival)
        for prev_p, curr_p in zip(prev_opp, curr_opp):
            if prev_p not in arrival and curr_p in arrival:
                reward -= OPP_ARRIVAL_PENALTY + n_already * OPP_ARRIVAL_SCALE
                n_already += 1

    # ── Menace : une bille à moi peut être capturée au prochain tour ─────────
    for curr_p in curr_mine:
        if _is_threatened(curr_p, curr_gs, my_color):
            reward -= THREAT_PENALTY

    return reward


def terminal_reward(winner_color: str | None, my_color: str) -> float:
    """Récompense terminale à ajouter quand la partie se termine."""
    if winner_color == my_color:
        return WIN_REWARD
    return LOSS_REWARD
