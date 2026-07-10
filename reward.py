"""
Récompenses denses pour le bot RL Mercury.
La récompense est calculée entre deux états successifs du même joueur
(i.e. entre deux tours du même bot, couvrant aussi les tours adverses).
"""

from mercury_legal_moves import (
    HOME_POSITIONS, START_POSITIONS, ARRIVAL_POSITIONS,
    _MAIN_PATH_IDX, _MAIN_PATH_LEN, _ALL_STARTS, ALL_COLORS,
)

# ── Coefficients du shaping ───────────────────────────────────────────────────
# ÉCHELLE ÷10 (vs la version précédente) : les retours s'étalent désormais sur ~±3
# au lieu de ~±30, ce qui rend la value loss du critique FITTABLE (le v_loss restait
# bloqué ~45 avec WIN=±30). La HIÉRARCHIE est strictement préservée — c'est l'ordre
# relatif des récompenses qui guide la politique, pas leur valeur absolue.
#     victoire ≫ arrivée > entrée > recul/capture adverse > progrès case-par-case
OWN_PROGRESS_COEF    = 0.06  # gain/perte de progrès sur mes billes (plus petit signal)
OPP_PROGRESS_COEF    = 0.1   # progrès perdu par un adversaire (capture + swap + 4)
OPP_ARRIVAL_PENALTY  = 0.015 # malus fixe par bille adverse entrant en zone d'arrivée
OPP_ARRIVAL_SCALE    = 0.01  # malus additionnel par bille déjà arrivée chez l'adversaire
THREAT_PENALTY       = 0.005 # malus si une bille à moi est à portée d'un adversaire
THREAT_RANGE         = 6     # distance (en cases) considérée comme menaçante — INCHANGÉ
# Capture subie (bille renvoyée au home) : le mode d'échec #1 diagnostiqué (~20/partie !).
# Avant, une bille capturée à mi-parcours ne coûtait que Δprogress×OWN_PROGRESS_COEF (~0.03)
# → le bot n'apprenait jamais à protéger ses billes (parties de ~200 coups, win vs random
# qui régressait de 0.82 à 0.62). On rend la capture coûteuse, proportionnellement au
# progrès perdu. Asymétrie VOULUE : se protéger (≈ −0.25) pèse plus que capturer un
# adversaire (+OPP_PROGRESS_COEF×progrès) — sa propre progression vaut plus que de gêner.
CAPTURE_PENALTY      = 0.10  # malus fixe : se faire renvoyer au home est mauvais en soi
CAPTURE_LOSS_COEF    = 0.20  # + proportionnel au progrès perdu (bille avancée = pire)
# Bonus d'entrée : faire entrer une bille est rare et débloquant (sans bille en jeu on
# passe son tour). Reste largement au-dessus de "advance-1" (≈0.0017 à cette échelle).
ENTRY_BONUS          = 0.1
# Bonus d'arrivée : 2ème plus grosse récompense après la victoire (4 arrivées = victoire).
ARRIVAL_BONUS        = 0.3

# ── Mode équipes 2v2 (teammate_color fourni à compute_reward) ─────────────────
# Le coéquipier SORT des termes « adversaire » (son progrès perdu ne rapporte plus,
# ses arrivées ne pénalisent plus) et son progrès devient POSITIF : l'équipe gagne
# quand les DEUX coéquipiers finissent. Coefs à ~moitié des « own » : ma propre course
# reste prioritaire (je ne contrôle pas ses cartes), mais capturer/faire capturer un
# pion allié coûte (delta de progrès négatif × ce coef — y compris si c'est MOI qui
# capture un pion allié via 7-split/atterrissage, le delta le pénalise mécaniquement).
TEAMMATE_PROGRESS_COEF = 0.03   # progrès (±) des billes du coéquipier
TEAMMATE_ARRIVAL_BONUS = 0.15   # bille du coéquipier entrant en zone d'arrivée

# ── Calibration WIN/LOSS ──────────────────────────────────────────────────────
# IMPORTANT — la 4ᵉ arrivée DÉCLENCHE la victoire : l'événement "arriver sa dernière
# bille" est donc payé deux fois (ARRIVAL_BONUS au dernier flush + WIN_REWARD terminal).
# WIN doit dominer la somme MAXIMALE des récompenses denses d'une partie gagnante :
#     4×ENTRY (0.4) + 4×ARRIVAL (1.2) + progrès own max (~0.24) + captures (~0.1-0.3)
#     ≈ +1.8 à +2.0
# On fixe WIN = +3 (≈ 1.5× le plafond dense) → strictement dominant, et retours ~±3
# (échelle confortable pour le critique). LOSS symétrique pour éviter le biais positif.
WIN_REWARD           = 3.0
LOSS_REWARD          = -3.0


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


def _is_threatened(my_pos: int, curr_gs: dict, my_color: str,
                   teammate_color: str | None = None) -> bool:
    """True si une bille adverse est à ≤ THREAT_RANGE cases derrière ma bille (sur le
    main path). En 2v2, le coéquipier n'est pas une menace (il PEUT me capturer mais
    ne le fera pas volontairement — approximation raisonnable)."""
    if my_pos not in _MAIN_PATH_IDX or my_pos in _ALL_STARTS:
        return False
    my_idx = _MAIN_PATH_IDX[my_pos]
    for color in ALL_COLORS:
        if color == my_color or color == teammate_color:
            continue
        for opp_p in _get_marbles(curr_gs, color):
            if opp_p in _MAIN_PATH_IDX:
                dist_behind = (my_idx - _MAIN_PATH_IDX[opp_p]) % _MAIN_PATH_LEN
                if 1 <= dist_behind <= THREAT_RANGE:
                    return True
    return False


def compute_reward(prev_gs: dict, curr_gs: dict, my_color: str,
                   teammate_color: str | None = None) -> float:
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

    2v2 (teammate_color fourni) : le coéquipier est un ALLIÉ — il sort des deux termes
    « adversaire » ci-dessus ; son progrès (±, y compris quand JE capture un de ses
    pions) est crédité via TEAMMATE_PROGRESS_COEF, ses arrivées via
    TEAMMATE_ARRIVAL_BONUS, et il n'est plus compté comme menace.
    """
    reward = 0.0

    prev_mine = _get_marbles(prev_gs, my_color)
    curr_mine = _get_marbles(curr_gs, my_color)
    my_arrival = ARRIVAL_POSITIONS[my_color]

    # ── Progrès de mes billes (positif = avance, négatif = capturé/swap arrière)
    for prev_p, curr_p in zip(prev_mine, curr_mine):
        delta = marble_progress(curr_p, my_color) - marble_progress(prev_p, my_color)
        reward += delta * OWN_PROGRESS_COEF
        # Capture subie : ma bille passe du plateau au home. Impossible par mon propre coup
        # (promote → arrivée, enter quitte le home) → c'est forcément une capture adverse.
        # Pénalité fixe + proportionnelle au progrès perdu (le mode d'échec #1 diagnostiqué).
        if prev_p not in HOME_POSITIONS[my_color] and curr_p in HOME_POSITIONS[my_color]:
            reward -= CAPTURE_PENALTY + marble_progress(prev_p, my_color) * CAPTURE_LOSS_COEF
        # Bonus d'entrée : sans cela le signal immédiat pour "enter" (≈0.075)
        # ne domine pas "advance-1" (≈0.017) malgré la valeur stratégique réelle.
        if prev_p in HOME_POSITIONS[my_color] and curr_p not in HOME_POSITIONS[my_color]:
            reward += ENTRY_BONUS
        # Bonus d'arrivée : la bille passe du plateau à ma zone d'arrivée.
        # Étape stratégique majeure → 2ème plus grosse récompense après la victoire.
        if prev_p not in my_arrival and curr_p in my_arrival:
            reward += ARRIVAL_BONUS

    # ── Progrès du coéquipier (2v2 uniquement) : allié, signal positif dédié ──
    if teammate_color is not None:
        tm_arrival = ARRIVAL_POSITIONS[teammate_color]
        prev_tm    = _get_marbles(prev_gs, teammate_color)
        curr_tm    = _get_marbles(curr_gs, teammate_color)
        for prev_p, curr_p in zip(prev_tm, curr_tm):
            delta = (marble_progress(curr_p, teammate_color)
                     - marble_progress(prev_p, teammate_color))
            reward += delta * TEAMMATE_PROGRESS_COEF
            if prev_p not in tm_arrival and curr_p in tm_arrival:
                reward += TEAMMATE_ARRIVAL_BONUS

    # ── Progrès perdu par les adversaires (capture + swap + card-4 adverse) ──
    # En sémantique semi-MDP (du tour T au tour T+1 du même bot), ce terme
    # capture explicitement le bénéfice d'un swap au Valet (la bille adverse
    # part vers ma case de départ, souvent un gros recul).
    # 2v2 : le coéquipier est exclu — sa perte de progrès ne rapporte RIEN
    # (elle coûte, via TEAMMATE_PROGRESS_COEF ci-dessus).
    for color in ALL_COLORS:
        if color == my_color or color == teammate_color:
            continue
        prev_opp = _get_marbles(prev_gs, color)
        curr_opp = _get_marbles(curr_gs, color)
        for prev_p, curr_p in zip(prev_opp, curr_opp):
            delta = marble_progress(curr_p, color) - marble_progress(prev_p, color)
            if delta < 0:
                reward += abs(delta) * OPP_PROGRESS_COEF

    # ── Adversaires entrant en zone d'arrivée ────────────────────────────────
    # 2v2 : les arrivées du coéquipier ne sont plus pénalisées (elles sont
    # récompensées plus haut — l'équipe gagne quand les DEUX finissent).
    for color in ALL_COLORS:
        if color == my_color or color == teammate_color:
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
        if _is_threatened(curr_p, curr_gs, my_color, teammate_color):
            reward -= THREAT_PENALTY

    return reward


def terminal_reward(winners, my_color: str) -> float:
    """Récompense terminale à ajouter quand la partie se termine.

    `winners` : liste de couleurs gagnantes (nouveau protocole `gameEnded.winners` —
    1 couleur en 1v3, les 2 couleurs de l'équipe gagnante en 2v2, [] si abandon).
    « J'ai gagné » = my_color ∈ winners : en 2v2 les DEUX coéquipiers touchent
    WIN_REWARD. Rétro-compat : accepte aussi une couleur seule ou None."""
    if winners is None:
        return LOSS_REWARD
    if isinstance(winners, str):
        winners = [winners]
    return WIN_REWARD if my_color in winners else LOSS_REWARD
