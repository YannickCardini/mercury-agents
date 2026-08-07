"""
Port fidèle de packages/shared/src/move-validator.ts (+ teams.ts) en Python.
Calcule le masque de coups légaux sans aller-retour serveur.

MODES DE JEU — le paramètre `teammate_color` (None par défaut) est le miroir exact du
champ optionnel `LegalMoveContext.teammateColor` côté TS :
  - None      → 1v3 historique, comportement bit-à-bit IDENTIQUE à l'ancien module
                (layout PER_SLOT=100 / ACTION_DIM=501).
  - une couleur → 2v2 équipes : active le Valet libre, le 7 partagé, la mise en jeu
                solidaire, et bascule sur le layout étendu PER_SLOT_2V2/ACTION_DIM_2V2.
En 2v2, l'appelant doit passer la couleur CONTRÔLÉE (cf. get_controlled_color) comme
`color`, ses positions comme `marble_positions`, et son coéquipier comme
`teammate_color` — exactement ce que fait buildLegalMoveContext dans game.ts.

IMPORTANT — invincibilité :
  Un pion est "invincible" entre son entrée (A/K sur sa case start) et son
  premier mouvement. Le serveur l'envoie dans gameState.players[].marbleInvincible
  (tableau parallèle à marblePositions). Un pion invincible :
    - bloque le chemin (avant ET arrière),
    - ne peut pas être reculé par un 4 (s'il est la source),
    - ne peut pas être échangé par un J (ni source, ni cible).
  C'est pourquoi toutes les fonctions reçoivent invincible_by_color
  (dict color -> liste de positions invincibles).
"""

MAIN_PATH = [
    9, 10, 25, 40, 55, 70, 85, 86, 87, 88, 89, 90, 105, 120, 135, 150,
    149, 148, 147, 146, 145, 160, 175, 190, 205, 220, 219, 218, 217, 216,
    201, 186, 171, 156, 141, 140, 139, 138, 137, 136, 121, 106, 91, 76,
    77, 78, 79, 80, 81, 66, 51, 36, 21, 6, 7, 8,
]

HOME_POSITIONS = {
  'red': [56, 57, 71, 72],
  'green': [161, 162, 176, 177],
  'blue': [154, 155, 169, 170],
  'orange': [49, 50, 64, 65],
}

START_POSITIONS = {
    'red':    9,
    'green':  135,
    'blue':   217,
    'orange': 91,
}

ARRIVAL_POSITIONS = {
    'red':    [38, 53, 68, 83],
    'green':  [118, 117, 116, 115],
    'blue':   [188, 173, 158, 143],
    'orange': [108, 109, 110, 111],
}

ALL_COLORS = ['red', 'green', 'blue', 'orange']


# ── Port de packages/shared/src/teams.ts (mode équipes 2v2) ──────────────────
# Équipes fixes, sans nom, déterminées par la couleur (= le siège) :
# red+blue vs green+orange (coéquipiers en diagonale).

TEAMMATE = {'red': 'blue', 'blue': 'red', 'green': 'orange', 'orange': 'green'}


def get_teammate_color(color: str) -> str:
    """Port de getTeammateColor()."""
    return TEAMMATE[color]


def same_team(a: str, b: str) -> bool:
    """Port de sameTeam() : une couleur est dans sa propre équipe."""
    return a == b or TEAMMATE[a] == b


def has_won(marble_positions: list[int], color: str) -> bool:
    """Port de hasWon() (board-config.ts) : les 4 pions en zone d'arrivée."""
    arrival = ARRIVAL_POSITIONS[color]
    return all(p in arrival for p in marble_positions)


def has_team_won(marbles_by_color: dict, color: str) -> bool:
    """Port de hasTeamWon() : les DEUX coéquipiers ont leurs 4 pions à l'abri."""
    teammate = TEAMMATE[color]
    return (has_won(marbles_by_color[color], color)
            and has_won(marbles_by_color[teammate], teammate))


def get_controlled_color(color: str, marble_positions: list[int]) -> str:
    """Port de getControlledColor() — le « switch » de fin de jeu (2v2) : un joueur
    qui a rentré ses 4 pions joue les pions de son coéquipier. Retourne la couleur
    des pions que ce joueur contrôle (en 1v3, l'appelant ne doit pas s'en servir)."""
    return TEAMMATE[color] if has_won(marble_positions, color) else color

CARD_MOVE_DISTANCE = {
    '2': 2, '3': 3, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9, '10': 10, 'Q': 12,
    # Joker joué comme déplacement : +18 (> 12, le max précédent). Cf. JOKER_MOVE_DISTANCE.
    'Joker': 18,
}

# Lookups rapides
_MAIN_PATH_IDX = {pos: idx for idx, pos in enumerate(MAIN_PATH)}
_MAIN_PATH_LEN = len(MAIN_PATH)
_ALL_ARRIVAL   = {p for positions in ARRIVAL_POSITIONS.values() for p in positions}
_ALL_HOME      = {p for positions in HOME_POSITIONS.values()    for p in positions}
_ALL_STARTS    = set(START_POSITIONS.values())

# Sentinelle pour les arguments invincible_by_color optionnels.
_EMPTY_INV = {c: [] for c in ALL_COLORS}


# ── Helpers de navigation ────────────────────────────────────────────────────

def is_on_main_path(pos: int) -> bool:
    return pos in _MAIN_PATH_IDX


def _forward(from_pos: int, steps: int) -> int | None:
    idx = _MAIN_PATH_IDX.get(from_pos)
    if idx is None:
        return None
    return MAIN_PATH[(idx + steps) % _MAIN_PATH_LEN]


def _backward(from_pos: int, steps: int) -> int | None:
    idx = _MAIN_PATH_IDX.get(from_pos)
    if idx is None:
        return None
    return MAIN_PATH[(idx - steps) % _MAIN_PATH_LEN]


def _color_at_position(pos: int, marbles_by_color: dict) -> str | None:
    """Port de colorAtPosition() : première couleur dont un pion occupe pos."""
    for color in marbles_by_color:
        if pos in marbles_by_color[color]:
            return color
    return None


def _is_invincible(pos: int, color: str, invincible_by_color: dict) -> bool:
    """Port de isInvincible() : pos est-elle dans la liste invincible de color."""
    return pos in invincible_by_color.get(color, [])


def _start_btw(from_pos: int, to_pos: int, color: str) -> bool:
    """True si la case START du joueur se trouve sur le chemin de from_pos → to_pos.
    Port de startPositionBtwFromAndTo()."""
    start = START_POSITIONS[color]
    if start == from_pos:
        return False
    if start == to_pos:
        return True
    idx = _MAIN_PATH_IDX[from_pos]
    while MAIN_PATH[idx] != to_pos:
        if MAIN_PATH[idx] == start:
            return True
        idx = (idx + 1) % _MAIN_PATH_LEN
    return False


def _path_is_clear(from_pos: int, steps: int, color: str,
                   all_marbles: list[int], marbles_by_color: dict,
                   invincible_by_color: dict) -> bool:
    """Port de pathIsClear().
    Le chemin est bloqué :
      - par sa PROPRE case de départ (la promotion est gérée en amont),
      - par tout pion INVINCIBLE rencontré (et seulement invincible).
    Un pion adverse non-invincible, même posé sur sa case start, NE bloque PAS.
    """
    from_idx = _MAIN_PATH_IDX.get(from_pos)
    if from_idx is None:
        return False
    own_start = START_POSITIONS[color]
    for i in range(1, steps + 1):
        pos = MAIN_PATH[(from_idx + i) % _MAIN_PATH_LEN]
        if pos == own_start:
            return False
        if pos in all_marbles:
            owner = _color_at_position(pos, marbles_by_color)
            if owner is not None and _is_invincible(pos, owner, invincible_by_color):
                return False
    return True


def _arrival_case(color: str, all_marbles: list[int],
                  from_pos: int, steps: int) -> int | None:
    """Port de getArrivelCaseIfCanPromote().
    Retourne la case d'arrivée cible ou None si le nombre de pas ne correspond pas.
    """
    available = [p for p in ARRIVAL_POSITIONS[color] if p not in all_marbles]
    if not available:
        return None
    start = START_POSITIONS[color]
    required = len(available) - 1          # pas dans la zone d'arrivée
    idx = _MAIN_PATH_IDX[from_pos]
    while MAIN_PATH[idx] != start:         # + pas jusqu'au start sur le chemin
        required += 1
        idx = (idx + 1) % _MAIN_PATH_LEN
    return available[-1] if required == steps else None


# ── Construction d'un coup (avant / arrière) ─────────────────────────────────

def _build_forward(card_value: str, from_pos: int, steps: int,
                   own_marbles: list[int], all_marbles: list[int],
                   color: str, marbles_by_color: dict,
                   invincible_by_color: dict) -> dict | None:
    """Port de buildMoveAction()."""
    to = _forward(from_pos, steps)
    if to is None:
        return None
    if _start_btw(from_pos, to, color):
        arrival = _arrival_case(color, all_marbles, from_pos, steps)
        if arrival is not None:
            return {'type': 'promote', 'from': from_pos, 'to': arrival}
    if to in own_marbles:
        return None
    if not _path_is_clear(from_pos, steps, color, all_marbles,
                          marbles_by_color, invincible_by_color):
        return None
    action_type = 'capture' if to in all_marbles else 'move'
    return {'type': action_type, 'from': from_pos, 'to': to}


def _build_backward(card_value: str, from_pos: int, steps: int,
                    own_marbles: list[int], all_marbles: list[int],
                    color: str, marbles_by_color: dict,
                    invincible_by_color: dict) -> dict | None:
    """Port de buildBackwardMoveAction().
    Le recul est bloqué si la destination est occupée par un pion à soi, ou si
    une case intermédiaire/destination contient un pion INVINCIBLE.
    """
    to = _backward(from_pos, steps)
    if to is None:
        return None
    if to in own_marbles:
        return None
    for i in range(1, steps + 1):
        pos = _backward(from_pos, i)
        if pos is None:
            return None
        owner = _color_at_position(pos, marbles_by_color)
        if owner is not None and _is_invincible(pos, owner, invincible_by_color):
            return None
    action_type = 'capture' if to in all_marbles else 'move'
    return {'type': action_type, 'from': from_pos, 'to': to}


def _build_forward_for_marble(card_value: str, from_pos: int, steps: int,
                              all_marbles: list[int], marbles_by_color: dict,
                              invincible_by_color: dict) -> dict | None:
    """Port de buildMoveActionForMarble() : comme _build_forward, mais pour un pion
    qui n'appartient pas forcément au joueur du contexte — on résout le PROPRIÉTAIRE
    de `from_pos` et on valide de SON point de vue (son start pour _path_is_clear/
    _start_btw, SES arrivées pour la promotion, SES pions pour le blocage
    d'atterrissage). Utilisé pour le 2e pion d'un 7 partagé avec le coéquipier (2v2)."""
    owner = _color_at_position(from_pos, marbles_by_color)
    if owner is None:
        return None
    return _build_forward(card_value, from_pos, steps,
                          list(marbles_by_color[owner]), all_marbles,
                          owner, marbles_by_color, invincible_by_color)


# ── Validation d'un coup ──────────────────────────────────────────────────────

def get_legal_action(card_value: str, marble_pos: int,
                     own_marbles: list[int], all_marbles: list[int],
                     color: str, marbles_by_color: dict,
                     invincible_by_color: dict | None = None) -> dict | None:
    """Port de getLegalAction().
    Retourne {'type', 'from', 'to'} ou None si le coup est illégal.
    (Le J est géré dans get_legal_mask car la cible du swap est apprise.)
    """
    # Robustesse : tout argument non-dict (None, ou un bool passé par erreur par un
    # appelant) est traité comme « aucun pion invincible » → ne JAMAIS crasher en
    # `.get`/`.items` sur un bool (cf. incident prod Joker).
    if not isinstance(invincible_by_color, dict):
        invincible_by_color = _EMPTY_INV

    # Garde de propriété (port de la garde ajoutée à getLegalAction) : sauf pour le
    # Valet (qui peut, en 2v2, échanger deux pions étrangers), le pion joué doit être
    # un pion contrôlé. No-op pour les appels du masque (qui itèrent déjà sur les
    # pions contrôlés) — c'est le miroir de la garde serveur sur `from`.
    if card_value != 'J' and marble_pos not in own_marbles:
        return None

    home  = HOME_POSITIONS[color]
    start = START_POSITIONS[color]

    # ── K : entrer ou avancer de 13 ───────────────────────────────────────────
    if card_value == 'K':
        if marble_pos in home:
            if start in own_marbles:
                return None
            return {'type': 'enter', 'from': marble_pos, 'to': start}
        if is_on_main_path(marble_pos):
            return _build_forward('K', marble_pos, 13, own_marbles, all_marbles,
                                  color, marbles_by_color, invincible_by_color)
        return None

    # ── A : entrer ou avancer de 1 ───────────────────────────────────────────
    if card_value == 'A':
        if marble_pos in home:
            if start in own_marbles:
                return None
            return {'type': 'enter', 'from': marble_pos, 'to': start}
        if is_on_main_path(marble_pos):
            return _build_forward('A', marble_pos, 1, own_marbles, all_marbles,
                                  color, marbles_by_color, invincible_by_color)
        return None

    # ── J : géré par get_legal_mask (cible apprise) ─────────────────────────
    if card_value == 'J':
        return None

    # ── 4 : reculer de 4 ────────────────────────────────────────────────────
    if card_value == '4':
        if not is_on_main_path(marble_pos):
            return None
        # Source invincible (fraîchement entrée, jamais bougé) → ne peut reculer.
        # NB : un pion non-invincible PEUT reculer même posé sur sa propre start.
        if _is_invincible(marble_pos, color, invincible_by_color):
            return None
        return _build_backward('4', marble_pos, 4, own_marbles, all_marbles,
                               color, marbles_by_color, invincible_by_color)

    # ── Joker : entrer un pion (comme A/K) OU avancer de 18 ─────────────────
    # (le rejeu qui suit est géré côté serveur ; le bot y réagit via un nouveau gameState)
    if card_value == 'Joker':
        if marble_pos in home:
            if start in own_marbles:
                return None
            return {'type': 'enter', 'from': marble_pos, 'to': start}
        if is_on_main_path(marble_pos):
            return _build_forward('Joker', marble_pos, 18, own_marbles, all_marbles,
                                  color, marbles_by_color, invincible_by_color)
        return None

    # ── 2,3,5,6,7,8,9,10,Q : avancer de N ───────────────────────────────────
    dist = CARD_MOVE_DISTANCE.get(card_value)
    if dist is not None and is_on_main_path(marble_pos):
        return _build_forward(card_value, marble_pos, dist, own_marbles, all_marbles,
                              color, marbles_by_color, invincible_by_color)
    return None


# ── Espace d'actions étendu ───────────────────────────────────────────────────
# Layout 1v3 (historique) : 5 slots de carte × 100 sous-actions + 1 discard = 501
#
# Sous-index (0..99) selon la valeur de la carte du slot :
#   '7' :
#     0..3   : pion unique, avance 7, marble_idx = sub
#     4..99  : split (marble_a, steps1, marble_b)
#              s' = sub - 4  ; marble_a = s'//24, steps1 = (s'%24)//4 + 1, marble_b = s'%4
#   'J' :
#     0..47  : (my_marble, target_idx) = (sub//12, sub%12)
#              target_idx indexe la liste fixe des 12 billes adverses (ALL_COLORS sauf ma couleur, 0..3)
#   autres :
#     0..3   : marble_idx = sub

PER_SLOT    = 100
N_SLOTS     = 5
DISCARD_IDX = N_SLOTS * PER_SLOT   # 500
ACTION_DIM  = DISCARD_IDX + 1      # 501

# ── Layout 2v2 DÉDIÉ (option (a) retenue) ─────────────────────────────────────
# Le 7 partagé (2e pion ∈ 8 pions de l'équipe) et le Valet libre (source ∈ 16 pions)
# débordent du layout 1v3 → layout étendu, sélectionné par teammate_color is not None.
# Le choix de la COMPLÉTUDE TOTALE (J : 16 sources, y compris ennemi↔ennemi) n'est pas
# un luxe : le serveur compte ces swaps comme coups légaux (canDiscard=false). Si le
# masque ne savait pas les exprimer alors qu'ils sont le SEUL coup légal, le fallback
# défausse serait rejeté (actionRejected) → timeout à chaque tour. Même logique pour
# l'entrée solidaire, qui a son propre index (seul coup légal quand elle s'applique).
# Le « switch » de fin de jeu, lui, est GRATUIT : les 4 slots pions deviennent ceux de
# la couleur contrôlée (l'appelant passe color=controlled), aucun changement de layout.
#
# Ordre d'ÉQUIPE des 16 pions : contrôlés 0..3, coéquipier 4..7, puis les 2 couleurs
# ennemies dans l'ordre ALL_COLORS (8..11, 12..15) — cf. _team_order().
#
# Sous-index (0..195) selon la valeur de la carte du slot :
#   '7' :
#     0..3    : pion unique (contrôlé), avance 7, marble_idx = sub
#     4..195  : split (m_a ∈ 0..3 contrôlé, s1 ∈ 1..6, m_b ∈ 0..7 équipe)
#               s' = sub - 4 ; m_a = s'//48, s1 = (s'%48)//8 + 1, m_b = s'%8
#   'J' :
#     0..191  : (source ∈ 0..15 ordre d'équipe) × (cible ∈ 0..11)
#               sub = src*12 + t ; les 12 cibles = pions des 3 couleurs ≠ couleur
#               source, dans l'ordre d'équipe
#   autres :
#     0..3    : marble_idx = sub (pion contrôlé)
#
# DISCARD_IDX_2V2 = 980 ; SOLIDAIRE_IDX = 981 (entrée solidaire forcée) ; dim = 982.

PER_SLOT_2V2    = 196
DISCARD_IDX_2V2 = N_SLOTS * PER_SLOT_2V2   # 980
SOLIDAIRE_IDX   = DISCARD_IDX_2V2 + 1      # 981
ACTION_DIM_2V2  = SOLIDAIRE_IDX + 1        # 982


def per_slot_for(teammate_color: str | None) -> int:
    return PER_SLOT if teammate_color is None else PER_SLOT_2V2


def discard_idx_for(teammate_color: str | None) -> int:
    return DISCARD_IDX if teammate_color is None else DISCARD_IDX_2V2


def action_dim_for(teammate_color: str | None) -> int:
    return ACTION_DIM if teammate_color is None else ACTION_DIM_2V2


def _team_order(color: str, teammate_color: str) -> list[str]:
    """Ordre d'équipe des couleurs : moi (contrôlé), coéquipier, puis les ennemis
    dans l'ordre ALL_COLORS. Base des index stables du layout 2v2."""
    return [color, teammate_color] + [c for c in ALL_COLORS
                                      if c not in (color, teammate_color)]


def _marble_slots(colors: list[str], marbles_by_color: dict) -> list[tuple[str, int]]:
    """Slots (couleur, position) des pions de `colors`, 4 par couleur, ordre stable.
    Position -1 = slot vide (défensif, le serveur envoie toujours 4 positions)."""
    slots: list[tuple[str, int]] = []
    for c in colors:
        positions = marbles_by_color.get(c, [])
        for i in range(4):
            slots.append((c, positions[i] if i < len(positions) else -1))
    return slots


def _opponent_marble_slots(my_color: str, marbles_by_color: dict) -> list[int]:
    """Ordre fixe des 12 billes adverses (pour un index stable du J)."""
    slots: list[int] = []
    for c in ALL_COLORS:
        if c == my_color:
            continue
        positions = marbles_by_color.get(c, [])
        for i in range(4):
            slots.append(positions[i] if i < len(positions) else -1)
    return slots


def _is_swappable(pos: int, marbles_by_color: dict, invincible_by_color: dict) -> bool:
    """Port du filtre swappableTargets de getLegalAction (cas J).
    Une cible est échangeable si elle est sur le main path, n'est PAS invincible,
    et n'est ni sur une case d'arrivée ni sur une case home.
    NB : contrairement à la source, une cible PEUT être sur une case start
    (le serveur ne filtre pas les starts pour la cible du J).
    """
    if pos < 0 or pos not in _MAIN_PATH_IDX:
        return False
    if pos in _ALL_ARRIVAL or pos in _ALL_HOME:
        return False
    owner = _color_at_position(pos, marbles_by_color)
    if owner is not None and _is_invincible(pos, owner, invincible_by_color):
        return False
    return True


def _build_7_split(marble_a_pos: int, steps1: int, marble_b_pos: int,
                   own_marbles: list[int], all_marbles: list[int],
                   color: str, marbles_by_color: dict,
                   invincible_by_color: dict,
                   teammate_color: str | None = None) -> dict | None:
    """Port de getLegalSplit7Action : pion A avance steps1, pion B avance 7-steps1.
    Le contexte du second demi-coup tient compte du déplacement du premier pion
    (et le premier pion n'est plus invincible après avoir bougé).

    2v2 (teammate_color fourni) : le PREMIER pion doit être un pion contrôlé
    (« own d'abord » — commencer par un pion du coéquipier est illégal), le SECOND
    peut être un pion contrôlé OU un pion du coéquipier, validé du point de vue de
    SON propriétaire (son start, ses arrivées). Pas d'immunité d'équipe : le second
    pion PEUT capturer un pion de sa propre équipe.
    """
    if not (1 <= steps1 <= 6):
        return None
    if marble_a_pos == marble_b_pos:
        return None
    if marble_a_pos not in _MAIN_PATH_IDX or marble_b_pos not in _MAIN_PATH_IDX:
        return None
    # Le PREMIER pion doit être contrôlé (garde de propriété serveur + règle 2v2
    # « le joueur actif bouge un de ses pions en premier »).
    if marble_a_pos not in own_marbles:
        return None
    # Le SECOND pion : contrôlé, ou pion du coéquipier en 2v2
    # (port de splitSecondMarbleCandidates).
    if marble_b_pos not in own_marbles:
        if teammate_color is None:
            return None
        if marble_b_pos not in marbles_by_color.get(teammate_color, []):
            return None

    action1 = _build_forward('7', marble_a_pos, steps1, own_marbles, all_marbles,
                             color, marbles_by_color, invincible_by_color)
    if action1 is None:
        return None
    # Un premier demi-coup PROMOTE est LÉGAL (getLegalSplit7Action ne le rejette
    # pas — c'est même le seul coup légal dans les fins de partie « 3 pions rentrés
    # + 1 à quelques cases de l'arrivée », fréquentes en 2v2 : sans lui le serveur
    # met canDiscard=false et rejette notre défausse → timeout à chaque tour).
    # Subtilité WIRE : game.ts re-dérive steps1 via MAIN_PATH.indexOf(action.to) →
    # le message sortant doit porter to = la case du MAIN PATH à steps1 pas (pas la
    # case d'arrivée) ; le serveur reconstruit le promote lui-même. D'où le champ
    # interne 'steps1', consommé par build_server_message.

    to1 = action1['to']
    # Un pion capturé par le premier demi-coup est renvoyé en réserve : il ne peut
    # pas être le second pion du split (garde `from2 === to1` du validateur TS).
    if marble_b_pos == to1:
        return None
    new_own = [to1 if p == marble_a_pos else p for p in own_marbles]
    new_all = [to1 if p == marble_a_pos else p for p in all_marbles]
    new_mbc = {
        c: [to1 if p == marble_a_pos else p for p in positions]
        for c, positions in marbles_by_color.items()
    }
    # Le premier pion a bougé → il n'est plus invincible (on le retire sans le
    # remplacer par to1, cf. ctx2 dans getLegalSplit7Action).
    new_inv = {
        c: [p for p in positions if p != marble_a_pos]
        for c, positions in invincible_by_color.items()
    }

    # `ForMarble` : si le second pion appartient au coéquipier, son mouvement est
    # validé de SON point de vue (cf. buildMoveActionForMarble). Quand le pion est
    # contrôlé, c'est strictement équivalent à l'ancien _build_forward direct.
    action2 = _build_forward_for_marble('7', marble_b_pos, 7 - steps1,
                                        new_all, new_mbc, new_inv)
    if action2 is None:
        return None

    split = {
        'type':      action1['type'],
        'from':      marble_a_pos,
        'to':        to1,
        'splitFrom': marble_b_pos,
        'splitTo':   action2['to'],
        'splitType': action2['type'],
        'steps1':    steps1,   # interne : encodage wire du `to` (cf. build_server_message)
    }
    # Propriétaire du 2e pion ≠ joueur actif (2v2) : champ interne, utile au
    # planificateur/suivi local — PAS envoyé au serveur (il recalcule et fait foi).
    owner2 = _color_at_position(marble_b_pos, marbles_by_color)
    if owner2 is not None and owner2 != color:
        split['splitMarbleColor'] = owner2
    return split


# ── Interface principale ──────────────────────────────────────────────────────

def _generate_moves(hand: list, marble_positions: list[int],
                    color: str, marbles_by_color: dict,
                    invincible_by_color: dict,
                    teammate_color: str | None,
                    mask: list, actions: list) -> None:
    """Remplit mask/actions avec les COUPS (sans fallback défausse/solidaire).
    Équivalent, sur toute la main, de findLegalMoveForCard × toutes les cibles.
    Le layout (1v3 ou 2v2) est déterminé par teammate_color."""
    all_marbles = [p for positions in marbles_by_color.values() for p in positions]
    per_slot    = per_slot_for(teammate_color)
    start       = START_POSITIONS[color]
    team_mode   = teammate_color is not None

    if team_mode:
        # Slots stables du layout 2v2 (ordre d'équipe, cf. bloc de doc ci-dessus).
        order        = _team_order(color, teammate_color)
        split_b_slots = _marble_slots([color, teammate_color], marbles_by_color)  # 8
        j_src_slots   = _marble_slots(order, marbles_by_color)                    # 16
        j_tgt_slots   = {
            c: _marble_slots([o for o in order if o != c], marbles_by_color)     # 12
            for c in order
        }
    else:
        split_b_slots = [(color, p) for p in marble_positions]                    # 4
        opp_slots     = _opponent_marble_slots(color, marbles_by_color)           # 12

    for card_idx, card in enumerate(hand):
        if card_idx >= N_SLOTS:
            break
        card_value = card['value']
        base       = card_idx * per_slot

        if card_value == '7':
            # Pion unique (sub 0..3)
            for m_idx, pos in enumerate(marble_positions):
                action = get_legal_action(card_value, pos, marble_positions,
                                          all_marbles, color, marbles_by_color,
                                          invincible_by_color)
                if action is not None:
                    i = base + m_idx
                    mask[i]    = True
                    actions[i] = (card, action)

            # Split — PAS de canonicalisation : un index "miroir"
            # (m_b, 7-s1, m_a) peut produire une action DIFFÉRENTE (type/to)
            # à cause du contexte mis à jour entre les deux demi-coups.
            # 1v3 : sub 4..99, m_b ∈ mes 4 pions. 2v2 : sub 4..195, m_b ∈ les
            # 8 pions de l'équipe (contrôlés 0..3, coéquipier 4..7).
            n_b = len(split_b_slots)
            for m_a in range(len(marble_positions)):
                pos_a = marble_positions[m_a]
                for s1 in range(1, 7):
                    for m_b, (_, pos_b) in enumerate(split_b_slots):
                        if pos_a == pos_b or pos_b < 0:
                            continue
                        split = _build_7_split(pos_a, s1, pos_b, marble_positions,
                                               all_marbles, color, marbles_by_color,
                                               invincible_by_color,
                                               teammate_color=teammate_color)
                        if split is not None:
                            sub = 4 + (m_a * 6 + (s1 - 1)) * n_b + m_b
                            i   = base + sub
                            mask[i]    = True
                            actions[i] = (card, split)

        elif card_value == 'J':
            if team_mode:
                # 2v2 : échange N'IMPORTE QUELLE paire de pions de couleurs
                # différentes — la source n'a plus besoin d'appartenir au joueur
                # actif (port du cas J de getLegalAction avec teammateColor).
                # Source : sur le main path, pas invincible. Cible : swappable
                # (main path, ni home ni arrivée, pas invincible) et couleur ≠
                # couleur source (garantie par construction des slots).
                for s_idx, (src_color, pos) in enumerate(j_src_slots):
                    if pos < 0 or pos not in _MAIN_PATH_IDX:
                        continue
                    if _is_invincible(pos, src_color, invincible_by_color):
                        continue
                    for t_idx, (_, target_pos) in enumerate(j_tgt_slots[src_color]):
                        if not _is_swappable(target_pos, marbles_by_color,
                                             invincible_by_color):
                            continue
                        sub = s_idx * 12 + t_idx
                        i   = base + sub
                        mask[i]    = True
                        actions[i] = (card, {
                            'type': 'swap', 'from': pos, 'to': target_pos,
                            'marbleColor': src_color,
                        })
            else:
                # 1v3 : (my_marble, target_idx) — choix appris.
                # Source : sur le main path, pas sur sa propre start, pas invincible.
                for m_idx, pos in enumerate(marble_positions):
                    if pos not in _MAIN_PATH_IDX or pos == start:
                        continue
                    if _is_invincible(pos, color, invincible_by_color):
                        continue
                    for t_idx, target_pos in enumerate(opp_slots):
                        if not _is_swappable(target_pos, marbles_by_color,
                                             invincible_by_color):
                            continue
                        sub = m_idx * 12 + t_idx
                        i   = base + sub
                        mask[i]    = True
                        actions[i] = (card, {
                            'type': 'swap', 'from': pos, 'to': target_pos,
                        })

        else:
            # Carte régulière (sub 0..3)
            for m_idx, pos in enumerate(marble_positions):
                action = get_legal_action(card_value, pos, marble_positions,
                                          all_marbles, color, marbles_by_color,
                                          invincible_by_color)
                if action is not None:
                    i = base + m_idx
                    mask[i]    = True
                    actions[i] = (card, action)


# Priorité de la carte sacrifiée pour l'entrée solidaire (port de
# SOLIDAIRE_CARD_PRIORITY) : le Roi d'abord, puis l'As, le Joker en dernier recours.
# (Cette fonction ne se déclenche que main ENTIÈREMENT bloquée : l'option d'avance de
# 13 du Roi est alors, elle aussi, nécessairement bloquée — l'ordre reste valable.)
_SOLIDAIRE_CARD_PRIORITY = ('K', 'A', 'Joker')


def find_solidaire_entry(hand: list, color: str, marbles_by_color: dict,
                         invincible_by_color: dict | None = None,
                         teammate_color: str | None = None,
                         hand_blocked: bool | None = None) -> tuple | None:
    """Port de findSolidaireEntry() — mise en jeu solidaire (2v2, OBLIGATOIRE) :
    quand la main du joueur actif est complètement bloquée mais qu'il possède un
    A/K/Joker, que son coéquipier a ≥ 1 pion en réserve ET que la case de départ du
    coéquipier est TOTALEMENT libre (aucun pion d'AUCUNE couleur — plus strict que
    l'entrée normale, qui capture), le joueur DOIT faire entrer un pion du coéquipier
    au lieu de se défausser (le serveur envoie canDiscard=false et REJETTE la
    défausse). Retourne (card, action_enter) ou None si l'exception ne s'applique pas.

    hand_blocked : si l'appelant sait déjà que la main est bloquée (True) ou non
    (False), évite de régénérer le masque. None = vérifier ici (port de isHandBlocked).

    Après le switch de fin de jeu, le teammate_color du contexte désigne le joueur
    fini : il n'a plus de pion en réserve → l'exception ne se re-déclenche jamais."""
    if teammate_color is None:
        return None
    if not isinstance(invincible_by_color, dict):
        invincible_by_color = _EMPTY_INV

    card = next((c for value in _SOLIDAIRE_CARD_PRIORITY
                 for c in hand if c['value'] == value), None)
    if card is None:
        return None

    teammate_home = HOME_POSITIONS[teammate_color]
    reserve_pos = next((p for p in marbles_by_color.get(teammate_color, [])
                        if p in teammate_home), None)
    if reserve_pos is None:
        return None

    start_pos = START_POSITIONS[teammate_color]
    if any(start_pos in positions for positions in marbles_by_color.values()):
        return None

    # isHandBlocked : aucun coup légal avec TOUTE la main (vérifié en dernier,
    # comme côté TS, car c'est le test le plus coûteux).
    if hand_blocked is None:
        probe_mask: list = [False] * ACTION_DIM_2V2
        probe_actions: list = [None] * ACTION_DIM_2V2
        _generate_moves(hand, marbles_by_color.get(color, []), color,
                        marbles_by_color, invincible_by_color, teammate_color,
                        probe_mask, probe_actions)
        hand_blocked = not any(probe_mask)
    if not hand_blocked:
        return None

    action = {'type': 'enter', 'from': reserve_pos, 'to': start_pos,
              'marbleColor': teammate_color}
    return (card, action)


def get_legal_mask(hand: list, marble_positions: list[int],
                   color: str, marbles_by_color: dict,
                   invincible_by_color: dict | None = None,
                   can_discard: bool = False,
                   teammate_color: str | None = None
                   ) -> tuple[list[bool], list]:
    """Retourne (mask, actions) :
      mask    : liste bool de longueur ACTION_DIM (1v3) ou ACTION_DIM_2V2 (2v2)
      actions : liste parallèle (None, 'discard', ou (card, action_dict))

    invincible_by_color : dict color -> [positions invincibles], reconstruit à
      partir de gameState.players[].marbleInvincible (parallèle à marblePositions).
      Si None, suppose aucun pion invincible (déconseillé : peut diverger du serveur).

    teammate_color : None = 1v3 (comportement bit-à-bit identique à l'historique).
      Sinon, mode 2v2 : `color`/`marble_positions` doivent être la couleur CONTRÔLÉE
      et ses positions (cf. get_controlled_color — le « switch » de fin de jeu),
      teammate_color son coéquipier. Active le Valet libre, le 7 partagé et l'entrée
      solidaire (index SOLIDAIRE_IDX, seul coup légal quand elle s'impose).
    """
    # Robustesse : tout non-dict (None ou bool passé par erreur) → aucun invincible.
    if not isinstance(invincible_by_color, dict):
        invincible_by_color = _EMPTY_INV

    discard_idx = discard_idx_for(teammate_color)
    dim         = action_dim_for(teammate_color)
    mask:    list = [False] * dim
    actions: list = [None] * dim

    if can_discard:
        mask[discard_idx]    = True
        actions[discard_idx] = 'discard'
        return mask, actions

    _generate_moves(hand, marble_positions, color, marbles_by_color,
                    invincible_by_color, teammate_color, mask, actions)

    if not any(mask):
        # 2v2 : l'entrée solidaire, quand elle s'applique, est OBLIGATOIRE et
        # remplace la défausse (le serveur rejetterait un discard → timeout).
        # hand_blocked=True : le masque tout juste généré est vide, inutile de refaire.
        solidaire = find_solidaire_entry(hand, color, marbles_by_color,
                                         invincible_by_color, teammate_color,
                                         hand_blocked=True)
        if solidaire is not None:
            mask[SOLIDAIRE_IDX]    = True
            actions[SOLIDAIRE_IDX] = solidaire
        else:
            mask[discard_idx]    = True
            actions[discard_idx] = 'discard'

    return mask, actions


def build_server_message(action_idx: int, hand: list, marble_positions: list[int],
                         color: str, marbles_by_color: dict,
                         invincible_by_color: dict | None = None,
                         teammate_color: str | None = None) -> dict:
    """Convertit un index d'action en message WebSocket pour le serveur.

    En 2v2 (teammate_color fourni), `color`/`marble_positions` = couleur CONTRÔLÉE
    et ses positions (comme pour get_legal_mask). Les champs internes marbleColor/
    splitMarbleColor ne sont PAS envoyés : le serveur recalcule et fait autorité
    (il valide card + from (+ to pour le J, splitFrom pour le 7))."""
    discard_idx = discard_idx_for(teammate_color)
    dim         = action_dim_for(teammate_color)

    if action_idx == discard_idx:
        return {
            'type': 'playAction',
            'action': {
                'type': 'discard',
                'from': 0,
                'to': 0,
                'cardPlayed': hand,
                'playerColor': color,
            },
        }

    mask, actions = get_legal_mask(hand, marble_positions, color, marbles_by_color,
                                   invincible_by_color, False,
                                   teammate_color=teammate_color)
    entry = actions[action_idx] if 0 <= action_idx < dim else None
    if entry is None or entry == 'discard' or not mask[action_idx]:
        return build_server_message(discard_idx, hand, marble_positions, color,
                                    marbles_by_color, invincible_by_color,
                                    teammate_color=teammate_color)

    card, action = entry
    server_action = {
        'type':        action['type'],
        'from':        action['from'],
        'to':          action['to'],
        'cardPlayed':  [card],
        'playerColor': color,
    }
    if 'splitFrom' in action:
        server_action['splitFrom'] = action['splitFrom']
        server_action['splitTo']   = action['splitTo']
        server_action['splitType'] = action['splitType']
        # Split de 7 : le serveur re-dérive steps1 via MAIN_PATH.indexOf(to) → `to`
        # doit être la case du MAIN PATH à steps1 pas, PAS la case d'arrivée quand le
        # 1er demi-coup est un promote (le serveur reconstruit le promote lui-même).
        # Pour un demi-coup move/capture, _forward(from, steps1) == to : inchangé.
        if 'steps1' in action:
            server_action['to'] = _forward(action['from'], action['steps1'])
    return {'type': 'playAction', 'action': server_action}
