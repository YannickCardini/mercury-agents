"""
Port fidèle de packages/shared/src/move-validator.ts en Python.
Calcule le masque de coups légaux sans aller-retour serveur.

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
    'red':    [3, 18, 33, 48],
    'green':  [13, 28, 43, 58],
    'blue':   [178, 193, 208, 223],
    'orange': [168, 183, 198, 213],
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


# ── Validation d'un coup ──────────────────────────────────────────────────────

def get_legal_action(card_value: str, marble_pos: int,
                     own_marbles: list[int], all_marbles: list[int],
                     color: str, marbles_by_color: dict,
                     invincible_by_color: dict | None = None) -> dict | None:
    """Port de getLegalAction().
    Retourne {'type', 'from', 'to'} ou None si le coup est illégal.
    (Le J est géré dans get_legal_mask car la cible du swap est apprise.)
    """
    if invincible_by_color is None:
        invincible_by_color = _EMPTY_INV

    home  = HOME_POSITIONS[color]
    start = START_POSITIONS[color]

    # ── K : entrer un pion ───────────────────────────────────────────────────
    if card_value == 'K':
        if marble_pos not in home:
            return None
        if start in own_marbles:
            return None
        return {'type': 'enter', 'from': marble_pos, 'to': start}

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
# Layout : 5 slots de carte × 100 sous-actions + 1 discard = 501
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
                   invincible_by_color: dict) -> dict | None:
    """Port de getLegalSplit7Action : pion A avance steps1, pion B avance 7-steps1.
    Le contexte du second demi-coup tient compte du déplacement du premier pion
    (et le premier pion n'est plus invincible après avoir bougé).
    """
    if not (1 <= steps1 <= 6):
        return None
    if marble_a_pos == marble_b_pos:
        return None
    if marble_a_pos not in _MAIN_PATH_IDX or marble_b_pos not in _MAIN_PATH_IDX:
        return None

    action1 = _build_forward('7', marble_a_pos, steps1, own_marbles, all_marbles,
                             color, marbles_by_color, invincible_by_color)
    if action1 is None:
        return None
    # Le serveur (game.ts) re-dérive steps1 via MAIN_PATH.indexOf(action.to) et
    # rejette si to n'est pas sur le main path → un premier demi-coup de type
    # promote est refusé. (Le SECOND demi-coup peut, lui, être un promote.)
    if action1['type'] == 'promote':
        return None

    to1 = action1['to']
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

    action2 = _build_forward('7', marble_b_pos, 7 - steps1, new_own, new_all,
                             color, new_mbc, new_inv)
    if action2 is None:
        return None

    return {
        'type':      action1['type'],
        'from':      marble_a_pos,
        'to':        to1,
        'splitFrom': marble_b_pos,
        'splitTo':   action2['to'],
        'splitType': action2['type'],
    }


# ── Interface principale ──────────────────────────────────────────────────────

def get_legal_mask(hand: list, marble_positions: list[int],
                   color: str, marbles_by_color: dict,
                   invincible_by_color: dict | None = None,
                   can_discard: bool = False
                   ) -> tuple[list[bool], list]:
    """Retourne (mask, actions) :
      mask    : liste bool de longueur ACTION_DIM (True = légal)
      actions : liste de ACTION_DIM entrées (None, 'discard', ou (card, action_dict))

    invincible_by_color : dict color -> [positions invincibles], reconstruit à
      partir de gameState.players[].marbleInvincible (parallèle à marblePositions).
      Si None, suppose aucun pion invincible (déconseillé : peut diverger du serveur).
    """
    if invincible_by_color is None:
        invincible_by_color = _EMPTY_INV

    all_marbles = [p for positions in marbles_by_color.values() for p in positions]
    mask    = [False] * ACTION_DIM
    actions: list = [None] * ACTION_DIM

    if can_discard:
        mask[DISCARD_IDX]    = True
        actions[DISCARD_IDX] = 'discard'
        return mask, actions

    opp_slots = _opponent_marble_slots(color, marbles_by_color)
    start     = START_POSITIONS[color]

    for card_idx, card in enumerate(hand):
        if card_idx >= N_SLOTS:
            break
        card_value = card['value']
        base       = card_idx * PER_SLOT

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

            # Split (sub 4..99) — PAS de canonicalisation : un index "miroir"
            # (m_b, 7-s1, m_a) peut produire une action DIFFÉRENTE (type/to)
            # à cause du contexte mis à jour entre les deux demi-coups.
            for m_a in range(len(marble_positions)):
                pos_a = marble_positions[m_a]
                for s1 in range(1, 7):
                    for m_b in range(len(marble_positions)):
                        if m_a == m_b:
                            continue
                        pos_b = marble_positions[m_b]
                        split = _build_7_split(pos_a, s1, pos_b, marble_positions,
                                               all_marbles, color, marbles_by_color,
                                               invincible_by_color)
                        if split is not None:
                            sub = 4 + m_a * 24 + (s1 - 1) * 4 + m_b
                            i   = base + sub
                            mask[i]    = True
                            actions[i] = (card, split)

        elif card_value == 'J':
            # (my_marble, target_idx) — choix appris.
            # Source : sur le main path, pas sur sa propre start, pas invincible.
            for m_idx, pos in enumerate(marble_positions):
                if pos not in _MAIN_PATH_IDX or pos == start:
                    continue
                if _is_invincible(pos, color, invincible_by_color):
                    continue
                for t_idx, target_pos in enumerate(opp_slots):
                    if not _is_swappable(target_pos, marbles_by_color, invincible_by_color):
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

    if not any(mask):
        mask[DISCARD_IDX]    = True
        actions[DISCARD_IDX] = 'discard'

    return mask, actions


def build_server_message(action_idx: int, hand: list, marble_positions: list[int],
                         color: str, marbles_by_color: dict,
                         invincible_by_color: dict | None = None) -> dict:
    """Convertit un index d'action en message WebSocket pour le serveur."""
    if action_idx == DISCARD_IDX:
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
                                   invincible_by_color, False)
    entry = actions[action_idx] if 0 <= action_idx < ACTION_DIM else None
    if not mask[action_idx] or entry is None or entry == 'discard':
        return build_server_message(DISCARD_IDX, hand, marble_positions, color,
                                    marbles_by_color, invincible_by_color)

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
    return {'type': 'playAction', 'action': server_action}
