"""
Planificateur de main pour Mercury — la pièce qui manquait : la PLANIFICATION.

Au lieu de choisir UNE carte au coup par coup (ce que fait le réseau réactif), on
recherche la meilleure SÉQUENCE de cartes de la main pour atteindre un objectif —
exactement le raisonnement humain : « il me faut 18 pour promote → 9+9, ou 9+6+3 ?
le chemin est-il libre ? ». On joue ensuite la PREMIÈRE carte du meilleur plan, et on
replanifie au tour suivant (horizon fuyant).

Approximation volontaire (= un humain) : les adversaires sont des obstacles FIXES entre
mes tours. On applique seulement MES coups, en réutilisant mercury_legal_moves (donc les
règles exactes : distances, blocages, promotions, captures, swaps, 7-split).

Profondeur bornée par la main (≤5 cartes) → recherche en faisceau tractable.
"""

from mercury_legal_moves import (
    get_legal_mask, ARRIVAL_POSITIONS, START_POSITIONS, HOME_POSITIONS,
    _MAIN_PATH_IDX, _MAIN_PATH_LEN, PER_SLOT, DISCARD_IDX, N_SLOTS,
    per_slot_for, discard_idx_for,
)
import time

from game_state_utils import invincible_by_color as _invincible_by_color
from reward import marble_progress

BEAM_WIDTH   = 48    # plans gardés à chaque profondeur (élargi 24→48 : recherche + fine)
THREAT_RANGE = 7     # bille adverse à ≤ N cases derrière = menace de capture
# Budget de temps DUR : en prod le planificateur (Python pur) bloque la boucle asyncio
# partagée par tous les bots. On borne la recherche pour ne jamais geler la boucle trop
# longtemps. Au pire on renvoie le meilleur plan trouvé jusque-là (≥ un plan à 1 coup).
MAX_PLAN_SECONDS = 0.08

# Poids du score d'un état de fin de plan (objectifs « humains »).
W_PROGRESS = 100.0   # somme des progrès de mes billes
W_ARRIVAL  = 50.0    # bonus : bille promue (en zone d'arrivée, sûre)
W_CAPTURE  = 120.0   # déni accumulé par capture (× progrès de la cible)
# Une capture prévue pour un de mes tours SUIVANTS est incertaine (la cible aura
# peut-être bougé) : décotée, pour qu'une capture jouable tout de suite ne soit jamais
# remise à plus tard au profit d'un coup sans urgence (ex. dégager un pieu).
W_CAPTURE_LATER = 0.75
# 2v2 : le coéquipier est un ALLIÉ. Son progrès compte dans le score de fin de plan
# (l'équipe gagne quand les DEUX finissent), pondéré à 0.5×W_PROGRESS : ma propre
# course reste prioritaire (je ne contrôle pas ses cartes). Capturer un pion du
# coéquipier n'est PAS crédité W_CAPTURE : son progrès disparaît du score → malus
# mécanique de W_PROGRESS_TEAMMATE × progrès perdu.
W_PROGRESS_TEAMMATE = 0.5 * W_PROGRESS
W_ARRIVAL_TEAMMATE  = 0.5 * W_ARRIVAL

# Préférence produit : le Roi doit privilégier faire entrer un pion ou promouvoir,
# plutôt qu'un simple déplacement de 13 en milieu de plateau. Biais SOUPLE (pas un
# filtre) : entrer ≈ W_PROGRESS*0.05≈5, avancer de 13 en milieu de plateau
# ≈13/56*0.65*W_PROGRESS≈15 — le bonus fait pencher vers l'entrée dans le cas
# générique sans jamais rivaliser avec un vrai gain de capture (W_CAPTURE=120).
W_KING_ENTER_BONUS   = 20.0
W_KING_PROMOTE_BONUS = 15.0   # déjà dominé par W_ARRIVAL ; renforce la consigne produit
# Une carte d'entrée est rare : l'As sert d'abord à faire entrer un pion. Sans ce
# biais, le moindre malus de barrage (ci-dessous) suffirait à « brûler » l'As pour
# avancer d'UNE case (entrer ≈ 5, avancer de 1 ≈ 1) — bien pire que gêner le coéquipier
# un tour. Calé pour donner à l'As la même marge que le Roi (≈ 10) face au barrage.
W_ACE_ENTER_BONUS    = 8.0

# ── Barrage d'une bille invincible ────────────────────────────────────────────
# Une bille qui vient d'entrer reste INVINCIBLE sur sa case de départ tant qu'elle n'a
# pas bougé : personne ne peut la dépasser ni s'y poser. Elle barre donc la route aux
# pions arrêtés juste derrière — ceux des adversaires (tant mieux)… et ceux du
# coéquipier. Les poids valent pour un pion complètement bloqué (collé au pieu) ;
# cf. _barrage pour la pondération par la distance.
#   - Coéquipier simplement gêné (il a d'autres pions à jouer) : malus < marge
#     d'entrée du Roi/de l'As → on entre quand même, mais on RETARDE l'entrée ou on
#     DÉGAGE la bille au coup suivant.
#   - Coéquipier coincé (tous ses pions en jeu sont derrière le pieu) : malus ×2,
#     assez pour renoncer à l'entrée (il risquerait de défausser toute sa main).
#   - Adversaires barrés : bonus → bloquer vaut parfois le coup, même en gênant
#     un peu le coéquipier.
W_BLOCK_ALLY    = 6.0
W_BLOCK_ENEMY   = 4.0
W_BLOCK_STUCK   = 2.0   # × si TOUS les pions en jeu de la couleur sont barrés
# Le barrage compte deux fois : à la fin de MON tour (certain : c'est ce que les
# autres subissent tout de suite) et en fin de plan (bille laissée sur place — plus
# long mais spéculatif, d'où le facteur).
W_BLOCK_END_FACTOR = 0.5


# ── Évaluation d'un état (mes billes) ─────────────────────────────────────────

def _threatened(pos: int, color: str, mbc: dict, teammate: str | None = None) -> bool:
    """v1 (grossier) : un adversaire est à ≤ THREAT_RANGE cases derrière.
    2v2 : le coéquipier n'est pas une menace (il peut me capturer mais ne le fera
    pas volontairement — approximation raisonnable)."""
    if pos not in _MAIN_PATH_IDX or pos == START_POSITIONS[color]:
        return False
    idx = _MAIN_PATH_IDX[pos]
    for c, positions in mbc.items():
        if c == color or c == teammate:
            continue
        for op in positions:
            if op in _MAIN_PATH_IDX:
                d = (idx - _MAIN_PATH_IDX[op]) % _MAIN_PATH_LEN
                if 1 <= d <= THREAT_RANGE:
                    return True
    return False


# Distances de capture EXACTES : un adversaire me prend s'il a une bille à exactement la
# distance d'une carte derrière moi (A,2,3,5,6,7,8,9,10,Q) — ou 4 cases DEVANT (carte 4 arrière).
_FWD_CAPTURE_DISTS = frozenset({1, 2, 3, 5, 6, 7, 8, 9, 10, 12})


def _capturable(pos: int, color: str, mbc: dict, teammate: str | None = None) -> bool:
    """v2 (exact) : une bille adverse peut-elle réellement atterrir sur `pos` au prochain
    tour ? (distance = celle d'une carte légale). Bien plus fidèle que « à ≤7 cases ».
    2v2 : le coéquipier est exclu (cf. _threatened)."""
    if pos not in _MAIN_PATH_IDX or pos == START_POSITIONS[color]:
        return False
    pi = _MAIN_PATH_IDX[pos]
    for c, positions in mbc.items():
        if c == color or c == teammate:
            continue
        for q in positions:
            qi = _MAIN_PATH_IDX.get(q)
            if qi is None:
                continue
            if (pi - qi) % _MAIN_PATH_LEN in _FWD_CAPTURE_DISTS:   # capture en avançant
                return True
            if (qi - pi) % _MAIN_PATH_LEN == 4:                    # capture au 4 (arrière)
                return True
    return False


def _make_score(threat_fn):
    """Évaluateur d'un état de fin de plan. `threat_fn(pos, color, mbc, teammate)` décide
    si ma bille risque d'être capturée → on suppose alors la PLUS AVANCÉE menacée capturée
    (snipe 1-ply : on ne sur-étend pas sa tête dans la zone de capture).
    v1 = _threatened, v2 = _capturable. En 2v2 (teammate fourni), le progrès des pions du
    coéquipier s'ajoute (pondéré W_PROGRESS_TEAMMATE/W_ARRIVAL_TEAMMATE)."""
    def score(my_pawns: list, color: str, mbc: dict, captured: float,
              teammate: str | None = None) -> float:
        arrival = ARRIVAL_POSITIONS[color]
        sniped, best = None, -1.0
        for i, p in enumerate(my_pawns):
            if p not in arrival and threat_fn(p, color, mbc, teammate):
                pr = marble_progress(p, color)
                if pr > best:
                    best, sniped = pr, i
        s = captured
        for i, p in enumerate(my_pawns):
            if i == sniped:
                continue                              # supposée capturée → 0 crédit
            prog = marble_progress(p, color)
            s += W_PROGRESS * prog
            if p in arrival:
                s += W_ARRIVAL
        if teammate is not None:
            tm_arrival = ARRIVAL_POSITIONS[teammate]
            for p in mbc.get(teammate, []):
                s += W_PROGRESS_TEAMMATE * marble_progress(p, teammate)
                if p in tm_arrival:
                    s += W_ARRIVAL_TEAMMATE
        return s
    return score


_score_board   = _make_score(_threatened)   # v1 (défaut, déployé)
score_board_v2 = _make_score(_capturable)   # v2 (menace exacte)


def _card_pref_bonus(card_value: str, action: dict) -> float:
    """Biais de préférence (pas de légalité) : le Roi doit privilégier entrer/promouvoir
    sur un simple déplacement de 13 en milieu de plateau, et l'As entrer plutôt
    qu'avancer d'une case — sans jamais interdire l'autre usage."""
    if card_value == 'K':
        if action['type'] == 'enter':
            return W_KING_ENTER_BONUS
        if action['type'] == 'promote':
            return W_KING_PROMOTE_BONUS
    elif card_value == 'A' and action['type'] == 'enter':
        return W_ACE_ENTER_BONUS
    return 0.0


# Cartes qui avancent un pion, par distance (A, 2, 3, 5…10, Q, K : 4 exemplaires ;
# Joker : 2). Part des cartes d'avance qu'un pion ne peut PLUS jouer quand un pieu
# se dresse `d` cases devant lui (toute distance ≥ d est barrée).
_FORWARD_CARDS = {1: 4, 2: 4, 3: 4, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4, 10: 4, 12: 4,
                  13: 4, 18: 2}
_BLOCK_RANGE   = max(_FORWARD_CARDS)
_BLOCKED_SHARE = [sum(n for dist, n in _FORWARD_CARDS.items() if dist >= d)
                  / sum(_FORWARD_CARDS.values()) for d in range(_BLOCK_RANGE + 1)]


def _barrage(inv: frozenset, color: str, mbc: dict, teammate: str | None) -> float:
    """Valeur, pour mon équipe, du barrage formé par MA bille invincible (0 si je n'en
    ai pas) : > 0 quand elle bloque surtout des adversaires, < 0 quand elle bloque
    surtout le coéquipier. Chaque pion barré pèse la part des cartes qu'il ne peut
    plus jouer (cf. _BLOCKED_SHARE) : collé au pieu = 1, à 10 cases ≈ 0.3."""
    start = START_POSITIONS[color]
    if start not in inv or start not in mbc[color]:
        return 0.0
    start_idx = _MAIN_PATH_IDX[start]
    value = 0.0
    for c, positions in mbc.items():
        if c == color:
            continue        # mes pions ne franchissent jamais ma case de départ
        own_start_idx = _MAIN_PATH_IDX[START_POSITIONS[c]]
        blocked, n_blocked, n_in_play = 0.0, 0, 0
        for p in positions:
            idx = _MAIN_PATH_IDX.get(p)
            if idx is None:
                continue
            n_in_play += 1
            d = (start_idx - idx) % _MAIN_PATH_LEN
            if not 1 <= d <= _BLOCK_RANGE:
                continue
            # Un pion qui atteint SA case de départ avant la mienne promeut : il ne
            # viendra jamais buter sur mon pieu.
            if 0 < (own_start_idx - idx) % _MAIN_PATH_LEN < d:
                continue
            blocked   += _BLOCKED_SHARE[d]
            n_blocked += 1
        if not n_blocked:
            continue
        if n_blocked == n_in_play:
            blocked *= W_BLOCK_STUCK
        value += -W_BLOCK_ALLY * blocked if c == teammate else W_BLOCK_ENEMY * blocked
    return value


# ── Application d'un de MES coups (adversaires = obstacles, capture = obstacle retiré) ──

def _apply(my_pawns: list, opp: dict, color: str, action: dict,
           teammate: str | None = None, inv: frozenset = frozenset()) -> tuple:
    """Retourne (new_my_pawns, new_opp, capture_gain, new_inv). `opp` = {color:
    [positions]} (toutes les couleurs ≠ la mienne, coéquipier inclus en 2v2) ; `inv` =
    cases occupées par une bille invincible (toutes couleurs).
    Réutilise les from/to/type déjà calculés par mercury_legal_moves, et applique le
    coup comme Game.updateMarblePositions côté serveur :
      - le pion déplacé est celui de la couleur portée par l'action (marbleColor /
        splitMarbleColor : coéquipier pour l'entrée solidaire ou le 2e pion d'un 7
        partagé), à défaut la mienne ;
      - les DEUX moitiés d'un 7 bougent avant toute capture (le 1er pion peut s'arrêter
        sur la case que le 2e quitte dans le même coup) ;
      - tout pion posé sur une case d'arrivée du coup est renvoyé au home, quel que
        soit le type de l'action (entrer sur un pion posé sur MA case de départ le
        capture) et quelle que soit sa couleur — pas d'immunité d'équipe.
    Seule la capture d'un pion ENNEMI est créditée W_CAPTURE : celle d'un pion du
    coéquipier (ou d'un des miens) coûte via le score de fin de plan, qui perd son
    progrès."""
    mp    = list(my_pawns)
    opp   = {c: list(v) for c, v in opp.items()}
    board = {color: mp, **opp}
    gain  = 0.0

    def find(pos, owner=None):
        """(couleur, index) du pion posé en `pos` — chez `owner` d'abord s'il est donné."""
        if owner in board and pos in board[owner]:
            return owner, board[owner].index(pos)
        for c, v in board.items():
            if pos in v:
                return c, v.index(pos)
        return None

    def send_victims_home(pos, mover):
        nonlocal gain
        for c, v in board.items():
            if pos not in v:
                continue
            for i, p in enumerate(v):
                if p != pos or (c, i) == mover:
                    continue
                home = next((h for h in HOME_POSITIONS[c] if h not in v), None)
                if home is None:
                    continue
                if c != color and c != teammate:
                    gain += W_CAPTURE * marble_progress(pos, c)
                v[i] = home

    if action['type'] == 'swap':
        # pion source (from) ↔ pion cible (to) : positions échangées, où que soient
        # les deux pions (Valet libre en 2v2 : la source peut être au coéquipier ou
        # à un adversaire, tant qu'un des deux pions est de mon équipe).
        # On NE récompense PAS le recul infligé à l'adversaire : le J ne doit servir
        # QUE la course de l'ÉQUIPE. Le gain réel (mon pion ou celui du coéquipier
        # saute vers l'avant, un ennemi menaçant s'éloigne de derrière nos billes)
        # est déjà capté par W_PROGRESS/W_PROGRESS_TEAMMATE et par threat_fn dans le
        # score de fin de plan → pas de « spite swaps ».
        src, tgt = find(action['from']), find(action['to'])
        if src and tgt:
            board[src[0]][src[1]] = action['to']
            board[tgt[0]][tgt[1]] = action['from']
        return mp, opp, gain, inv - {action['from'], action['to']}

    # move / capture / promote / enter (+ 7-split : 2e demi-coup)
    owner = action.get('marbleColor', color)
    mover = find(action['from'], owner)
    if mover:
        board[mover[0]][mover[1]] = action['to']
    inv = inv - {action['from']}
    if action['type'] == 'enter':
        inv = inv | {action['to']}      # invincible jusqu'à son premier mouvement

    split = None
    if 'splitTo' in action:
        split = find(action['splitFrom'], action.get('splitMarbleColor', owner))
        if split:
            board[split[0]][split[1]] = action['splitTo']
        inv = inv - {action['splitFrom']}

    send_victims_home(action['to'], mover)
    if split:
        send_victims_home(action['splitTo'], split)
    return mp, opp, gain, inv


# ── Recherche en faisceau sur la main ─────────────────────────────────────────

def _slot_of(idx: int, hand: list, card: dict, per_slot: int) -> int:
    """Slot de main consommé par l'action d'index `idx`. Cas général : idx // per_slot.
    Les index HORS layout de slot (SOLIDAIRE_IDX en 2v2) sont résolus par IDENTITÉ de
    la carte (les entrées du masque référencent les objets carte de la main)."""
    slot = idx // per_slot
    if slot < N_SLOTS:
        return slot
    for k, c in enumerate(hand):
        if c is card:
            return k
    # Repli défensif (carte reconstruite ?) : première carte de même valeur.
    return next((k for k, c in enumerate(hand) if c.get('value') == card.get('value')), 0)


def _legal_entries(hand: list, my_pawns: list, color: str, mbc: dict,
                   teammate: str | None = None, inv: frozenset = frozenset()) -> list:
    """(slot, card, action_dict) légaux pour la main restante (adversaires fixes).
    Seule l'invincibilité de MES billes est simulée (je décide quand elles bougent :
    ni 4 ni J tant qu'elles n'ont pas avancé) ; celle des autres est ignorée, ils
    auront bougé d'ici là → planification optimiste sur les coups profonds.
    Inclut l'entrée solidaire (2v2) — forcée par le serveur quand elle s'impose."""
    mask, actions = get_legal_mask(hand, my_pawns, color, mbc,
                                   invincible_by_color={
                                       color: [p for p in my_pawns if p in inv]},
                                   can_discard=False, teammate_color=teammate)
    per_slot = per_slot_for(teammate)
    out = []
    for j, ok in enumerate(mask):
        if not ok:
            continue
        e = actions[j]
        if e is None or e == 'discard':
            continue
        out.append((_slot_of(j, hand, e[0], per_slot), e[0], e[1]))
    return out


def _search(game_state: dict, color: str, mask: list, actions: list,
            score_fn=None, beam_width: int = None, topk: int = None,
            teammate_color: str | None = None):
    """Renvoie le meilleur nœud {my, opp, first_idx, score}. Si `topk` est fourni, renvoie
    plutôt la LISTE des `topk` meilleurs nœuds par score (pour un re-classement externe,
    ex. par un réseau de valeur).

    2v2 (teammate_color fourni) : `color` = couleur CONTRÔLÉE (celle du masque), le
    coéquipier est scoré comme allié et n'est plus une menace."""
    score_fn = score_fn or _score_board
    bw       = beam_width or BEAM_WIDTH
    tm       = teammate_color
    per_slot = per_slot_for(tm)
    pool     = []                          # tous les nœuds vus (pour le top-K éventuel)

    mbc0      = {p['color']: list(p['marblePositions']) for p in game_state['players']}
    my_pawns0 = mbc0[color]
    opp0      = {c: v for c, v in mbc0.items() if c != color}
    hand0     = game_state.get('hand', [])
    inv0      = frozenset(p for positions in _invincible_by_color(game_state).values()
                          for p in positions)

    def board_of(my_pawns, opp):
        """{couleur: positions} dans l'ordre des joueurs du serveur (le validateur
        résout le propriétaire d'une case en parcourant les couleurs dans cet ordre)."""
        return {c: my_pawns if c == color else opp[c] for c in mbc0}

    def make_node(parent, card, action, child_hand, first_idx):
        """Nœud obtenu en jouant `card`/`action` depuis `parent` (None = racine)."""
        if parent is None:
            parent = {'my': my_pawns0, 'opp': opp0, 'inv': inv0, 'captured': 0.0,
                      'bonus': 0.0, 'turn_block': None}
        mp, opp, gain, inv = _apply(parent['my'], parent['opp'], color, action,
                                    teammate=tm, inv=parent['inv'])
        if parent['turn_block'] is not None:      # coup d'un tour suivant
            gain *= W_CAPTURE_LATER
        mbc     = board_of(mp, opp)
        barrage = _barrage(inv, color, mbc, tm)
        # Barrage subi par les autres à la fin de MON tour : figé dès que le tour se
        # termine, encore ouvert (None) tant qu'un Joker me fait rejouer.
        turn_block = parent['turn_block']
        if turn_block is None and not (card['value'] == 'Joker' and child_hand):
            turn_block = barrage
        node = {'my': mp, 'opp': opp, 'inv': inv, 'hand': child_hand,
                'captured': parent['captured'] + gain,
                'bonus': parent['bonus'] + _card_pref_bonus(card['value'], action),
                'turn_block': turn_block, 'first_idx': first_idx}
        node['score'] = (score_fn(mp, color, mbc, node['captured'], teammate=tm)
                         + node['bonus']
                         + (barrage if turn_block is None else turn_block)
                         + W_BLOCK_END_FACTOR * barrage)
        return node

    # ── Racine : on étend les actions FOURNIES (vraie invincibilité / can_discard) ──
    beam = []
    for i, ok in enumerate(mask):
        if not ok:
            continue
        e = actions[i]
        if e is None or e == 'discard':
            continue
        slot = _slot_of(i, hand0, e[0], per_slot)
        beam.append(make_node(None, e[0], e[1], hand0[:slot] + hand0[slot + 1:], i))

    if not beam:
        return [] if topk else None
    pool.extend(beam)

    best     = max(beam, key=lambda n: n['score'])
    frontier = sorted(beam, key=lambda n: n['score'], reverse=True)[:bw]

    # ── Approfondissement : on consomme la main restante carte par carte ──
    deadline  = time.monotonic() + MAX_PLAN_SECONDS
    timed_out = False
    while frontier and not timed_out:
        nxt = []
        for node in frontier:
            if time.monotonic() > deadline:   # budget vérifié PAR nœud → blocage borné serré
                timed_out = True
                break
            if not node['hand']:
                continue
            mbc = board_of(node['my'], node['opp'])
            for slot, card, action in _legal_entries(node['hand'], node['my'],
                                                     color, mbc, teammate=tm,
                                                     inv=node['inv']):
                child = make_node(node, card, action,
                                  node['hand'][:slot] + node['hand'][slot + 1:],
                                  node['first_idx'])
                nxt.append(child)
                if child['score'] > best['score']:
                    best = child
        pool.extend(nxt)
        frontier = sorted(nxt, key=lambda n: n['score'], reverse=True)[:bw]

    if topk:
        import heapq
        return heapq.nlargest(topk, pool, key=lambda n: n['score'])
    return best


def plan_hand_pick(game_state: dict, color: str,
                   mask: list, actions: list, score_fn=None, beam_width: int = None,
                   teammate_color: str | None = None) -> int:
    """Sélecteur d'action compatible avec heuristic_pick : renvoie l'index (dans `mask`)
    du PREMIER coup du meilleur PLAN trouvé sur la main. Repli sur la défausse si aucun
    coup légal. En 2v2, `color` = couleur CONTRÔLÉE et teammate_color son coéquipier
    (mêmes valeurs que celles passées à get_legal_mask pour construire `mask`)."""
    best = _search(game_state, color, mask, actions, score_fn, beam_width,
                   teammate_color=teammate_color)
    if best is None:
        return discard_idx_for(teammate_color)
    return best['first_idx']
