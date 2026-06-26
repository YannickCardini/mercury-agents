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
    _MAIN_PATH_IDX, _MAIN_PATH_LEN, PER_SLOT, DISCARD_IDX,
)
import time

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


# ── Évaluation d'un état (mes billes) ─────────────────────────────────────────

def _threatened(pos: int, color: str, mbc: dict) -> bool:
    """v1 (grossier) : un adversaire est à ≤ THREAT_RANGE cases derrière."""
    if pos not in _MAIN_PATH_IDX or pos == START_POSITIONS[color]:
        return False
    idx = _MAIN_PATH_IDX[pos]
    for c, positions in mbc.items():
        if c == color:
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


def _capturable(pos: int, color: str, mbc: dict) -> bool:
    """v2 (exact) : une bille adverse peut-elle réellement atterrir sur `pos` au prochain
    tour ? (distance = celle d'une carte légale). Bien plus fidèle que « à ≤7 cases »."""
    if pos not in _MAIN_PATH_IDX or pos == START_POSITIONS[color]:
        return False
    pi = _MAIN_PATH_IDX[pos]
    for c, positions in mbc.items():
        if c == color:
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
    """Évaluateur d'un état de fin de plan. `threat_fn(pos, color, mbc)` décide si ma bille
    risque d'être capturée → on suppose alors la PLUS AVANCÉE menacée capturée (snipe 1-ply :
    on ne sur-étend pas sa tête dans la zone de capture). v1 = _threatened, v2 = _capturable."""
    def score(my_pawns: list, color: str, mbc: dict, captured: float) -> float:
        arrival = ARRIVAL_POSITIONS[color]
        sniped, best = None, -1.0
        for i, p in enumerate(my_pawns):
            if p not in arrival and threat_fn(p, color, mbc):
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
        return s
    return score


_score_board   = _make_score(_threatened)   # v1 (défaut, déployé)
score_board_v2 = _make_score(_capturable)   # v2 (menace exacte)


# ── Application d'un de MES coups (adversaires = obstacles, capture = obstacle retiré) ──

def _apply(my_pawns: list, opp: dict, color: str, action: dict) -> tuple:
    """Retourne (new_my_pawns, new_opp, capture_gain). `opp` = {color: [positions]}.
    Réutilise les from/to/type déjà calculés par mercury_legal_moves."""
    mp  = list(my_pawns)
    opp = {c: list(v) for c, v in opp.items()}
    gain = 0.0

    def move_pawn(frm, to):
        for i, p in enumerate(mp):
            if p == frm:
                mp[i] = to
                return

    def remove_opp(pos):
        nonlocal gain
        for c, v in opp.items():
            for i, p in enumerate(v):
                if p == pos:
                    gain += W_CAPTURE * marble_progress(pos, c)
                    v[i] = HOME_POSITIONS[c][0]   # renvoyée au home → plus un obstacle
                    return

    t = action['type']

    if t == 'swap':
        # mon pion (from) ↔ bille adverse (to) : positions échangées.
        # On NE récompense PAS le recul infligé à l'adversaire : le J ne doit servir
        # QUE ma propre course. Mon gain (mon pion saute vers `to`) est déjà crédité par
        # W_PROGRESS dans le score de fin de plan → le planificateur ne joue le J que
        # quand il fait réellement avancer mon pion, jamais pour seulement embêter un
        # joueur (les « spite swaps » d'un adversaire avancé mais sans bénéfice pour moi).
        owner = None
        for c, v in opp.items():
            for i, p in enumerate(v):
                if p == action['to']:
                    owner = (c, i)
                    break
        move_pawn(action['from'], action['to'])
        if owner is not None:
            c, i = owner
            opp[c][i] = action['from']   # l'adverse recule (effet de bord), non récompensé
        return mp, opp, gain

    # move / capture / promote / enter (+ 7-split : 2e demi-coup)
    if t == 'capture':
        remove_opp(action['to'])
    move_pawn(action['from'], action['to'])
    if 'splitTo' in action:
        if action.get('splitType') == 'capture':
            remove_opp(action['splitTo'])
        move_pawn(action['splitFrom'], action['splitTo'])
    return mp, opp, gain


# ── Recherche en faisceau sur la main ─────────────────────────────────────────

def _legal_entries(hand: list, my_pawns: list, color: str, mbc: dict) -> list:
    """(slot, card, action_dict) légaux pour la main restante (adversaires fixes, pas
    d'invincibilité simulée → planification optimiste sur les coups profonds)."""
    mask, actions = get_legal_mask(hand, my_pawns, color, mbc,
                                   invincible_by_color=None, can_discard=False)
    out = []
    for j, ok in enumerate(mask):
        if not ok:
            continue
        e = actions[j]
        if e is None or e == 'discard':
            continue
        out.append((j // PER_SLOT, e[0], e[1]))
    return out


def _search(game_state: dict, color: str, mask: list, actions: list,
            score_fn=None, beam_width: int = None, topk: int = None):
    """Renvoie le meilleur nœud {my, opp, first_idx, score}. Si `topk` est fourni, renvoie
    plutôt la LISTE des `topk` meilleurs nœuds par score (pour un re-classement externe,
    ex. par un réseau de valeur)."""
    score_fn = score_fn or _score_board
    bw       = beam_width or BEAM_WIDTH
    pool     = []                          # tous les nœuds vus (pour le top-K éventuel)

    mbc0      = {p['color']: list(p['marblePositions']) for p in game_state['players']}
    my_pawns0 = mbc0[color]
    opp0      = {c: v for c, v in mbc0.items() if c != color}
    hand0     = game_state.get('hand', [])

    # ── Racine : on étend les actions FOURNIES (vraie invincibilité / can_discard) ──
    beam = []
    for i, ok in enumerate(mask):
        if not ok:
            continue
        e = actions[i]
        if e is None or e == 'discard':
            continue
        slot = i // PER_SLOT
        mp, opp, gain = _apply(my_pawns0, opp0, color, e[1])
        child_hand = hand0[:slot] + hand0[slot + 1:]
        node = {'my': mp, 'opp': opp, 'hand': child_hand,
                'captured': gain, 'first_idx': i}
        node['score'] = score_fn(mp, color, {color: mp, **opp}, gain)
        beam.append(node)

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
            mbc = {color: node['my'], **node['opp']}
            for slot, _card, action in _legal_entries(node['hand'], node['my'], color, mbc):
                mp, opp, gain = _apply(node['my'], node['opp'], color, action)
                child_hand = node['hand'][:slot] + node['hand'][slot + 1:]
                child = {'my': mp, 'opp': opp, 'hand': child_hand,
                         'captured': node['captured'] + gain,
                         'first_idx': node['first_idx']}
                child['score'] = score_fn(mp, color, {color: mp, **opp}, child['captured'])
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
                   mask: list, actions: list, score_fn=None, beam_width: int = None) -> int:
    """Sélecteur d'action compatible avec heuristic_pick : renvoie l'index (dans `mask`)
    du PREMIER coup du meilleur PLAN trouvé sur la main. Repli sur la défausse si aucun
    coup légal."""
    best = _search(game_state, color, mask, actions, score_fn, beam_width)
    if best is None:
        return DISCARD_IDX
    return best['first_idx']
