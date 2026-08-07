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


def _king_pref_bonus(card_value: str, action: dict) -> float:
    """Biais de préférence (pas de légalité) : le Roi doit privilégier entrer/promouvoir
    sur un simple déplacement de 13 en milieu de plateau, sans jamais l'interdire."""
    if card_value != 'K':
        return 0.0
    if action['type'] == 'enter':
        return W_KING_ENTER_BONUS
    if action['type'] == 'promote':
        return W_KING_PROMOTE_BONUS
    return 0.0


# ── Application d'un de MES coups (adversaires = obstacles, capture = obstacle retiré) ──

def _apply(my_pawns: list, opp: dict, color: str, action: dict,
           teammate: str | None = None) -> tuple:
    """Retourne (new_my_pawns, new_opp, capture_gain). `opp` = {color: [positions]}
    (toutes les couleurs ≠ la mienne, coéquipier inclus en 2v2).
    Réutilise les from/to/type déjà calculés par mercury_legal_moves.

    2v2 : les pions déplacés peuvent appartenir au coéquipier (entrée solidaire,
    2e pion d'un 7 partagé) voire à personne de mon équipe (J libre) → résolution
    GÉNÉRIQUE du propriétaire par position. Capturer un pion du coéquipier n'est
    PAS crédité W_CAPTURE (le score de fin de plan perd son progrès → malus) ; idem
    pour un de MES pions (2e demi-coup d'un 7 atterrissant sur le 1er camp)."""
    mp  = list(my_pawns)
    opp = {c: list(v) for c, v in opp.items()}
    gain = 0.0

    def move_pawn(frm, to):
        """Déplace le pion posé en `frm`, où qu'il soit (moi d'abord, puis les autres)."""
        for i, p in enumerate(mp):
            if p == frm:
                mp[i] = to
                return
        for v in opp.values():
            for i, p in enumerate(v):
                if p == frm:
                    v[i] = to
                    return

    def remove_victim(pos):
        """Renvoie au home le pion capturé en `pos`. Crédit W_CAPTURE seulement si
        c'est un pion ENNEMI — un allié (coéquipier ou moi-même) coûte via le score."""
        nonlocal gain
        for i, p in enumerate(mp):
            if p == pos:
                free = next((h for h in HOME_POSITIONS[color] if h not in mp),
                            HOME_POSITIONS[color][0])
                mp[i] = free
                return
        for c, v in opp.items():
            for i, p in enumerate(v):
                if p == pos:
                    if teammate is None or c != teammate:
                        gain += W_CAPTURE * marble_progress(pos, c)
                    v[i] = HOME_POSITIONS[c][0]   # renvoyée au home → plus un obstacle
                    return

    t = action['type']

    if t == 'swap':
        # pion source (from) ↔ pion cible (to) : positions échangées.
        # On NE récompense PAS le recul infligé à l'adversaire : le J ne doit servir
        # QUE la course de l'ÉQUIPE. Le gain réel (mon pion ou celui du coéquipier
        # saute vers `to`, un ennemi menaçant s'éloigne de derrière nos billes) est
        # déjà capté par W_PROGRESS/W_PROGRESS_TEAMMATE et par threat_fn dans le
        # score de fin de plan → pas de « spite swaps ». En 2v2 la source peut être
        # n'importe quel pion (J libre, y compris ennemi↔ennemi défensif).
        owner = None                      # occupant de `to` chez les AUTRES couleurs
        for c, v in opp.items():
            for i, p in enumerate(v):
                if p == action['to']:
                    owner = (c, i)
                    break
        to_was_mine = owner is None and action['to'] in mp
        move_pawn(action['from'], action['to'])
        if owner is not None:
            c, i = owner
            opp[c][i] = action['from']   # l'autre extrémité recule/avance, non récompensé
        elif to_was_mine:
            # Cible = un de MES pions (source étrangère, J libre 2v2) : move_pawn a
            # déplacé la source vers `to`, mon pion prend la place laissée en `from`.
            for i, p in enumerate(mp):
                if p == action['to']:
                    mp[i] = action['from']
                    break
        return mp, opp, gain

    # move / capture / promote / enter (+ 7-split : 2e demi-coup)
    if t == 'capture':
        remove_victim(action['to'])
    move_pawn(action['from'], action['to'])
    if 'splitTo' in action:
        if action.get('splitType') == 'capture':
            remove_victim(action['splitTo'])
        move_pawn(action['splitFrom'], action['splitTo'])
    return mp, opp, gain


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
                   teammate: str | None = None) -> list:
    """(slot, card, action_dict) légaux pour la main restante (adversaires fixes, pas
    d'invincibilité simulée → planification optimiste sur les coups profonds).
    Inclut l'entrée solidaire (2v2) — forcée par le serveur quand elle s'impose."""
    mask, actions = get_legal_mask(hand, my_pawns, color, mbc,
                                   invincible_by_color=None, can_discard=False,
                                   teammate_color=teammate)
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

    # ── Racine : on étend les actions FOURNIES (vraie invincibilité / can_discard) ──
    beam = []
    for i, ok in enumerate(mask):
        if not ok:
            continue
        e = actions[i]
        if e is None or e == 'discard':
            continue
        slot = _slot_of(i, hand0, e[0], per_slot)
        mp, opp, gain = _apply(my_pawns0, opp0, color, e[1], teammate=tm)
        child_hand = hand0[:slot] + hand0[slot + 1:]
        king = _king_pref_bonus(e[0]['value'], e[1])
        node = {'my': mp, 'opp': opp, 'hand': child_hand,
                'captured': gain, 'bonus': king, 'first_idx': i}
        node['score'] = score_fn(mp, color, {color: mp, **opp}, gain, teammate=tm) + king
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
            for slot, card, action in _legal_entries(node['hand'], node['my'],
                                                     color, mbc, teammate=tm):
                mp, opp, gain = _apply(node['my'], node['opp'], color, action,
                                       teammate=tm)
                child_hand = node['hand'][:slot] + node['hand'][slot + 1:]
                king = _king_pref_bonus(card['value'], action)
                child = {'my': mp, 'opp': opp, 'hand': child_hand,
                         'captured': node['captured'] + gain,
                         'bonus': node['bonus'] + king,
                         'first_idx': node['first_idx']}
                child['score'] = score_fn(mp, color, {color: mp, **opp},
                                          child['captured'], teammate=tm) + child['bonus']
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
