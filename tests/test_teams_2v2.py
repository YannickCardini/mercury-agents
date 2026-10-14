"""
Tests du mode équipes 2v2 — miroir Python des tests TS du backend :
  teams.test.ts, jack-team.test.ts, seven-team.test.ts, solidaire.test.ts
(mêmes cas, mêmes géométries), plus les invariants du layout 2v2 et des helpers
mode-aware (detect_winners).

Lancer : python -m unittest discover tests -v
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mercury_legal_moves import (  # noqa: E402
    get_legal_mask, get_legal_action, build_server_message, find_solidaire_entry,
    get_teammate_color, same_team, has_team_won, get_controlled_color,
    _build_7_split,
    HOME_POSITIONS, START_POSITIONS, ARRIVAL_POSITIONS, ALL_COLORS,
    DISCARD_IDX_2V2, SOLIDAIRE_IDX, ACTION_DIM_2V2, ACTION_DIM,
)
from game_state_utils import detect_winners, detect_winner  # noqa: E402


def card(value: str, cid: str | None = None) -> dict:
    return {'id': cid or f'test-{value}', 'value': value,
            'suit': '🃏' if value == 'Joker' else '♠'}


def empty_by_color() -> dict:
    return {c: [] for c in ALL_COLORS}


def legal_entries(mask, actions):
    """Les (index, card, action) légaux d'un masque (hors discard/solidaire)."""
    out = []
    for i, ok in enumerate(mask):
        if ok and actions[i] not in (None, 'discard'):
            out.append((i, actions[i][0], actions[i][1]))
    return out


def team_mask(hand, color, marbles, invincible=None, can_discard=False):
    """get_legal_mask en contexte 2v2 (miroir de buildCtx(teamMode) des tests TS)."""
    return get_legal_mask(
        hand, marbles[color], color, marbles,
        invincible_by_color=invincible,
        can_discard=can_discard,
        teammate_color=get_teammate_color(color),
    )


# ── teams.test.ts ─────────────────────────────────────────────────────────────

class TestTeams(unittest.TestCase):
    def test_get_teammate_color(self):
        self.assertEqual(get_teammate_color('red'), 'blue')
        self.assertEqual(get_teammate_color('blue'), 'red')
        self.assertEqual(get_teammate_color('green'), 'orange')
        self.assertEqual(get_teammate_color('orange'), 'green')

    def test_same_team(self):
        self.assertTrue(same_team('red', 'blue'))
        self.assertTrue(same_team('blue', 'red'))
        self.assertTrue(same_team('red', 'red'))
        self.assertTrue(same_team('orange', 'green'))
        self.assertFalse(same_team('red', 'green'))
        self.assertFalse(same_team('blue', 'orange'))

    def test_has_team_won_requires_both_teammates(self):
        marbles = {
            **empty_by_color(),
            'red':    list(ARRIVAL_POSITIONS['red']),
            'blue':   list(ARRIVAL_POSITIONS['blue']),
            'green':  list(HOME_POSITIONS['green']),
            'orange': list(HOME_POSITIONS['orange']),
        }
        self.assertTrue(has_team_won(marbles, 'red'))
        self.assertTrue(has_team_won(marbles, 'blue'),
                        "symétrique pour l'autre membre de l'équipe")
        self.assertFalse(has_team_won(marbles, 'green'))

        # Un seul membre fini → l'équipe n'a PAS gagné (le OU historique devient un ET)
        partial = {**marbles, 'blue': [217] + list(ARRIVAL_POSITIONS['blue'][1:])}
        self.assertFalse(has_team_won(partial, 'red'))
        self.assertFalse(has_team_won(partial, 'blue'))

    def test_get_controlled_color(self):
        # Rouge a fini → il contrôle les pions de bleu
        self.assertEqual(get_controlled_color('red', list(ARRIVAL_POSITIONS['red'])), 'blue')
        # Rouge en cours de partie → ses propres pions
        self.assertEqual(
            get_controlled_color('red', [9, 25] + list(ARRIVAL_POSITIONS['red'][2:])),
            'red')
        self.assertEqual(get_controlled_color('green', list(ARRIVAL_POSITIONS['green'])),
                         'orange')


# ── jack-team.test.ts ─────────────────────────────────────────────────────────

class TestJackTeam(unittest.TestCase):
    def _swaps(self, mask, actions):
        return [a for _, _, a in legal_entries(mask, actions) if a['type'] == 'swap']

    def test_2v2_swap_teammate_marble_with_enemy(self):
        # Rouge joue le Valet ; blue (coéquipier) en 10, green (adverse) en 25,
        # aucun pion rouge en jeu : la source n'a pas à être un pion contrôlé.
        marbles = {**empty_by_color(), 'red': list(HOME_POSITIONS['red']),
                   'blue': [10], 'green': [25]}
        mask, actions = team_mask([card('J')], 'red', marbles)
        swaps = self._swaps(mask, actions)
        self.assertEqual(sorted((s['from'], s['to']) for s in swaps),
                         [(10, 25), (25, 10)],
                         "échange blue↔green sans pion du joueur actif")
        by_from = {s['from']: s for s in swaps}
        self.assertEqual(by_from[10]['marbleColor'], 'blue',
                         "marbleColor = couleur du pion source")

    def test_2v2_enemy_enemy_swap_rejected(self):
        # green et orange sont tous deux adverses de l'équipe red+blue : le serveur
        # REJETTE cet échange (actionRejected → tour perdu au timeout si on l'envoie).
        marbles = {**empty_by_color(), 'red': [40],
                   'green': [10], 'orange': [25]}
        mask, actions = team_mask([card('J')], 'red', marbles)
        pairs = {(s['from'], s['to']) for s in self._swaps(mask, actions)}
        self.assertNotIn((10, 25), pairs)
        self.assertNotIn((25, 10), pairs)
        self.assertIn((40, 10), pairs, "un pion de l'équipe reste échangeable")
        self.assertIn((10, 40), pairs, "source adverse ↔ cible de l'équipe : autorisé")

    def test_2v2_same_color_pair_rejected(self):
        marbles = {**empty_by_color(), 'red': list(HOME_POSITIONS['red']),
                   'green': [10, 25]}
        mask, actions = team_mask([card('J')], 'red', marbles)
        self.assertEqual(self._swaps(mask, actions), [],
                         "deux pions de MÊME couleur ne s'échangent pas")

    def test_2v2_invincible_protected_as_source_and_target(self):
        # Paire autorisée par la règle d'équipe (blue = coéquipier, orange = adverse) :
        # seule l'invincibilité bloque. Orange fraîchement entré sur son start (91).
        marbles = {**empty_by_color(), 'red': list(HOME_POSITIONS['red']),
                   'blue': [10], 'orange': [START_POSITIONS['orange']]}
        invincible = {**empty_by_color(), 'orange': [START_POSITIONS['orange']]}
        mask, actions = team_mask([card('J')], 'red', marbles, invincible=invincible)
        self.assertEqual(self._swaps(mask, actions), [],
                         "pieu protégé du J, comme source et comme cible")

    def test_1v3_source_must_be_own(self):
        marbles = {**empty_by_color(), 'red': [10], 'green': [25]}
        mask, actions = get_legal_mask([card('J')], marbles['red'], 'red', marbles)
        swaps = self._swaps(mask, actions)
        self.assertEqual([(s['from'], s['to']) for s in swaps], [(10, 25)],
                         "1v3 : source propre uniquement")

    def test_2v2_swap_exists_without_own_source_but_not_in_1v3(self):
        # Tous les pions rouges en réserve : aucune source propre possible.
        # (le pendant Python de findLegalMoveForCard : la défausse ne doit pas
        # être proposée en 2v2 alors qu'un swap coéquipier↔adverse est jouable)
        marbles = {**empty_by_color(), 'red': list(HOME_POSITIONS['red']),
                   'blue': [10], 'green': [25]}
        mask, actions = team_mask([card('J')], 'red', marbles)
        self.assertTrue(self._swaps(mask, actions), '2v2 : swap blue↔green trouvé')
        self.assertFalse(mask[DISCARD_IDX_2V2], '2v2 : pas de défausse possible')

        # Adverse ↔ adverse seul : aucun coup, la défausse redevient le seul choix.
        enemies = {**empty_by_color(), 'red': list(HOME_POSITIONS['red']),
                   'green': [10], 'orange': [25]}
        mask_e, actions_e = team_mask([card('J')], 'red', enemies)
        self.assertEqual(self._swaps(mask_e, actions_e), [])
        self.assertTrue(mask_e[DISCARD_IDX_2V2], '2v2 : défausse forcée')

        mask13, actions13 = get_legal_mask([card('J')], marbles['red'], 'red', marbles)
        self.assertEqual(self._swaps(mask13, actions13), [],
                         '1v3 : aucun coup (source propre requise)')
        self.assertTrue(mask13[len(mask13) - 1], '1v3 : défausse forcée')

    def test_ownership_guard_on_move_cards(self):
        # L'exploit historique : envoyer `from` = pion adverse avec une carte de déplacement.
        marbles = {**empty_by_color(), 'red': [10], 'green': [25]}
        all_marbles = [10, 25]
        self.assertIsNone(get_legal_action('Q', 25, [10], all_marbles, 'red', marbles))
        self.assertIsNotNone(get_legal_action('Q', 10, [10], all_marbles, 'red', marbles))


# ── Roi (K) à double usage : entrer OU avancer de 13 (changement de règle serveur) ──
# Rappel géométrie (MAIN_PATH) : 9(start red)→10→25→...→79→80→81→66→51→36→21→6→7→8→9,
# arrivées red = [38, 53, 68, 83].

class TestKingDualUse(unittest.TestCase):
    def test_king_still_enters_from_home(self):
        # Non-régression : K sur un pion en réserve continue de faire entrer.
        marbles = {**empty_by_color(), 'red': [HOME_POSITIONS['red'][0]]}
        action = get_legal_action('K', HOME_POSITIONS['red'][0],
                                  marbles['red'], marbles['red'], 'red', marbles)
        self.assertEqual(action, {'type': 'enter',
                                  'from': HOME_POSITIONS['red'][0],
                                  'to': START_POSITIONS['red']})

    def test_king_advances_13_on_board(self):
        marbles = {**empty_by_color(), 'red': [10]}
        action = get_legal_action('K', 10, [10], [10], 'red', marbles)
        self.assertEqual(action, {'type': 'move', 'from': 10, 'to': 135})

    def test_king_captures_at_13(self):
        marbles = {**empty_by_color(), 'red': [10], 'blue': [135]}
        all_marbles = [10, 135]
        action = get_legal_action('K', 10, [10], all_marbles, 'red', marbles)
        self.assertEqual(action, {'type': 'capture', 'from': 10, 'to': 135})

    def test_king_promotes_at_13(self):
        # Depuis 79, +13 tombe exactement sur la dernière case d'arrivée red (83).
        marbles = {**empty_by_color(), 'red': [79]}
        action = get_legal_action('K', 79, [79], [79], 'red', marbles)
        self.assertEqual(action, {'type': 'promote', 'from': 79, 'to': 83})

    def test_king_blocked_by_own_marble_at_destination(self):
        marbles = {**empty_by_color(), 'red': [10, 135]}
        all_marbles = [10, 135]
        self.assertIsNone(get_legal_action('K', 10, [10, 135], all_marbles, 'red',
                                           marbles))


# ── seven-team.test.ts ────────────────────────────────────────────────────────
# Rappels géométrie (MAIN_PATH) : 9→10→25→40→55→70→85→86→87→88→89→90→105→120…
# et …220→219→218→217(start blue)→216… ; arrivées blue = [188,173,158,143].

class TestSevenTeam(unittest.TestCase):
    def _split(self, m_a, s1, m_b, own_color, marbles, teammate='auto',
               invincible=None):
        if teammate == 'auto':
            teammate = get_teammate_color(own_color)
        all_marbles = [p for ps in marbles.values() for p in ps]
        return _build_7_split(m_a, s1, m_b, marbles[own_color], all_marbles,
                              own_color, marbles, invincible or empty_by_color(),
                              teammate_color=teammate)

    def test_split_own_plus_teammate(self):
        marbles = {**empty_by_color(), 'red': [10], 'blue': [86]}
        action = self._split(10, 1, 86, 'red', marbles)
        self.assertIsNotNone(action)
        self.assertEqual(action['from'], 10)
        self.assertEqual(action['to'], 25)
        self.assertEqual(action['splitFrom'], 86)
        self.assertEqual(action['splitTo'], 120)
        self.assertEqual(action['splitMarbleColor'], 'blue',
                         'le second pion appartient au coéquipier')

    def test_starting_with_teammate_marble_is_illegal(self):
        marbles = {**empty_by_color(), 'red': [10], 'blue': [86]}
        self.assertIsNone(self._split(86, 1, 10, 'red', marbles),
                          'own d\'abord : commencer par un pion du coéquipier est illégal')

    def test_teammate_marble_promotes_in_its_own_arrival(self):
        # Blue en 220 : 3 pas jusqu'à son start (217) + 3 arrivées à franchir = 6 pas.
        marbles = {**empty_by_color(), 'red': [10], 'blue': [220]}
        action = self._split(10, 1, 220, 'red', marbles)
        self.assertIsNotNone(action)
        self.assertEqual(action['splitType'], 'promote')
        self.assertIn(action['splitTo'], ARRIVAL_POSITIONS['blue'],
                      f"splitTo ({action['splitTo']}) doit être une arrivée BLUE")
        self.assertEqual(action['splitTo'], 143)
        self.assertEqual(action['splitMarbleColor'], 'blue')

    def test_second_marble_can_capture_own_team(self):
        # Red bouge 10→55 (3 pas), puis blue 86 avance de 4 → 90, occupé par l'autre red.
        marbles = {**empty_by_color(), 'red': [10, 90], 'blue': [86]}
        action = self._split(10, 3, 86, 'red', marbles)
        self.assertIsNotNone(action)
        self.assertEqual(action['to'], 55)
        self.assertEqual(action['splitTo'], 90)
        self.assertEqual(action['splitType'], 'capture',
                         'atterrir sur un pion allié le capture (pas d\'immunité)')

    def test_first_marble_lands_on_current_square_of_second(self):
        # Bug de prod côté serveur, corrigé depuis : red en 10, coéquipier blue une
        # case devant (25). Les deux moitiés bougent dans le MÊME coup : red avance
        # dans la case que blue quitte — ce n'est pas une capture.
        marbles = {**empty_by_color(), 'red': [10], 'blue': [25]}
        action = self._split(10, 1, 25, 'red', marbles)
        self.assertIsNotNone(action, 'le split reste légal même si to1 == from2')
        self.assertEqual((action['to'], action['type']), (25, 'move'))
        self.assertEqual((action['splitFrom'], action['splitTo'], action['splitType']),
                         (25, 87, 'move'))
        self.assertEqual(action['splitMarbleColor'], 'blue')

    def test_both_halves_cannot_land_on_same_square(self):
        # Red en 10 et blue en 7 : avec 2+5, les deux pions convergent sur 40.
        marbles = {**empty_by_color(), 'red': [10], 'blue': [7]}
        self.assertIsNone(self._split(10, 2, 7, 'red', marbles),
                          'to1 == to2 est illégal')

    def test_teammate_invincible_marble_ahead_does_not_block_split(self):
        # Red 218 est bloqué seul par blue 217 (invincible, start blue) ; mais blue
        # quitte 217 dans le même coup → le split est légal, donc pas de défausse.
        marbles = {**empty_by_color(), 'red': [218], 'blue': [217, 205]}
        invincible = {**empty_by_color(), 'blue': [217]}
        self.assertIsNotNone(self._split(218, 1, 217, 'red', marbles,
                                         invincible=invincible))
        mask, _ = team_mask([card('7')], 'red', marbles, invincible=invincible)
        self.assertFalse(mask[DISCARD_IDX_2V2])
        self.assertFalse(any(mask[:4]), 'le 7 entier, lui, reste bloqué')

    def test_1v3_split_limited_to_own_marbles(self):
        marbles = {**empty_by_color(), 'red': [10], 'blue': [86]}
        self.assertIsNone(self._split(10, 1, 86, 'red', marbles, teammate=None))

    def test_promote_first_split_is_legal_and_wire_encoded(self):
        # Cas réel observé en partie 2v2 (rejet serveur avant correctif) : red a
        # rentré 3 pions (83/68/53), le 4e est en 8 — à 1 pas de sa dernière arrivée
        # (38). Le SEUL coup légal du 7 est le split « promote d'abord » : red@8
        # promeut (s1=1), le pion blue@40 avance des 6 restants (40→88).
        # getLegalSplit7Action (TS) l'ACCEPTE ; sans lui le serveur met
        # canDiscard=false et rejette la défausse → timeout à chaque tour.
        marbles = {**empty_by_color(), 'red': [83, 68, 53, 8],
                   'blue': [143, 158, 190, 40]}
        action = self._split(8, 1, 40, 'red', marbles)
        self.assertIsNotNone(action, 'promote-first split doit être légal')
        self.assertEqual(action['type'], 'promote')
        self.assertEqual(action['to'], 38, 'le pion promeut dans SA zone d\'arrivée')
        self.assertEqual((action['splitFrom'], action['splitTo']), (40, 88))

        # Encodage WIRE : game.ts re-dérive steps1 via MAIN_PATH.indexOf(to) → le
        # message doit porter to=9 (case du main path à 1 pas), PAS 38 (l'arrivée).
        mask, actions = team_mask([card('7')], 'red', marbles)
        idx = next(i for i, ok in enumerate(mask)
                   if ok and actions[i] not in (None, 'discard')
                   and actions[i][1].get('splitFrom') == 40
                   and actions[i][1]['type'] == 'promote')
        msg = build_server_message(idx, [card('7')], marbles['red'], 'red', marbles,
                                   teammate_color='blue')
        self.assertEqual(msg['action']['to'], 9)
        self.assertEqual(msg['action']['from'], 8)
        self.assertEqual(msg['action']['splitFrom'], 40)

    def test_mask_index_layout_2v2(self):
        # Le split red@10 (s1=1) + blue@86 doit occuper l'index attendu du layout :
        # slot 0, sub = 4 + (m_a*6 + (s1-1))*8 + m_b = 4 + 0 + 4 (blue = slot équipe 4).
        marbles = {**empty_by_color(), 'red': [10], 'blue': [86]}
        mask, actions = team_mask([card('7')], 'red', marbles)
        self.assertEqual(len(mask), ACTION_DIM_2V2)
        idx = 4 + (0 * 6 + 0) * 8 + 4
        self.assertTrue(mask[idx])
        _, action = actions[idx]
        self.assertEqual((action['from'], action['to'],
                          action['splitFrom'], action['splitTo']),
                         (10, 25, 86, 120))


# ── promote-invincible-block.test.ts ──────────────────────────────────────────

class TestPromoteInvincibleBlock(unittest.TestCase):
    # Green start = 135. Pion green en 8 : 15 pas jusqu'à 135 + 3 cases d'arrivée
    # = 18 pas, exactement la distance du Joker. Le trajet passe par 9 (start red).

    def test_invincible_marble_on_the_way_blocks_promotion(self):
        marbles = {**empty_by_color(), 'green': [8], 'red': [9]}
        invincible = {**empty_by_color(), 'red': [9]}
        self.assertIsNone(get_legal_action('Joker', 8, [8], [8, 9], 'green', marbles,
                                           invincible))
        self.assertIsNotNone(get_legal_action('Joker', 8, [8], [8, 9], 'green',
                                              marbles),
                             'sans invincibilité, la promotion reste légale')

    def test_own_invincible_marble_on_own_start_does_not_block(self):
        marbles = {**empty_by_color(), 'green': [8, 135]}
        invincible = {**empty_by_color(), 'green': [135]}
        action = get_legal_action('Joker', 8, [8, 135], [8, 135], 'green', marbles,
                                  invincible)
        self.assertIsNotNone(action)
        self.assertEqual((action['type'], action['to']), ('promote', 115))


# ── solidaire.test.ts ─────────────────────────────────────────────────────────

class TestSolidaire(unittest.TestCase):
    # Main [K] bloquée pour red avec un seul pion en 10 : le K n'est pas en réserve,
    # et red n'a aucun pion en réserve. Depuis le changement de règle serveur, le K
    # peut aussi avancer de 13 quand il est déjà en jeu — un pion orange INVINCIBLE
    # une case devant (25) bloque ce nouveau déplacement (cf. BLOCKED_RED_INVINCIBLE,
    # à passer en invincible_by_color pour les tests qui exigent une main VRAIMENT
    # bloquée).
    BLOCKED_RED = {'red': [10], 'orange': [25]}
    BLOCKED_RED_INVINCIBLE = {**empty_by_color(), 'orange': [25]}

    def _solidaire(self, hand, color, marbles, invincible=None, teammate='auto'):
        if teammate == 'auto':
            teammate = get_teammate_color(color)
        return find_solidaire_entry(hand, color, marbles,
                                    invincible_by_color=invincible,
                                    teammate_color=teammate)

    def test_forced_entry_of_teammate_marble(self):
        marbles = {**empty_by_color(), **self.BLOCKED_RED,
                   'blue': [HOME_POSITIONS['blue'][0], HOME_POSITIONS['blue'][1], 86]}
        entry = self._solidaire([card('K')], 'red', marbles,
                                invincible=self.BLOCKED_RED_INVINCIBLE)
        self.assertIsNotNone(entry)
        c, action = entry
        self.assertEqual(action['type'], 'enter')
        self.assertEqual(action['to'], START_POSITIONS['blue'],
                         'entrée sur le start DU COÉQUIPIER')
        self.assertIn(action['from'], HOME_POSITIONS['blue'],
                      'depuis une case de réserve blue')
        self.assertEqual(action['marbleColor'], 'blue')
        self.assertEqual(c['value'], 'K')

    def test_teammate_start_occupied_by_any_color(self):
        # « Totalement libre » : même un pion adverse sur le start bloque l'exception
        # (l'entrée normale, elle, aurait capturé).
        marbles = {**empty_by_color(), **self.BLOCKED_RED,
                   'blue':   [HOME_POSITIONS['blue'][0], 86],
                   'orange': [START_POSITIONS['blue']]}
        self.assertIsNone(self._solidaire([card('K')], 'red', marbles))

    def test_hand_not_blocked(self):
        # La Q peut avancer le pion red de 12 : la main n'est pas bloquée.
        marbles = {**empty_by_color(), **self.BLOCKED_RED,
                   'blue': [HOME_POSITIONS['blue'][0]]}
        self.assertIsNone(self._solidaire([card('K'), card('Q')], 'red', marbles))

    def test_no_enter_card_in_hand(self):
        # Red fraîchement entré (invincible sur son start 9) : le 4 ne peut pas le
        # reculer → main [4] bloquée, mais aucune carte d'entrée.
        marbles = {**empty_by_color(), 'red': [START_POSITIONS['red']],
                   'blue': [HOME_POSITIONS['blue'][0]]}
        invincible = {**empty_by_color(), 'red': [START_POSITIONS['red']]}
        self.assertIsNone(self._solidaire([card('4')], 'red', marbles,
                                          invincible=invincible))

    def test_teammate_without_reserve(self):
        marbles = {**empty_by_color(), **self.BLOCKED_RED, 'blue': [86, 87]}
        self.assertIsNone(self._solidaire([card('K')], 'red', marbles))

    def test_1v3_never_triggers(self):
        marbles = {**empty_by_color(), **self.BLOCKED_RED,
                   'blue': [HOME_POSITIONS['blue'][0]]}
        self.assertIsNone(self._solidaire([card('K')], 'red', marbles, teammate=None))

    def test_card_priority_A_K_Joker(self):
        # Pions red tous à l'abri (aucun coup possible avec aucune carte).
        marbles = {**empty_by_color(),
                   'red':  list(ARRIVAL_POSITIONS['red']),
                   'blue': [HOME_POSITIONS['blue'][0], 86]}
        pick = lambda hand: self._solidaire(hand, 'red', marbles)[0]['value']
        self.assertEqual(pick([card('Joker'), card('K'), card('A')]), 'A')
        self.assertEqual(pick([card('Joker'), card('K')]), 'K')
        self.assertEqual(pick([card('Joker')]), 'Joker')

    def test_no_double_trigger_after_switch(self):
        # Red a fini : le ctx « switché » a color=blue et teammate=red.
        # Red n'a plus de pion en réserve → l'exception ne se déclenche pas.
        marbles = {**empty_by_color(),
                   'red':  list(ARRIVAL_POSITIONS['red']),
                   'blue': [10]}
        self.assertIsNone(find_solidaire_entry([card('K')], 'blue', marbles,
                                               teammate_color='red'))

    def test_mask_solidaire_is_only_legal_action(self):
        # ⚠️ Le point critique prod : quand la solidaire s'impose, le masque ne doit
        # proposer QUE l'index solidaire — jamais la défausse (rejetée par le serveur
        # → timeout à chaque tour).
        marbles = {**empty_by_color(), **self.BLOCKED_RED,
                   'blue': [HOME_POSITIONS['blue'][0], HOME_POSITIONS['blue'][1], 86]}
        mask, actions = team_mask([card('K')], 'red', marbles,
                                  invincible=self.BLOCKED_RED_INVINCIBLE)
        self.assertTrue(mask[SOLIDAIRE_IDX])
        self.assertFalse(mask[DISCARD_IDX_2V2])
        self.assertEqual(sum(mask), 1)

        msg = build_server_message(SOLIDAIRE_IDX, [card('K')], marbles['red'], 'red',
                                   marbles, invincible_by_color=self.BLOCKED_RED_INVINCIBLE,
                                   teammate_color='blue')
        action = msg['action']
        self.assertEqual(action['type'], 'enter')
        self.assertEqual(action['from'], HOME_POSITIONS['blue'][0])
        self.assertEqual(action['to'], START_POSITIONS['blue'])
        self.assertEqual(action['cardPlayed'][0]['value'], 'K')
        self.assertNotIn('marbleColor', action,
                         'champ interne non envoyé : le serveur recalcule et fait foi')


# ── Helpers mode-aware (gameEnded.winners / detect_winners) ──────────────────

def _gs(mode: str, marbles: dict) -> dict:
    return {'gameMode': mode,
            'players': [{'color': c, 'marblePositions': list(ps)}
                        for c, ps in marbles.items()]}


class TestModeAwareHelpers(unittest.TestCase):
    def test_detect_winners_2v2_requires_both_teammates(self):
        # Un joueur fini seul NE termine PAS la partie en 2v2 (le bot doit continuer
        # à jouer — il contrôle les pions du coéquipier).
        marbles = {**{c: list(HOME_POSITIONS[c]) for c in ALL_COLORS},
                   'red': list(ARRIVAL_POSITIONS['red'])}
        self.assertEqual(detect_winners(_gs('2v2', marbles)), [])
        self.assertIsNone(detect_winner(_gs('2v2', marbles)))

        marbles['blue'] = list(ARRIVAL_POSITIONS['blue'])
        self.assertEqual(sorted(detect_winners(_gs('2v2', marbles))), ['blue', 'red'])

    def test_detect_winners_1v3_single_color(self):
        marbles = {**{c: list(HOME_POSITIONS[c]) for c in ALL_COLORS},
                   'red': list(ARRIVAL_POSITIONS['red'])}
        self.assertEqual(detect_winners(_gs('1v3', marbles)), ['red'])
        # Rétro-compat : gameMode absent = 1v3.
        gs = _gs('1v3', marbles)
        del gs['gameMode']
        self.assertEqual(detect_winners(gs), ['red'])


if __name__ == '__main__':
    unittest.main()
