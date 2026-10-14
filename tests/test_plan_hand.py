"""
Tests du planificateur de main (plan_hand.py) : application des coups « comme le
serveur » et gestion du BARRAGE d'une bille invincible (ne pas bloquer son coéquipier
sans raison, mais savoir bloquer quand ça vaut le coup).

Géométrie utilisée (red = le bot, blue = coéquipier, green/orange = adversaires) :
la case de départ red est 9 ; juste derrière, dans le sens de la marche :
  … 51 → 36 → 21 → 6 → 7 → 8 → [9]      (51 est à 6 cases du pieu, 8 à 1 case)

Lancer : python -m unittest discover tests -v
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mercury_legal_moves import (  # noqa: E402
    get_legal_mask, HOME_POSITIONS, START_POSITIONS, ARRIVAL_POSITIONS,
)
from plan_hand import plan_hand_pick, _apply, _barrage, _search  # noqa: E402

RED_HOME, BLUE_HOME = HOME_POSITIONS['red'], HOME_POSITIONS['blue']
GREEN_HOME, ORANGE_HOME = HOME_POSITIONS['green'], HOME_POSITIONS['orange']
RED_START = START_POSITIONS['red']


def game_state(hand: list[str], red, blue, green=None, orange=None,
               invincible: tuple = ()) -> dict:
    """gameState minimal, joueurs dans l'ordre du serveur ; `invincible` = cases."""
    marbles = {'red': red, 'green': green or GREEN_HOME,
               'blue': blue, 'orange': orange or ORANGE_HOME}
    return {
        'gameMode': '2v2',
        'hand': [{'id': f'c{i}', 'value': v, 'suit': '♠'} for i, v in enumerate(hand)],
        'players': [{'color': c, 'marblePositions': list(ps),
                     'marbleInvincible': [p in invincible for p in ps]}
                    for c, ps in marbles.items()],
    }


def legal(gs: dict):
    mbc = {p['color']: p['marblePositions'] for p in gs['players']}
    inv = {p['color']: [pos for pos, i in zip(p['marblePositions'],
                                              p['marbleInvincible']) if i]
           for p in gs['players']}
    return get_legal_mask(gs['hand'], mbc['red'], 'red', mbc,
                          invincible_by_color=inv, teammate_color='blue')


def pick(gs: dict) -> tuple[str, dict]:
    """(valeur de la carte, action) du coup choisi par le planificateur pour red."""
    mask, actions = legal(gs)
    card, action = actions[plan_hand_pick(gs, 'red', mask, actions,
                                          teammate_color='blue')]
    return card['value'], action


class TestApplyLikeServer(unittest.TestCase):
    def test_split_first_marble_lands_on_square_left_by_teammate(self):
        # 7 partagé 1+6 : red 10→25 pendant que blue quitte 25 pour 87.
        action = {'type': 'move', 'from': 10, 'to': 25, 'splitFrom': 25,
                  'splitTo': 87, 'splitType': 'move', 'splitMarbleColor': 'blue'}
        mp, opp, gain, _ = _apply([10], {'blue': [25], 'green': []}, 'red', action,
                                  teammate='blue')
        self.assertEqual((mp, opp['blue'], gain), ([25], [87], 0.0))

    def test_entering_captures_enemy_standing_on_my_start(self):
        action = {'type': 'enter', 'from': RED_HOME[0], 'to': RED_START}
        mp, opp, gain, inv = _apply([RED_HOME[0]], {'green': [RED_START], 'blue': []},
                                    'red', action, teammate='blue')
        self.assertEqual(mp, [RED_START])
        self.assertEqual(opp['green'], [GREEN_HOME[0]], 'adversaire renvoyé au home')
        self.assertGreater(gain, 0, 'capture créditée')
        self.assertEqual(inv, {RED_START}, 'la bille entrée est invincible')

    def test_entering_on_teammate_marble_is_not_rewarded(self):
        action = {'type': 'enter', 'from': RED_HOME[0], 'to': RED_START}
        _, opp, gain, _ = _apply([RED_HOME[0]], {'blue': [RED_START], 'green': []},
                                 'red', action, teammate='blue')
        self.assertEqual(opp['blue'], [BLUE_HOME[0]], "pas d'immunité d'équipe")
        self.assertEqual(gain, 0.0)

    def test_moving_clears_invincibility(self):
        action = {'type': 'move', 'from': RED_START, 'to': 25}
        _, _, _, inv = _apply([RED_START], {}, 'red', action,
                              inv=frozenset({RED_START}))
        self.assertEqual(inv, frozenset())


class TestBarrageValue(unittest.TestCase):
    INV = frozenset({RED_START})

    def _value(self, blue=(), green=(), orange=()):
        mbc = {'red': [RED_START], 'green': list(green), 'blue': list(blue),
               'orange': list(orange)}
        return _barrage(self.INV, 'red', mbc, 'blue')

    def test_no_invincible_marble_no_barrage(self):
        mbc = {'red': [RED_START], 'blue': [8], 'green': [], 'orange': []}
        self.assertEqual(_barrage(frozenset(), 'red', mbc, 'blue'), 0.0)

    def test_blocking_teammate_costs_more_when_closer(self):
        far, close = self._value(blue=[51, 201]), self._value(blue=[8, 201])
        self.assertLess(close, far)
        self.assertLess(far, 0)

    def test_teammate_with_no_other_marble_in_play_is_worse(self):
        self.assertLess(self._value(blue=[6]), self._value(blue=[6, 201]))

    def test_blocking_enemies_is_worth_something(self):
        self.assertGreater(self._value(green=[8]), 0)

    def test_marble_out_of_reach_or_ahead_is_not_blocked(self):
        # 138 est à 19 cases derrière le pieu (hors de portée du Joker), 10 est devant.
        self.assertEqual(self._value(blue=[138, 10]), 0.0)

    def test_marble_about_to_promote_before_my_start_is_not_blocked(self):
        # Orange en 106 : sa case de départ (91) est 1 case devant lui — il rentre
        # chez lui avant d'atteindre mon pieu.
        self.assertEqual(self._value(orange=[106]), 0.0)


class TestBarrageDecisions(unittest.TestCase):
    """Red : un pion en 90 (hors de portée de ce qui se passe derrière son départ),
    les autres en réserve. Un pion blue attend 3 cases derrière le départ red (6).
    `201` = un second pion blue, loin, qui n'est pas barré."""

    RED = [90] + RED_HOME[:3]

    def test_entry_is_deferred_while_teammate_is_right_behind(self):
        # Le 9 d'abord : le coéquipier a une chance de passer avant que j'entre.
        gs = game_state(['K', '9'], self.RED, blue=[6, 201] + BLUE_HOME[:2])
        card, action = pick(gs)
        self.assertEqual((card, action['type']), ('9', 'move'))

    def test_without_teammate_behind_the_king_enters_first(self):
        gs = game_state(['K', '9'], self.RED, blue=[201] + BLUE_HOME[:3])
        self.assertEqual(pick(gs)[1]['type'], 'enter')

    def test_still_enters_when_teammate_is_only_hindered(self):
        # Dernière carte : une entrée vaut mieux que 13 cases, le coéquipier a un
        # autre pion à jouer en attendant.
        gs = game_state(['K'], self.RED, blue=[6, 201] + BLUE_HOME[:2])
        self.assertEqual(pick(gs)[1]['type'], 'enter')

    def test_does_not_enter_when_teammate_would_be_stuck(self):
        # Le SEUL pion en jeu du coéquipier est collé derrière mon départ : entrer le
        # priverait de presque toutes ses cartes → le Roi avance plutôt de 13.
        gs = game_state(['K'], self.RED, blue=[6] + BLUE_HOME[:3])
        card, action = pick(gs)
        self.assertEqual((card, action['type'], action['from']), ('K', 'move', 90))

    def test_blocking_is_worth_it_when_it_also_traps_an_enemy(self):
        # Même position, mais un pion green (seul en jeu) est collé au pieu lui aussi.
        gs = game_state(['K'], self.RED, blue=[6] + BLUE_HOME[:3],
                        green=[8] + GREEN_HOME[:3])
        self.assertEqual(pick(gs)[1]['type'], 'enter')

    def test_ace_is_not_wasted_to_spare_a_hindered_teammate(self):
        gs = game_state(['A'], self.RED, blue=[6, 201] + BLUE_HOME[:2])
        self.assertEqual(pick(gs)[1]['type'], 'enter')

    def test_invincible_marble_moves_away_when_it_blocks_teammate(self):
        red = [90, RED_START] + RED_HOME[:2]
        gs = game_state(['5', '9'], red, blue=[6, 201] + BLUE_HOME[:2],
                        invincible=(RED_START,))
        self.assertEqual(pick(gs)[1]['from'], RED_START, 'on dégage le pieu en premier')

    def test_capture_available_now_comes_before_clearing_the_way(self):
        # Même position, mais le 5 capture un pion orange bien avancé (149) : une
        # capture ne se remet pas à plus tard, le pieu sera dégagé au tour suivant.
        red = [90, RED_START] + RED_HOME[:2]
        gs = game_state(['5', '9'], red, blue=[6, 201] + BLUE_HOME[:2],
                        orange=[149] + ORANGE_HOME[:3], invincible=(RED_START,))
        card, action = pick(gs)
        self.assertEqual((card, action['type'], action['from']), ('5', 'capture', 90))

    def test_invincible_marble_stays_when_it_blocks_an_enemy(self):
        red = [RED_START, 90] + RED_HOME[:2]
        gs = game_state(['5', '9'], red, blue=[201] + BLUE_HOME[:3],
                        green=[8] + GREEN_HOME[:3], invincible=(RED_START,))
        self.assertEqual(pick(gs)[1]['from'], 90, "le pieu reste, l'autre pion avance")

    def test_joker_entry_followed_by_replay_does_not_count_as_a_block(self):
        # Joker = je rejoue : entrer puis dégager dans le même tour ne bloque personne.
        gs = game_state(['Joker', '5'], RED_HOME, blue=[6] + BLUE_HOME[:3])
        mask, actions = legal(gs)
        best = _search(gs, 'red', mask, actions, teammate_color='blue')
        self.assertEqual(best['turn_block'], 0.0)
        self.assertNotIn(RED_START, best['my'])

    def test_fresh_marble_cannot_be_planned_backwards(self):
        # Une bille invincible ne recule pas au 4 : le plan « entrer puis 4 » (qui
        # la poserait à 4 cases de l'arrivée) n'existe pas côté serveur.
        gs = game_state(['K', '4'], RED_HOME, blue=BLUE_HOME)
        mask, actions = legal(gs)
        best = _search(gs, 'red', mask, actions, teammate_color='blue')
        self.assertIn(RED_START, best['my'])


if __name__ == '__main__':
    unittest.main()
