"""
NON-RÉGRESSION 1v3 : avec teammate_color=None, get_legal_mask doit produire un masque
bit-à-bit IDENTIQUE à l'implémentation d'avant le portage 2v2, sur les 300 états de
référence de fixture_1v3_masks.json (générés par gen_1v3_fixture.py AVANT modification).

TROIS divergences intentionnelles sont tolérées :
  1. Le correctif du split de 7 dont le 1er demi-coup est un PROMOTE : l'ancien port le
     rejetait à tort (le validateur TS getLegalSplit7Action l'accepte, et le serveur
     compte ce split dans canDiscard — sans lui, la défausse du bot est rejetée →
     timeout à chaque tour ; observé en partie réelle, surtout en 2v2 mais possible en
     1v3).
  2. Le changement de règle serveur du Roi (K) : en plus de faire entrer un pion, il
     peut désormais avancer un pion déjà en jeu de 13 cases (move/capture/promote) —
     la fixture, gelée AVANT ce changement, ne contient pas ces coups.
  3. Une promotion dont le trajet jusqu'à la case de départ croise un pion INVINCIBLE
     est illégale (getArrivelCaseIfCanPromote) : l'ancien port l'acceptait à tort, le
     serveur rejetait le coup.
Le test vérifie que TOUT ajout correspond exactement à l'une des deux premières formes,
et que les seules suppressions possibles sont la défausse-fallback déplacée par ces
nouveaux coups et les promotions bloquées de la 3e forme.
"""

import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mercury_legal_moves import (  # noqa: E402
    get_legal_mask, ACTION_DIM, DISCARD_IDX, PER_SLOT,
    MAIN_PATH, START_POSITIONS,
)

FIXTURE = pathlib.Path(__file__).parent / "fixture_1v3_masks.json"


def _invincible_before_own_start(case: dict, idx: int) -> bool:
    """Divergence 3 : l'index est un coup à un seul pion dont le trajet jusqu'à SA
    case de départ croise un pion invincible."""
    if idx >= DISCARD_IDX or idx % PER_SLOT >= 4:
        return False
    pos = case['marbles_by_color'][case['color']][idx % PER_SLOT]
    if pos not in MAIN_PATH:
        return False
    invincible = {p for ps in case['invincible_by_color'].values() for p in ps}
    start = START_POSITIONS[case['color']]
    i = MAIN_PATH.index(pos)
    while True:
        i = (i + 1) % len(MAIN_PATH)
        if MAIN_PATH[i] == start:
            return False
        if MAIN_PATH[i] in invincible:
            return True


class TestRegression1v3(unittest.TestCase):
    def test_masks_identical_to_reference(self):
        cases = json.loads(FIXTURE.read_text())
        self.assertEqual(len(cases), 300)
        n_promote_splits = 0
        for k, c in enumerate(cases):
            mask, actions = get_legal_mask(
                c['hand'], c['marbles_by_color'][c['color']], c['color'],
                c['marbles_by_color'],
                invincible_by_color=c['invincible_by_color'],
                can_discard=c['can_discard'],
                teammate_color=None,
            )
            self.assertEqual(len(mask), ACTION_DIM, f"cas {k}: dimension")
            got = {i for i, ok in enumerate(mask) if ok}
            ref = set(c['mask_true_idx'])

            added = got - ref
            for i in sorted(added):
                self.assertLess(i, DISCARD_IDX, f"cas {k}: ajout {i} hors slots")
                card, action = actions[i]
                if card['value'] == 'K':
                    self.assertIn(action['type'], ('move', 'capture', 'promote'),
                                 f"cas {k}: ajout {i} K hors move/capture/promote")
                    continue
                self.assertEqual(card['value'], '7', f"cas {k}: ajout {i} hors carte 7")
                self.assertIn('splitFrom', action, f"cas {k}: ajout {i} pas un split")
                self.assertEqual(action['type'], 'promote',
                                 f"cas {k}: ajout {i} pas un promote-first")
                self.assertGreaterEqual(i % PER_SLOT, 4,
                                        f"cas {k}: ajout {i} hors sous-index split")
                n_promote_splits += 1

            removed = {i for i in ref - got
                       if not _invincible_before_own_start(c, i)}
            if removed:
                self.assertTrue(added, f"cas {k}: suppression sans ajout : {removed}")
                self.assertEqual(removed, {DISCARD_IDX},
                                 f"cas {k}: seule la défausse-fallback peut disparaître")

    def test_dimension_and_discard_unchanged(self):
        # Le layout 1v3 lui-même ne bouge pas.
        self.assertEqual(ACTION_DIM, 501)
        self.assertEqual(DISCARD_IDX, 500)
        self.assertEqual(PER_SLOT, 100)


if __name__ == '__main__':
    unittest.main()
