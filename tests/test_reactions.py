"""
Réactions emoji en mode 2v2 (main.py) : le coéquipier est un ALLIÉ — ses
entrées/promotions/captures déclenchent de la joie, jamais la tristesse/colère des
tables adverses. En 1v3, comportement bit-à-bit identique à l'historique (tout
non-soi = enemy, tables inchangées).
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from main import (  # noqa: E402
    _reaction_relation,
    REACTION_ACTION_TABLE, REACTION_ACTION_TABLE_ALLY,
    REACTION_BROADCAST_TABLE, REACTION_BROADCAST_TABLE_ALLY,
)

# Palette autorisée par le serveur (REACTION_EMOJIS, packages/shared/src/types.ts) :
# tout emoji hors liste est silencieusement ignoré par handleReaction. Les 6
# premiers sont gratuits, les 9 autres viennent de la boutique (SHOP_CATALOG).
SERVER_PALETTE = {'👏', '😂', '😮', '😥', '🔥', '🤔', '😡', '😎', '😴', '⏰', '🥱', '🦧', '😉', '😘', '😊'}

SAD_OR_ANGRY = {'😥', '😡'}
HAPPY = {'👏', '🔥', '😎', '😂'}


class TestReactionRelation(unittest.TestCase):
    def test_1v3_everyone_is_enemy(self):
        self.assertEqual(_reaction_relation('red', 'red', '1v3'), 'own')
        # blue est le coéquipier 2v2 de red, mais en 1v3 c'est un adversaire.
        self.assertEqual(_reaction_relation('red', 'blue', '1v3'), 'enemy')
        self.assertEqual(_reaction_relation('red', 'green', '1v3'), 'enemy')

    def test_2v2_teammate_is_ally(self):
        self.assertEqual(_reaction_relation('red', 'blue', '2v2'), 'ally')
        self.assertEqual(_reaction_relation('blue', 'red', '2v2'), 'ally')
        self.assertEqual(_reaction_relation('green', 'orange', '2v2'), 'ally')
        self.assertEqual(_reaction_relation('red', 'green', '2v2'), 'enemy')
        self.assertEqual(_reaction_relation('red', 'orange', '2v2'), 'enemy')
        self.assertEqual(_reaction_relation('red', 'red', '2v2'), 'own')

    def test_unknown_author_is_enemy(self):
        self.assertEqual(_reaction_relation('red', None, '2v2'), 'enemy')


class TestReactionTables(unittest.TestCase):
    def _check_entries(self, entries, label):
        for key, (prob, candidates) in entries:
            self.assertTrue(0 < prob <= 1, f"{label}[{key}]: prob {prob}")
            self.assertTrue(candidates, f"{label}[{key}]: vide")
            for emoji, weight in candidates:
                self.assertIn(emoji, SERVER_PALETTE,
                              f"{label}[{key}]: {emoji} hors palette serveur")
                self.assertGreater(weight, 0, f"{label}[{key}]: poids {weight}")

    def test_all_tables_use_server_palette(self):
        self._check_entries(REACTION_ACTION_TABLE.items(), 'ACTION')
        self._check_entries(REACTION_ACTION_TABLE_ALLY.items(), 'ACTION_ALLY')
        self._check_entries(REACTION_BROADCAST_TABLE.items(), 'BROADCAST')
        self._check_entries(REACTION_BROADCAST_TABLE_ALLY.items(), 'BROADCAST_ALLY')
        self.assertEqual(set(REACTION_BROADCAST_TABLE), SERVER_PALETTE)

    def test_ally_tables_cover_same_keys(self):
        # Tout type d'action / emoji reconnu pour un adversaire l'est pour l'allié :
        # sinon le bot deviendrait muet face à son coéquipier sur ces événements.
        action_types = {atype for atype, _ in REACTION_ACTION_TABLE}
        self.assertEqual(set(REACTION_ACTION_TABLE_ALLY), action_types)
        self.assertEqual(set(REACTION_BROADCAST_TABLE_ALLY),
                         set(REACTION_BROADCAST_TABLE))

    def test_ally_success_is_celebrated_not_resented(self):
        # Le cœur de la demande : quand l'ALLIÉ rentre/promeut/capture, jamais de
        # tristesse ni de colère (contrairement aux tables adverses), et au moins
        # une réponse clairement positive.
        for atype in ('enter', 'promote', 'capture', 'swap'):
            _, candidates = REACTION_ACTION_TABLE_ALLY[atype]
            emojis = {e for e, _ in candidates}
            self.assertFalse(emojis & SAD_OR_ANGRY,
                             f"ally {atype}: réponse négative {emojis & SAD_OR_ANGRY}")
            self.assertTrue(emojis & HAPPY, f"ally {atype}: aucune réponse positive")
        # Contre-point : les mêmes événements côté ADVERSAIRE restent négatifs.
        for atype in ('enter', 'promote', 'capture'):
            _, candidates = REACTION_ACTION_TABLE[(atype, False)]
            self.assertTrue({e for e, _ in candidates} & (SAD_OR_ANGRY | {'😮'}),
                            f"enemy {atype}: devrait rester contrarié/surpris")

    def test_ally_emojis_never_answered_with_anger(self):
        # On ne répond jamais 😡 ni 🥱 au 🔥/😎/👏 du coéquipier (jalousie réservée
        # aux adversaires) — on l'amplifie.
        for received in ('🔥', '😎', '👏'):
            _, candidates = REACTION_BROADCAST_TABLE_ALLY[received]
            emojis = {e for e, _ in candidates}
            self.assertNotIn('😡', emojis, f"ally broadcast {received}")
            self.assertTrue(emojis & HAPPY, f"ally broadcast {received}")


if __name__ == '__main__':
    unittest.main()
