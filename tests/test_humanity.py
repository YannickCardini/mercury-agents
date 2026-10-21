"""
Comportements « humains » des bots : persona figée par identité (emojis
possédés, bavardage, rythme), temps de réflexion selon le coup, choix
d'identité qui évite les bots déjà croisés par les humains de la file, et pool
relu en base (ajout d'un bot sans redéploiement).
"""

import asyncio
import json
import pathlib
import random
import statistics
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import main  # noqa: E402
from persona import (  # noqa: E402
    CHATTINESS_TIERS, FREE_EMOJIS, OWNERSHIP_TIERS, PAID_EMOJIS, PERSONAS_PATH,
    Persona, load_personas, parse_persona, persona_for,
)
from pool import BotPool  # noqa: E402

from tests.test_reactions import SERVER_PALETTE  # noqa: E402


class TestFrozenPersonas(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))

    def test_every_pool_identity_has_a_frozen_persona(self):
        pool = {ident["botId"] for ident in main.BOT_IDENTITIES}
        self.assertTrue(pool <= set(self.raw), f"manquants : {pool - set(self.raw)}")

    def test_schema_and_palette(self):
        self.assertEqual(set(FREE_EMOJIS) | set(PAID_EMOJIS), SERVER_PALETTE)
        styles = {name: mult for name, mult, _ in CHATTINESS_TIERS}
        for bot_id, data in self.raw.items():
            persona = Persona.from_dict(data)
            self.assertEqual(persona.chattiness, styles[persona.style], bot_id)
            self.assertTrue(persona.owned <= set(PAID_EMOJIS), bot_id)
            if persona.favorite is not None:
                self.assertTrue(persona.can_use(persona.favorite), bot_id)
            self.assertTrue(0.75 <= persona.pace <= 1.35, bot_id)

    def test_pool_mixes_characters(self):
        styles = [p["style"] for p in self.raw.values()]
        for name, _, share in CHATTINESS_TIERS:
            self.assertLessEqual(abs(styles.count(name) - share * len(styles)), 1, name)
        owned = [len(p["owned"]) for p in self.raw.values()]
        for name, (low, high), share in OWNERSHIP_TIERS:
            count = sum(1 for n in owned if low <= n <= high)
            self.assertLessEqual(abs(count - share * len(owned)), 1, name)

    def test_unknown_bot_gets_a_stable_fallback(self):
        self.assertEqual(persona_for("unknown-bot"), persona_for("unknown-bot"))
        self.assertEqual(load_personas(pathlib.Path("/nonexistent/personas.json")), {})


class TestPersonaFromDatabase(unittest.TestCase):
    DB = {"style": "quiet", "chattiness": 0.5, "owned": ["😎", "⏰"], "favorite": "😎", "pace": 0.8}

    def test_database_persona_wins_over_local_file(self):
        persona = persona_for("1", self.DB)
        self.assertEqual(persona.owned, frozenset({"😎", "⏰"}))
        self.assertEqual(persona.pace, 0.8)
        self.assertNotEqual(persona, persona_for("1"), "personas.json diffère pour le bot 1")

    def test_new_bot_from_database(self):
        self.assertEqual(persona_for("104729384756102938475", self.DB).style, "quiet")

    def test_invalid_database_persona_falls_back(self):
        for bad in ({**self.DB, "owned": ["👏"]}, {**self.DB, "style": "loud"},
                    {**self.DB, "favorite": "🦧"}, {"style": "quiet"}, "nope"):
            self.assertIsNone(parse_persona(bad), bad)
        self.assertEqual(persona_for("1", {"style": "loud"}), persona_for("1"))


class TestBotPool(unittest.TestCase):
    def _pool(self, responses, ttl=60.0):
        calls = []

        async def fetch():
            calls.append(1)
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return BotPool("http://api", "secret", ["1", "2"], ttl=ttl, fetch=fetch), calls

    def test_static_until_backend_answers(self):
        pool, _ = self._pool([RuntimeError("404")])
        self.assertEqual(pool.ids(), ["1", "2"])
        asyncio.run(pool.refresh())
        self.assertEqual(pool.ids(), ["1", "2"])
        self.assertEqual(pool.source, "static")

    def test_backend_pool_with_new_bot_and_persona(self):
        persona = TestPersonaFromDatabase.DB
        pool, _ = self._pool([[{"id": "1"}, {"id": "104729384756102938475", "persona": persona}]])
        asyncio.run(pool.refresh())
        self.assertEqual(pool.ids(), ["1", "104729384756102938475"])
        self.assertEqual(pool.persona_data("104729384756102938475"), persona)
        self.assertIsNone(pool.persona_data("1"))
        self.assertEqual(pool.source, "backend")

    def test_refreshes_at_most_once_per_ttl(self):
        pool, calls = self._pool([[{"id": "1"}], [{"id": "1"}, {"id": "9"}]], ttl=60.0)
        asyncio.run(pool.refresh())
        asyncio.run(pool.refresh())
        self.assertEqual(len(calls), 1)

    def test_failure_keeps_last_known_pool(self):
        pool, _ = self._pool([[{"id": "7"}, {"id": "8"}], RuntimeError("down"), []], ttl=0.0)
        asyncio.run(pool.refresh())
        asyncio.run(pool.refresh())
        self.assertEqual(pool.ids(), ["7", "8"])
        asyncio.run(pool.refresh())
        self.assertEqual(pool.ids(), ["7", "8"], "une liste vide ne vide pas le pool")

    def test_dispatch_picks_from_pool(self):
        ident = main._next_free_identity({"7": {}}, frozenset(), ["7", "104729384756102938475"])
        self.assertEqual(ident, {"botId": "104729384756102938475"})


class TestPersonaReactions(unittest.TestCase):
    def _bot(self, **persona):
        bot = main.InferenceBot({"botId": "1"})
        base = {"style": "chatty", "chattiness": 2.0, "owned": [], "favorite": None, "pace": 1.0}
        bot.persona = Persona.from_dict({**base, **persona})
        return bot

    def test_never_sends_an_unowned_paid_emoji(self):
        bot = self._bot(owned=["😎"])
        tables = [*main.REACTION_ACTION_TABLE.values(), *main.REACTION_ACTION_TABLE_ALLY.values(),
                  *main.REACTION_BROADCAST_TABLE.values(), *main.REACTION_BROADCAST_TABLE_ALLY.values()]
        seen = set()
        random.seed(3)
        for _ in range(200):
            for _, candidates in tables:
                emoji = bot._pick_weighted(candidates, 1.0)
                if emoji:
                    seen.add(emoji)
        self.assertTrue(seen <= set(FREE_EMOJIS) | {"😎"}, seen)
        self.assertIn("😎", seen)

    def test_only_paid_candidates_and_none_owned_means_silence(self):
        bot = self._bot()
        self.assertIsNone(bot._pick_weighted([("🦧", 100)], 1.0))

    def test_chattiness_scales_reaction_rate(self):
        random.seed(5)
        candidates = [("👏", 1)]
        silent = self._bot(style="silent", chattiness=0.15)
        chatty = self._bot(style="chatty", chattiness=2.0)
        n = 20_000
        silent_rate = sum(bool(silent._pick_weighted(candidates, 0.1)) for _ in range(n)) / n
        chatty_rate = sum(bool(chatty._pick_weighted(candidates, 0.1)) for _ in range(n)) / n
        self.assertAlmostEqual(silent_rate, 0.015, delta=0.005)
        self.assertAlmostEqual(chatty_rate, 0.2, delta=0.02)

    def test_favorite_is_preferred(self):
        random.seed(9)
        bot = self._bot(favorite="🔥")
        picks = [bot._pick_weighted([("👏", 50), ("🔥", 50)], 1.0) for _ in range(4_000)]
        self.assertGreater(picks.count("🔥") / len(picks), 0.7)


class TestThinkDelay(unittest.TestCase):
    def _samples(self, n_legal, first_turn=False, pace=1.0, count=4_000, seed=1):
        rng = random.Random(seed)
        return [main.think_delay(n_legal, first_turn, pace, rng) for _ in range(count)]

    def test_bounds(self):
        for n in (1, 2, 8, 64, 400):
            for first in (False, True):
                for d in self._samples(n, first, pace=1.35, count=1_000):
                    self.assertGreaterEqual(d, main.THINK_MIN)
                    self.assertLessEqual(d, main.THINK_MAX)
        self.assertLess(main.THINK_MAX, 30, "très en dessous du timeout serveur (~60 s)")

    def test_forced_move_is_quick(self):
        forced = self._samples(1)
        self.assertLess(statistics.median(forced), 1.3)

    def test_median_grows_with_choice(self):
        medians = [statistics.median(self._samples(n)) for n in (2, 8, 64)]
        self.assertLess(medians[0], medians[1])
        self.assertLess(medians[1], medians[2])
        self.assertAlmostEqual(medians[2], 5.4, delta=0.6)

    def test_first_turn_and_pace_slow_down(self):
        self.assertGreater(statistics.median(self._samples(8, first_turn=True)),
                           statistics.median(self._samples(8)) + 1.0)
        self.assertGreater(statistics.median(self._samples(8, pace=1.3)),
                           statistics.median(self._samples(8, pace=0.8)))


class TestIdentityChoice(unittest.TestCase):
    def test_avoids_recent_bots_when_possible(self):
        avoid = frozenset(str(i) for i in range(1, 26))
        with mock.patch.object(main.random, "choice", side_effect=lambda seq: seq[0]):
            ident = main._next_free_identity({}, avoid)
        self.assertNotIn(ident["botId"], avoid)

    def test_falls_back_to_any_free_identity(self):
        everyone = frozenset(ident["botId"] for ident in main.BOT_IDENTITIES)
        active = {ident["botId"]: {} for ident in main.BOT_IDENTITIES[1:]}
        ident = main._next_free_identity(active, everyone)
        self.assertEqual(ident["botId"], main.BOT_IDENTITIES[0]["botId"])

    def test_none_when_pool_is_busy(self):
        active = {ident["botId"]: {} for ident in main.BOT_IDENTITIES}
        self.assertIsNone(main._next_free_identity(active))


if __name__ == "__main__":
    unittest.main()
