"""
Tests de la boucle de tour du bot (main.InferenceBot) face à un faux WebSocket :
un coup refusé par le serveur (actionRejected) ne doit plus coûter le tour entier.

Lancer : python -m unittest discover tests -v
"""

import asyncio
import contextlib
import io
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import main  # noqa: E402
from mercury_legal_moves import HOME_POSITIONS  # noqa: E402

GAME_ENDED = {'type': 'gameEnded', 'winners': [], 'reason': 'abandoned'}
REJECTED = {'type': 'actionRejected', 'reason': 'Invalid action'}


def game_state_msg(hand: list[str], message: str = 'New turn') -> dict:
    """Tour de red (le bot) : un seul pion en jeu (90), donc un coup légal par carte."""
    marbles = {'red': [90] + HOME_POSITIONS['red'][:3], 'green': HOME_POSITIONS['green'],
               'blue': HOME_POSITIONS['blue'], 'orange': HOME_POSITIONS['orange']}
    return {'type': 'gameState', 'message': message, 'gameState': {
        'gameMode': '2v2', 'currentTurn': 'red', 'canDiscard': False,
        'hand': [{'id': f'c{i}', 'value': v, 'suit': '♠'} for i, v in enumerate(hand)],
        'players': [{'color': c, 'name': 'Bot' if c == 'red' else c,
                     'marblePositions': ps, 'marbleInvincible': [False] * 4}
                    for c, ps in marbles.items()],
    }}


class FakeWS:
    """Serveur scripté : `on_play(n)` renvoie les messages émis en réponse au n-ième
    message de jeu du bot (playAction ou turnTimeout)."""

    def __init__(self, first: dict, on_play):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.queue.put_nowait(first)
        self.sent: list[dict] = []
        self.on_play = on_play

    async def send(self, raw: str) -> None:
        msg = json.loads(raw)
        self.sent.append(msg)
        if msg['type'] in ('playAction', 'turnTimeout'):
            for reply in self.on_play(len(self.game_msgs())):
                self.queue.put_nowait(reply)

    def game_msgs(self) -> list[dict]:
        return [m for m in self.sent if m['type'] in ('playAction', 'turnTimeout')]

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        return json.dumps(await self.queue.get())


class TestRejectedAction(unittest.TestCase):
    def setUp(self):
        self._think = (main.THINK_MIN, main.THINK_MAX)
        main.THINK_MIN, main.THINK_MAX = 0.0, 0.001

    def tearDown(self):
        main.THINK_MIN, main.THINK_MAX = self._think

    def _run(self, hand: list[str], on_play) -> list[dict]:
        bot = main.InferenceBot({'botId': '1'})
        bot.name = 'Bot'
        ws = FakeWS(game_state_msg(hand), on_play)
        with contextlib.redirect_stdout(io.StringIO()):      # logs du bot
            asyncio.run(asyncio.wait_for(bot.play_one_game(ws), timeout=5))
        return ws.game_msgs()

    def test_rejected_move_is_replaced_by_another_one_in_the_same_turn(self):
        sent = self._run(['5', '9'], lambda n: [REJECTED] if n == 1 else [GAME_ENDED])
        self.assertEqual([m['type'] for m in sent], ['playAction', 'playAction'])
        first, second = (m['action']['cardPlayed'][0]['id'] for m in sent)
        self.assertNotEqual(first, second, 'le coup refusé ne doit pas être renvoyé')

    def test_gives_the_turn_to_the_server_when_every_move_is_rejected(self):
        sent = self._run(['5', '9'], lambda n: [REJECTED] if n <= 2 else [GAME_ENDED])
        self.assertEqual([m['type'] for m in sent],
                         ['playAction', 'playAction', 'turnTimeout'])

    def test_stops_retrying_after_max_attempts(self):
        hand = ['2', '3', '5', '6', '9']
        limit = main.MAX_ATTEMPTS_PER_TURN
        sent = self._run(hand, lambda n: [REJECTED] if n <= limit else [GAME_ENDED])
        self.assertEqual([m['type'] for m in sent],
                         ['playAction'] * limit + ['turnTimeout'])

    def test_not_your_turn_is_not_retried(self):
        late = {'type': 'actionRejected', 'reason': 'Not your turn'}
        sent = self._run(['5', '9'], lambda n: [late, GAME_ENDED])
        self.assertEqual(len(sent), 1)

    def test_connection_update_does_not_replay_an_answered_turn(self):
        update = game_state_msg(['5', '9'], message='Connection update')
        sent = self._run(['5', '9'], lambda n: [update, GAME_ENDED])
        self.assertEqual(len(sent), 1)


if __name__ == '__main__':
    unittest.main()
