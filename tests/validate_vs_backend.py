"""
Validation d'intégration : 4 bots planificateurs jouent des parties complètes contre le
backend local. Ils rejoignent le matchmaking en INVITÉS (sans authToken) : aucun compte
bot requis, ZÉRO écriture en base (pas de userId → pas de points Elo). Vérifie :
  - zéro actionRejected, zéro timeout (coup imposé par le serveur),
  - le bot continue de jouer après avoir rentré ses 4 pions (switch 2v2),
  - la partie se termine sur gameEnded avec 2 winners (2v2) / 1 winner (1v3).
Sur actionRejected, le contexte complet (gameState + message envoyé) est dumpé en JSON
dans le répertoire courant pour diagnostic.

Prérequis : backend local lancé avec le mode voulu, ex. depuis mercury/backend :
  GAME_MODE=2v2 TRAIN_MODE=true npx tsx src/index.ts     # ou GAME_MODE=1v3

Usage : python tests/validate_vs_backend.py [nb_parties]
"""

import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import websockets  # noqa: E402

from mercury_legal_moves import (  # noqa: E402
    get_legal_mask, build_server_message,
    get_controlled_color, get_teammate_color, SOLIDAIRE_IDX,
)
from plan_hand import plan_hand_pick  # noqa: E402
from game_state_utils import (  # noqa: E402
    marbles_by_color, invincible_by_color,
)

WS_URL = "ws://localhost:8080"


class ValBot:
    def __init__(self, name: str):
        self.name = name
        self.color = None
        self.mode = None
        self.rejected = 0        # actionRejected reçus (DOIT rester 0)
        self.timeouts = 0        # coups imposés par timeout serveur sur MES tours (DOIT rester 0)
        self.turns = 0
        self.switch_turns = 0    # tours joués APRÈS avoir rentré mes 4 pions (2v2)
        self.solidaire_plays = 0
        self.winners = None
        self.last_sent = None    # (gs, idx, msg_out) du dernier coup envoyé (diagnostic)

    async def run(self):
        async with websockets.connect(WS_URL, max_size=None) as ws:
            await ws.send(json.dumps({"type": "joinMatchmaking",
                                      "playerName": self.name}))
            async for raw in ws:
                msg = json.loads(raw)
                t = msg.get("type")

                if t == "actionRejected":
                    self.rejected += 1
                    print(f"  !! [{self.name}] actionRejected: {msg.get('reason')}")
                    if self.last_sent is not None:
                        fn = f"rejected_{self.name}_{self.turns}.json"
                        with open(fn, "w") as f:
                            json.dump(self.last_sent, f, indent=1)
                        print(f"  !! contexte dumpé dans {fn}")
                    continue
                if t == "actionPlayed":
                    a = msg.get("action", {})
                    if a.get("playerColor") == self.color and msg.get("isTimeout"):
                        self.timeouts += 1
                        print(f"  !! [{self.name}] timeout serveur (coup imposé)")
                    continue
                if t == "gameEnded":
                    self.winners = msg.get("winners")
                    return
                if t != "gameState":
                    continue

                gs = msg["gameState"]
                self.mode = gs.get("gameMode", self.mode)
                if self.color is None:
                    for p in gs.get("players", []):
                        if p.get("name") == self.name:
                            self.color = p["color"]
                if self.color is None or gs.get("currentTurn") != self.color:
                    continue
                hand = gs.get("hand", [])
                if not hand:
                    continue  # main vide : le serveur passe tout seul

                mbc = marbles_by_color(gs)
                inv = invincible_by_color(gs)
                if self.mode == "2v2":
                    controlled = get_controlled_color(self.color, mbc[self.color])
                    teammate = get_teammate_color(controlled)
                    if controlled != self.color:
                        self.switch_turns += 1
                else:
                    controlled, teammate = self.color, None

                mask, actions = get_legal_mask(
                    hand, mbc[controlled], controlled, mbc,
                    invincible_by_color=inv,
                    can_discard=gs.get("canDiscard", False),
                    teammate_color=teammate,
                )
                if not any(mask):
                    continue
                idx = plan_hand_pick(gs, controlled, mask, actions,
                                     teammate_color=teammate)
                if teammate is not None and idx == SOLIDAIRE_IDX:
                    self.solidaire_plays += 1
                out = build_server_message(
                    idx, hand, mbc[controlled], controlled, mbc,
                    invincible_by_color=inv, teammate_color=teammate,
                )
                self.turns += 1
                if self.turns > 2500:
                    print(f"  !! [{self.name}] plafond de coups atteint, abandon")
                    return
                self.last_sent = {"bot": self.name, "color": self.color,
                                  "controlled": controlled, "idx": idx,
                                  "out": out, "gs": gs}
                await ws.send(json.dumps(out))


async def one_game(game_no: int) -> dict:
    bots = [ValBot(f"VAL{game_no}-{c}") for c in ("A", "B", "C", "D")]
    await asyncio.wait_for(asyncio.gather(*(b.run() for b in bots)), timeout=600)
    mode = next(b.mode for b in bots)
    winners = next((b.winners for b in bots if b.winners is not None), None)
    rejected = sum(b.rejected for b in bots)
    timeouts = sum(b.timeouts for b in bots)
    switch = sum(b.switch_turns for b in bots)
    solidaire = sum(b.solidaire_plays for b in bots)
    turns = sum(b.turns for b in bots)
    print(f"[partie {game_no}] mode={mode} winners={winners} coups={turns} "
          f"rejected={rejected} timeouts={timeouts} "
          f"tours_apres_switch={switch} solidaires={solidaire}")
    ok = (rejected == 0 and timeouts == 0 and winners is not None
          and len(winners) == (2 if mode == "2v2" else 1))
    if mode == "2v2":
        colors = {b.color for b in bots}
        w = set(winners or [])
        ok = ok and w <= colors and len(w) == 2
    return {"ok": ok, "mode": mode, "switch": switch, "solidaire": solidaire}


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    results = [await one_game(i + 1) for i in range(n)]
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n=> {n_ok}/{len(results)} parties valides "
          f"(tours_apres_switch cumulés={sum(r['switch'] for r in results)}, "
          f"solidaires={sum(r['solidaire'] for r in results)})")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
