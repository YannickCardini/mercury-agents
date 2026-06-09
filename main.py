"""
Service d'agents Mercury — piloté par webhook depuis le backend.

Architecture :
  Backend matchmaking ──POST /dispatch──> ce service
  Le service choisit une identité libre, spawne un bot qui rejoint la
  matchmaking et joue une partie. Le timing (10 s, +15 s, etc.) est géré
  côté backend : il sait quand il a besoin d'un joueur.

Endpoints :
  POST /dispatch   header X-Bot-Secret      → 200 dispatché / 503 occupé
  GET  /health                              → healthcheck (Azure)
  GET  /status                              → bots actifs / disponibles

Conçu pour Azure App Service / Container Apps : config par env, pas d'état
persistant local, healthcheck inclus.
"""

import asyncio
import json
import os
import pathlib
import random
import time

import httpx
import torch
import websockets
from aiohttp import web

from mercury_legal_moves import (
    get_legal_mask, build_server_message,
    ARRIVAL_POSITIONS,
)
from model import MercuryNet, encode_state_for, load_net
from plan_hand import plan_hand_pick


# ── Configuration ─────────────────────────────────────────────────────────────

API_URL    = os.environ.get("MERCURY_API_URL", "http://localhost:8080")
WS_URL     = os.environ.get("MERCURY_WS_URL",  "ws://localhost:8080")
BOT_SECRET = os.environ.get("MERCURY_BOT_SECRET", "change-me-before-prod")
MODEL_PATH = pathlib.Path(os.environ.get("MERCURY_MODEL_PATH", "model.pt"))

# Agent joué en prod :
#   'plan' (défaut) → planificateur de main (recherche multi-cartes, niveau humain vs random,
#                     bat le réseau ~0.72). N'utilise PAS le réseau.
#   'net'           → ancien réseau réactif (argmax). Repli si besoin.
AGENT_MODE = os.environ.get("MERCURY_AGENT", "plan")

LISTEN_HOST = os.environ.get("MERCURY_AGENT_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("MERCURY_AGENT_PORT", os.environ.get("PORT", "8000")))

# Délais simulés pour imiter un joueur humain (secondes)
THINK_MIN = float(os.environ.get("MERCURY_THINK_MIN", "0.5"))
THINK_MAX = float(os.environ.get("MERCURY_THINK_MAX", "8.5"))

# Identités des bots. Les comptes correspondants existent déjà en base ;
# le nom est récupéré depuis la réponse de /api/auth/bot.
BOT_IDENTITIES: list[dict] = [{"botId": str(i)} for i in range(1, 28)]


# ── Modèle ────────────────────────────────────────────────────────────────────

def load_model():
    # En mode 'plan', le réseau n'est PAS utilisé → on tolère son absence/incompatibilité
    # (le planificateur est auto-suffisant). En mode 'net', il reste obligatoire.
    if not MODEL_PATH.exists():
        if AGENT_MODE == "plan":
            print(f"[model] {MODEL_PATH} absent — mode planificateur (sans réseau)")
            return None
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
    try:
        # load_net auto-détecte l'architecture (MercuryNet 138-dim courant OU MercuryNetLegacy
        # 54-dim des anciens modèles) → compatible quel que soit le modèle déployé.
        net = load_net(MODEL_PATH, map_location="cpu", eval_mode=True)
        print(f"[model] chargé depuis {MODEL_PATH} ({type(net).__name__})")
        return net
    except Exception as e:
        if AGENT_MODE == "plan":
            print(f"[model] chargement impossible ({e!r}) — mode planificateur (sans réseau)")
            return None
        raise


# ── Helpers d'état ────────────────────────────────────────────────────────────

def _has_won(positions: list[int], color: str) -> bool:
    return all(p in ARRIVAL_POSITIONS[color] for p in positions)


def _detect_winner(gs: dict) -> str | None:
    for p in gs["players"]:
        if _has_won(p["marblePositions"], p["color"]):
            return p["color"]
    return None


def _marbles_by_color(gs: dict) -> dict:
    return {p["color"]: p["marblePositions"] for p in gs["players"]}


def _invincible_by_color(gs: dict) -> dict:
    """Reconstruit {color: [positions invincibles]} depuis marbleInvincible,
    tableau parallèle à marblePositions envoyé par le serveur.
    Indispensable pour que le masque légal d'inférence soit IDENTIQUE à celui
    de l'entraînement (cf. bot_rl.py)."""
    out = {}
    for p in gs["players"]:
        positions = p["marblePositions"]
        inv_flags = p.get("marbleInvincible")
        # Robustesse : si marbleInvincible est absent, faux, ou pas une liste → défaut zéros
        if not isinstance(inv_flags, list):
            inv_flags = [False] * len(positions)
        out[p["color"]] = [pos for pos, inv in zip(positions, inv_flags) if inv]
    return out


# ── Bot d'inférence ───────────────────────────────────────────────────────────

class InferenceBot:
    """Joue une partie en utilisant le réseau gelé, action greedy (argmax)."""

    def __init__(self, net: MercuryNet, identity: dict):
        self.net    = net
        self.bot_id = identity["botId"]
        self.name: str   = self.bot_id  # remplacé après authenticate()
        self.picture: str = ""           # remplacé après authenticate()
        self.color: str | None = None

    async def authenticate(self, client: httpx.AsyncClient) -> None:
        r = await client.post(
            f"{API_URL}/api/auth/bot",
            json={"secret": BOT_SECRET, "botId": self.bot_id},
            timeout=10.0,
        )
        r.raise_for_status()
        data = r.json()
        self.name   = data.get("name", self.bot_id)
        self.picture = data.get("picture", "")

    def _resolve_color(self, gs: dict) -> bool:
        for p in gs["players"]:
            if p.get("userId") == self.bot_id:
                self.color = p["color"]
                return True
        return False

    def _select_action(self, state_enc: torch.Tensor, mask: list[bool]) -> int:
        legal = torch.tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            dist, _ = self.net(state_enc, legal)
            return int(torch.argmax(dist.probs).item())

    async def _join(self, ws) -> None:
        await ws.send(json.dumps({
            "type":       "joinMatchmaking",
            "playerName": self.name,
            "browserId":  self.bot_id,
            "userId":     self.bot_id,
            "picture":    self.picture,
        }))

    async def _maybe_react(self, ws, action: dict) -> None:
        atype = action.get("type")
        actor = action.get("playerColor")

        emoji = None
        if atype == "capture" and actor != self.color and random.random() < 1 / 15:
            emoji = "😡"
        elif atype == "promote" and actor == self.color and random.random() < 1 / 30:
            emoji = "😎"
        elif atype == "capture" and actor == self.color and random.random() < 1 / 15:
            emoji = "🔥"
        elif atype == "enter" and actor == self.color and random.random() < 1 / 20:
            emoji = "👏"

        if emoji:
            await ws.send(json.dumps({
                "type":      "reaction",
                "emoji":     emoji,
                "fromColor": self.color,
            }))

    async def play_one_game(self, ws) -> None:
        await self._join(ws)
        anim_done = json.dumps({"type": "animationDone"})

        async for raw in ws:
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "actionPlayed":
                await ws.send(anim_done)
                if self.color:
                    await self._maybe_react(ws, msg.get("action", {}))
                continue

            if mtype == "gameEnded":
                return

            if mtype != "gameState":
                continue

            gs = msg["gameState"]
            if not self._resolve_color(gs):
                continue

            if _detect_winner(gs) is not None:
                return

            if gs["currentTurn"] != self.color:
                continue

            hand = gs.get("hand", [])
            mbc  = _marbles_by_color(gs)
            inv_by_color = _invincible_by_color(gs)
            mask, actions = get_legal_mask(
                hand, mbc[self.color], self.color, mbc,
                invincible_by_color=inv_by_color,
                can_discard=gs.get("canDiscard", False),
            )
            if not any(mask):
                continue

            # Sélection robuste : un plantage du sélecteur NE DOIT PAS déconnecter le bot
            # (sinon il quitte la partie). En cas d'erreur → 1er coup légal de repli.
            t0 = time.monotonic()
            try:
                if AGENT_MODE == "plan":
                    action = plan_hand_pick(gs, self.color, mask, actions)
                else:
                    state_enc = encode_state_for(self.net, gs, self.color)
                    action    = self._select_action(state_enc, mask)
            except Exception as e:
                print(f"[{self.name}] erreur sélection ({e!r}) → coup légal de repli", flush=True)
                action = next(i for i, ok in enumerate(mask) if ok)
            msg_out   = build_server_message(
                action, hand, mbc[self.color], self.color, mbc,
                invincible_by_color=inv_by_color,
            )
            # Exponential distribution clipped to [THINK_MIN, THINK_MAX]: small delays are common, large ones rare
            lam = 3.0 / (THINK_MAX - THINK_MIN)
            target = THINK_MIN + min(random.expovariate(lam), THINK_MAX - THINK_MIN)
            # Le temps de CALCUL fait partie du "temps de réflexion" : on l'absorbe dans le
            # délai (avant, calcul + delay s'additionnaient → risque de dépasser le timeout
            # serveur). On dort le reste seulement.
            await asyncio.sleep(max(0.0, target - (time.monotonic() - t0)))
            await ws.send(json.dumps(msg_out))


async def run_bot_session(net: MercuryNet, identity: dict) -> None:
    """Cycle complet : auth → WS → 1 partie → close. Robuste aux erreurs."""
    bot = InferenceBot(net, identity)
    print("connection du bot", bot.bot_id)
    try:
        async with httpx.AsyncClient() as client:
            await bot.authenticate(client)
        async with websockets.connect(WS_URL) as ws:
            await bot.play_one_game(ws)
        print(f"[{bot.name}] partie terminée")
    except Exception as e:
        print(f"[bot {identity['botId']}] erreur de session : {e}")


# ── Serveur HTTP (webhook depuis le backend) ──────────────────────────────────

def _cleanup_active(active: dict) -> None:
    for bid in list(active):
        if active[bid].done():
            del active[bid]


def _next_free_identity(active: dict) -> dict | None:
    free = [ident for ident in BOT_IDENTITIES if ident["botId"] not in active]
    if not free:
        return None
    return random.choice(free)


async def dispatch_handler(request: web.Request) -> web.Response:
    if request.headers.get("X-Bot-Secret") != BOT_SECRET:
        return web.json_response({"error": "unauthorized"}, status=401)

    active: dict = request.app["active"]
    _cleanup_active(active)

    ident = _next_free_identity(active)
    if ident is None:
        return web.json_response({"status": "busy", "active": list(active)},
                                 status=503)

    net = request.app["net"]
    active[ident["botId"]] = asyncio.create_task(run_bot_session(net, ident))
    print(f"[dispatch] bot {ident['botId']}")
    return web.json_response({
        "status": "dispatched",
        "botId":  ident["botId"],
    })


async def health_handler(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def status_handler(request: web.Request) -> web.Response:
    active: dict = request.app["active"]
    _cleanup_active(active)
    return web.json_response({
        "active":    list(active),
        "available": [i["botId"] for i in BOT_IDENTITIES
                      if i["botId"] not in active],
    })


def make_app() -> web.Application:
    app = web.Application()
    app["net"]    = load_model()
    app["active"] = {}
    app.router.add_post("/dispatch", dispatch_handler)
    app.router.add_get("/health",    health_handler)
    app.router.add_get("/status",    status_handler)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host=LISTEN_HOST, port=LISTEN_PORT)
