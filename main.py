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
)
from model import MercuryNet, encode_state_for, load_net
from plan_hand import plan_hand_pick
from game_state_utils import (
    detect_winner as _detect_winner,
    marbles_by_color as _marbles_by_color,
    invincible_by_color as _invincible_by_color,
)


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

# Délai minimum entre deux réactions (serveur ignore les envois sous 2 s)
REACT_COOLDOWN = 2.1

# (action_type, is_own_action) → (base_probability, [(emoji, weight)])
REACTION_ACTION_TABLE: dict[tuple[str, bool], tuple[float, list[tuple[str, int]]]] = {
    ("enter",   True):  (1/30, [("👏", 60), ("🔥", 25), ("😎", 10), ("😂",  5)]),
    ("enter",   False): (1/50, [("😮", 50), ("🤔", 30), ("😥", 15), ("😡",  5)]),
    ("move",    True):  (1/90, [("🤔", 50), ("🔥", 25), ("😎", 15), ("👏", 10)]),
    ("move",    False): (1/100, [("😴", 40), ("🥱", 35), ("🤔", 20), ("😮",  5)]),
    ("capture", True):  (1/10, [("🔥", 50), ("😎", 25), ("👏", 15), ("😂", 10)]),
    ("capture", False): (1/8,  [("😡", 55), ("😥", 25), ("😮", 15), ("😂",  5)]),
    ("promote", True):  (1/12, [("😎", 50), ("👏", 25), ("🔥", 20), ("😂",  5)]),
    ("promote", False): (1/25, [("😥", 40), ("😡", 25), ("😮", 25), ("🤔", 10)]),
    ("discard", True):  (1/25, [("😥", 40), ("🤔", 35), ("🥱", 15), ("😴", 10)]),
    ("discard", False): (1/50, [("😴", 40), ("🥱", 35), ("🤔", 15), ("😮", 10)]),
    ("swap",    True):  (1/12, [("😎", 40), ("🔥", 30), ("😂", 20), ("👏", 10)]),
    ("swap",    False): (1/10, [("🦧", 45), ("😮", 30), ("😥", 20), ("😂",  5)]),
}

# emoji reçu → (base_probability, [(emoji_réponse, weight)])
REACTION_BROADCAST_TABLE: dict[str, tuple[float, list[tuple[str, int]]]] = {
    "👏": (1/20, [("👏", 50), ("😂", 30), ("😎", 20)]),
    "😂": (1/18, [("😂", 55), ("👏", 20), ("🤔", 15), ("😮", 10)]),
    "😮": (1/25, [("😮", 50), ("😂", 25), ("🤔", 25)]),
    "😥": (1/82, [("😥", 40), ("🤔", 35), ("😮", 25)]),
    "🔥": (1/80, [("🔥", 50), ("😡", 30), ("😮", 20)]),
    "🤔": (1/30, [("🤔", 60), ("😂", 25), ("😴", 15)]),
    "😡": (1/85, [("😡", 40), ("😂", 30), ("🦧", 20), ("🥱", 10)]),
    "😎": (1/98, [("😡", 40), ("😮", 30), ("🥱", 20), ("🤔", 10)]),
    "😴": (1/35, [("😴", 50), ("🥱", 35), ("⏰", 15)]),
    "⏰": (1/25, [("⏰", 45), ("😴", 30), ("🥱", 25)]),
    "🥱": (1/30, [("🥱", 50), ("😴", 30), ("😂", 20)]),
    "🦧": (1/20, [("🦧", 100)]),
}


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
        # load_net auto-détecte l'architecture (MercuryNet 143-dim courant OU MercuryNetLegacy
        # 54-dim des anciens modèles) → compatible quel que soit le modèle déployé.
        net = load_net(MODEL_PATH, map_location="cpu", eval_mode=True)
        print(f"[model] chargé depuis {MODEL_PATH} ({type(net).__name__})")
        return net
    except Exception as e:
        if AGENT_MODE == "plan":
            print(f"[model] chargement impossible ({e!r}) — mode planificateur (sans réseau)")
            return None
        raise


# ── Bot d'inférence ───────────────────────────────────────────────────────────

class InferenceBot:
    """Joue une partie en utilisant le réseau gelé, action greedy (argmax)."""

    def __init__(self, net: MercuryNet, identity: dict):
        self.net    = net
        self.bot_id = identity["botId"]
        self.name: str          = self.bot_id  # remplacé après authenticate()
        self.picture: str       = ""            # remplacé après authenticate()
        self.session_token: str = ""            # remplacé après authenticate()
        self.user_id: str       = self.bot_id  # ID serveur, remplacé après authenticate()
        self.color: str | None  = None
        self._resolve_warned    = False        # diag _resolve_color logué une seule fois
        self._last_react_time: float = 0.0

    async def authenticate(self, client: httpx.AsyncClient) -> None:
        r = await client.post(
            f"{API_URL}/api/auth/bot",
            json={"secret": BOT_SECRET, "botId": self.bot_id},
            timeout=10.0,
        )
        r.raise_for_status()
        data = r.json()
        self.name          = data.get("name", self.bot_id)
        self.picture       = data.get("picture", "")
        self.user_id       = data.get("userId", self.bot_id)
        self.session_token = data.get("sessionToken", "")
        if not self.session_token:
            raise RuntimeError(
                f"/api/auth/bot n'a pas renvoyé de sessionToken pour botId={self.bot_id} "
                f"— le backend déployé n'est peut-être pas à jour."
            )
        print(f"[{self.name}] authToken OK (len={len(self.session_token)})", flush=True)

    def _resolve_color(self, gs: dict) -> bool:
        players = gs.get("players", [])
        # 1) Voie nominale : userId serveur (dérivé de l'authToken).
        for p in players:
            if self.user_id and p.get("userId") == self.user_id:
                self.color = p["color"]
                return True
        # 2) Repli par nom de profil. Depuis le durcissement sécurité, le serveur
        #    dérive l'identité de l'authToken et n'expose plus forcément userId dans
        #    le gameState diffusé → l'ancien match userId échouait silencieusement et
        #    le bot ne jouait jamais (timeout à chaque tour). Le nom, envoyé via
        #    playerName au join, reste un identifiant fiable côté affichage.
        for p in players:
            if self.name and p.get("name") == self.name:
                self.color = p["color"]
                return True
        # Échec total : on logue l'état réel UNE fois pour diagnostiquer en prod.
        if not self._resolve_warned:
            self._resolve_warned = True
            summary = [{"color": p.get("color"), "userId": p.get("userId"),
                        "name": p.get("name")} for p in players]
            print(f"[{self.name}] _resolve_color échec — self.user_id={self.user_id!r} "
                  f"self.name={self.name!r} players={summary}", flush=True)
        return False

    def _select_action(self, state_enc: torch.Tensor, mask: list[bool]) -> int:
        legal = torch.tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            dist, _ = self.net(state_enc, legal)
            return int(torch.argmax(dist.probs).item())

    async def _join(self, ws) -> None:
        await ws.send(json.dumps({
            "type":       "joinMatchmaking",
            "authToken":  self.session_token,
            "playerName": self.name,
            "picture":    self.picture,
        }))

    def _pick_weighted(
        self,
        candidates: list[tuple[str, int]],
        base_prob: float,
    ) -> str | None:
        if random.random() >= base_prob:
            return None
        total = sum(w for _, w in candidates)
        r = random.uniform(0, total)
        acc = 0
        for emoji, w in candidates:
            acc += w
            if acc > r:
                return emoji
        return candidates[-1][0]

    async def _schedule_reaction(self, ws, emoji: str, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            now = time.monotonic()
            if now - self._last_react_time < REACT_COOLDOWN:
                return
            await ws.send(json.dumps({
                "type":      "reaction",
                "emoji":     emoji,
                "fromColor": self.color,
            }))
            self._last_react_time = now
        except Exception:
            pass

    async def _maybe_react_action(self, ws, action: dict) -> None:
        atype  = action.get("type")
        is_own = action.get("playerColor") == self.color
        entry  = REACTION_ACTION_TABLE.get((atype, is_own))
        if entry is None:
            return
        base_prob, candidates = entry
        emoji = self._pick_weighted(candidates, base_prob)
        if emoji:
            asyncio.create_task(
                self._schedule_reaction(ws, emoji, random.uniform(1.0, 4.0))
            )

    async def _maybe_react_broadcast(self, ws, broadcast: dict) -> None:
        if broadcast.get("author") == self.color:
            return
        received = broadcast.get("emoji")
        entry = REACTION_BROADCAST_TABLE.get(received)
        if entry is None:
            return
        base_prob, candidates = entry
        emoji = self._pick_weighted(candidates, base_prob)
        if emoji:
            asyncio.create_task(
                self._schedule_reaction(ws, emoji, random.uniform(1.5, 4.0))
            )

    async def play_one_game(self, ws) -> None:
        await self._join(ws)
        anim_done = json.dumps({"type": "animationDone"})

        async for raw in ws:
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "actionPlayed":
                await ws.send(anim_done)
                if self.color:
                    await self._maybe_react_action(ws, msg.get("action", {}))
                continue

            if mtype == "reactionBroadcast":
                if self.color:
                    await self._maybe_react_broadcast(ws, msg)
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
