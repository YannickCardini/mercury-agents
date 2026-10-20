"""
Service d'agents Mercury — piloté par webhook depuis le backend.

Architecture :
  Backend matchmaking ──POST /dispatch──> ce service
  Le service choisit une identité libre (de préférence hors `avoidUserIds`,
  les bots déjà croisés récemment par les humains de la file), spawne un bot
  qui rejoint la matchmaking et joue une partie. Le timing d'arrivée est
  scripté côté backend (matchmaking-script.ts) : il sait quand il a besoin
  d'un joueur. Un bot renvoyé de la file (socket fermée avant `welcome`)
  libère simplement son identité.

Endpoints :
  POST /dispatch   header X-Bot-Secret      → 200 dispatché / 503 occupé
  GET  /health                              → healthcheck (Azure)
  GET  /status                              → bots actifs / disponibles

Conçu pour Azure App Service / Container Apps : config par env, pas d'état
persistant local, healthcheck inclus.
"""

import asyncio
import json
import math
import os
import random
import time

import httpx
import websockets
from aiohttp import web

from mercury_legal_moves import (
    get_legal_mask, build_server_message,
    get_controlled_color, get_teammate_color, same_team,
)
from plan_hand import plan_hand_pick
from persona import FAVORITE_WEIGHT, persona_for
from game_state_utils import (
    detect_winners as _detect_winners,
    marbles_by_color as _marbles_by_color,
    invincible_by_color as _invincible_by_color,
)


# ── Configuration ─────────────────────────────────────────────────────────────

API_URL    = os.environ.get("MERCURY_API_URL", "http://localhost:8080")
WS_URL     = os.environ.get("MERCURY_WS_URL",  "ws://localhost:8080")
BOT_SECRET = os.environ.get("MERCURY_BOT_SECRET", "change-me-before-prod")

# Modes de jeu supportés — le backend décide (env GAME_MODE côté serveur) et l'annonce
# via le body du dispatch puis gameState.gameMode (source de vérité EN partie).
GAME_MODES        = ("1v3", "2v2")
DEFAULT_GAME_MODE = "2v2"   # défaut serveur

LISTEN_HOST = os.environ.get("MERCURY_AGENT_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("MERCURY_AGENT_PORT", os.environ.get("PORT", "8000")))

# Temps de réflexion simulé (secondes), bornes surchargeables par l'environnement.
# Le serveur impose un coup au bout d'environ 60 s : rester très en dessous.
THINK_MIN = float(os.environ.get("MERCURY_THINK_MIN", "0.5"))
THINK_MAX = float(os.environ.get("MERCURY_THINK_MAX", "14"))
# Coup unique ou forcé : le temps de voir la carte et de la poser.
THINK_FORCED = (0.6, 1.5)
# Rare hésitation (distraction, relecture du plateau) : chance, délai ajouté.
THINK_HESITATION = (0.04, (3.0, 6.0))


def think_delay(n_legal: int, first_turn: bool, pace: float,
                rng: random.Random | None = None) -> float:
    """Temps de réflexion d'un humain pour un coup parmi `n_legal` possibles.

    Un coup forcé part vite ; sinon la médiane croît avec le logarithme du nombre
    de coups (2 coups ≈ 1,9 s, 64 coups ≈ 5,4 s), avec une dispersion log-normale
    et le rythme propre au bot (persona). Le premier tour de la partie est un peu
    plus long (on découvre sa main et le plateau)."""
    rng = rng or random
    if n_legal <= 1:
        delay = rng.uniform(*THINK_FORCED)
    else:
        median = 1.2 + 0.7 * math.log2(n_legal)
        delay = median * math.exp(0.45 * rng.gauss(0.0, 1.0)) * pace
    if first_turn:
        delay += rng.uniform(1.0, 3.0)
    chance, extra = THINK_HESITATION
    if rng.random() < chance:
        delay += rng.uniform(*extra)
    return max(THINK_MIN, min(THINK_MAX, delay))

# Identités des bots. Les comptes correspondants existent déjà en base ;
# le nom est récupéré depuis la réponse de /api/auth/bot.
BOT_IDENTITIES: list[dict] = [{"botId": str(i)} for i in range(1, 28)]

# Délai minimum entre deux réactions (serveur ignore les envois sous 2 s)
REACT_COOLDOWN = 2.1

# Temps de réaction avant d'envoyer un emoji, en secondes : (min, max, le plus
# fréquent) d'un tirage triangulaire — souvent proche du mode, parfois plus long.
# Après une ACTION, les joueurs ne voient le résultat qu'une fois la carte posée
# (~1,6 s) et le pion déplacé : on laisse d'abord passer l'animation.
REACT_ACTION_DELAY = (4.0, 10.0, 5.5)
# En réponse à un EMOJI (affiché tout de suite) : le temps de le lire et de répondre.
REACT_REPLY_DELAY  = (2.5, 7.0, 3.5)

# Coups tentés au maximum dans un même tour. Quand le serveur refuse un coup
# (actionRejected), le tour reste ouvert : on rejoue aussitôt le meilleur coup
# restant au lieu de laisser filer les 60 s du tour. Au-delà, on demande au serveur
# d'imposer son coup tout de suite (cf. InferenceBot._give_up_turn).
MAX_ATTEMPTS_PER_TURN = 4


def _ts() -> str:
    """Horodatage court pour corréler les logs agent ↔ logs serveur."""
    return time.strftime("%H:%M:%S")

# ── Réactions emoji ───────────────────────────────────────────────────────────
# La relation à l'auteur d'une action/réaction est à TROIS niveaux :
#   'own'   → moi (playerColor == ma couleur — inchangé en 2v2 : c'est bien le
#             joueur qui a joué la carte, même si le pion déplacé est au coéquipier)
#   'ally'  → mon coéquipier (2v2 uniquement : red+blue vs green+orange)
#   'enemy' → tout le reste (= tous les autres en 1v3, comportement historique)
# Les tables historiques (own/enemy) sont INCHANGÉES ; les tables _ALLY ne sont
# consultées qu'en 2v2. Palette = REACTION_EMOJIS du serveur (15 emojis, dont
# 9 de boutique) ; chaque bot ne pioche que dans les emojis de sa persona.

def _reaction_relation(my_color: str, other_color: str | None, game_mode: str) -> str:
    """Relation emoji à l'auteur d'une action ou d'une réaction (cf. bloc ci-dessus)."""
    if other_color == my_color:
        return "own"
    if game_mode == "2v2" and other_color is not None and same_team(other_color, my_color):
        return "ally"
    return "enemy"


# (action_type, is_own_action) → (base_probability, [(emoji, weight)])
# True = 'own', False = 'enemy' (l'allié 2v2 a sa table dédiée ci-dessous).
REACTION_ACTION_TABLE: dict[tuple[str, bool], tuple[float, list[tuple[str, int]]]] = {
    ("enter",   True):  (1/30, [("👏", 60), ("🔥", 25), ("😎", 10), ("😊",  8), ("😂",  5)]),
    ("enter",   False): (1/50, [("😮", 50), ("🤔", 30), ("😥", 15), ("😡",  5)]),
    ("move",    True):  (1/90, [("🤔", 50), ("🔥", 25), ("😎", 15), ("👏", 10)]),
    ("move",    False): (1/100, [("😴", 40), ("🥱", 35), ("🤔", 20), ("😮",  5)]),
    ("capture", True):  (1/10, [("🔥", 50), ("😎", 25), ("👏", 15), ("😂", 10)]),
    ("capture", False): (1/8,  [("😡", 55), ("😥", 25), ("😮", 15), ("😂",  5)]),
    ("promote", True):  (1/12, [("😎", 50), ("👏", 25), ("🔥", 20), ("😊",  8), ("😂",  5)]),
    ("promote", False): (1/25, [("😥", 40), ("😡", 25), ("😮", 25), ("🤔", 10)]),
    ("discard", True):  (1/25, [("😥", 40), ("🤔", 35), ("🥱", 15), ("😴", 10)]),
    ("discard", False): (1/50, [("😴", 40), ("🥱", 35), ("🤔", 15), ("😮", 10)]),
    ("swap",    True):  (1/12, [("😎", 40), ("🔥", 30), ("😂", 20), ("😉", 12), ("👏", 10)]),
    ("swap",    False): (1/10, [("🦧", 45), ("😮", 30), ("😥", 20), ("😂",  5)]),
}

# action_type → (base_probability, [(emoji, weight)]) quand l'auteur est mon
# COÉQUIPIER (2v2). Ses succès sont NOS succès : entrée/promotion/capture → joie
# (👏🔥😎), jamais la tristesse/colère des tables « enemy » ; sa défausse → compassion.
# NB : une capture alliée peut, rarement, être du friendly-fire (7 partagé) — l'action
# diffusée ne porte pas la victime, on assume le cas courant (capture d'un ennemi).
REACTION_ACTION_TABLE_ALLY: dict[str, tuple[float, list[tuple[str, int]]]] = {
    "enter":   (1/35, [("👏", 55), ("🔥", 25), ("😎", 15), ("😊", 10), ("😂",  5)]),
    "move":    (1/110, [("🤔", 40), ("👏", 25), ("😎", 20), ("🥱", 15)]),
    "capture": (1/12, [("🔥", 50), ("👏", 25), ("😂", 15), ("😎", 10)]),
    "promote": (1/12, [("👏", 50), ("🔥", 25), ("😎", 20), ("😘",  8), ("😂",  5)]),
    "discard": (1/40, [("😥", 45), ("🤔", 30), ("😮", 15), ("🥱", 10)]),
    "swap":    (1/12, [("😎", 40), ("🔥", 30), ("👏", 20), ("😉", 12), ("😂", 10)]),
}

# emoji reçu → (base_probability, [(emoji_réponse, weight)])
# Table historique = réponses à un ADVERSAIRE (et tout le monde en 1v3).
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
    "😉": (1/25, [("😉", 40), ("😂", 35), ("🤔", 25)]),
    "😘": (1/30, [("😮", 40), ("😂", 35), ("😉", 25)]),
    "😊": (1/30, [("😊", 50), ("👏", 30), ("😂", 20)]),
}

# Réponses à l'emoji de mon COÉQUIPIER (2v2) : on fait équipe — on amplifie ses
# 🔥/👏/😎 (au lieu de la jalousie 😡 des tables adverses), on console ses 😥/😡,
# et on rejoint ses pitreries. Probabilités un peu plus hautes : on se répond
# davantage entre coéquipiers qu'à des adversaires.
REACTION_BROADCAST_TABLE_ALLY: dict[str, tuple[float, list[tuple[str, int]]]] = {
    "👏": (1/18, [("👏", 60), ("😂", 25), ("🔥", 15), ("😊", 10)]),
    "😂": (1/16, [("😂", 60), ("👏", 25), ("😎", 15)]),
    "😮": (1/25, [("😮", 50), ("😂", 30), ("🤔", 20)]),
    "😥": (1/30, [("😥", 40), ("👏", 30), ("🤔", 30)]),
    "🔥": (1/25, [("🔥", 60), ("👏", 25), ("😎", 15)]),
    "🤔": (1/30, [("🤔", 55), ("😂", 25), ("👏", 20)]),
    "😡": (1/40, [("😥", 35), ("🤔", 30), ("😮", 20), ("😂", 15)]),
    "😎": (1/40, [("👏", 40), ("🔥", 30), ("😂", 20), ("😎", 10)]),
    "😴": (1/35, [("😂", 40), ("😴", 30), ("⏰", 30)]),
    "⏰": (1/30, [("⏰", 45), ("😂", 30), ("😴", 25)]),
    "🥱": (1/30, [("🥱", 40), ("😂", 35), ("😴", 25)]),
    "🦧": (1/20, [("🦧", 60), ("😂", 40)]),
    "😉": (1/20, [("😉", 45), ("😂", 35), ("😎", 20)]),
    "😘": (1/18, [("😘", 45), ("😊", 30), ("😂", 25)]),
    "😊": (1/20, [("😊", 50), ("👏", 30), ("😘", 20)]),
}


# ── Bot d'inférence ───────────────────────────────────────────────────────────

class InferenceBot:
    """Joue une partie en utilisant le planificateur de main (plan_hand.plan_hand_pick) :
    recherche en faisceau de la meilleure séquence de cartes sur la main courante."""

    def __init__(self, identity: dict, game_mode: str = DEFAULT_GAME_MODE):
        # Hint du dispatch — écrasé par gameState.gameMode (source de vérité) en partie.
        self.game_mode = game_mode
        self.bot_id = identity["botId"]
        # Habitudes du bot (emojis, bavardage, rythme), les mêmes à chaque partie.
        self.persona = persona_for(self.bot_id)
        self._turns_played = 0
        # Vrai dès que le serveur a confirmé le siège (`welcome`) : sinon une
        # fermeture de socket est un renvoi de la file, pas une fin de partie.
        self.welcomed = False
        self.name: str          = self.bot_id  # remplacé après authenticate()
        self.picture: str       = ""            # remplacé après authenticate()
        self.session_token: str = ""            # remplacé après authenticate()
        self.user_id: str       = self.bot_id  # ID serveur, remplacé après authenticate()
        self.color: str | None  = None
        self._resolve_warned    = False        # diag _resolve_color logué une seule fois
        self._last_react_time: float = 0.0
        # Mon tour en cours : {"gs", "rejected" (index refusés par le serveur),
        # "sent" / "sent_msg" (index et message du dernier coup envoyé)}. None hors
        # de mon tour.
        self._turn: dict | None = None

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
        """Tirage d'une réaction selon la persona : probabilité modulée par son
        bavardage, réponses limitées aux emojis gratuits et à ceux qu'elle
        possède, emoji fétiche favorisé."""
        persona = self.persona
        if random.random() >= min(1.0, base_prob * persona.chattiness):
            return None
        usable = [(emoji, w * (FAVORITE_WEIGHT if emoji == persona.favorite else 1))
                  for emoji, w in candidates if persona.can_use(emoji)]
        if not usable:
            return None
        total = sum(w for _, w in usable)
        r = random.uniform(0, total)
        acc = 0
        for emoji, w in usable:
            acc += w
            if acc > r:
                return emoji
        return usable[-1][0]

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
        atype    = action.get("type")
        # Relation jugée sur playerColor (le joueur qui a joué la carte) — en 2v2 le
        # pion déplacé peut appartenir au coéquipier (switch/solidaire/7 partagé),
        # mais on réagit au JOUEUR, pas au pion. Un promote du coéquipier → joie
        # (table ALLY), plus jamais la tristesse/colère de la table adverse.
        relation = _reaction_relation(self.color, action.get("playerColor"),
                                      self.game_mode)
        if relation == "ally":
            entry = REACTION_ACTION_TABLE_ALLY.get(atype)
        else:
            entry = REACTION_ACTION_TABLE.get((atype, relation == "own"))
        if entry is None:
            return
        base_prob, candidates = entry
        emoji = self._pick_weighted(candidates, base_prob)
        if emoji:
            asyncio.create_task(
                self._schedule_reaction(ws, emoji, random.triangular(*REACT_ACTION_DELAY))
            )

    async def _maybe_react_broadcast(self, ws, broadcast: dict) -> None:
        author = broadcast.get("author")
        if author == self.color:
            return
        received = broadcast.get("emoji")
        if _reaction_relation(self.color, author, self.game_mode) == "ally":
            entry = REACTION_BROADCAST_TABLE_ALLY.get(received)
        else:
            entry = REACTION_BROADCAST_TABLE.get(received)
        if entry is None:
            return
        base_prob, candidates = entry
        emoji = self._pick_weighted(candidates, base_prob)
        if emoji:
            asyncio.create_task(
                self._schedule_reaction(ws, emoji, random.triangular(*REACT_REPLY_DELAY))
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
                # BREAKING protocole : `winner` n'existe plus, `winners` est une liste
                # (1 couleur en 1v3, les 2 de l'équipe en 2v2, [] si abandon). Le bot
                # d'inférence n'exploite pas le résultat : fin de partie = on sort.
                print(f"[{_ts()}][{self.name}] gameEnded reçu "
                      f"(winners={msg.get('winners')}, reason={msg.get('reason')})", flush=True)
                return

            if mtype == "welcome":
                # Confirme le seat (lancement normal ou re-seat après restart serveur).
                self.welcomed = True
                print(f"[{_ts()}][{self.name}] welcome — gameId={msg.get('gameId')} "
                      f"couleur={msg.get('myColor')}", flush=True)
                continue

            if mtype == "actionRejected":
                await self._on_action_rejected(ws, msg)
                continue

            if mtype != "gameState":
                continue

            gs = msg["gameState"]
            # Source de vérité EN partie (écrase le hint du dispatch).
            self.game_mode = gs.get("gameMode", self.game_mode)
            if not self._resolve_color(gs):
                continue

            # Garde-fou local MODE-AWARE uniquement : en 2v2 un joueur qui a rentré
            # ses 4 pions ne termine PAS la partie (il joue ceux du coéquipier).
            # La vraie fin de partie est le message gameEnded ci-dessus.
            if _detect_winners(gs):
                # Sortie silencieuse = tours ratés côté serveur si le garde-fou
                # se trompe : toujours loguer pourquoi on quitte la partie.
                print(f"[{_ts()}][{self.name}] fin détectée localement "
                      f"(winners={_detect_winners(gs)}) — je quitte la partie", flush=True)
                return

            if gs["currentTurn"] != self.color:
                self._turn = None
                continue
            if not gs.get("hand"):
                self._turn = None
                continue    # main vide : le serveur passe mon tour tout seul
            # « Connection update » rediffuse le tour EN COURS (un joueur s'est
            # (dé)connecté), ce n'est pas un nouveau tour : si j'y ai déjà répondu,
            # ne pas rejouer à partir de cet état, périmé dès que mon coup est appliqué.
            if msg.get("message") == "Connection update" and self._turn is not None:
                continue

            self._turn = {"gs": gs, "rejected": set(), "sent": None}
            await self._play_turn(ws, think=True)

    async def _play_turn(self, ws, think: bool) -> None:
        """Choisit et envoie un coup pour le tour en cours (self._turn), en écartant
        les coups déjà refusés par le serveur pendant ce tour. `think` : simuler le
        temps de réflexion d'un humain (1re tentative seulement)."""
        turn = self._turn
        gs   = turn["gs"]
        hand = gs["hand"]
        mbc  = _marbles_by_color(gs)
        inv_by_color = _invincible_by_color(gs)

        # 2v2 — « switch » de fin de jeu : quand j'ai rentré mes 4 pions, mes
        # cartes contrôlent les pions du coéquipier. Tout le contexte (couleur,
        # positions, règles d'équipe) bascule sur la couleur contrôlée, comme
        # buildLegalMoveContext côté serveur. En 1v3 : controlled = ma couleur.
        if self.game_mode == "2v2":
            controlled = get_controlled_color(self.color, mbc[self.color])
            teammate   = get_teammate_color(controlled)
        else:
            controlled, teammate = self.color, None

        if not turn["rejected"]:
            print(f"[{_ts()}][{self.name}] à moi de jouer ({self.color}"
                  f"{f', contrôle {controlled}' if controlled != self.color else ''}) "
                  f"— main={len(hand)} cartes", flush=True)

        mask, actions = get_legal_mask(
            hand, mbc[controlled], controlled, mbc,
            invincible_by_color=inv_by_color,
            can_discard=gs.get("canDiscard", False),
            teammate_color=teammate,
        )
        for idx in turn["rejected"]:
            mask[idx] = False
        if not any(mask) or len(turn["rejected"]) >= MAX_ATTEMPTS_PER_TURN:
            await self._give_up_turn(ws)
            return

        # Sélection robuste : un plantage du planificateur NE DOIT PAS déconnecter le bot
        # (sinon il quitte la partie). En cas d'erreur → 1er coup légal de repli
        # (inclut l'entrée solidaire forcée, seule action du masque quand elle s'impose).
        t0 = time.monotonic()
        try:
            action = plan_hand_pick(gs, controlled, mask, actions,
                                    teammate_color=teammate)
        except Exception as e:
            print(f"[{self.name}] erreur sélection ({e!r}) → coup légal de repli", flush=True)
            action = None
        if action is None or not mask[action]:
            action = next(i for i, ok in enumerate(mask) if ok)
        msg_out   = build_server_message(
            action, hand, mbc[controlled], controlled, mbc,
            invincible_by_color=inv_by_color,
            teammate_color=teammate,
            legal=(mask, actions),
        )
        compute_s = time.monotonic() - t0
        if compute_s > 5.0:
            print(f"[{_ts()}][{self.name}] ⚠️ sélection lente : {compute_s:.1f}s "
                  f"(risque de timeout serveur à ~65s)", flush=True)
        if think:
            # Réflexion « humaine » : rapide sur un coup forcé, plus longue quand
            # le choix est vaste (cf. think_delay), au rythme propre du bot.
            target = think_delay(sum(1 for ok in mask if ok),
                                 first_turn=self._turns_played == 0,
                                 pace=self.persona.pace)
            self._turns_played += 1
            # Le temps de CALCUL fait partie du "temps de réflexion" : on l'absorbe dans le
            # délai (avant, calcul + delay s'additionnaient → risque de dépasser le timeout
            # serveur). On dort le reste seulement.
            await asyncio.sleep(max(0.0, target - (time.monotonic() - t0)))
        turn["sent"] = action
        turn["sent_msg"] = msg_out
        await ws.send(json.dumps(msg_out))
        print(f"[{_ts()}][{self.name}] coup envoyé "
              f"type={msg_out.get('action', {}).get('type')} "
              f"calcul={compute_s:.1f}s total={time.monotonic() - t0:.1f}s", flush=True)

    async def _on_action_rejected(self, ws, msg: dict) -> None:
        """Le serveur a refusé mon coup. Le tour reste ouvert côté serveur : sans
        nouvelle tentative, il attend la fin du timer (60 s) avant d'imposer un coup.
        On écarte le coup refusé et on rejoue aussitôt le meilleur coup restant."""
        turn = self._turn
        reason = msg.get("reason")
        if turn is None or turn["sent"] is None or reason == "Not your turn":
            print(f"[{_ts()}][{self.name}] actionRejected ignoré (reason={reason!r})",
                  flush=True)
            return
        # Un rejet = le masque local diverge du validateur serveur : on logue de quoi
        # rejouer la position hors ligne (tests/diff_vs_ts_validator.py).
        gs, sent = turn["gs"], turn["sent_msg"]["action"]
        print(f"[{_ts()}][{self.name}] ⚠️ actionRejected (reason={reason!r}) "
              f"coup={json.dumps({k: sent.get(k) for k in ('type', 'from', 'to', 'splitFrom')})} "
              f"carte={[c.get('value') for c in sent.get('cardPlayed', [])]} "
              f"main={[c.get('value') for c in gs['hand']]} "
              f"canDiscard={gs.get('canDiscard')} "
              f"pions={json.dumps(_marbles_by_color(gs))} "
              f"invincibles={json.dumps(_invincible_by_color(gs))}", flush=True)
        turn["rejected"].add(turn["sent"])
        turn["sent"] = None
        await self._play_turn(ws, think=False)

    async def _give_up_turn(self, ws) -> None:
        """Plus aucun coup à tenter : `turnTimeout` (le message qu'envoie le front à la
        fin de son timer) fait imposer par le serveur SON coup légal immédiatement,
        au lieu de bloquer la table jusqu'à la fin du tour."""
        print(f"[{_ts()}][{self.name}] ⚠️ aucun coup accepté par le serveur "
              f"({len(self._turn['rejected'])} refus) → turnTimeout, coup imposé", flush=True)
        self._turn["sent"] = None
        await ws.send(json.dumps({"type": "turnTimeout"}))


async def run_bot_session(identity: dict,
                          game_mode: str = DEFAULT_GAME_MODE) -> None:
    """Cycle complet : auth → WS → 1 partie → close. Robuste aux erreurs."""
    bot = InferenceBot(identity, game_mode=game_mode)
    print(f"connection du bot {bot.bot_id} (mode {game_mode})")
    try:
        async with httpx.AsyncClient() as client:
            await bot.authenticate(client)
        async with websockets.connect(WS_URL) as ws:
            await bot.play_one_game(ws)
        if bot.welcomed:
            print(f"[{bot.name}] partie terminée")
        else:
            # Arrivé après le lancement ou l'annulation de la file : le serveur
            # ferme la socket (close 1000) et l'identité redevient libre.
            print(f"[{bot.name}] renvoyé de la file avant le lancement", flush=True)
    except Exception as e:
        print(f"[bot {identity['botId']}] erreur de session : {e}")


# ── Serveur HTTP (webhook depuis le backend) ──────────────────────────────────

def _cleanup_active(active: dict) -> None:
    for bid in list(active):
        if active[bid]["task"].done():
            del active[bid]


def _next_free_identity(active: dict, avoid: frozenset[str] = frozenset()) -> dict | None:
    """Identité libre au hasard, de préférence hors `avoid` (bots croisés
    récemment par les humains de la file) ; à défaut, n'importe laquelle."""
    free = [ident for ident in BOT_IDENTITIES if ident["botId"] not in active]
    if not free:
        return None
    fresh = [ident for ident in free if ident["botId"] not in avoid]
    return random.choice(fresh or free)


async def dispatch_handler(request: web.Request) -> web.Response:
    if request.headers.get("X-Bot-Secret") != BOT_SECRET:
        return web.json_response({"error": "unauthorized"}, status=401)

    # Le backend POSTe désormais {"gameMode": "1v3" | "2v2"} (avant : body {}) pour
    # indiquer le mode de la partie. Body absent/invalide → défaut serveur (2v2).
    # Ce hint est ensuite écrasé par gameState.gameMode, la source de vérité en partie.
    game_mode = DEFAULT_GAME_MODE
    # userId d'un bot = son botId (document Users) : liste directement comparable.
    avoid: frozenset[str] = frozenset()
    try:
        body = await request.json()
        if isinstance(body, dict) and body.get("gameMode") in GAME_MODES:
            game_mode = body["gameMode"]
        if isinstance(body, dict) and isinstance(body.get("avoidUserIds"), list):
            avoid = frozenset(str(x) for x in body["avoidUserIds"])
    except Exception:
        pass

    active: dict = request.app["active"]
    _cleanup_active(active)

    ident = _next_free_identity(active, avoid)
    if ident is None:
        return web.json_response({"status": "busy", "active": list(active)},
                                 status=503)

    active[ident["botId"]] = {
        "task": asyncio.create_task(run_bot_session(ident, game_mode)),
        "gameMode": game_mode,
    }
    print(f"[dispatch] bot {ident['botId']} (mode {game_mode})")
    return web.json_response({
        "status":   "dispatched",
        "botId":    ident["botId"],
        "gameMode": game_mode,
    })


async def health_handler(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def status_handler(request: web.Request) -> web.Response:
    active: dict = request.app["active"]
    _cleanup_active(active)
    return web.json_response({
        "active":    list(active),
        # Mode de jeu de chaque bot actif (hint du dispatch de sa partie en cours).
        "gameModes": {bid: entry["gameMode"] for bid, entry in active.items()},
        "available": [i["botId"] for i in BOT_IDENTITIES
                      if i["botId"] not in active],
    })


def make_app() -> web.Application:
    app = web.Application()
    app["active"] = {}
    app.router.add_post("/dispatch", dispatch_handler)
    app.router.add_get("/health",    health_handler)
    app.router.add_get("/status",    status_handler)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host=LISTEN_HOST, port=LISTEN_PORT)