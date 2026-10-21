"""
Pool des identités de bots, relu depuis le backend Mercury.

Les bots sont des comptes en base (`isBot: true`, avec leur persona). Le
backend les liste via GET /api/auth/bots (en-tête X-Bot-Secret) : un bot ajouté
par backend/scripts/add-bot.ts devient jouable dans la minute, sans
redéployer ce service. Si le backend est injoignable (ou pas encore à jour),
on garde la dernière liste connue, à défaut la liste statique de secours.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

import httpx

# Relecture au plus une fois par minute : un dispatch n'attend jamais le réseau
# plus d'une fois par minute.
POOL_TTL_SECONDS = 60.0
POOL_TIMEOUT_SECONDS = 5.0

Fetch = Callable[[], Awaitable[list[dict]]]


class BotPool:
    def __init__(self, api_url: str, secret: str, fallback_ids: list[str],
                 ttl: float = POOL_TTL_SECONDS, fetch: Fetch | None = None):
        self._api_url = api_url.rstrip("/")
        self._secret = secret
        self._ttl = ttl
        self._fetch = fetch or self._fetch_from_backend
        # botId → persona (dict) ou None (persona de repli côté agent).
        self._bots: dict[str, dict | None] = {bot_id: None for bot_id in fallback_ids}
        self._fetched_at: float | None = None
        self._lock = asyncio.Lock()
        self.source = "static"

    async def _fetch_from_backend(self) -> list[dict]:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{self._api_url}/api/auth/bots",
                                 headers={"X-Bot-Secret": self._secret},
                                 timeout=POOL_TIMEOUT_SECONDS)
        r.raise_for_status()
        return r.json()["bots"]

    def _fresh(self) -> bool:
        return self._fetched_at is not None and time.monotonic() - self._fetched_at < self._ttl

    async def refresh(self) -> None:
        """Relit le pool s'il date de plus de `ttl` secondes. Ne lève jamais."""
        if self._fresh():
            return
        async with self._lock:
            if self._fresh():
                return
            try:
                bots = await self._fetch()
                parsed = {str(b["id"]): b.get("persona") for b in bots if b.get("id")}
                if parsed:
                    added = sorted(set(parsed) - set(self._bots), key=lambda i: (len(i), i))
                    if added and self.source == "backend":
                        print(f"[pool] nouveau(x) bot(s) : {', '.join(added)}", flush=True)
                    self._bots = parsed
                    self.source = "backend"
                else:
                    print("[pool] liste vide reçue du backend : pool inchangé", flush=True)
            except Exception as e:
                print(f"[pool] relecture impossible ({e!r}) : pool inchangé "
                      f"({len(self._bots)} bots, source {self.source})", flush=True)
            self._fetched_at = time.monotonic()

    def ids(self) -> list[str]:
        return list(self._bots)

    def persona_data(self, bot_id: str) -> dict | None:
        return self._bots.get(bot_id)
