"""
Personnalité de chaque bot, liée à vie à son identité (botId).

Un humain a ses habitudes : certains n'envoient presque jamais d'emoji, d'autres
réagissent à tout ; certains ont acheté des emojis en boutique, d'autres se
contentent des gratuits ; certains jouent vite, d'autres prennent leur temps.
Pour qu'un même bot se comporte pareil d'une partie à l'autre, sa persona est
tirée UNE seule fois (tools/generate_personas.py) puis figée dans
personas.json, versionné : un changement ultérieur des règles de tirage ne
change jamais le caractère d'un bot existant.
"""

from __future__ import annotations

import json
import pathlib
import random
from dataclasses import dataclass

# Palette du serveur (REACTION_EMOJIS, packages/shared/src/types.ts de Mercury) :
# les gratuits sont ouverts à tous, les payants s'achètent en boutique
# (SHOP_CATALOG). Le serveur laisse passer les payants d'un bot : c'est la
# persona qui décide s'il les « possède ».
FREE_EMOJIS: tuple[str, ...] = ("👏", "😂", "😮", "😥", "🔥", "🤔")
PAID_EMOJIS: tuple[str, ...] = ("😡", "😎", "😴", "⏰", "🥱", "🦧", "😉", "😘", "😊")

PERSONAS_PATH = pathlib.Path(__file__).with_name("personas.json")

# (style, multiplicateur des probabilités de réaction, part des bots)
CHATTINESS_TIERS: tuple[tuple[str, float, float], ...] = (
    ("silent", 0.15, 0.20),
    ("quiet",  0.5,  0.30),
    ("normal", 1.0,  0.30),
    ("chatty", 2.0,  0.20),
)
# (palier, nombre d'emojis payants possédés (min, max), part des bots)
OWNERSHIP_TIERS: tuple[tuple[str, tuple[int, int], float], ...] = (
    ("none", (0, 0), 0.45),
    ("few",  (1, 2), 0.35),
    ("many", (3, 6), 0.20),
)
# Poids multiplié de l'emoji fétiche quand il fait partie des réponses possibles.
FAVORITE_WEIGHT = 3


@dataclass(frozen=True)
class Persona:
    style: str
    chattiness: float
    owned: frozenset[str]
    favorite: str | None
    pace: float

    def can_use(self, emoji: str) -> bool:
        return emoji in FREE_EMOJIS or emoji in self.owned

    @staticmethod
    def from_dict(data: dict) -> "Persona":
        return Persona(
            style=str(data["style"]),
            chattiness=float(data["chattiness"]),
            owned=frozenset(data["owned"]),
            favorite=data.get("favorite"),
            pace=float(data["pace"]),
        )


def _draw_tier(rng: random.Random, tiers) -> str:
    r, acc = rng.random(), 0.0
    for tier in tiers:
        acc += tier[2]
        if r < acc:
            return tier[0]
    return tiers[-1][0]


def draw_persona(rng: random.Random, style: str | None = None, ownership: str | None = None) -> dict:
    """Tire une persona (forme JSON). Réservé au générateur et au repli.
    `style` / `ownership` imposés : le générateur répartit ces paliers par quotas."""
    style = style or _draw_tier(rng, CHATTINESS_TIERS)
    chattiness = {name: mult for name, mult, _ in CHATTINESS_TIERS}[style]

    ownership = ownership or _draw_tier(rng, OWNERSHIP_TIERS)
    low, high = {name: span for name, span, _ in OWNERSHIP_TIERS}[ownership]
    owned = sorted(rng.sample(PAID_EMOJIS, rng.randint(low, high)))

    favorite = rng.choice(list(FREE_EMOJIS) + owned) if rng.random() < 0.5 else None
    pace = round(rng.uniform(0.75, 1.35), 2)
    return {"style": style, "chattiness": chattiness, "owned": owned,
            "favorite": favorite, "pace": pace}


def seed_for(bot_id: str) -> random.Random:
    # Graine texte : stable d'un process à l'autre (indépendante de PYTHONHASHSEED).
    return random.Random(f"mercury-persona:{bot_id}")


def load_personas(path: pathlib.Path = PERSONAS_PATH) -> dict[str, Persona]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"⚠️ {path.name} introuvable : personas de repli pour tous les bots", flush=True)
        return {}
    return {bot_id: Persona.from_dict(data) for bot_id, data in raw.items()}


_PERSONAS = load_personas()


def persona_for(bot_id: str) -> Persona:
    persona = _PERSONAS.get(bot_id)
    if persona is None:
        # Bot ajouté sans relancer le générateur : persona déterministe (la même
        # que le générateur lui donnerait), à figer avec tools/generate_personas.py.
        print(f"⚠️ bot {bot_id} absent de personas.json : persona de repli", flush=True)
        persona = Persona.from_dict(draw_persona(seed_for(bot_id)))
        _PERSONAS[bot_id] = persona
    return persona
