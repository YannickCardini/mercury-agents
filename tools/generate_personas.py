"""
Fige les personas des bots dans personas.json.

N'ajoute que les botIds absents : une persona déjà écrite n'est JAMAIS
retirée, pour qu'un bot garde le même caractère à vie. À relancer après
l'ajout d'identités au pool (BOT_IDENTITIES de main.py), puis commiter.

    python tools/generate_personas.py          # botIds 1 à 27
    python tools/generate_personas.py 32       # botIds 1 à 32
"""

import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from persona import CHATTINESS_TIERS, OWNERSHIP_TIERS, PERSONAS_PATH, draw_persona, seed_for  # noqa: E402


def quota(tiers, existing: list[str], new_count: int, rng: random.Random) -> list[str]:
    """Paliers des nouveaux bots, au plus près des parts visées sur l'ensemble du
    pool (un tirage indépendant peut, sur 27 bots, ne donner que 2 bavards),
    puis mélangés pour ne pas suivre l'ordre des botIds."""
    counts = {tier[0]: existing.count(tier[0]) for tier in tiers}
    total = len(existing) + new_count
    picked = []
    for _ in range(new_count):
        deficit = {tier[0]: tier[2] * total - counts[tier[0]] for tier in tiers}
        name = max(deficit, key=deficit.get)
        counts[name] += 1
        picked.append(name)
    rng.shuffle(picked)
    return picked


def _ownership_of(persona: dict) -> str:
    n = len(persona["owned"])
    return next(name for name, (low, high), _ in OWNERSHIP_TIERS if low <= n <= high)


def generate(count: int, path: pathlib.Path = PERSONAS_PATH) -> list[str]:
    """Complète `path` pour les botIds 1..count ; renvoie les ids ajoutés."""
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    missing = [str(i) for i in range(1, count + 1) if str(i) not in existing]
    rng = random.Random(f"mercury-persona-quotas:{len(existing)}")
    styles = quota(CHATTINESS_TIERS, [p["style"] for p in existing.values()], len(missing), rng)
    ownerships = quota(OWNERSHIP_TIERS, [_ownership_of(p) for p in existing.values()], len(missing), rng)
    added = []
    for bot_id, style, ownership in zip(missing, styles, ownerships):
        existing[bot_id] = draw_persona(seed_for(bot_id), style=style, ownership=ownership)
        added.append(bot_id)
    ordered = dict(sorted(existing.items(), key=lambda kv: (len(kv[0]), kv[0])))
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return added


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 27
    added = generate(count)
    print(f"{len(added)} persona(s) ajoutée(s) : {', '.join(added) or 'aucune'}")
