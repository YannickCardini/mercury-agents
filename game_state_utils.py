"""Helpers de lecture du gameState serveur, partagés par l'inférence (main.py) et
l'entraînement (bot_rl.py). Auparavant dupliqués à l'identique dans les deux fichiers."""

from mercury_legal_moves import ARRIVAL_POSITIONS


def has_won(marble_positions: list[int], color: str) -> bool:
    """True si les 4 billes de `color` sont toutes en zone d'arrivée."""
    arrival = ARRIVAL_POSITIONS[color]
    return all(p in arrival for p in marble_positions)


def detect_winner(game_state: dict) -> str | None:
    """Couleur gagnante si une est déjà arrivée au complet, sinon None."""
    for p in game_state["players"]:
        if has_won(p["marblePositions"], p["color"]):
            return p["color"]
    return None


def marbles_by_color(game_state: dict) -> dict:
    return {p["color"]: p["marblePositions"] for p in game_state["players"]}


def invincible_by_color(game_state: dict) -> dict:
    """Reconstruit {color: [positions invincibles]} depuis marbleInvincible, tableau
    parallèle à marblePositions envoyé par le serveur. Indispensable pour que le masque
    légal d'inférence soit IDENTIQUE à celui de l'entraînement."""
    out = {}
    for p in game_state["players"]:
        positions = p["marblePositions"]
        inv_flags = p.get("marbleInvincible") or [False] * len(positions)
        out[p["color"]] = [pos for pos, inv in zip(positions, inv_flags) if inv]
    return out
