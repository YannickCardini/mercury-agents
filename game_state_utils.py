"""Helpers de lecture du gameState serveur, utilisés par le service d'inférence (main.py)."""

from mercury_legal_moves import has_won, has_team_won, get_teammate_color


def detect_winners(game_state: dict) -> list[str]:
    """Couleurs gagnantes si la partie est finie, sinon []. MODE-AWARE via
    gameState.gameMode (source de vérité serveur) :
      - '1v3' (ou absent, rétro-compat) : [couleur] dès qu'une couleur a ses 4 pions
        en zone d'arrivée.
      - '2v2' : un joueur qui a rentré ses 4 pions ne termine PAS la partie ; gagnants
        = les DEUX couleurs d'une équipe seulement quand LES DEUX coéquipiers ont fini
        (8 pions à l'abri). Sans ce test, le bot quittait la partie en plein milieu.
    Garde-fou local uniquement : la vraie fin de partie est le message `gameEnded`
    (champ `winners`), qui fait autorité."""
    mode = game_state.get("gameMode", "1v3")
    mbc = marbles_by_color(game_state)
    if mode == "2v2":
        for p in game_state["players"]:
            color = p["color"]
            if has_team_won(mbc, color):
                return [color, get_teammate_color(color)]
        return []
    for p in game_state["players"]:
        if has_won(p["marblePositions"], p["color"]):
            return [p["color"]]
    return []


def detect_winner(game_state: dict) -> str | None:
    """Compat historique : première couleur gagnante ou None. Mode-aware (délègue à
    detect_winners) — en 2v2, ne renvoie une couleur que si l'ÉQUIPE entière a fini."""
    winners = detect_winners(game_state)
    return winners[0] if winners else None


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
