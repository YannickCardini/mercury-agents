"""
Simulateur de jeu Mercury local — cœur de moteur.

But : permettre (1) la RECHERCHE ADVERSE (simuler les coups des adversaires, pas juste
les miens), (2) la MESURE rapide planificateur-vs-planificateur, (3) l'ENTRAÎNEMENT sans
aller-retour serveur (100× plus rapide).

Ce fichier ne contient QUE le cœur dont je suis sûr, parce qu'il s'appuie sur
mercury_legal_moves (port fidèle du move-validator du serveur) :
  - apply_move : effet d'un coup sur le plateau, pour N'IMPORTE QUEL joueur ;
  - legal_actions : coups légaux d'un joueur (via get_legal_mask) ;
  - winner / is_terminal.

⚠️ La DISTRIBUTION des cartes (tailles de manche 5+4+4, reshuffle) et l'ORDRE DES TOURS
(passe/défausse quand pas de coup) ne sont PAS encore ici : ils doivent être calqués sur
le serveur (game.ts) pour être fidèles, sinon l'entraînement/mesure serait subtilement
faux. → on les ajoute une fois la source serveur en main (cf. note en bas).
"""

from mercury_legal_moves import (
    get_legal_mask, ARRIVAL_POSITIONS, HOME_POSITIONS, ALL_COLORS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _owner_at(marbles: dict, pos: int) -> str | None:
    for c, ps in marbles.items():
        if pos in ps:
            return c
    return None


def _send_home(marbles: dict, invincible: dict, owner: str, pos: int) -> None:
    """Renvoie la bille `owner` qui était en `pos` vers une case home libre de sa couleur
    (effet d'une capture). Mute marbles/invincible en place."""
    free = [h for h in HOME_POSITIONS[owner] if h not in marbles[owner]]
    dest = free[0] if free else HOME_POSITIONS[owner][0]
    marbles[owner][marbles[owner].index(pos)] = dest
    invincible[owner].discard(pos)


# ── Application d'un coup (n'importe quel joueur) ──────────────────────────────

def apply_move(marbles: dict, invincible: dict, color: str, action: dict) -> tuple:
    """Applique `action` (from/to/type + splitFrom/splitTo/splitType pour le 7) jouée par
    `color`. Renvoie (new_marbles, new_invincible) — COPIES, les entrées ne sont pas mutées.

    Règles d'effet (cohérentes avec le serveur) :
      - move/capture/promote : ma bille from→to ; si capture, la bille adverse en `to`
        repart au home de SA couleur ;
      - enter : home→start, la bille devient INVINCIBLE ;
      - swap (Valet) : ma bille et la bille adverse échangent leurs positions ;
      - 7-split : on applique les DEUX demi-coups ;
      - toute bille qui BOUGE perd son invincibilité.
    """
    M   = {c: list(ps) for c, ps in marbles.items()}
    INV = {c: set(s)  for c, s in invincible.items()}

    def _move_one(frm: int, to: int, etype: str) -> None:
        if etype == 'capture':
            victim = _owner_at(M, to)
            if victim is not None and victim != color:
                _send_home(M, INV, victim, to)      # libère `to` AVANT d'y poser ma bille
        M[color][M[color].index(frm)] = to
        INV[color].discard(frm)                     # ma bille a bougé → plus invincible

    t = action['type']
    if t == 'enter':
        M[color][M[color].index(action['from'])] = action['to']
        INV[color].add(action['to'])                # fraîchement entrée → invincible
    elif t == 'swap':
        frm, to = action['from'], action['to']
        victim  = _owner_at(M, to)
        M[color][M[color].index(frm)] = to
        if victim is not None and victim != color:
            M[victim][M[victim].index(to)] = frm
            INV[victim].discard(to)
        INV[color].discard(frm)
    else:
        _move_one(action['from'], action['to'], t)
        if 'splitTo' in action:                     # 2e demi-coup du 7
            _move_one(action['splitFrom'], action['splitTo'],
                      action.get('splitType', 'move'))
    return M, INV


# ── Coups légaux / fin de partie ──────────────────────────────────────────────

def legal_actions(marbles: dict, invincible: dict, color: str,
                  hand: list, can_discard: bool = False) -> tuple:
    """(mask, actions) légaux du joueur `color` — délègue au moteur de règles fidèle."""
    inv = {c: list(s) for c, s in invincible.items()}
    return get_legal_mask(hand, marbles[color], color, marbles,
                          invincible_by_color=inv, can_discard=can_discard)


def winner(marbles: dict) -> str | None:
    for c, ps in marbles.items():
        if all(p in ARRIVAL_POSITIONS[c] for p in ps):
            return c
    return None


def is_terminal(marbles: dict) -> bool:
    return winner(marbles) is not None


# ── Construction d'un état neuf (billes au home) ──────────────────────────────

def initial_marbles() -> dict:
    """Toutes les billes dans leurs 4 cases home."""
    return {c: list(HOME_POSITIONS[c]) for c in ALL_COLORS}


def empty_invincible() -> dict:
    return {c: set() for c in ALL_COLORS}


def to_game_state(marbles: dict, color: str, hand: list, can_discard: bool = False) -> dict:
    """Convertit l'état-sim en gameState (format serveur) → réutilisable par encode_state,
    plan_hand, etc. `marbleInvincible` omis ici (les helpers acceptent l'absence)."""
    return {
        'players':     [{'color': c, 'marblePositions': marbles[c]} for c in ALL_COLORS],
        'currentTurn': color,
        'hand':        hand,
        'canDiscard':  can_discard,
    }


# ── Paquet de cartes (port fidèle de deck.ts) ─────────────────────────────────

import random as _random
from mercury_legal_moves import PER_SLOT

_DECK_VALUES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
_SUITS       = ['H', 'D', 'C', 'S']   # la couleur n'affecte pas le jeu (4 cartes / rang)
CARDS_PER_HAND = 5                    # cf. constants.ts


DECK_SIZE = 54   # 52 cartes standard + 2 Jokers (cf. deck.ts)


def fresh_deck() -> list:
    """54 cartes = 4 rangs × 13 valeurs + 2 Jokers (comme deck.ts ; suit non-significatif)."""
    deck = [{'id': str(i), 'suit': s, 'value': v}
            for i, (s, v) in enumerate((s, v) for s in _SUITS for v in _DECK_VALUES)]
    deck.append({'id': '52', 'suit': 'JOKER', 'value': 'Joker'})
    deck.append({'id': '53', 'suit': 'JOKER', 'value': 'Joker'})
    return deck


# ── Politiques utilitaires ────────────────────────────────────────────────────

def random_policy(gs: dict, color: str, mask: list, actions: list) -> int:
    return _random.choice([i for i, ok in enumerate(mask) if ok])


# ── Partie complète (port fidèle de game.ts) ──────────────────────────────────

class SimGame:
    """Partie locale fidèle à game.ts : distribution 5/4/4 + reshuffle, ordre des tours,
    règle jouer-sinon-défausser, victoire = toutes mes billes en arrivée.

    Une `policy` est `fn(game_state, color, mask, actions) -> index_action` — exactement
    la même signature que plan_hand_pick / heuristic_pick / le réseau. On peut donc faire
    jouer n'importe quel agent, mesurer (planif vs réseau) et entraîner, SANS serveur."""

    def __init__(self, colors=None, seed=None):
        self.colors      = list(colors) if colors else list(ALL_COLORS)
        self.rng         = _random.Random(seed)
        self.marbles     = initial_marbles()
        self.invincible  = empty_invincible()
        self.hands       = {c: [] for c in self.colors}
        self.discarded   = []
        self.deck        = fresh_deck()    # game.ts construit un Deck → resetDeck → 52
        self.round       = 0
        self.turn        = 0
        self.first_player = 0
        self.cur         = 0

    def _deal(self) -> None:
        self.round += 1
        if not self.deck:                          # isEmpty → reset
            self.deck = fresh_deck()
        self.rng.shuffle(self.deck)
        per = CARDS_PER_HAND if len(self.deck) == DECK_SIZE else CARDS_PER_HAND - 1   # 5 si plein, sinon 4
        for c in self.colors:
            take = min(per, len(self.deck))
            self.hands[c] = self.deck[:take]
            self.deck     = self.deck[take:]

    def _start_new_round(self) -> None:
        self.first_player = (self.first_player + 1) % len(self.colors)
        self.cur          = self.first_player
        self._deal()

    def _play_one_turn(self, policies: dict) -> bool:
        """Joue le tour du joueur courant. Renvoie True si le coup déclenche un REJEU
        (Joker effectivement joué + main non vide) → la boucle ne passe pas au suivant."""
        color = self.colors[self.cur]
        if not self.hands[color]:                  # main vide → pass
            return False
        mask, actions = legal_actions(self.marbles, self.invincible, color,
                                      self.hands[color], can_discard=False)
        gs  = to_game_state(self.marbles, color, self.hands[color])
        idx = policies[color](gs, color, mask, actions)
        entry = actions[idx] if 0 <= idx < len(actions) else None
        if entry == 'discard' or entry is None or not mask[idx]:   # défausse (forcée)
            self.discarded.extend(self.hands[color])
            self.hands[color] = []
            return False
        card, action = entry
        self.marbles, self.invincible = apply_move(self.marbles, self.invincible, color, action)
        slot = idx // PER_SLOT                      # la carte jouée est au slot idx//100
        if slot < len(self.hands[color]):
            self.discarded.append(self.hands[color].pop(slot))
        # Rejeu : un Joker joué (entrée ou +18) offre un tour de plus tant qu'il reste des cartes.
        return card.get('value') == 'Joker' and bool(self.hands[color])

    def play(self, policies: dict, max_turns: int = 4000) -> str | None:
        """Joue jusqu'à victoire (ou max_turns en garde-fou). Renvoie la couleur gagnante."""
        self._deal()
        while winner(self.marbles) is None and self.turn < max_turns:
            if all(not self.hands[c] for c in self.colors):
                self._start_new_round()
                continue
            replay = self._play_one_turn(policies)
            if not replay:                          # Joker → même joueur rejoue (cf. game.ts)
                self.cur = (self.cur + 1) % len(self.colors)
            self.turn += 1
        return winner(self.marbles)


# ── Harnais de mesure rapide ──────────────────────────────────────────────────

def duel(policy_a, policy_b, n_games: int = 20, seed: int = 0,
         max_turns: int = 4000) -> dict:
    """A vs B en 2-vs-2 (diagonale). On ALTERNE les côtés à chaque partie pour NEUTRALISER
    le biais de position (le côté red+blue gagne ~0.65 même à forces égales). Renvoie le
    win-rate de A parmi les parties décidées (0.50 = égalité, > 0.50 = A plus fort)."""
    cs    = list(ALL_COLORS)
    diag1 = {cs[0], cs[2]}                  # red+blue
    wins_a = decided = 0
    turns  = []
    for k in range(n_games):
        a_side = diag1 if k % 2 == 0 else (set(cs) - diag1)   # A change de côté 1 partie/2
        g   = SimGame(seed=seed + k)
        pol = {c: (policy_a if c in a_side else policy_b) for c in cs}
        w   = g.play(pol, max_turns=max_turns)
        turns.append(g.turn)
        if w is not None:
            decided += 1
            if w in a_side:
                wins_a += 1
    return {
        'winrate_a': (wins_a / decided) if decided else float('nan'),
        'decided':   decided,
        'n_games':   n_games,
        'avg_turns': sum(turns) // len(turns),
    }
