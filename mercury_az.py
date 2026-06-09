"""
AlphaZero-lite pour Mercury : entraîner un ÉVALUATEUR DE POSITION (réseau de valeur) sur
du self-play SIMULÉ, pour remplacer le score fait-main du planificateur → un agent plus
fort (tier T2). Le simulateur rend la génération de données rapide (le blocage historique).

Pipeline :
  1. generate_data : self-play simulé, on échantillonne (état encodé du joueur courant,
     label = 1 si ce joueur gagne la partie).
  2. ValueNet : petit MLP 138 → V(état) = proba de gagner.
  3. train : BCE, split PAR PARTIE (pas de fuite entre états corrélés).
  4. (étape suivante) brancher V comme évaluateur des feuilles du planificateur + mesurer.
"""

import random
import torch
import torch.nn as nn

import mercury_sim as sim
from mercury_sim import SimGame, to_game_state
from model import encode_state, STATE_DIM
from plan_hand import _search, DISCARD_IDX


# ── Réseau de valeur ──────────────────────────────────────────────────────────

class ValueNet(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(STATE_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).squeeze(-1)        # logit ; sigmoid(logit) = P(gagner)


# ── Génération de données (self-play simulé) ──────────────────────────────────

def generate_data(n_games: int, policies: dict, seed: int = 0,
                  sample_rate: float = 0.25, max_turns: int = 4000):
    """Joue n_games parties ; échantillonne des états (POV joueur courant), labellisés par
    l'issue. État encodé MAIN VIDE (évaluateur de position, cohérent avec une feuille de
    plan où la main est jouée). Renvoie (X[N,138], y[N])."""
    rng = random.Random(seed)
    X, C, G, win = [], [], [], {}
    for k in range(n_games):
        g = SimGame(seed=seed + k)
        g._deal()
        while sim.winner(g.marbles) is None and g.turn < max_turns:
            if all(not g.hands[c] for c in g.colors):
                g._start_new_round()
                continue
            color = g.colors[g.cur]
            if g.hands[color] and rng.random() < sample_rate:
                X.append(encode_state(to_game_state(g.marbles, color, []), color))
                C.append(color)
                G.append(k)
            g._play_one_turn(policies)
            g.cur  = (g.cur + 1) % len(g.colors)
            g.turn += 1
        win[k] = sim.winner(g.marbles)
    y = torch.tensor([1.0 if win[gi] == c else 0.0 for c, gi in zip(C, G)])
    return torch.stack(X), y, torch.tensor(G)


# ── Entraînement ──────────────────────────────────────────────────────────────

def train_value(X, y, games, hidden=256, epochs=40, lr=1e-3, batch=512, seed=0):
    """Entraîne ValueNet. Split PAR PARTIE (val = 20% des parties) pour éviter la fuite
    entre états d'une même partie. Renvoie (net, val_accuracy, base_rate)."""
    torch.manual_seed(seed)
    uniq = games.unique()
    perm = uniq[torch.randperm(len(uniq))]
    val_games = set(perm[: max(1, len(uniq)//5)].tolist())
    val_mask  = torch.tensor([int(g) in val_games for g in games])
    Xtr, ytr = X[~val_mask], y[~val_mask]
    Xva, yva = X[val_mask],  y[val_mask]

    net = ValueNet(hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    n = len(Xtr)
    for ep in range(epochs):
        net.train()
        for s in range(0, n, batch):
            idx = slice(s, s + batch)
            opt.zero_grad()
            loss = lossf(net(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(net(Xva)) > 0.5).float()
        acc  = (pred == yva).float().mean().item()
    base = max(yva.mean().item(), 1 - yva.mean().item())   # accuracy d'un prédicteur constant
    return net, acc, base


# ── Intégration : planificateur dont les feuilles sont notées par le réseau ───

def make_value_policy(net, k: int = 16):
    """Politique : le planificateur génère ses `k` meilleurs plans (score fait-main), puis
    le RÉSEAU DE VALEUR les re-classe (1 forward batché → rapide) et on joue le 1er coup du
    plan le mieux noté. C'est l'évaluateur appris greffé sur la recherche fait-main."""
    net.eval()

    def pick(gs, color, mask, actions):
        cand = _search(gs, color, mask, actions, topk=k)
        if not cand:
            return DISCARD_IDX
        encs = [encode_state(to_game_state({color: n['my'], **n['opp']}, color, []), color)
                for n in cand]
        with torch.no_grad():
            vals = net(torch.stack(encs))
        return cand[int(torch.argmax(vals))]['first_idx']

    return pick


if __name__ == "__main__":
    import time
    from plan_hand import plan_hand_pick

    # Self-play de planificateurs RÉDUITS (beam 16) : termine proprement, bonnes positions,
    # assez rapide pour générer un corpus en quelques minutes.
    pol = {c: (lambda gs, col, m, a: plan_hand_pick(gs, col, m, a, beam_width=16))
           for c in sim.ALL_COLORS}

    N = 150
    print(f"[AZ] génération {N} parties (planif réduit self-play)…", flush=True)
    t0 = time.time()
    X, y, G = generate_data(N, pol, seed=0, sample_rate=0.25)
    print(f"[AZ] {len(X)} états, {int(G.max())+1} parties, {time.time()-t0:.0f}s  "
          f"(taux de victoire moyen {y.mean():.2f})", flush=True)

    print("[AZ] entraînement du réseau de valeur…", flush=True)
    net, acc, base = train_value(X, y, G)
    print(f"[AZ] val_accuracy={acc:.3f}  (prédicteur constant={base:.3f})  "
          f"→ {'le réseau APPREND' if acc > base + 0.03 else 'pas mieux que constant'}", flush=True)

    torch.save(net.state_dict(), "value_net.pt")
    print("[AZ] réseau sauvé → value_net.pt", flush=True)
