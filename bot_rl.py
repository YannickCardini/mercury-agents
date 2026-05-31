"""
Bot Mercury — entraînement PPO en self-play.
4 bots partagent le même réseau et jouent simultanément contre le serveur local.

Lancer :
  cd /home/loubard/Documents/mercury
  python bot_rl.py
"""

import asyncio, json, pathlib, random, threading
import concurrent.futures, functools
from collections import deque
import torch
import torch.nn.functional as F
import httpx
import websockets

from mercury_legal_moves import (
    get_legal_mask, build_server_message,
    ARRIVAL_POSITIONS, HOME_POSITIONS, ALL_COLORS,
)
from reward  import compute_reward, terminal_reward
from model   import MercuryNet, encode_state, ACTION_DIM, STATE_DIM

# ── Configuration ─────────────────────────────────────────────────────────────

API_URL    = "http://localhost:8080"
WS_URL     = "ws://localhost:8080"
BOT_SECRET = "change-me-before-prod"
MODEL_PATH = pathlib.Path("model.pt")
MODEL_DIR  = pathlib.Path("models")

GAMMA        = 0.99
GAE_LAMBDA   = 0.95
CLIP_EPS     = 0.2
LR           = 3e-4
ENTROPY_COEF = 0.015  # optimum en U inversé : 0.01 figeait l'agent (entropie ~0.10,
                      # eval~0.70) ; 0.025 sur-explorait (entropie ~0.40, eval baissée à 0.64).
                      # 0.015 = compromis exploration/exploitation entre les deux extrêmes.
VALUE_COEF   = 0.5
UPDATE_EVERY = 512  
PPO_EPOCHS   = 4
BATCH_SIZE   = 64

POOL_SIZE      = 5    # snapshots du passé conservés en mémoire
SNAPSHOT_EVERY = 10   # sauvegarder un snapshot tous les N updates
WINRATE_WINDOW = 200  # fenêtre glissante pour la win-rate des bots learner
LEARNER_PROB   = 0.6  # proba qu'un bot joue le réseau courant (→ learner) par partie
                      # ⇒ ~2-3 des 4 bots alimentent le gradient ; le reste = pool gelé

# ── Évaluation vs bots aléatoires ───────────────────────────────────────────────
# Le win_rate vs pool tend mécaniquement vers 0.25 en self-play symétrique (on joue
# contre soi-même). Pour une métrique ABSOLUE de progression, on lance périodiquement
# une partie isolée : 1 learner greedy (la politique réellement déployée, cf. main.py)
# contre 3 bots aléatoires. Matchmaking FIFO ⇒ les 4 bots d'éval, rejoignant ensemble
# pendant que les bots de training sont en partie, sont regroupés dans la même partie.
EVAL_EVERY        = 10   # lancer une partie d'éval tous les N updates PPO
EVAL_WINDOW       = 50   # fenêtre glissante de l'eval win-rate
EVAL_LEARNER_COLOR = ALL_COLORS[0]  # couleur du bot greedy dans le quatuor d'éval

COLORS = ALL_COLORS  # 4 joueurs : 1 learner (red) + 3 adversaires


# ── Normalisation des retours (robustesse du critique) ────────────────────────

class RunningMeanStd:
    """Moyenne/variance courantes (algo de Welford) sur les retours.
    Le critique apprend à prédire des retours STANDARDISÉS (moy 0, écart-type 1),
    ce qui borne l'amplitude de la value loss quelle que soit l'échelle des rewards
    — complément de la réduction ÷10 : robuste même si cette estimation dérive."""

    def __init__(self):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = 1e-4

    def update(self, x: torch.Tensor):
        batch_mean  = float(x.mean())
        batch_var   = float(x.var(unbiased=False))
        batch_count = x.numel()
        delta      = batch_mean - self.mean
        tot        = self.count + batch_count
        self.mean += delta * batch_count / tot
        m_a        = self.var * self.count
        m_b        = batch_var * batch_count
        m2         = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.var   = m2 / tot
        self.count = tot

    @property
    def std(self) -> float:
        return (self.var ** 0.5) + 1e-8

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


# ── Rollout par bot (trajectoire d'une partie) ────────────────────────────────

class RolloutBuffer:
    """Rollout local d'un bot. Les frontières d'épisode restent cohérentes
    pour le calcul de l'avantage (GAE)."""

    def __init__(self):
        self.states:     list[torch.Tensor] = []
        self.masks:      list[torch.Tensor] = []
        self.actions:    list[int]          = []
        self.log_probs:  list[float]        = []
        self.values:     list[float]        = []
        self.rewards:    list[float]        = []
        self.dones:      list[float]        = []
        self.next_vals:  list[float]        = []

    def add(self, state, mask, action, log_prob, value,
            reward, done, next_value):
        self.states.append(state)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(float(done))
        self.next_vals.append(next_value)

    def __len__(self):
        return len(self.states)

    def compute_gae(self, gamma: float, lam: float
                    ) -> tuple[list[float], list[float]]:
        """Avantages GAE(λ) et retours = avantages + valeurs."""
        n          = len(self.states)
        advantages = [0.0] * n
        gae        = 0.0
        for t in reversed(range(n)):
            non_term = 1.0 - self.dones[t]
            delta    = (self.rewards[t]
                        + gamma * self.next_vals[t] * non_term
                        - self.values[t])
            gae      = delta + gamma * lam * non_term * gae
            advantages[t] = gae
        returns = [a + v for a, v in zip(advantages, self.values)]
        return advantages, returns


# ── Buffer agrégé (prêt pour PPO) ─────────────────────────────────────────────

class TrainingBuffer:
    """Transitions pré-calculées (avantages GAE déjà figés) prêtes pour PPO."""

    def __init__(self):
        self.states:     list[torch.Tensor] = []
        self.masks:      list[torch.Tensor] = []
        self.actions:    list[int]          = []
        self.log_probs:  list[float]        = []
        self.advantages: list[float]        = []
        self.returns:    list[float]        = []

    def extend_from_rollout(self, rollout: RolloutBuffer,
                            gamma: float, lam: float):
        advs, rets = rollout.compute_gae(gamma, lam)
        self.states.extend(rollout.states)
        self.masks.extend(rollout.masks)
        self.actions.extend(rollout.actions)
        self.log_probs.extend(rollout.log_probs)
        self.advantages.extend(advs)
        self.returns.extend(rets)

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.__init__()


# ── Agent PPO ─────────────────────────────────────────────────────────────────

class PPOAgent:
    def __init__(self):
        MODEL_DIR.mkdir(exist_ok=True)
        self.net        = MercuryNet()
        self.optimizer  = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.buffer     = TrainingBuffer()
        self.n_updates  = 0
        self.n_games    = 0
        self.pool: list[MercuryNet] = []   # snapshots gelés (pool d'adversaires)
        self._inference_lock = threading.Lock()  # protège net pendant updates
        # Stats courantes des retours : le critique prédit des retours standardisés.
        self.ret_rms = RunningMeanStd()
        # Fenêtre glissante des résultats des bots learner (1 = gagné, 0 = perdu).
        # En self-play symétrique elle tend vers ~0.25 (on joue contre ses clones) :
        # c'est un indicateur de stabilité, PAS de progression absolue.
        self.recent_wins = deque(maxlen=WINRATE_WINDOW)
        # Win-rate du learner greedy vs 3 bots aléatoires : LA métrique de progression
        # absolue. Doit grimper bien au-dessus de 0.25 si l'agent apprend réellement.
        self.eval_wins = deque(maxlen=EVAL_WINDOW)
        self._load()

    def _load(self):
        if MODEL_PATH.exists():
            try:
                self.net.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
                print(f"Modèle chargé depuis {MODEL_PATH}")
            except RuntimeError:
                print(f"Checkpoint incompatible (STATE_DIM changé ?), repartir de zéro.")
                MODEL_PATH.unlink()

    def _save_snapshot(self):
        """Gèle une copie du réseau courant dans le pool et sur disque."""
        snap = MercuryNet()
        snap.load_state_dict({k: v.clone() for k, v in self.net.state_dict().items()})
        snap.eval()
        self.pool.append(snap)
        if len(self.pool) > POOL_SIZE:
            self.pool.pop(0)
        path = MODEL_DIR / f"snapshot_{self.n_updates:06d}.pt"
        torch.save(snap.state_dict(), path)
        # Garder uniquement les POOL_SIZE derniers sur disque
        snaps = sorted(MODEL_DIR.glob("snapshot_*.pt"))
        for old in snaps[:-POOL_SIZE]:
            old.unlink(missing_ok=True)

    def sample_opponent_net(self) -> MercuryNet:
        """Tire aléatoirement un snapshot GELÉ du pool (jamais le réseau courant).
        Le choix courant-vs-pool est fait en amont dans RLBot._reset_episode ; ici on
        ne renvoie que des adversaires figés pour que `game_net is agent.net` reste un
        test fiable du rôle learner. Repli sur le réseau courant si le pool est vide."""
        if not self.pool:
            return self.net
        return random.choice(self.pool)

    def select_action(self, state: torch.Tensor,
                      mask: list[bool]) -> tuple[int, float, float]:
        legal = torch.tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            with self._inference_lock:  # protège contre les updates simultanées
                dist, value = self.net(state, legal)
                action      = dist.sample()
                log_prob    = dist.log_prob(action)
        # Le critique prédit des retours STANDARDISÉS → dénormaliser pour que GAE
        # (compute_gae) opère dans l'espace brut des récompenses (cohérence rewards/values).
        value = self.ret_rms.denormalize(value.item())
        return action.item(), log_prob.item(), value

    def update(self):
        n = len(self.buffer)
        if n < 32:
            return

        states     = torch.stack(self.buffer.states)
        masks      = torch.stack(self.buffer.masks)
        actions    = torch.tensor(self.buffer.actions,    dtype=torch.long)
        old_lps    = torch.tensor(self.buffer.log_probs,  dtype=torch.float32)
        advantages = torch.tensor(self.buffer.advantages, dtype=torch.float32)
        returns    = torch.tensor(self.buffer.returns,    dtype=torch.float32)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Le critique prédit des retours STANDARDISÉS : on met à jour les stats courantes
        # puis on normalise les cibles. La value loss reste ainsi d'amplitude O(1) quelle
        # que soit l'échelle des récompenses (complément robuste de la réduction ÷10).
        self.ret_rms.update(returns)
        returns_norm = self.ret_rms.normalize(returns)

        # Métriques accumulées sur tous les mini-batches de l'update (pour le log).
        sum_p_loss = sum_v_loss = sum_entropy = sum_kl = 0.0
        n_batches  = 0

        with self._inference_lock:  # bloque les inférences pendant les updates
            for _ in range(PPO_EPOCHS):
                perm = torch.randperm(n)
                for start in range(0, n, BATCH_SIZE):
                    idx = perm[start:start + BATCH_SIZE]

                    dist, values = self.net(states[idx], masks[idx])
                    log_probs    = dist.log_prob(actions[idx])
                    entropy      = dist.entropy().mean()

                    ratio   = (log_probs - old_lps[idx]).exp()
                    adv     = advantages[idx]
                    surr    = torch.min(ratio * adv,
                                        ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv)
                    p_loss  = -surr.mean()
                    # values et returns_norm sont tous deux en espace STANDARDISÉ.
                    v_loss  = F.mse_loss(values, returns_norm[idx])
                    loss    = p_loss + VALUE_COEF * v_loss - ENTROPY_COEF * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                    self.optimizer.step()

                    # KL approx (old - new) : indicateur de l'amplitude du pas de politique.
                    with torch.no_grad():
                        approx_kl = (old_lps[idx] - log_probs).mean()
                    sum_p_loss += p_loss.item()
                    sum_v_loss += v_loss.item()
                    sum_entropy += entropy.item()
                    sum_kl     += approx_kl.item()
                    n_batches  += 1

        self.buffer.clear()
        self.n_updates += 1
        torch.save(self.net.state_dict(), MODEL_PATH)
        if self.n_updates % SNAPSHOT_EVERY == 0:
            self._save_snapshot()

        nb = max(n_batches, 1)
        win_rate = (sum(self.recent_wins) / len(self.recent_wins)
                    if self.recent_wins else float('nan'))
        print(f"[PPO] update #{self.n_updates}  parties={self.n_games}  "
              f"transitions={n}  pool={len(self.pool)}  "
              f"p_loss={sum_p_loss / nb:+.4f}  v_loss={sum_v_loss / nb:.4f}  "
              f"ent={sum_entropy / nb:.3f}  kl={sum_kl / nb:+.4f}  "
              f"win_rate={win_rate:.3f} (n={len(self.recent_wins)})  → {MODEL_PATH}")


# ── Bot individuel ────────────────────────────────────────────────────────────

def _has_won(marble_positions: list[int], color: str) -> bool:
    arrival = ARRIVAL_POSITIONS[color]
    return all(p in arrival for p in marble_positions)

def _detect_winner(game_state: dict) -> str | None:
    for player in game_state['players']:
        if _has_won(player['marblePositions'], player['color']):
            return player['color']
    return None

def _marbles_by_color(game_state: dict) -> dict:
    return {p['color']: p['marblePositions'] for p in game_state['players']}

def _invincible_by_color(game_state: dict) -> dict:
    """Reconstruit {color: [positions invincibles]} depuis marbleInvincible,
    tableau parallèle à marblePositions envoyé par le serveur."""
    out = {}
    for p in game_state['players']:
        positions = p['marblePositions']
        inv_flags = p.get('marbleInvincible') or [False] * len(positions)
        out[p['color']] = [pos for pos, inv in zip(positions, inv_flags) if inv]
    return out


class RLBot:
    def __init__(self, agent: PPOAgent, bot_id: str, name: str,
                 executor: concurrent.futures.Executor = None):
        self.agent      = agent
        self.is_learner = True        # (re)décidé par partie dans _reset_episode
        self.color      = None        # assigned by the server
        self.bot_id     = bot_id
        self.name       = name
        self.game_net   = agent.net   # réseau utilisé pour cette partie
        self._executor  = executor
        self._reset_episode()

    def _resolve_color(self, gs: dict) -> bool:
        for p in gs['players']:
            if p.get('userId') == self.bot_id:
                self.color = p['color']
                return True
        return False

    def _reset_episode(self):
        self.color      = None
        self._pending   = None
        self._game_over = False  # garde anti-double-fin (gameState gagnant + gameEnded)
        self.rollout    = RolloutBuffer()
        # Rôle tiré PAR PARTIE (self-play avec pool conservé) : chaque bot joue soit
        # le réseau courant (→ learner, accumule le gradient), soit un snapshot gelé du
        # pool (→ adversaire figé, diversité / anti-oubli). On ne fige plus le rôle par
        # couleur : ainsi 2-3 bots learner en moyenne alimentent le buffer chaque partie.
        if random.random() < LEARNER_PROB or not self.agent.pool:
            self.game_net = self.agent.net
        else:
            self.game_net = self.agent.sample_opponent_net()
        # is_learner ⇔ "joue le réseau courant" : c'est cette condition (et non un rôle
        # fixe) qui décide si la trajectoire est utilisée pour le gradient.
        self.is_learner = (self.game_net is self.agent.net)

    def _flush_rollout(self):
        """Transfère le rollout vers le buffer d'entraînement (learner uniquement)."""
        if self.is_learner and len(self.rollout) > 0:
            self.agent.buffer.extend_from_rollout(self.rollout, GAMMA, GAE_LAMBDA)
        self.rollout = RolloutBuffer()

    def _flush_pending(self, curr_gs: dict | None, done: bool,
                       winner: str | None, next_value: float):
        """Ajoute la transition en attente au rollout local (learner uniquement)."""
        if self._pending is None:
            return
        if self.is_learner:
            p = self._pending
            r = compute_reward(p['gs'], curr_gs, self.color) if curr_gs is not None else 0.0
            if done:
                r += terminal_reward(winner, self.color)
            self.rollout.add(
                p['enc'], p['mask'], p['act'], p['lp'], p['val'],
                r, done, next_value,
            )
        self._pending = None

    def _select_action_sync(self, state_enc: torch.Tensor,
                            mask: list[bool]) -> tuple[int, float, float]:
        """Inférence bloquante (exécutée dans un thread)."""
        if self.is_learner:
            return self.agent.select_action(state_enc, mask)
        legal = torch.tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            with self.agent._inference_lock:  # protège contre les updates
                dist, value = self.game_net(state_enc, legal)
                action      = dist.sample()
                log_prob    = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    async def _select_action(self, loop, state_enc: torch.Tensor,
                             mask: list[bool]) -> tuple[int, float, float]:
        """Wrapper async : délègue l'inférence au thread pool."""
        # Utiliser partial pour éviter les problèmes de sérialisation
        fn = functools.partial(self._select_action_sync, state_enc, mask)
        return await loop.run_in_executor(self._executor, fn)

    async def _join(self, ws):
        await ws.send(json.dumps({
            "type":       "joinMatchmaking",
            "playerName": self.name,
            "browserId":  self.bot_id,
            "userId":     self.bot_id,
        }))

    def _on_game_end(self, gs: dict | None, winner: str | None) -> bool:
        """Fin de partie : flush la trajectoire et signale qu'il faut fermer la WS.
        On NE PEUT PAS re-join sur la même WebSocket : une fois la partie lancée,
        le MultiWsMessenger du Game capte tous les messages entrants de cette WS
        (handler permanent), et le handler initial de session (index.ts) est
        `{ once: true }` — donc un joinMatchmaking renvoyé sur la même socket
        n'atteint jamais le matchmaking. Il faut fermer puis rouvrir.
        Retourne True si la fin a été traitée (→ l'appelant doit return)."""
        if self._game_over:
            return True  # déjà traité (un gameState gagnant a précédé gameEnded)
        self._game_over = True
        self._flush_pending(curr_gs=gs, done=True, winner=winner, next_value=0.0)
        self._flush_rollout()
        self.agent.n_games += 1
        won = (winner == self.color)
        # Seuls les bots jouant le réseau courant (learner) comptent dans la win-rate :
        # mesurer les snapshots gelés brouillerait la métrique de progression.
        if self.is_learner:
            self.agent.recent_wins.append(1 if won else 0)
        result = "GAGNÉ" if won else "perdu"
        # print(f"[{self.name}] {result} (gagnant : {winner})")
        return True

    async def run(self, ws):
        self._reset_episode()  # nouvelle WS = nouvelle partie : réarme _game_over, rollout, etc.
        await self._join(ws)
        loop = asyncio.get_event_loop()

        async for raw in ws:
            try:
                msg = json.loads(raw)

                # actionPlayed : le serveur fait autorité sur le timing
                # (animationDone est ignoré côté serveur) → rien à faire.
                if msg.get("type") == "actionPlayed":
                    continue

                if msg.get("type") == "gameEnded":
                    self._on_game_end(gs=None, winner=msg.get("winner"))
                    return  # ferme la WS → run_bot reconnecte pour la partie suivante

                if msg.get("type") != "gameState":
                    continue

                gs = msg["gameState"]
                if not self._resolve_color(gs):
                    continue  # our join isn't reflected yet
                winner     = _detect_winner(gs)
                is_my_turn = gs["currentTurn"] == self.color

                # ── Fin de partie détectée via gameState ────────────────────
                if winner is not None:
                    self._on_game_end(gs=gs, winner=winner)
                    return  # ferme la WS → run_bot reconnecte pour la partie suivante

                # ── Passer si ce n'est pas mon tour ─────────────────────────
                if not is_my_turn:
                    continue

                # ── Mon tour : décider, jouer, gérer le pending ─────────────
                hand         = gs.get("hand", [])
                can_discard  = gs.get("canDiscard", False)
                mbc          = _marbles_by_color(gs)
                my_marbles   = mbc[self.color]
                inv_by_color = _invincible_by_color(gs)

                mask, _ = get_legal_mask(
                    hand, my_marbles, self.color, mbc,
                    invincible_by_color=inv_by_color,
                    can_discard=can_discard,
                )
                if not any(mask):
                    continue  # ne devrait pas arriver

                state_enc               = encode_state(gs, self.color)
                action, log_prob, value = await self._select_action(loop, state_enc, mask)

                # Flush de la décision précédente avec le V(s) courant comme next_value
                self._flush_pending(curr_gs=gs, done=False,
                                    winner=None, next_value=value)

                msg_out = build_server_message(
                    action, hand, my_marbles, self.color, mbc,
                    invincible_by_color=inv_by_color,
                )
                await ws.send(json.dumps(msg_out))

                self._pending = {
                    'gs':   gs,
                    'enc':  state_enc,
                    'mask': torch.tensor(mask, dtype=torch.bool),
                    'act':  action,
                    'lp':   log_prob,
                    'val':  value,
                }

            except Exception as e:
                # Une erreur isolée (message inattendu, etc.) ne doit PAS fermer
                # la WS : sinon les 3 autres bots restent bloqués 180s.
                print(f"[{self.name}] erreur traitement message: {e!r}")
                continue


# ── Bot d'évaluation (greedy learner OU random, sans gradient) ──────────────────

class EvalBot:
    """Bot d'une partie d'évaluation. Ne touche NI au buffer NI à recent_wins.
      - is_greedy=True  : joue argmax sur le réseau courant (politique déployée).
      - is_greedy=False : joue un coup légal uniformément au hasard (baseline).
    Seul le résultat du bot greedy est enregistré (agent.eval_wins)."""

    def __init__(self, agent: PPOAgent, bot_id: str, name: str, is_greedy: bool):
        self.agent     = agent
        self.is_greedy = is_greedy
        self.bot_id    = bot_id
        self.name      = name
        self.color     = None
        self._over     = False

    def _resolve_color(self, gs: dict) -> bool:
        for p in gs['players']:
            if p.get('userId') == self.bot_id:
                self.color = p['color']
                return True
        return False

    async def _join(self, ws):
        await ws.send(json.dumps({
            "type":       "joinMatchmaking",
            "playerName": self.name,
            "browserId":  self.bot_id,
            "userId":     self.bot_id,
        }))

    def _pick_action(self, gs: dict, mask: list[bool]) -> int:
        legal_idx = [i for i, ok in enumerate(mask) if ok]
        if not self.is_greedy:
            return random.choice(legal_idx)
        # Greedy : argmax de la politique du réseau courant (cf. main.py / prod).
        state_enc = encode_state(gs, self.color)
        legal     = torch.tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            with self.agent._inference_lock:
                dist, _ = self.agent.net(state_enc, legal)
                return int(torch.argmax(dist.probs).item())

    def _on_end(self, winner: str | None):
        if self._over:
            return
        self._over = True
        if self.is_greedy:
            self.agent.eval_wins.append(1 if winner == self.color else 0)

    async def run(self, ws):
        await self._join(ws)
        async for raw in ws:
            try:
                msg = json.loads(raw)
                if msg.get("type") == "actionPlayed":
                    continue
                if msg.get("type") == "gameEnded":
                    self._on_end(msg.get("winner"))
                    return
                if msg.get("type") != "gameState":
                    continue

                gs = msg["gameState"]
                if not self._resolve_color(gs):
                    continue
                winner = _detect_winner(gs)
                if winner is not None:
                    self._on_end(winner)
                    return
                if gs["currentTurn"] != self.color:
                    continue

                hand         = gs.get("hand", [])
                mbc          = _marbles_by_color(gs)
                my_marbles   = mbc[self.color]
                inv_by_color = _invincible_by_color(gs)
                mask, _ = get_legal_mask(
                    hand, my_marbles, self.color, mbc,
                    invincible_by_color=inv_by_color,
                    can_discard=gs.get("canDiscard", False),
                )
                if not any(mask):
                    continue

                action  = self._pick_action(gs, mask)
                msg_out = build_server_message(
                    action, hand, my_marbles, self.color, mbc,
                    invincible_by_color=inv_by_color,
                )
                await ws.send(json.dumps(msg_out))
            except Exception as e:
                print(f"[{self.name}] erreur éval: {e!r}")
                continue


# ── Boucle d'entraînement ─────────────────────────────────────────────────────

async def run_training():
    agent = PPOAgent()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    # Plus de rôle fixe par couleur : chaque bot tire son rôle PAR PARTIE dans
    # _reset_episode (réseau courant → learner, ou snapshot du pool → adversaire figé).
    # En moyenne ~LEARNER_PROB × 4 bots alimentent le gradient à chaque partie.

    async def run_bot(color: str):
        bot_id     = f"rl-bot-{color}"
        name       = f"RL-{color.capitalize()}"
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{API_URL}/api/auth/bot",
                json={"secret": BOT_SECRET, "botId": bot_id, "name": name},
            )
        bot = RLBot(agent, bot_id, name, executor=executor)
        while True:
            try:
                async with websockets.connect(WS_URL) as ws:
                    await bot.run(ws)
            except Exception as e:
                print(f"[{name}] connexion perdue ({e}), reconnexion…")
                await asyncio.sleep(2)

    async def run_eval_game():
        """Une partie d'éval isolée : 1 learner greedy + 3 bots aléatoires.
        Les 4 bots rejoignent en même temps ; matchmaking FIFO ⇒ même partie
        (les 4 bots de training sont alors occupés dans leur propre partie)."""
        bots = []
        for color in COLORS:
            is_greedy = (color == EVAL_LEARNER_COLOR)
            bot_id = f"eval-bot-{color}"
            name   = f"EVAL-{'NET' if is_greedy else 'RND'}-{color.capitalize()}"
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{API_URL}/api/auth/bot",
                    json={"secret": BOT_SECRET, "botId": bot_id, "name": name},
                )
            bots.append(EvalBot(agent, bot_id, name, is_greedy=is_greedy))

        async def play(bot: EvalBot):
            try:
                async with websockets.connect(WS_URL) as ws:
                    await bot.run(ws)
            except Exception as e:
                print(f"[{bot.name}] éval connexion perdue ({e})")

        await asyncio.gather(*(play(b) for b in bots))
        if agent.eval_wins:
            rate = sum(agent.eval_wins) / len(agent.eval_wins)
            print(f"[EVAL] vs random : win_rate={rate:.3f} "
                  f"(n={len(agent.eval_wins)}, dernière={agent.eval_wins[-1]})")

    async def update_loop():
        last_eval = 0
        while True:
            await asyncio.sleep(2)
            if len(agent.buffer) >= UPDATE_EVERY:
                agent.update()
                if agent.n_updates - last_eval >= EVAL_EVERY:
                    last_eval = agent.n_updates
                    # Lancer en tâche de fond : ne pas bloquer la boucle d'update.
                    asyncio.create_task(run_eval_game())

    await asyncio.gather(
        *[run_bot(color) for color in COLORS],
        update_loop(),
    )


if __name__ == "__main__":
    asyncio.run(run_training())
