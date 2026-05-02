"""
Bot Mercury — entraînement PPO en self-play.
4 bots partagent le même réseau et jouent simultanément contre le serveur local.

Lancer :
  cd /home/loubard/Documents/mercury
  python bot_rl.py
"""

import asyncio, json, pathlib, random, threading
import concurrent.futures, functools
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
ENTROPY_COEF = 0.01
VALUE_COEF   = 0.5
UPDATE_EVERY = 2048  # transitions agrégées avant chaque update PPO (4× plus stable)
PPO_EPOCHS   = 4
BATCH_SIZE   = 64

POOL_SIZE      = 5   # snapshots du passé conservés en mémoire
SNAPSHOT_EVERY = 10  # sauvegarder un snapshot tous les N updates

COLORS = ['red', 'green', 'blue']


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
        """Tire aléatoirement un réseau du pool (ou le réseau courant si pool vide)."""
        if not self.pool:
            return self.net
        # 50 % courant, 50 % pool → le pool se diversifie sans trop décrocher
        candidates = self.pool + [self.net]
        return random.choice(candidates)

    def select_action(self, state: torch.Tensor,
                      mask: list[bool]) -> tuple[int, float, float]:
        legal = torch.tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            with self._inference_lock:  # protège contre les updates simultanées
                dist, value = self.net(state, legal)
                action      = dist.sample()
                log_prob    = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

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
                    v_loss  = F.mse_loss(values, returns[idx])
                    loss    = p_loss + VALUE_COEF * v_loss - ENTROPY_COEF * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                    self.optimizer.step()

        self.buffer.clear()
        self.n_updates += 1
        torch.save(self.net.state_dict(), MODEL_PATH)
        if self.n_updates % SNAPSHOT_EVERY == 0:
            self._save_snapshot()
        print(f"[PPO] update #{self.n_updates}  parties={self.n_games}  "
              f"transitions={n}  pool={len(self.pool)}  sauvegardé → {MODEL_PATH}")


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


class RLBot:
    def __init__(self, agent: PPOAgent, bot_id: str, name: str,
                 is_learner: bool = True,
                 executor: concurrent.futures.Executor = None):
        self.agent      = agent
        self.is_learner = is_learner  # False → adversaire du pool, pas de gradient
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
        self.color    = None
        self._pending = None
        self.rollout  = RolloutBuffer()
        # L'adversaire tire un réseau aléatoire (pool ou courant) à chaque partie
        if not self.is_learner:
            self.game_net = self.agent.sample_opponent_net()

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

    async def run(self, ws):
        await self._join(ws)
        loop = asyncio.get_event_loop()
        anim_done = json.dumps({"type": "animationDone"})

        async for raw in ws:
            msg = json.loads(raw)

            # Débloquer le serveur dès qu'une action est animée
            if msg.get("type") == "actionPlayed":
                await ws.send(anim_done)
                continue

            if msg.get("type") == "gameEnded":
                winner = msg.get("winner")
                # Flush terminal : dernière transition avec la récompense terminale
                self._flush_pending(curr_gs=None, done=True,
                                    winner=winner, next_value=0.0)
                self._flush_rollout()
                self.agent.n_games += 1
                result = "GAGNÉ" if winner == self.color else "perdu"
                print(f"[{self.name}] {result} (gagnant : {winner})")
                self._reset_episode()
                return  # ferme le WS, le while True externe reconnecte

            if msg.get("type") != "gameState":
                continue

            gs = msg["gameState"]
            if not self._resolve_color(gs):
                continue  # our join isn't reflected yet
            winner     = _detect_winner(gs)
            is_my_turn = gs["currentTurn"] == self.color

            # ── Fin de partie détectée via gameState ────────────────────────
            if winner is not None:
                self._flush_pending(curr_gs=gs, done=True,
                                    winner=winner, next_value=0.0)
                self._flush_rollout()
                self.agent.n_games += 1
                result = "GAGNÉ" if winner == self.color else "perdu"
                print(f"[{self.name}] {result} (gagnant : {winner})")
                self._reset_episode()
                await self._join(ws)
                continue

            # ── Passer si ce n'est pas mon tour ─────────────────────────────
            if not is_my_turn:
                continue

            # ── Mon tour : décider, jouer, gérer le pending ─────────────────
            hand        = gs.get("hand", [])
            can_discard = gs.get("canDiscard", False)
            mbc         = _marbles_by_color(gs)
            my_marbles  = mbc[self.color]

            mask, _ = get_legal_mask(hand, my_marbles, self.color, mbc, can_discard)
            if not any(mask):
                continue  # ne devrait pas arriver

            state_enc               = encode_state(gs, self.color)
            action, log_prob, value = await self._select_action(loop, state_enc, mask)

            # Flush de la décision précédente avec le V(s) courant comme next_value
            self._flush_pending(curr_gs=gs, done=False,
                                winner=None, next_value=value)

            msg_out = build_server_message(action, hand, my_marbles, self.color, mbc)
            await ws.send(json.dumps(msg_out))

            self._pending = {
                'gs':   gs,
                'enc':  state_enc,
                'mask': torch.tensor(mask, dtype=torch.bool),
                'act':  action,
                'lp':   log_prob,
                'val':  value,
            }


# ── Boucle d'entraînement ─────────────────────────────────────────────────────

async def run_training():
    agent = PPOAgent()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    # Premier bot = learner (accumule le gradient).
    # Les autres sont des adversaires qui tirent aléatoirement dans le pool
    # de snapshots → diversité des opposants, stabilité de l'entraînement.
    LEARNER_COLOR = COLORS[0]

    async def run_bot(color: str):
        is_learner = (color == LEARNER_COLOR)
        bot_id     = f"rl-bot-{color}"
        name       = f"RL-{color.capitalize()}"
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{API_URL}/api/auth/bot",
                json={"secret": BOT_SECRET, "botId": bot_id, "name": name},
            )
        bot = RLBot(agent, bot_id, name, is_learner=is_learner, executor=executor)
        while True:
            try:
                async with websockets.connect(WS_URL) as ws:
                    await bot.run(ws)
            except Exception as e:
                print(f"[{name}] connexion perdue ({e}), reconnexion…")
                await asyncio.sleep(2)

    async def update_loop():
        while True:
            await asyncio.sleep(2)
            if len(agent.buffer) >= UPDATE_EVERY:
                agent.update()

    await asyncio.gather(
        *[run_bot(color) for color in COLORS],
        update_loop(),
    )


if __name__ == "__main__":
    asyncio.run(run_training())
