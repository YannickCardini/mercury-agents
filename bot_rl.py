"""
Bot Mercury — entraînement PPO en self-play.
4 bots partagent le même réseau et jouent simultanément contre le serveur local.

Lancer :
  cd /home/loubard/Documents/mercury
  python bot_rl.py
"""

import asyncio, json, os, pathlib, random, shutil, threading
import concurrent.futures, functools
from collections import deque
import torch
import torch.nn.functional as F
import httpx
import websockets

from mercury_legal_moves import (
    get_legal_mask, build_server_message, DISCARD_IDX,
    ARRIVAL_POSITIONS, HOME_POSITIONS, ALL_COLORS,
)
from reward  import compute_reward, terminal_reward
from model   import (
    MercuryNet, encode_state, encode_state_for, load_net, ACTION_DIM, STATE_DIM,
)
from heuristic_bot import heuristic_pick
from plan_hand     import plan_hand_pick

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

POOL_SIZE      = 18   # snapshots du passé conservés (élargi 5→18 : plus de diversité
                      # d'adversaires, réduit l'oubli catastrophique et l'oscillation)
SNAPSHOT_EVERY = 25   # snapshot tous les N updates (espacé 10→25 : couvre un historique
                      # plus large avec le même nombre de snapshots conservés)
WINRATE_WINDOW = 200  # fenêtre glissante pour la win-rate des bots learner
LEARNER_PROB   = 0.6  # proba qu'un bot joue le réseau courant (→ learner) par partie
                      # ⇒ ~2-3 des 4 bots alimentent le gradient ; le reste = pool gelé
# Parmi les bots NON-learner, proba de jouer le bot HEURISTIQUE scripté au lieu d'un
# snapshot du pool. Apporte un adversaire cohérent (≠ self-play) → robustesse + ancre
# anti-dérive. ~0.08/bot ⇒ ~0.3 heuristique/partie (1 seul ⇒ pas de gridlock).
HEUR_OPPONENT_PROB = 0.2

# ── Évaluation vs bots aléatoires ───────────────────────────────────────────────
# Le win_rate vs pool tend mécaniquement vers 0.25 en self-play symétrique (on joue
# contre soi-même). Pour une métrique ABSOLUE de progression, on lance périodiquement
# une partie isolée : 1 learner greedy (la politique réellement déployée, cf. main.py)
# contre 3 bots aléatoires. Matchmaking FIFO ⇒ les 4 bots d'éval, rejoignant ensemble
# pendant que les bots de training sont en partie, sont regroupés dans la même partie.
EVAL_EVERY        = 10   # lancer une partie d'éval tous les N updates PPO
EVAL_WINDOW       = 50   # fenêtre glissante de l'eval win-rate
EVAL_LEARNER_COLOR = ALL_COLORS[0]  # couleur du bot greedy dans le quatuor d'éval

# ── Évaluation vs bot HEURISTIQUE (adversaire cohérent) ─────────────────────────
# vs random est devenu un simple plancher (il a fini son rôle : trouver le bug de
# capture). Le vrai juge — celui qui corrèle avec le niveau humain — est le win-rate
# d'1 net greedy contre 3 bots heuristiques (jeu cohérent, cf. heuristic_bot.py).
HEUR_EVERY = 10   # lancer une partie net vs 3 heuristiques tous les N updates PPO

# Plafond de coups d'une partie d'éval : au-delà, on l'abandonne (comptée comme défaite).
# Évite les parties dégénérées qui tournent en rond (gridlock) et bloquent l'isolated_lock /
# saturent le serveur. Une partie normale fait ~150-190 coups → 400 laisse de la marge.
# Une partie à 4 joueurs COMPÉTENTS est longue (interférence/captures mutuelles) : ~400-600
# coups, vs ~177 contre du random. 700 ne coupe donc que le vrai gridlock, pas une partie
# normalement longue. S'applique à TOUS les bots d'éval (sinon un bot non plafonné dans une
# partie gridlockée bloque l'isolated_lock à vie).
EVAL_MOVE_CAP = 700
# Sécurité ultime : si une partie isolée traîne au-delà de ça (s), on l'abandonne et on
# libère le verrou (évite tout deadlock résiduel, ex. WS qui stalle sans coup joué).
ISOLATED_GAME_TIMEOUT = 300
# Plafond de coups d'une partie d'ENTRAÎNEMENT : au-delà, le bot abandonne (sans polluer
# le buffer) et ferme la WS. ~150-300 coups/bot normal → 800 ne coupe que le gridlock.
TRAIN_MOVE_CAP = 800

# ── Duel vs l'agent de référence (la version 0.82 en prod) ──────────────────────
# Métrique du VRAI critère : un candidat ne remplace la prod que s'il bat le ref en
# duel direct. Partie isolée : 2 bots jouent le réseau courant, 2 jouent le ref figé
# (positions en diagonale). On mesure la win-rate du réseau courant.
REF_MODEL_PATH = pathlib.Path("model_ref_082.pt")  # référence figée (agent 0.82 prod)
DUEL_EVERY     = 20   # lancer un duel tous les N updates PPO
PLAN_DUEL_EVERY = 20  # duel planificateur-de-main vs réseau réactif (mesure : planifier
                      # bat-il jouer au coup par coup ?). 0.50 = égalité, > 0.50 = le plan gagne.
DUEL_WINDOW    = 50   # fenêtre glissante de la win-rate de duel
DUEL_REPLACE_THRESHOLD = 0.55  # seuil de supériorité pour considérer un remplacement prod
# Couleurs jouées par le réseau COURANT dans le duel (les 2 autres jouent le ref).
# En diagonale pour neutraliser tout biais de position de départ.
DUEL_CURRENT_COLORS = {ALL_COLORS[0], ALL_COLORS[2]}

COLORS = ALL_COLORS  # 4 joueurs : 1 learner (red) + 3 adversaires


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
        # Fenêtre glissante des résultats des bots learner (1 = gagné, 0 = perdu).
        # En self-play symétrique elle tend vers ~0.25 (on joue contre ses clones) :
        # c'est un indicateur de stabilité, PAS de progression absolue.
        self.recent_wins = deque(maxlen=WINRATE_WINDOW)
        # Win-rate du learner greedy vs 3 bots aléatoires : métrique absolue (thermomètre).
        self.eval_wins = deque(maxlen=EVAL_WINDOW)
        # Diagnostic du mode d'échec vs random (fenêtre glissante, bot net uniquement) :
        # pourquoi perd-on les ~18 % ? finies = billes en arrivée en fin de partie (0-4),
        # captured = mes billes renvoyées au home (capturées), length = nb de mes coups,
        # discards = mains jetées faute de coup utile.
        self.eval_finished = deque(maxlen=EVAL_WINDOW)
        self.eval_captured = deque(maxlen=EVAL_WINDOW)
        self.eval_length   = deque(maxlen=EVAL_WINDOW)
        self.eval_discards = deque(maxlen=EVAL_WINDOW)
        # Métrique COHÉRENTE (le vrai juge) : net greedy vs 3 heuristiques. Mêmes diagnostics.
        self.heur_wins     = deque(maxlen=EVAL_WINDOW)
        self.heur_finished = deque(maxlen=EVAL_WINDOW)
        self.heur_captured = deque(maxlen=EVAL_WINDOW)
        self.heur_length   = deque(maxlen=EVAL_WINDOW)
        self.heur_discards = deque(maxlen=EVAL_WINDOW)
        self.best_heur     = 0.0   # meilleur win-rate vs heuristique → auto-save
        # Win-rate du réseau courant vs l'agent de référence (0.82 prod) en duel direct :
        # LE critère de remplacement prod. Un candidat doit dépasser DUEL_REPLACE_THRESHOLD.
        self.duel_wins = deque(maxlen=DUEL_WINDOW)
        self.best_duel = 0.0   # meilleur duel_rate vu → déclenche l'auto-save model_best.pt
        # Win-rate du PLANIFICATEUR DE MAIN vs le réseau réactif (2 vs 2). Mesure si la
        # planification multi-cartes bat le choix au coup par coup. 0.50 = égalité.
        self.plan_wins = deque(maxlen=DUEL_WINDOW)
        # Niveau ABSOLU du planificateur : vs 3 random (à comparer au 0.83 du réseau et au
        # ~0.999 d'un humain). C'est la jauge qui dit si on s'approche du niveau humain.
        self.plan_rnd_wins = deque(maxlen=EVAL_WINDOW)
        # Filet de sécurité : meilleur win-rate vs random vu → auto-save model_best_eval.pt.
        # La win_rate self-play (~0.25) est aveugle aux régressions absolues ; ce best
        # capture la VRAIE meilleure politique vs random et l'empêche d'être perdue.
        self.best_eval = 0.0
        # Réseau de référence figé pour le duel (chargé dans sa propre architecture via
        # load_net : le ref est un ancien MercuryNetLegacy 2×256).
        self.ref_net = load_net(REF_MODEL_PATH) if REF_MODEL_PATH.exists() else None
        if self.ref_net is None:
            print(f"[duel] {REF_MODEL_PATH} introuvable → duel désactivé")
        self._load()
        self._load_best_thresholds()

    def _load_best_thresholds(self):
        """Recharge les records best_* depuis les fichiers best déjà sauvegardés.
        SANS ça, best_*=0 à chaque relance → le 1er éval écrase le fichier best avec un
        modèle potentiellement MOINS BON (le filet se sabote). Le seuil DOIT persister
        pour que `model_best_eval.pt` reste le meilleur de TOUS les runs."""
        for path, attr, key in (("model_best_eval.pt", "best_eval", "eval_rate"),
                                ("model_best.pt",      "best_duel", "duel_rate"),
                                ("model_best_heur.pt", "best_heur", "heur_rate")):
            p = pathlib.Path(path)
            if not p.exists():
                continue
            try:
                ck = torch.load(p, weights_only=True)
                if isinstance(ck, dict) and key in ck:
                    setattr(self, attr, float(ck[key]))
                    print(f"[best] seuil {attr}={float(ck[key]):.3f} rechargé ({path})")
            except Exception as e:
                print(f"[best] lecture impossible {path}: {e!r}")

    def _load(self):
        if not MODEL_PATH.exists():
            return
        try:
            ckpt = torch.load(MODEL_PATH, weights_only=True)
            # Nouveau format : dict {net, n_updates, n_games}.
            # Ancien format (rétro-compat) : state_dict brut du réseau.
            if isinstance(ckpt, dict) and 'net' in ckpt:
                self.net.load_state_dict(ckpt['net'])
                self.n_updates = ckpt.get('n_updates', 0)
                self.n_games   = ckpt.get('n_games', 0)
                print(f"Checkpoint chargé depuis {MODEL_PATH} "
                      f"(updates={self.n_updates}, games={self.n_games})")
            else:
                self.net.load_state_dict(ckpt)
                print(f"Modèle chargé depuis {MODEL_PATH} (ancien format state_dict brut)")
        except RuntimeError:
            # STATE_DIM a changé (passage à l'encodage 138-dim) → l'ancien checkpoint n'est
            # plus chargeable. On met de côté les modèles déployables (au lieu de les
            # supprimer) pour ne rien perdre, puis on repart de zéro.
            backup = MODEL_PATH.with_name("model_legacy54_backup.pt")
            if not backup.exists():
                os.replace(MODEL_PATH, backup)   # déplace model.pt → backup
                print(f"Checkpoint incompatible (nouvel encodage) → sauvegardé dans "
                      f"{backup}. Entraînement repart de zéro.")
            else:
                MODEL_PATH.unlink()
                print("Checkpoint incompatible (nouvel encodage). Entraînement repart de zéro.")
            # Préserver aussi l'ancien model_best.pt (save_best l'écraserait dès le 1er duel).
            best, best_bak = pathlib.Path("model_best.pt"), pathlib.Path("model_best_legacy54_backup.pt")
            if best.exists() and not best_bak.exists():
                shutil.copy2(best, best_bak)
                print(f"Ancien model_best.pt sauvegardé dans {best_bak}.")

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

    def _save_checkpoint(self):
        """Sauvegarde ATOMIQUE de {net, n_updates, n_games}.
        Écriture fichier temp + os.replace : une coupure de courant pendant l'écriture
        laisse soit l'ancien fichier intact, soit le nouveau complet — jamais un fichier
        tronqué."""
        ckpt = {
            'net':       self.net.state_dict(),
            'n_updates': self.n_updates,
            'n_games':   self.n_games,
        }
        tmp = MODEL_PATH.with_suffix('.pt.tmp')
        torch.save(ckpt, tmp)
        os.replace(tmp, MODEL_PATH)   # atomique sur le même système de fichiers

    def save_best(self, duel_rate: float):
        """Sauvegarde model_best.pt si `duel_rate` bat le record (capture le pic, qu'on
        perdait jusqu'ici quand la métrique oscillait). Écriture atomique."""
        if duel_rate <= self.best_duel:
            return
        self.best_duel = duel_rate
        best_path = pathlib.Path("model_best.pt")
        ckpt = {'net': self.net.state_dict(),
                'n_updates': self.n_updates, 'n_games': self.n_games,
                'duel_rate': duel_rate}
        tmp = best_path.with_suffix('.pt.tmp')
        torch.save(ckpt, tmp)
        os.replace(tmp, best_path)
        print(f"[BEST] nouveau record duel={duel_rate:.3f} → {best_path}")

    def save_best_eval(self, eval_rate: float):
        """Sauvegarde model_best_eval.pt si `eval_rate` (win-rate vs random) bat le record.
        Filet absolu : capture la meilleure politique vs random, que la win_rate self-play
        (toujours ~0.25) est incapable de signaler. Écriture atomique."""
        if eval_rate <= self.best_eval:
            return
        self.best_eval = eval_rate
        best_path = pathlib.Path("model_best_eval.pt")
        ckpt = {'net': self.net.state_dict(),
                'n_updates': self.n_updates, 'n_games': self.n_games,
                'eval_rate': eval_rate}
        tmp = best_path.with_suffix('.pt.tmp')
        torch.save(ckpt, tmp)
        os.replace(tmp, best_path)
        print(f"[BEST-EVAL] nouveau record vs random={eval_rate:.3f} → {best_path}")

    def save_best_heur(self, heur_rate: float):
        """Sauvegarde model_best_heur.pt si `heur_rate` (win-rate vs heuristique) bat le
        record. C'est LE critère qui corrèle avec le niveau humain. Écriture atomique."""
        if heur_rate <= self.best_heur:
            return
        self.best_heur = heur_rate
        best_path = pathlib.Path("model_best_heur.pt")
        ckpt = {'net': self.net.state_dict(),
                'n_updates': self.n_updates, 'n_games': self.n_games,
                'heur_rate': heur_rate}
        tmp = best_path.with_suffix('.pt.tmp')
        torch.save(ckpt, tmp)
        os.replace(tmp, best_path)
        print(f"[BEST-HEUR] nouveau record vs heuristique={heur_rate:.3f} → {best_path}")

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

        # Le critique prédit directement les retours BRUTS. Inutile de normaliser : avec
        # l'échelle de récompenses ÷10 (reward.py), les retours sont déjà d'amplitude ~±3,
        # parfaitement fittable. (Une normalisation par moyenne courante finissait par se
        # figer — count → ∞ — et désalignait le critique.)

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
                    v_loss  = F.mse_loss(values, returns[idx])
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
        self._save_checkpoint()   # atomique : {net, n_updates, n_games}
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
        self.agent         = agent
        self.is_learner    = True        # (re)décidé par partie dans _reset_episode
        self.color         = None        # assigned by the server
        self.bot_id        = bot_id
        self.name          = name
        self.session_token = ""          # défini avant chaque partie par run_bot
        self.user_id       = bot_id      # ID serveur, défini avant chaque partie par run_bot
        self.game_net      = agent.net   # réseau utilisé pour cette partie
        self._executor     = executor
        self._reset_episode()

    def _resolve_color(self, gs: dict) -> bool:
        for p in gs['players']:
            if p.get('userId') == self.user_id:
                self.color = p['color']
                return True
        return False

    def _reset_episode(self):
        self.color      = None
        self._pending   = None
        self._game_over = False  # garde anti-double-fin (gameState gagnant + gameEnded)
        self._turns     = 0      # nb de mes coups (plafonné par TRAIN_MOVE_CAP)
        self.rollout    = RolloutBuffer()
        # Rôle tiré PAR PARTIE (self-play avec pool conservé) : chaque bot joue soit
        # le réseau courant (→ learner, accumule le gradient), soit un snapshot gelé du
        # pool (→ adversaire figé, diversité / anti-oubli). On ne fige plus le rôle par
        # couleur : ainsi 2-3 bots learner en moyenne alimentent le buffer chaque partie.
        if random.random() < LEARNER_PROB or not self.agent.pool:
            self.game_net = self.agent.net
            self.role     = 'learner'
        elif random.random() < HEUR_OPPONENT_PROB:
            # Adversaire SCRIPTÉ cohérent (≠ self-play). Un seul par partie en moyenne ⇒
            # pas de gridlock. Aucune inférence réseau, aucun gradient.
            self.game_net = None
            self.role     = 'heuristic'
        else:
            self.game_net = self.agent.sample_opponent_net()
            self.role     = 'pool'
        # is_learner ⇔ "joue le réseau courant" : c'est cette condition (et non un rôle
        # fixe) qui décide si la trajectoire est utilisée pour le gradient.
        self.is_learner = (self.role == 'learner')

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
            "type":      "joinMatchmaking",
            "authToken": self.session_token,
            "name":      self.name,
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

                mask, actions = get_legal_mask(
                    hand, my_marbles, self.color, mbc,
                    invincible_by_color=inv_by_color,
                    can_discard=can_discard,
                )
                if not any(mask):
                    continue  # ne devrait pas arriver

                self._turns += 1
                if self._turns > TRAIN_MOVE_CAP:
                    # Partie dégénérée (gridlock) → on abandonne SANS flush (pas de pollution
                    # du buffer) et on ferme la WS. run_bot reconnecte pour la suivante.
                    print(f"[{self.name}] partie >{TRAIN_MOVE_CAP} coups → abandon (anti-gridlock)")
                    return

                # Adversaire heuristique : coup scripté, pas d'inférence réseau, pas de
                # gradient (is_learner=False). On joue et on attend le message suivant.
                if self.role == 'heuristic':
                    action  = heuristic_pick(gs, self.color, mask, actions)
                    msg_out = build_server_message(
                        action, hand, my_marbles, self.color, mbc,
                        invincible_by_color=inv_by_color,
                    )
                    await ws.send(json.dumps(msg_out))
                    continue

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


# ── Bot d'évaluation (greedy sur un réseau donné OU random, sans gradient) ──────

class EvalBot:
    """Bot d'une partie d'éval/duel. Ne touche NI au buffer NI à recent_wins.
      - mode='random'    : joue un coup légal uniformément au hasard (baseline).
      - mode='heuristic' : joue le bot scripté cohérent (heuristic_bot.heuristic_pick).
      - mode='plan'      : joue le planificateur de main (plan_hand.plan_hand_pick).
      - mode='net'       : joue argmax sur le réseau `net` fourni (greedy, comme la prod).
    Si `result_sink` (deque) est fourni, le résultat du bot (1=gagné, 0=perdu) y est
    ajouté en fin de partie. Sert à la fois pour l'éval vs random (1 net + 3 random,
    sink sur le net) et le duel (2 courant + 2 ref, sink sur le courant).
    `agent` n'est utilisé que pour le verrou d'inférence (protège net pendant updates)."""

    def __init__(self, agent: PPOAgent, bot_id: str, name: str, mode: str,
                 net=None, result_sink=None, collect_stats: bool = False):
        self.agent         = agent
        self.mode          = mode          # 'random' | 'heuristic' | 'plan' | 'net'
        self.net           = net           # réseau à jouer si mode == 'net'
        self.result_sink   = result_sink   # deque où enregistrer le résultat (ou None)
        self.bot_id        = bot_id
        self.name          = name
        self.session_token = ""            # défini avant chaque partie par _play_isolated_game
        self.user_id       = bot_id        # ID serveur, défini avant chaque partie par _play_isolated_game
        self.color         = None
        self._over       = False
        self._won_color  = None          # couleur gagnante de la partie (lue après coup)
        # Diagnostic du mode d'échec (bot net d'éval uniquement). Compteurs accumulés
        # sur la partie, relus après coup par run_eval_game.
        self.collect_stats = collect_stats
        self.turns_played  = 0           # nb de mes coups joués
        self.discards      = 0           # nb de mains jetées
        self.captured      = 0           # nb de mes billes renvoyées au home (capturées)
        self.finished      = 0           # mes billes en arrivée en fin de partie (0-4)
        self._prev_my      = None        # mes positions au gameState précédent

    def _resolve_color(self, gs: dict) -> bool:
        for p in gs['players']:
            if p.get('userId') == self.user_id:
                self.color = p['color']
                return True
        return False

    async def _join(self, ws):
        await ws.send(json.dumps({
            "type":      "joinMatchmaking",
            "authToken": self.session_token,
            "name":      self.name,
        }))

    def _pick_action(self, gs: dict, mask: list[bool], actions: list = None) -> int:
        legal_idx = [i for i, ok in enumerate(mask) if ok]
        if self.mode == 'random':
            return random.choice(legal_idx)
        if self.mode == 'heuristic':
            return heuristic_pick(gs, self.color, mask, actions)
        if self.mode == 'plan':
            return plan_hand_pick(gs, self.color, mask, actions)
        # Greedy : argmax de la politique du réseau fourni (cf. main.py / prod).
        # encode_state_for route vers l'encodeur correct : le ref de duel est un modèle
        # LEGACY (entrée 54) tandis que le réseau courant utilise l'encodage 138-dim.
        state_enc = encode_state_for(self.net, gs, self.color)
        legal     = torch.tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            with self.agent._inference_lock:  # le réseau courant peut être en cours d'update
                dist, _ = self.net(state_enc, legal)
                return int(torch.argmax(dist.probs).item())

    def _track_captures(self, gs: dict):
        """Capture subie = une de mes billes passe de 'en jeu' à 'home' entre deux
        gameStates (dans ce jeu une bille ne retourne au home QUE si elle est capturée :
        mon propre coup ne l'y renvoie jamais — promote va en arrivée, enter quitte le
        home). Appelé sur CHAQUE gameState (les captures arrivent aux tours adverses)."""
        my = _marbles_by_color(gs).get(self.color)
        if my is None:
            return
        home = HOME_POSITIONS[self.color]
        if self._prev_my is not None:
            for prev_p, cur_p in zip(self._prev_my, my):
                if prev_p not in home and cur_p in home:
                    self.captured += 1
        self._prev_my = list(my)

    def _on_end(self, winner: str | None):
        if self._over:
            return
        self._over = True
        self._won_color = winner
        if self.result_sink is not None:
            self.result_sink.append(1 if winner == self.color else 0)
        if self.collect_stats and self._prev_my is not None:
            arrival = ARRIVAL_POSITIONS[self.color]
            self.finished = sum(1 for p in self._prev_my if p in arrival)

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
                if self.collect_stats:
                    self._track_captures(gs)   # sur chaque état (captures aux tours adverses)
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
                mask, actions = get_legal_mask(
                    hand, my_marbles, self.color, mbc,
                    invincible_by_color=inv_by_color,
                    can_discard=gs.get("canDiscard", False),
                )
                if not any(mask):
                    continue

                action  = self._pick_action(gs, mask, actions)
                self.turns_played += 1
                if self.collect_stats and action == DISCARD_IDX:
                    self.discards += 1
                if self.turns_played >= EVAL_MOVE_CAP:
                    # Cap pour TOUS les bots (pas que collect_stats) : sinon un bot non
                    # plafonné dans une partie gridlockée bloque l'isolated_lock à vie.
                    # Abandon comptée comme défaite, fermeture de la WS.
                    self._on_end(None)
                    return
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
    # Sérialise éval et duel : deux parties isolées ne doivent jamais être en file
    # de matchmaking en même temps (FIFO les mélangerait dans une même partie).
    isolated_lock = asyncio.Lock()

    # Plus de rôle fixe par couleur : chaque bot tire son rôle PAR PARTIE dans
    # _reset_episode (réseau courant → learner, ou snapshot du pool → adversaire figé).
    # En moyenne ~LEARNER_PROB × 4 bots alimentent le gradient à chaque partie.

    async def run_bot(color: str):
        bot_id = f"rl-bot-{color}"
        name   = f"RL-{color.capitalize()}"
        bot    = RLBot(agent, bot_id, name, executor=executor)
        while True:
            # Auth avant chaque partie : token valide pour la durée d'une partie uniquement.
            # Robuste : on réessaie si le serveur est lent/redémarre.
            async with httpx.AsyncClient(timeout=30.0) as client:
                while True:
                    try:
                        r = await client.post(
                            f"{API_URL}/api/auth/bot",
                            json={"secret": BOT_SECRET, "botId": bot_id, "name": name},
                        )
                        r.raise_for_status()
                        data = r.json()
                        bot.session_token = data["sessionToken"]
                        bot.user_id       = data.get("userId", bot_id)
                        break
                    except Exception as e:
                        print(f"[{name}] serveur injoignable à l'auth ({e!r}), retry dans 3s…")
                        await asyncio.sleep(3)
            try:
                async with websockets.connect(WS_URL, max_size=None) as ws:
                    await bot.run(ws)
            except Exception as e:
                print(f"[{name}] connexion perdue ({e}), reconnexion…")
                await asyncio.sleep(2)

    async def _play_isolated_game(bots: list):
        """Lance les 4 bots fournis en parallèle sur des WS séparées. Le matchmaking
        FIFO les regroupe (les 4 bots de training sont occupés ailleurs). Sert pour
        l'éval vs random ET le duel vs ref. Sérialisé via isolated_lock pour qu'éval et
        duel ne se mélangent jamais dans la file de matchmaking."""
        async with isolated_lock:
            for bot in bots:
                async with httpx.AsyncClient() as client:
                    r = await client.post(
                        f"{API_URL}/api/auth/bot",
                        json={"secret": BOT_SECRET, "botId": bot.bot_id, "name": bot.name},
                    )
                    r.raise_for_status()
                    data = r.json()
                    bot.session_token = data["sessionToken"]
                    bot.user_id       = data.get("userId", bot.bot_id)

            async def play(bot: EvalBot):
                try:
                    async with websockets.connect(WS_URL, max_size=None) as ws:
                        await bot.run(ws)
                except Exception as e:
                    print(f"[{bot.name}] partie isolée perdue ({e})")

            try:
                await asyncio.wait_for(
                    asyncio.gather(*(play(b) for b in bots)),
                    timeout=ISOLATED_GAME_TIMEOUT,
                )
            except asyncio.TimeoutError:
                print(f"[isolated] partie isolée > {ISOLATED_GAME_TIMEOUT}s → abandon "
                      f"(verrou libéré)")

    async def run_eval_game():
        """Éval isolée : 1 bot greedy (réseau courant) + 3 bots aléatoires."""
        bots = []
        net_bot = None
        for color in COLORS:
            is_net = (color == EVAL_LEARNER_COLOR)
            bot = EvalBot(
                agent, f"eval-bot-{color}",
                f"EVAL-{'NET' if is_net else 'RND'}-{color.capitalize()}",
                mode=('net' if is_net else 'random'),
                net=(agent.net if is_net else None),
                result_sink=(agent.eval_wins if is_net else None),
                collect_stats=is_net,
            )
            if is_net:
                net_bot = bot
            bots.append(bot)
        await _play_isolated_game(bots)
        # Diagnostic : n'enregistrer les stats que si la partie a bien abouti pour le net.
        if net_bot is not None and net_bot._over:
            agent.eval_finished.append(net_bot.finished)
            agent.eval_captured.append(net_bot.captured)
            agent.eval_length.append(net_bot.turns_played)
            agent.eval_discards.append(net_bot.discards)
        if agent.eval_wins:
            rate = sum(agent.eval_wins) / len(agent.eval_wins)
            # Filet absolu : ne déclencher le best qu'avec assez d'échantillons (évite un
            # faux record sur une fenêtre quasi vide).
            if len(agent.eval_wins) >= 20:
                agent.save_best_eval(rate)
            print(f"[EVAL] vs random : win_rate={rate:.3f} "
                  f"(n={len(agent.eval_wins)}, dernière={agent.eval_wins[-1]})")
            if agent.eval_finished:
                avg = lambda d: sum(d) / len(d)
                print(f"[EVAL] diag TOUTES (moy/partie n={len(agent.eval_finished)}) : "
                      f"billes_finies={avg(agent.eval_finished):.2f}/4  "
                      f"captures_subies={avg(agent.eval_captured):.2f}  "
                      f"coups={avg(agent.eval_length):.1f}  "
                      f"discards={avg(agent.eval_discards):.2f}")
                # Détail sur les parties PERDUES : c'est là que sont les ~18 % à expliquer.
                # Les deques sont alignées avec eval_wins (un append par partie aboutie).
                losses = [(f, c, l, d) for w, f, c, l, d in zip(
                            agent.eval_wins, agent.eval_finished, agent.eval_captured,
                            agent.eval_length, agent.eval_discards) if w == 0]
                if losses:
                    n = len(losses)
                    print(f"[EVAL] diag PERDUES (n={n}) : "
                          f"billes_finies={sum(x[0] for x in losses)/n:.2f}/4  "
                          f"captures_subies={sum(x[1] for x in losses)/n:.2f}  "
                          f"coups={sum(x[2] for x in losses)/n:.1f}  "
                          f"discards={sum(x[3] for x in losses)/n:.2f}")

    async def run_heuristic_eval_game():
        """Éval COHÉRENTE (le vrai juge) : 1 bot greedy (réseau courant) + 3 heuristiques."""
        bots = []
        net_bot = None
        for color in COLORS:
            is_net = (color == EVAL_LEARNER_COLOR)
            bot = EvalBot(
                agent, f"heur-bot-{color}",
                f"HEUR-{'NET' if is_net else 'BOT'}-{color.capitalize()}",
                mode=('net' if is_net else 'heuristic'),
                net=(agent.net if is_net else None),
                result_sink=(agent.heur_wins if is_net else None),
                collect_stats=is_net,
            )
            if is_net:
                net_bot = bot
            bots.append(bot)
        await _play_isolated_game(bots)
        if net_bot is not None and net_bot._over:
            agent.heur_finished.append(net_bot.finished)
            agent.heur_captured.append(net_bot.captured)
            agent.heur_length.append(net_bot.turns_played)
            agent.heur_discards.append(net_bot.discards)
        if agent.heur_wins:
            rate = sum(agent.heur_wins) / len(agent.heur_wins)
            if len(agent.heur_wins) >= 20:
                agent.save_best_heur(rate)
            print(f"[EVAL-H] vs heuristique : win_rate={rate:.3f} "
                  f"(n={len(agent.heur_wins)}, dernière={agent.heur_wins[-1]})")
            if agent.heur_finished:
                avg = lambda d: sum(d) / len(d)
                print(f"[EVAL-H] diag TOUTES (moy/partie n={len(agent.heur_finished)}) : "
                      f"billes_finies={avg(agent.heur_finished):.2f}/4  "
                      f"captures_subies={avg(agent.heur_captured):.2f}  "
                      f"coups={avg(agent.heur_length):.1f}  "
                      f"discards={avg(agent.heur_discards):.2f}")
                losses = [(f, c, l, d) for w, f, c, l, d in zip(
                            agent.heur_wins, agent.heur_finished, agent.heur_captured,
                            agent.heur_length, agent.heur_discards) if w == 0]
                if losses:
                    n = len(losses)
                    print(f"[EVAL-H] diag PERDUES (n={n}) : "
                          f"billes_finies={sum(x[0] for x in losses)/n:.2f}/4  "
                          f"captures_subies={sum(x[1] for x in losses)/n:.2f}  "
                          f"coups={sum(x[2] for x in losses)/n:.1f}  "
                          f"discards={sum(x[3] for x in losses)/n:.2f}")

    async def run_duel_game():
        """Duel isolé : 2 bots jouent le réseau COURANT (greedy), 2 jouent le ref figé,
        en diagonale. On enregistre UNE SEULE valeur par partie : 1 si le camp courant
        a gagné, 0 si le camp ref a gagné (tête-à-tête). Ainsi duel_wins vaut 0.50 à
        réseaux égaux (2 vs 2 sur 4 joueurs), et > 0.50 ⇔ le courant est supérieur."""
        if agent.ref_net is None:
            return
        bots = []
        for color in COLORS:
            is_current = (color in DUEL_CURRENT_COLORS)
            net = agent.net if is_current else agent.ref_net
            # Pas de result_sink : on lit le gagnant après la partie (1 valeur/partie).
            bots.append(EvalBot(
                agent, f"duel-bot-{color}",
                f"DUEL-{'CUR' if is_current else 'REF'}-{color.capitalize()}",
                mode='net', net=net, result_sink=None,
            ))
        await _play_isolated_game(bots)
        # Le gagnant est connu via le _over/color des bots : on prend le premier bot
        # Tous les bots voient le même gagnant (stocké dans _won_color). On en déduit
        # le camp vainqueur (courant vs ref).
        winner_color = next((b._won_color for b in bots if b._won_color is not None), None)
        if winner_color is None:
            return  # partie non aboutie (déconnexion) → ne pas polluer la métrique
        agent.duel_wins.append(1 if winner_color in DUEL_CURRENT_COLORS else 0)
        rate = sum(agent.duel_wins) / len(agent.duel_wins)
        agent.save_best(rate)   # capture le pic si record
        verdict = "SUPÉRIEUR ✓" if rate > DUEL_REPLACE_THRESHOLD else ""
        print(f"[DUEL] courant vs ref(0.82) : win_rate={rate:.3f} "
              f"(n={len(agent.duel_wins)}, 0.50=égalité) {verdict}")

    async def run_plan_duel_game():
        """Mesure : 2 bots PLANIFICATEUR de main vs 2 bots RÉSEAU réactif (en diagonale).
        win_rate = part des parties gagnées par le camp planificateur. 0.50 = égalité ;
        > 0.50 ⇒ planifier la main bat jouer au coup par coup → le levier qu'on cherche."""
        bots = []
        for color in COLORS:
            is_plan = (color in DUEL_CURRENT_COLORS)   # 2 plan / 2 réseau, en diagonale
            bots.append(EvalBot(
                agent, f"plan-bot-{color}",
                f"PLAN-{'PLN' if is_plan else 'NET'}-{color.capitalize()}",
                mode=('plan' if is_plan else 'net'),
                net=(None if is_plan else agent.net),
                result_sink=None,
                collect_stats=is_plan,   # cap anti-gridlock + longueur de partie
            ))
        await _play_isolated_game(bots)
        winner_color = next((b._won_color for b in bots if b._won_color is not None), None)
        if winner_color is None:
            return
        agent.plan_wins.append(1 if winner_color in DUEL_CURRENT_COLORS else 0)
        rate  = sum(agent.plan_wins) / len(agent.plan_wins)
        coups = next((b.turns_played for b in bots if b.collect_stats), 0)
        verdict = "PLAN > RÉSEAU ✓" if rate > 0.55 else ""
        print(f"[PLAN-DUEL] planificateur vs réseau : win_rate={rate:.3f} "
              f"(n={len(agent.plan_wins)}, 0.50=égalité, coups~{coups}) {verdict}")

    async def run_plan_vs_random_game():
        """Niveau ABSOLU du planificateur : 1 planificateur greedy vs 3 random. À comparer
        au 0.83 du réseau et au ~0.999 d'un humain → dit si on s'approche du niveau humain."""
        bots = []
        plan_bot = None
        for color in COLORS:
            is_plan = (color == EVAL_LEARNER_COLOR)
            bot = EvalBot(
                agent, f"planrnd-bot-{color}",
                f"PLANRND-{'PLN' if is_plan else 'RND'}-{color.capitalize()}",
                mode=('plan' if is_plan else 'random'),
                net=None,
                result_sink=(agent.plan_rnd_wins if is_plan else None),
                collect_stats=is_plan,
            )
            if is_plan:
                plan_bot = bot
            bots.append(bot)
        await _play_isolated_game(bots)
        if agent.plan_rnd_wins:
            rate  = sum(agent.plan_rnd_wins) / len(agent.plan_rnd_wins)
            coups = plan_bot.turns_played if plan_bot else 0
            print(f"[PLAN-RND] planificateur vs random : win_rate={rate:.3f} "
                  f"(n={len(agent.plan_rnd_wins)}, coups~{coups})  [humain≈0.999, réseau≈0.83]")

    async def update_loop():
        # L'éval heuristique 1-vs-3 est désactivée (gridlock structurel à 4 joueurs cohérents).
        # L'heuristique sert désormais d'adversaire d'ENTRAÎNEMENT (cf. _reset_episode), et le
        # benchmark cohérent reste le DUEL vs ref. run_heuristic_eval_game reste défini au cas
        # où on voudrait un jour un duel 2-vs-2 heuristique.
        last_eval = last_duel = last_plan = 0
        while True:
            await asyncio.sleep(2)
            if len(agent.buffer) >= UPDATE_EVERY:
                agent.update()
                if agent.n_updates - last_eval >= EVAL_EVERY:
                    last_eval = agent.n_updates
                    asyncio.create_task(run_eval_game())  # tâche de fond, non bloquant
                if agent.n_updates - last_duel >= DUEL_EVERY:
                    last_duel = agent.n_updates
                    asyncio.create_task(run_duel_game())
                if agent.n_updates - last_plan >= PLAN_DUEL_EVERY:
                    last_plan = agent.n_updates
                    asyncio.create_task(run_plan_duel_game())
                    asyncio.create_task(run_plan_vs_random_game())

    await asyncio.gather(
        *[run_bot(color) for color in COLORS],
        update_loop(),
    )


if __name__ == "__main__":
    asyncio.run(run_training())
