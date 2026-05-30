# Mercury Agent Service

Service de bots d'IA pour le jeu Mercury. Expose une API HTTP pilotée par webhook depuis le backend.

## Endpoints

| Méthode | Route       | Description                        |
|---------|-------------|------------------------------------|
| POST    | /dispatch   | Spawne un bot (header X-Bot-Secret)|
| GET     | /health     | Healthcheck Azure                  |
| GET     | /status     | Bots actifs / disponibles          |

---

## Architecture technique

### `main.py` — Serveur d'inférence (production)

Serveur HTTP/WebSocket déployé sur Azure. Gère 27 identités de bots disponibles en parallèle pour remplir le matchmaking.

**Cycle de vie d'une partie :**
1. Le backend appelle `POST /dispatch` avec l'ID de partie
2. Un bot disponible ouvre un WebSocket vers le serveur Mercury
3. À chaque tour du bot :
   - L'état de jeu (54 features) est encodé en vecteur numérique
   - Le réseau de neurones calcule la distribution de probabilité sur les 501 actions légales
   - Le bot joue l'action avec la **probabilité maximale (greedy)**
   - Un délai simulé (1–8.5 secondes) est ajouté pour imiter un joueur humain
4. Fin de partie → bot libéré et disponible à nouveau

> En production, le modèle est **figé** : aucun apprentissage n'a lieu pendant les parties. Pour améliorer le modèle, il faut entraîner via `bot_rl.py` puis redéployer.

---

### `bot_rl.py` — Entraînement par reinforcement learning (offline)

Script d'entraînement à lancer **localement** avec un serveur Mercury local sur `localhost:8080`. Lance 4 bots en self-play simultané pour entraîner le réseau par l'algorithme **PPO (Proximal Policy Optimization)**.

**Organisation des bots :**
- **1 bot learner (rouge)** : accumule de l'expérience et met à jour le réseau
- **3 bots adversaires** : piochent dans un pool de snapshots passés pour diversifier les opposants

**Boucle d'entraînement :**
1. Les 4 bots jouent des parties complètes en parallèle
2. Le learner collecte ses transitions : `(état, action, log_prob, valeur, reward)`
3. Toutes les **512 transitions** collectées → une mise à jour PPO
4. Tous les **10 updates** → snapshot sauvegardé dans `models/snapshot_XXXXXX.pt`

**Hyperparamètres PPO :**

| Paramètre        | Valeur | Rôle                                      |
|------------------|--------|-------------------------------------------|
| `GAMMA`          | 0.99   | Facteur de discount (importance du futur) |
| `GAE_LAMBDA`     | 0.95   | Lissage de l'avantage (GAE)               |
| `CLIP_EPS`       | 0.2    | Plafond du ratio de politique (stabilité) |
| `LR`             | 3e-4   | Learning rate Adam                        |
| `ENTROPY_COEF`   | 0.01   | Bonus d'entropie (encourage l'exploration)|
| `VALUE_COEF`     | 0.5    | Poids de la loss du critique              |
| `UPDATE_EVERY`   | 512   | Transitions accumulées avant update       |
| `PPO_EPOCHS`     | 4      | Epochs de gradient par update             |
| `BATCH_SIZE`     | 64     | Taille des mini-batches                   |
| `POOL_SIZE`      | 5      | Nombre de snapshots adversaires conservés |
| `SNAPSHOT_EVERY` | 10     | Fréquence de sauvegarde des snapshots     |

**Lancer l'entraînement :**
```bash
python -m venv torch_env
source torch_env/bin/activate
pip install -r requirements.txt

export MERCURY_API_URL=http://localhost:8080
export MERCURY_WS_URL=ws://localhost:8080
export MERCURY_BOT_SECRET=change-me-before-prod

python bot_rl.py
# Logs : "update 1 | p_loss=... v_loss=... ent=..."
# Snapshots : models/snapshot_000010.pt, snapshot_000020.pt, ...
```

Après entraînement, copier le meilleur snapshot en `model.pt` et pousser sur `main` pour redéployer.

---

### `model.py` — Réseau de neurones

**Encodage de l'état (54 dimensions) :**

Le plateau est une grille dont seules **68 cases** sont actives pour le déplacement, plus **8 cases par couleur** (4 HOME + 4 ARRIVAL). La normalisation des positions brutes utilise donc `/68.0`.

| Features          | Dimensions | Contenu                                                    |
|-------------------|------------|------------------------------------------------------------|
| Positions propres | [0:4]      | 4 marbles du bot (normalisé /68)                           |
| Positions adv.    | [4:16]     | 12 marbles adversaires (3 couleurs × 4, normalisé /68)     |
| Progression own   | [16:20]    | Progrès normalisé (0→1) de mes 4 marbles vers l'arrivée    |
| Progression adv.  | [20:32]    | Progrès normalisé des 12 marbles adverses (3 couleurs × 4) |
| Arrivées          | [32:36]    | Marbles arrivées par couleur (4 couleurs, normalisé /4)    |
| Tour actuel       | [36:40]    | One-hot encoding (4 couleurs)                              |
| Cartes en main    | [40:53]    | Présence de chaque rang (2..A), one-hot 13 dims             |
| canDiscard        | [53]       | Flag booléen                                               |

**Architecture du réseau :**
```
Entrée (54) → FC(256, ReLU) → FC(256, ReLU) → Policy head (501 actions)
                                             → Value head (1 scalaire)
```
- Les actions illégales sont masquées (`-inf` dans le softmax) avant l'échantillonnage
- Total : ~209 000 paramètres

---

### `reward.py` — Système de récompenses

Récompenses **denses** calculées à chaque transition (pas seulement en fin de partie) pour accélérer l'apprentissage.

| Récompense                  | Coefficient | Condition                                                  |
|-----------------------------|-------------|------------------------------------------------------------|
| Progression de ses marbles    | `+0.6`      | Marble qui avance sur le plateau (normalisé 0→1)           |
| Entrée en jeu d'une marble    | `+1.0`      | Marble quittant la maison pour la case départ              |
| **Arrivée d'une marble**      | `+3.0`      | Marble propre entrant en zone d'arrivée (une fois)         |
| Recul d'une marble adverse    | `+1.0`      | Adversaire capturé, swappé ou reculé par carte 4           |
| Arrivée d'un adversaire       | `-0.15`     | Adverse entrant en zone d'arrivée (scalé par déjà arrivés) |
| Menace (proximité)            | `-0.05`     | Marble propre à moins de 6 cases d'un adversaire           |
| **Victoire**                  | `+10.0`     | Terminale                                                  |
| **Défaite**                   | `-4.0`      | Terminale                                                  |

> **Calibration :** le reward de victoire (`+10.0`) reste le signal dominant. L'**arrivée d'une
> bille** (`+3.0`) est la 2ème plus grosse récompense : faire rentrer ses billes en zone d'arrivée
> est l'une des étapes décisives du jeu (4 billes arrivées = victoire), donc bien au-dessus de
> l'entrée en jeu et du recul adverse (`+1.0`). Le bonus est déclenché une seule fois, au passage
> plateau → zone d'arrivée (l'avancement dans les slots reste couvert par le progrès dense).
> L'entrée en jeu (`+1.0`) est intentionnellement bien supérieure à l'avancement d'une case
> (~`+0.007`), pour éviter que l'agent utilise l'As pour avancer plutôt que pour faire entrer une bille.

---

### `mercury_legal_moves.py` — Moteur de règles

Calcule le masque des **501 actions légales** à partir de la main et des positions actuelles.

**Espace d'actions :** 5 emplacements de carte × 100 sous-actions + 1 défausse = 501 actions

| Carte | Comportement                                                     |
|-------|------------------------------------------------------------------|
| K, A  | Faire entrer une marble depuis la maison                         |
| A     | Avancer de 1 case                                                |
| 2–Q   | Avancer de N cases                                               |
| 4     | Reculer de 4 cases                                               |
| 7     | Avancer de 7 (avec option de diviser entre deux marbles)         |
| J     | Échanger une marble propre avec une marble adverse               |

---

## Déploiement Azure App Service

### Secrets GitHub à configurer

Dans **Settings → Secrets and variables → Actions** du repo GitHub :

| Secret                        | Valeur                                              |
|-------------------------------|-----------------------------------------------------|
| `AZURE_WEBAPP_NAME`           | Nom de ton App Service (ex: `mercury-agent`)        |
| `AZURE_WEBAPP_PUBLISH_PROFILE`| Contenu du fichier publish profile (portail Azure)  |

### Variables d'environnement Azure (App Settings)

Dans **App Service → Configuration → Application settings** :

| Variable               | Description                              | Exemple                        |
|------------------------|------------------------------------------|--------------------------------|
| `MERCURY_API_URL`      | URL HTTP du backend                      | `https://ton-backend.azurewebsites.net` |
| `MERCURY_WS_URL`       | URL WebSocket du backend                 | `wss://ton-backend.azurewebsites.net`   |
| `MERCURY_BOT_SECRET`   | Secret partagé avec le backend           | (valeur secrète)               |
| `MERCURY_MODEL_PATH`   | Chemin vers le modèle (optionnel)        | `model.pt`                     |

> Le port est géré automatiquement via la variable `PORT` injectée par Azure.

### Récupérer le publish profile

1. Portail Azure → App Service → **Get publish profile**
2. Copier le contenu du fichier téléchargé dans le secret `AZURE_WEBAPP_PUBLISH_PROFILE`

## Lancement local

```bash
python -m venv torch_env
source torch_env/bin/activate
pip install -r requirements.txt
MERCURY_BOT_SECRET=secret python main.py
```