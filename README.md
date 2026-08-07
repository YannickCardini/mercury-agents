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

**Agent joué en prod** : le **planificateur de main** (`plan_hand.py`) — recherche en faisceau de
la meilleure séquence de cartes sur la main courante.

**Cycle de vie d'une partie :**
1. Le backend appelle `POST /dispatch` avec l'ID de partie
2. Un bot disponible ouvre un WebSocket vers le serveur Mercury
3. À chaque tour du bot :
   - Le planificateur choisit une action parmi les 501 actions légales calculées par
     `mercury_legal_moves.py`
   - Un délai simulé (0.5–8.5 secondes) est ajouté pour imiter un joueur humain
4. Fin de partie → bot libéré et disponible à nouveau

---

### `plan_hand.py` — Planificateur de main

Recherche en faisceau (beam search) sur les séquences de cartes jouables, replanifiée à chaque
tour (horizon glissant). C'est l'agent de production par défaut — pur Python, aucun réseau de
neurones impliqué.

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

### Autres modules

| Fichier                | Rôle                                                                                           |
|-------------------------|-------------------------------------------------------------------------------------------------|
| `game_state_utils.py`  | Helpers partagés sur le gameState (détection de victoire, billes par couleur, invincibilité).  |
| `reward.py`             | `marble_progress()` — progrès normalisé d'une bille, utilisé par le scoring du planificateur.  |

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
|------------------------|-------------------------------------------|--------------------------------|
| `MERCURY_API_URL`      | URL HTTP du backend                      | `https://ton-backend.azurewebsites.net` |
| `MERCURY_WS_URL`       | URL WebSocket du backend                 | `wss://ton-backend.azurewebsites.net`   |
| `MERCURY_BOT_SECRET`   | Secret partagé avec le backend           | (valeur secrète)               |

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
