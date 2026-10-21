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
1. Le backend appelle `POST /dispatch` avec le body `{"gameMode": "2v2", "avoidUserIds": [...]}`.
   Le moment de chaque appel suit le script d'arrivée de la file, tiré côté backend
   (`backend/src/session/matchmaking-script.ts` de Mercury).
2. Le service choisit une identité libre dans le pool, de préférence hors `avoidUserIds`
   (les bots déjà croisés récemment par les humains de la file ; le userId d'un bot est son botId).
   Il répond 503 si toutes les identités sont occupées.
   - Le pool est relu au plus une fois par minute via `GET /api/auth/bots` (`pool.py`).
   - Repli : la dernière liste connue, à défaut les ids 1 à 27.
3. Le bot s'authentifie (`POST /api/auth/bot`), ouvre un WebSocket et rejoint la file comme un humain.
   Si le serveur n'a plus besoin de lui (partie déjà lancée, file annulée), il ferme la
   socket (close 1000) : le bot loggue « renvoyé de la file » et libère son identité.
4. À chaque tour du bot :
   - Le planificateur choisit une action parmi les 501 actions légales calculées par
     `mercury_legal_moves.py`
   - Un temps de réflexion humain est simulé (`think_delay`) :
     - environ 1 s pour un coup forcé ;
     - sinon une médiane qui croît avec le nombre de coups possibles (2 coups ≈ 1,9 s, 64 ≈ 5,4 s),
       modulée par le rythme du bot ;
     - un peu plus long au premier tour, avec de rares hésitations ;
     - borné par `MERCURY_THINK_MIN` / `MERCURY_THINK_MAX` (0,5 s / 14 s par défaut).
5. Fin de partie → bot libéré et disponible à nouveau

**Personas** (`persona.py`) : chaque identité a un caractère fixe, le même à chaque partie,
stocké sur son compte en base (`botPersona`) et transmis par le pool.
- Bavardage : discret, réservé, normal ou bavard. Il multiplie les probabilités de réaction.
- Emojis payants « possédés » : aucun, quelques-uns ou beaucoup. Le bot ne pioche que dans
  les gratuits et les siens ; le serveur laisse passer les payants d'un bot.
- Un éventuel emoji fétiche.
- Un rythme de jeu.

`personas.json` garde les personas des 27 premiers bots : c'est la source de leur import en base,
et le repli tant que le backend n'en fournit pas.

### Ajouter un bot (sans redéploiement)

Depuis `backend/` de Mercury : le script crée le compte (id au format Google, avatar 512×512 sur
le Blob, persona) et le service le prend en compte dans la minute.

```bash
npx tsx scripts/add-bot.ts --name "Prénom Nom" --photo chemin/photo.jpg            # dry-run
npx tsx scripts/add-bot.ts --name "Prénom Nom" --photo chemin/photo.jpg --apply    # écrit
```

Traits optionnels : `--style silent|quiet|normal|chatty`, `--owned 😎,⏰` (ou `none`),
`--favorite 😎` (ou `none`), `--pace 0.75..1.35`. Les autres sont tirés pour équilibrer le pool.

---

### `plan_hand.py` — Planificateur de main

Recherche en faisceau (beam search) sur les séquences de cartes jouables, replanifiée à chaque
tour (horizon glissant). C'est l'agent de production par défaut — pur Python, aucun réseau de
neurones impliqué.

---

### `mercury_legal_moves.py` — Moteur de règles

Calcule le masque des **501 actions légales** à partir de la main et des positions actuelles.

**Espace d'actions :** 5 emplacements de carte × 100 sous-actions + 1 défausse = 501 actions

| Carte       | Comportement                                                |
|-------------|--------------------------------------------------------------|
| K, A, Joker | Faire entrer une marble depuis la maison                   |
| K           | Avancer de 13 cases (si la marble est déjà en jeu)          |
| A           | Avancer de 1 case                                            |
| Joker       | Avancer de 18 cases                                          |
| 2–Q         | Avancer de N cases                                           |
| 4           | Reculer de 4 cases                                           |
| 7           | Avancer de 7 (avec option de diviser entre deux marbles)     |
| J           | Échanger une marble propre avec une marble adverse           |

---

### Autres modules

| Fichier                | Rôle                                                                                           |
|-------------------------|-------------------------------------------------------------------------------------------------|
| `persona.py`           | Personas figées par identité (emojis possédés, bavardage, rythme), lues dans `personas.json`.  |
| `pool.py`              | Pool des identités relu en base (`GET /api/auth/bots`), avec repli sur la dernière liste connue. |
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

## Tests

```bash
python -m unittest discover tests          # tests unitaires (règles, planificateur, boucle de tour)
python tests/diff_vs_ts_validator.py 3000  # port Python vs move-validator.ts du jeu (~/Documents/mercury)
```

Le test différentiel exécute le vrai validateur TypeScript (fonctions pures, sans backend
ni base) sur des positions aléatoires : chaque coup que le bot peut envoyer doit être
accepté par le serveur, et chaque coup légal côté serveur doit exister dans le masque.
À relancer après toute évolution des règles côté jeu — un écart se traduit en partie par
un `actionRejected`.
