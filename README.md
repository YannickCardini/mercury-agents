# Mercury Agent Service

Service de bots d'IA pour le jeu Mercury. Expose une API HTTP pilotée par webhook depuis le backend.

## Endpoints

| Méthode | Route       | Description                        |
|---------|-------------|------------------------------------|
| POST    | /dispatch   | Spawne un bot (header X-Bot-Secret)|
| GET     | /health     | Healthcheck Azure                  |
| GET     | /status     | Bots actifs / disponibles          |

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
