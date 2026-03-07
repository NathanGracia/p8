# Guide de déploiement — VPS

Architecture cible :

```
Internet (HTTPS)
      |
   Caddy (reverse proxy existant)
      |
  ┌───┴────────────┐
  │                │
p8.nathangracia.com   p8-api.nathangracia.com
      |                        |
  p8-streamlit:8501        p8-api:8000
      └────── http://p8-api:8000 (réseau Docker interne)
```

---

## 1. Cloner le repo sur le VPS

```bash
cd /opt
git clone <url-du-repo> p8
cd p8
```

---

## 2. Transférer les fichiers lourds

Le dataset et le modèle ne sont pas dans le repo Git (trop lourds).
À faire depuis ta machine Windows :

```bash
# Ouvrir WSL ou Git Bash

# Modèle (le meilleur, ~150MB)
rsync -avP \
  "/mnt/c/Users/nathan/Documents/OpenClassrooms/p8/models/unet_mobilenet_20260209_132821.h5" \
  user@vps-ip:/opt/p8/models/

# Dataset Cityscapes (~15GB — prendre un café)
rsync -avP --progress \
  /mnt/c/Users/nathan/Documents/OpenClassrooms/p8/data/ \
  user@vps-ip:/opt/p8/data/
```

Vérifier l'arborescence sur le VPS :
```bash
ls /opt/p8/models/    # → unet_mobilenet_20260209_132821.h5
ls /opt/p8/data/      # → gtFine/  leftImg8bit/
```

---

## 3. Trouver le nom du réseau Docker de Caddy

```bash
docker network ls
# Chercher le réseau auquel appartient le container caddy
docker inspect caddy | grep -A5 Networks
```

Le nom est probablement `caddy_default`, `caddy` ou autre.
Si ce n'est pas `caddy`, mettre à jour la section `networks` dans `docker-compose.yml` :

```yaml
networks:
  caddy:
    external: true
    name: <nom-réel-du-réseau>
```

---

## 4. Ajouter les subdomains dans le Caddyfile

Ouvrir le Caddyfile existant (probablement dans `/opt/caddy/` ou `/etc/caddy/`) :

```bash
# Trouver le Caddyfile
docker inspect caddy | grep Caddyfile
# ou
find /opt -name Caddyfile 2>/dev/null
```

Ajouter à la fin du Caddyfile :

```
p8.nathangracia.com {
    reverse_proxy p8-streamlit:8501
}

p8-api.nathangracia.com {
    reverse_proxy p8-api:8000
}
```

Recharger Caddy :
```bash
docker exec caddy caddy reload --config /etc/caddy/Caddyfile
```

---

## 5. Builder et lancer les containers

```bash
cd /opt/p8

# Premier build (long — ~10-15min à cause de tensorflow-cpu)
docker compose up -d --build

# Suivre les logs au démarrage
docker compose logs -f
```

Le message indiquant que tout est OK :
```
p8-api      | INFO:     Application startup complete.
p8-api      | INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 6. Vérifier

```bash
# Santé de l'API (depuis le VPS)
curl http://localhost:8000   # ne répondra pas (pas de port publié)
docker exec p8-streamlit curl -s http://p8-api:8000/  # teste le réseau interne

# Containers up
docker compose ps
```

Depuis le navigateur :
- App : https://p8.nathangracia.com
- API docs : https://p8-api.nathangracia.com/docs

---

## Commandes utiles

```bash
# Voir les logs
docker compose logs -f p8-api
docker compose logs -f p8-streamlit

# Redémarrer un service
docker compose restart p8-api

# Mettre à jour après un git pull
git pull
docker compose up -d --build

# Arrêter
docker compose down
```

---

## Espace disque

| Élément | Taille estimée |
|---------|---------------|
| Image Docker API (tensorflow-cpu) | ~2-3 GB |
| Image Docker Streamlit | ~300 MB |
| Modèle `.h5` | ~150 MB |
| Dataset Cityscapes | ~15 GB |

Le VPS a 32 GB libres — suffisant.

---

## Dépannage

**L'API ne démarre pas** :
```bash
docker compose logs p8-api
# Souvent : modèle introuvable → vérifier le chemin dans models/
```

**Streamlit affiche "API inaccessible"** :
```bash
# Vérifier que les deux containers sont sur le même réseau
docker inspect p8-api | grep Networks
docker inspect p8-streamlit | grep Networks
```

**Caddy 502 Bad Gateway** :
```bash
# Vérifier que les containers tournent
docker compose ps
# Vérifier le nom du réseau dans docker-compose.yml
docker network ls
```
