# Image de base
FROM python:3.10-slim

# Dossier de travail
WORKDIR /app

# Copier le code du déploiement
COPY deployment/ /app/

# Copier le modèle
COPY dl-model/eye_state_model.keras /app/dl-model/eye_state_model.keras

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port FastAPI
# EXPOSE 8000

# Démarrer l’API 
CMD ["bash", "start.sh"]