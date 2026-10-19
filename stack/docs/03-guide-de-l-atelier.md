# Guide de l'atelier : ce qui tourne sur votre VM

Chaque participant dispose d'une VM Ubuntu 22.04 de taille Standard_B12ms (12 vCPU, 48 Go de mémoire, sans GPU). Tous les services tournent dans des conteneurs Docker, dans le dossier /opt/ai-lab/stack.

## Services

- `web` : nginx, sert l'interface de chat sur le port 80 et relaie les appels /api vers le service RAG
- `rag-api` : application FastAPI qui découpe les documents, interroge la base vectorielle, construit le prompt et diffuse la réponse
- `chroma` : base vectorielle ChromaDB, accessible uniquement depuis le réseau Docker
- `ollama` : serveur de modèles, charge qwen2.5:7b pour la génération et nomic-embed-text pour les embeddings
- `ollama-init` : tâche ponctuelle qui télécharge les modèles au premier démarrage

## Commandes utiles en SSH

- `docker compose ps` dans /opt/ai-lab/stack : état des services
- `docker compose logs -f rag-api` : journaux de l'application RAG
- `curl localhost/api/health` : état des modèles et de l'indexation
- `sudo scripts/set-openllmetry.sh on <tenant> <token>` : activer OpenLLMetry
- `sudo scripts/set-openllmetry.sh off` : le désactiver
- `scripts/generate-traffic.sh 10` : envoyer dix questions pour produire des traces

## Exercices proposés

1. Poser la même question avec puis sans l'option « Chercher dans les documents » et comparer les réponses.
2. Ajouter un de vos documents dans la colonne de gauche et lui poser une question dessus.
3. Dans l'onglet Paramètres, faire varier la température, le top K et le top P, puis comparer plusieurs réponses à la même question.
4. Modifier le message système (par exemple imposer des réponses en une phrase) et observer l'effet.
5. Activer OpenLLMetry, envoyer quelques questions, puis retrouver dans Dynatrace les traces : la durée de la recherche et celle de la génération, et le prompt complet envoyé au modèle.
