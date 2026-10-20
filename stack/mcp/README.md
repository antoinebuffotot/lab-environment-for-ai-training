# Exercice : connecter l'agent à un serveur MCP Dynatrace

Ce dossier (`/workspace/mcp`) est le dossier par défaut du VS Code web (`http://<vm>:8080/`).
L'agent (`http://<vm>/agent.html`) lit le fichier **`mcp.json`** de ce dossier : à vous de le créer.

## 1. Créer un token plateforme dans Dynatrace

1. **Access Tokens** > **Generate new token**.
2. Scopes : `mcp-gateway:servers:invoke` et `mcp-gateway:servers:read`, plus les droits de lecture voulus
   (par exemple `storage:logs:read`, `storage:entities:read`, `storage:metrics:read`, `storage:events:read`,
   `storage:spans:read`, `document:documents:read`, `davis:analyzers:read`, `davis:analyzers:execute`).
3. Copiez le token (`dt0s16.…`), il n'est affiché qu'une fois.

Votre utilisateur doit lui aussi avoir ces permissions : le token ne peut pas faire plus que vous.

## 2. Créer `mcp.json`

Copiez `mcp.json.example` en `mcp.json`, puis remplacez l'URL de votre environnement et le token.
URL du serveur : `https://<environnement>.apps.dynatrace.com/platform-reserved/mcp-gateway/v0.1/servers/dynatrace-mcp/mcp`

- `url` : serveur MCP **distant** (HTTP). Les serveurs locaux (stdio) ne sont pas gérés.
- `headers` : en-têtes envoyés au serveur. `${NOM}` est remplacé par la variable d'environnement `NOM`.
- `tools` (facultatif) : liste des outils autorisés. À utiliser : le modèle choisit mieux avec quelques outils, et cela évite les outils qui écrivent dans Dynatrace.

Le fichier contient un secret : il reste sur votre VM (ignoré par git), ne le partagez pas.

## 3. Vérifier

Ouvrez la page de l'agent, onglet **Outils** : une ligne « Serveurs MCP » indique l'état de la connexion
(erreur 401 : token invalide ; 403 : droits manquants, vérifiez `mcp-gateway:servers:invoke`).
Aucun redémarrage n'est nécessaire (rechargement automatique dans la minute).
Puis posez une question, par exemple : « Quels problèmes sont ouverts dans Dynatrace ? ».
Avec OpenLLMetry actif, chaque appel MCP apparaît dans la trace comme un span d'outil.
