# OpenLLMetry et observabilité des applications LLM

OpenLLMetry est un projet open source (Traceloop) qui ajoute à OpenTelemetry l'instrumentation des applications d'IA générative. Il s'installe avec le SDK Traceloop et produit des traces, des métriques et des logs au format OpenTelemetry (OTLP).

## Ce que le SDK sait faire

Par défaut, le SDK détecte les bibliothèques d'IA présentes dans l'application (clients de modèles, bases vectorielles, frameworks d'orchestration) et transforme chaque appel en span, avec le nom du modèle, le nombre de jetons, la durée et, si on l'autorise, le texte des prompts et des réponses.

## Ce que trace cet atelier

L'atelier utilise une configuration volontairement réduite : seules deux choses sont envoyées.

- **l'activité courante** : un workflow `rag.chat` et ses deux étapes, `rag.retrieve` (embedding de la question puis recherche dans ChromaDB) et `rag.generate` (appel au modèle qwen2.5:7b). La durée de chaque span montre où le temps passe ;
- **le prompt** : l'ensemble des messages envoyés au modèle (message système, historique, passages retrouvés et question), portés par le span `rag.generate` dans les attributs `gen_ai.prompt.N.role` et `gen_ai.prompt.N.content`, avec le modèle et les réglages de la requête (température, top K, top P).

Ne sont pas envoyés : les instrumentations automatiques (Ollama, ChromaDB, HTTP), le texte des réponses, les métriques et les logs.

## Dans cet atelier

L'application RAG est écrite en Python (FastAPI). Quand l'option OpenLLMetry est activée au déploiement, le SDK Traceloop est initialisé au démarrage et envoie les traces vers l'endpoint OTLP du tenant Dynatrace. Quand l'option est désactivée, l'application fonctionne de la même façon mais n'émet aucune télémétrie.

Changer la température, le top K, le top P ou le message système dans l'interface modifie le prompt et les réglages visibles dans la trace de la question suivante.
