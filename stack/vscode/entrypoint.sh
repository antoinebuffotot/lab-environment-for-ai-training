#!/bin/sh
# Configure the Continue chat for the local Ollama models, then start code-server on /workspace/mcp.
set -e
OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
LLM_MODEL="${LLM_MODEL:-qwen2.5:7b}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"

mkdir -p /root/.continue /root/.local/share/code-server/User /workspace/mcp

cat > /root/.continue/config.yaml <<YAML
name: Local Ollama
version: 1.0.0
schema: v1
models:
  - name: ${LLM_MODEL} (local)
    provider: ollama
    model: ${LLM_MODEL}
    apiBase: ${OLLAMA_HOST}
    roles:
      - chat
      - edit
      - apply
  - name: ${EMBED_MODEL} (local)
    provider: ollama
    model: ${EMBED_MODEL}
    apiBase: ${OLLAMA_HOST}
    roles:
      - embed
YAML

SETTINGS=/root/.local/share/code-server/User/settings.json
[ -f "$SETTINGS" ] || cat > "$SETTINGS" <<JSON
{
  "continue.telemetryEnabled": false,
  "workbench.startupEditor": "none",
  "security.workspace.trust.enabled": false
}
JSON

exec /usr/bin/entrypoint.sh --bind-addr 0.0.0.0:8080 --auth password /workspace/mcp
