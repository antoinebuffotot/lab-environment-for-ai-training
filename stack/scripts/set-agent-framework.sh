#!/bin/bash
# Choose the framework of the AI agent on a running VM (same web page, same tools, same events):
#   sudo ./set-agent-framework.sh python      hand-written agent loop (default)
#   sudo ./set-agent-framework.sh langchain   LangChain / LangGraph agent
set -e
STACK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$STACK_DIR/.env"

case "${1:-}" in
  python|langchain) value="$1" ;;
  *) echo "usage: $0 python | langchain"; exit 1 ;;
esac

{ grep -v "^AGENT_FRAMEWORK=" "$ENV_FILE" || true; } > "$ENV_FILE.tmp"
echo "AGENT_FRAMEWORK=$value" >> "$ENV_FILE.tmp"
mv "$ENV_FILE.tmp" "$ENV_FILE"
chmod 600 "$ENV_FILE"

cd "$STACK_DIR"
docker compose up -d --force-recreate agent-api
echo "Agent framework : $value"
curl -fs http://localhost/agent/api/health | grep -o '"framework":"[^"]*"' || true
