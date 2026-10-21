# code-server (VS Code in the browser) + the Continue extension (AI chat) wired to the local Ollama model
FROM codercom/code-server:latest
USER root
RUN code-server --install-extension Continue.continue
# git (to fetch Dynatrace skills) and dtctl (Dynatrace CLI, configured by the participants in the terminal).
# If the dtctl download fails the image still builds.
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && (DTCTL_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/dynatrace-oss/dtctl/main/install.sh | sh \
        || echo "WARNING: dtctl installation failed")
# Node.js 20 and dt-evals (LLM-as-a-judge evaluations of the GenAI traces, run by the participants in the terminal).
# If the download fails the image still builds.
RUN (curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
      && apt-get install -y --no-install-recommends nodejs \
      && npm install -g @dynatrace-oss/dt-evals) \
    || echo "WARNING: Node.js / dt-evals installation failed"; \
    rm -rf /var/lib/apt/lists/*
COPY entrypoint.sh /usr/local/bin/vscode-entrypoint.sh
RUN chmod +x /usr/local/bin/vscode-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/vscode-entrypoint.sh"]
