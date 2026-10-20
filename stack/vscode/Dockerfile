# code-server (VS Code in the browser) + the Continue extension (AI chat) wired to the local Ollama model
FROM codercom/code-server:latest
USER root
RUN code-server --install-extension Continue.continue
COPY entrypoint.sh /usr/local/bin/vscode-entrypoint.sh
RUN chmod +x /usr/local/bin/vscode-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/vscode-entrypoint.sh"]
