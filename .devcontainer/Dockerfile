FROM mcr.microsoft.com/devcontainers/base:noble

ENV DEBIAN_FRONTEND=noninteractive

# Update system + deps
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-full \
    git \
    curl \
    unzip \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Install uv system-wide
# -------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

# -------------------------
# Install Claude Code
# -------------------------
RUN sudo -u vscode bash -c "curl -fsSL https://claude.ai/install.sh | bash"
