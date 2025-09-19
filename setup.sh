#!/usr/bin/env bash
set -e  # exit on first error
set -x  # echo commands as they run

# Update package lists and install basic tools
apt update
apt install -y nano tmux curl git python3-pip

# Install uv (if not already installed)
# Using the official installer script
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session (and future shells)
    export PATH="$HOME/.cargo/bin:$PATH"
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
fi

source $HOME/.local/bin/env

# Make sure uv is working
uv --version

# Sync dependencies
# 1. First sync while skipping flash-attn (to avoid the long build)
uv sync --no-install-package flash-attn

# 2. Install everything else (flash-attn will be included now)
uv sync
