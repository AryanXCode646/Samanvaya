#!/bin/bash
# Samanvaya — One-Command Web Portal Launcher (Linux/macOS)
# Usage: bash start.sh

set -e

CYAN='\033[0;36m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "  ${CYAN}${BOLD}🌙 Samanvaya (समान्वय) — Web Portal Launcher${NC}"
echo -e "  ${CYAN}═══════════════════════════════════════════${NC}"
echo -e "  ${GREEN}🚀  Streamlit Portal → http://localhost:8501${NC}"
echo ""

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Activate virtualenv if present
if [ -f "$ROOT/.venv/bin/activate" ]; then
  source "$ROOT/.venv/bin/activate"
elif [ -f "$ROOT/venv/bin/activate" ]; then
  source "$ROOT/venv/bin/activate"
fi

echo -e "${GREEN}Starting Samanvaya Streamlit Portal...${NC}"
streamlit run "$ROOT/app.py" --server.port 8501
