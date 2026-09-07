#!/usr/bin/env bash
# ==============================================================================
# Samanvaya (समान्वय) - One-Command Automated Installer & Environment Setup
# ISRO Chandrayaan-2 Optical Planetary Registration (SIH PS 26166)
# ==============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "  ███████╗ █████╗ ███╗   ███╗ █████╗ ███╗   ██╗██╗   ██╗ █████╗ ██╗   ██╗ █████╗ "
echo "  ██╔════╝██╔══██╗████╗ ████║██╔══██╗████╗  ██║██║   ██║██╔══██╗╚██╗ ██╔╝██╔══██╗"
echo "  ███████╗███████║██╔████╔██║███████║██╔██╗ ██║██║   ██║███████║ ╚████╔╝ ███████║"
echo "  ╚════██║██╔══██║██║╚██╔╝██║██╔══██║██║╚██╗██║╚██╗ ██╔╝██╔══██║  ╚██╔╝  ██╔══██║"
echo "  ███████║██║  ██║██║ ╚═╝ ██║██║  ██║██║ ╚████║ ╚████╔╝ ██║  ██║   ██║   ██║  ██║"
echo "  ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝  ╚═══╝  ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝"
echo "  ISRO Chandrayaan-2 Planetary Image Registration Framework (SIH PS 26166)"
echo -e "${NC}"

# Check Python 3 version
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}❌ Error: python3 is not installed. Please install Python 3.9+ to continue.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✔ Detected Python ${PYTHON_VERSION}${NC}"

# Create or reuse virtual environment
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${BLUE}📦 Creating virtual environment in ./${VENV_DIR}...${NC}"
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo -e "${BLUE}⚡ Activating virtual environment...${NC}"
source "${VENV_DIR}/bin/activate"

# Upgrade pip and install core dependencies
echo -e "${BLUE}📥 Installing dependencies from requirements.txt...${NC}"
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Install Samanvaya package in editable mode
echo -e "${BLUE}🔧 Registering 'samanvaya' CLI...${NC}"
pip install -e .

# Run test suite to verify installation
echo -e "${BLUE}🧪 Running verification test suite...${NC}"
pytest tests/ -q

echo ""
echo -e "${GREEN}==============================================================================${NC}"
echo -e "${GREEN}🎉 Samanvaya has been successfully installed and verified!${NC}"
echo -e "${GREEN}==============================================================================${NC}"
echo ""
echo -e "To launch the ${YELLOW}Interactive Web Portal${NC}:"
echo -e "    ${BLUE}source venv/bin/activate && streamlit run app.py${NC}   (or: ${BLUE}make run${NC})"
echo ""
echo -e "To run ${YELLOW}Headless GeoTIFF Registration${NC}:"
echo -e "    ${BLUE}samanvaya align -s source.tif -r reference.tif -o output/${NC}"
echo ""
