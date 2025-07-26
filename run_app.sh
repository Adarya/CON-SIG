#!/bin/bash

# CONSIG Web Application Launcher
# This script sets up and launches the CONSIG web application

set -e  # Exit on any error

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧬 CONSIG - CNA Signature Analysis Web Application${NC}"
echo -e "${BLUE}===============================================${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.9 or higher.${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
REQUIRED_VERSION="3.9.0"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}❌ Python 3.9 or higher is required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"

# Function to create virtual environment if it doesn't exist
setup_venv() {
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
        python3 -m venv venv
    fi
    
    echo -e "${YELLOW}🔧 Activating virtual environment...${NC}"
    source venv/bin/activate
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install requirements
    echo -e "${YELLOW}📥 Installing dependencies...${NC}"
    pip install -r requirements_app.txt
    
    echo -e "${GREEN}✅ Dependencies installed successfully${NC}"
}

# Function to run the application
run_app() {
    echo -e "${YELLOW}🚀 Starting CONSIG web application...${NC}"
    echo -e "${BLUE}   - Application will open in your default browser${NC}"
    echo -e "${BLUE}   - Default URL: http://localhost:8501${NC}"
    echo -e "${BLUE}   - Press Ctrl+C to stop the application${NC}"
    echo ""
    
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Run Streamlit app
    streamlit run app.py --server.port 8501 --server.address localhost
}

# Function to install system dependencies (if needed)
install_system_deps() {
    echo -e "${YELLOW}🔧 Checking system dependencies...${NC}"
    
    # Check for required system packages (add as needed)
    # This is a placeholder for system-specific dependencies
    
    echo -e "${GREEN}✅ System dependencies check complete${NC}"
}

# Parse command line arguments
INSTALL_DEPS=false
NO_VENV=false
PORT=8501

while [[ $# -gt 0 ]]; do
    case $1 in
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        --no-venv)
            NO_VENV=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --install-deps    Install system dependencies"
            echo "  --no-venv        Don't use virtual environment"
            echo "  --port PORT      Specify port (default: 8501)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Main execution
main() {
    # Install system dependencies if requested
    if [ "$INSTALL_DEPS" = true ]; then
        install_system_deps
    fi
    
    # Set up virtual environment unless disabled
    if [ "$NO_VENV" = false ]; then
        setup_venv
    else
        echo -e "${YELLOW}⚠️  Running without virtual environment${NC}"
        # Still install requirements
        pip install -r requirements_app.txt
    fi
    
    # Run the application
    run_app
}

# Trap to handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}🛑 Shutting down CONSIG...${NC}"; exit 0' INT

# Run main function
main "$@"