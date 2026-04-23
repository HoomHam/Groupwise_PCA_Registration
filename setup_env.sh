#!/bin/bash
set -e

echo "=== Setting up environment for Groupwise PCA Registration ==="

# Install Python 3.11 via Homebrew if not already present
if ! brew list python@3.11 &>/dev/null; then
    echo "Installing Python 3.11..."
    brew install python@3.11
else
    echo "Python 3.11 already installed."
fi

PYTHON=/opt/homebrew/bin/python3.11

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv .venv
else
    echo "Virtual environment already exists."
fi

source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing packages (antspyx may take a few minutes)..."
pip install -r requirements.txt

echo ""
echo "=== Done! ==="
echo "To activate the environment in future sessions, run:"
echo "    source .venv/bin/activate"
