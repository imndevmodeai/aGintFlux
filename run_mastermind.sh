#!/bin/bash
# run_mastermind.sh - Script to run the MasterMind_AI system

# MasterMind_AI Run Script
# This script provides a convenient way to run the MasterMind_AI system with various options

# Function to display usage
show_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --interactive     Run in interactive mode"
    echo "  --agentic        Enable agentic capabilities"
    echo "  --visualize      Enable visualization"
    echo "  --query TEXT     Process a specific query"
    echo "  --simulate       Run Tesla waveform simulation"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --interactive"
    echo "  $0 --query \"What is quantum computing?\""
    echo "  $0 --interactive --agentic --visualize"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check if requirements are installed
if [ ! -f "requirements.installed" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    touch requirements.installed
fi

# Parse command line arguments
INTERACTIVE=false
AGENTIC=false
VISUALIZE=false
QUERY=""
SIMULATE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --interactive)
            INTERACTIVE=true
            shift
            ;;
        --agentic)
            AGENTIC=true
            shift
            ;;
        --visualize)
            VISUALIZE=true
            shift
            ;;
        --query)
            QUERY="$2"
            shift 2
            ;;
        --simulate)
            SIMULATE=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Build command
CMD="python main.py"

if [ "$INTERACTIVE" = true ]; then
    CMD="$CMD --interactive"
fi

if [ "$AGENTIC" = true ]; then
    CMD="$CMD --agentic"
fi

if [ "$VISUALIZE" = true ]; then
    CMD="$CMD --visualize"
fi

if [ ! -z "$QUERY" ]; then
    CMD="$CMD --query \"$QUERY\""
fi

if [ "$SIMULATE" = true ]; then
    CMD="$CMD --simulate"
fi

# Run the command
echo "Running MasterMind_AI with command: $CMD"
eval $CMD

# Deactivate virtual environment
deactivate 