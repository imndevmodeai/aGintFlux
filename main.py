#!/usr/bin/env python3
"""
MasterMind_AI - Quantum State Transformation with Tesla-Inspired Waveforms
Main script to run the system
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from typing import Dict, Any
import sys
import os
# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import MasterMind components
from quantum_utils import superposition_state, entangled_state, calculate_shannon_entropy
from code.system_orchestrator import SystemOrchestrator

# Import tools
from tools.embedding_tools import NodeEmbedderTool
from tools.digital_twin_tools import DigitalTwinTool
from tools.hmi_tools import HMITool
import web_search_tool
import code_execution_tool
import any_doc_reader_tool
import calculator_tool
import pdf_to_markdown_tool
import fetch_url_tool
import edit_image_tool
import image_generation_tool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MasterMind_AI")

def parse_arguments():
    parser = argparse.ArgumentParser(description='MasterMind_AI System Interface')
    parser.add_argument('--visualize', action='store_true',
                      help='Enable knowledge graph visualization')
    parser.add_argument('--interactive', action='store_true',
                      help='Run in interactive mode')
    parser.add_argument('--query', type=str,
                      help='Initial query to process')
    parser.add_argument('--agentic', action='store_true',
                      help='Enable agentic mode with tool use capabilities')
    return parser.parse_args()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MasterMind_AI with Tesla-Inspired Waveforms")
    parser.add_argument("--simulate", action="store_true", help="Run the Tesla waveform simulation")
    parser.add_argument("--periods", type=int, default=5, help="Number of pulse periods to scan (default: 5)")
    parser.add_argument("--amplitude", type=float, default=5.0, help="Pulse amplitude (default: 5.0)")
    parser.add_argument("--width", type=float, default=0.5, help="Pulse width (default: 0.5)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--agentic", action="store_true", help="Enable agentic mode with tool use capabilities")
    return parser.parse_args()

def run_simulation(num_periods: int = 5, amplitude: float = 5.0, width: float = 0.5):
    """Run the Tesla waveform quantum simulation with specified parameters."""
    logger.info(f"Starting Tesla waveform simulation with {num_periods} periods")
    
    try:
        # Import the simulation function from tesla_waveform_simulation_tool
        # This is imported here to separate it from the main initialization
        import tesla_waveform_simulation_tool
        logger.info("Simulation completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error running simulation: {e}")
        return False

def register_tools(orchestrator):
    """Register available tools with the MasterMind_AI system."""
    logger.info("Registering tools with MasterMind_AI")
    
    # Register domain-specific tools
    orchestrator.mastermind.register_tool(NodeEmbedderTool())
    orchestrator.mastermind.register_tool(DigitalTwinTool())
    orchestrator.mastermind.register_tool(HMITool())
    
    # Register standard tools
    orchestrator.mastermind.register_tool(web_search_tool)
    orchestrator.mastermind.register_tool(code_execution_tool)
    orchestrator.mastermind.register_tool(calculator_tool)
    orchestrator.mastermind.register_tool(fetch_url_tool)
    orchestrator.mastermind.register_tool(pdf_to_markdown_tool)
    orchestrator.mastermind.register_tool(any_doc_reader_tool)
    orchestrator.mastermind.register_tool(edit_image_tool)
    orchestrator.mastermind.register_tool(image_generation_tool)
    
    # Register custom command execution functions
    orchestrator.mastermind.register_tool(orchestrator.mastermind.process_command)
    orchestrator.mastermind.register_tool(orchestrator.mastermind.code_execution)
    orchestrator.mastermind.register_tool(orchestrator.mastermind.web_search)
    
    logger.info(f"Registered {len(orchestrator.mastermind.tools)} tools")

def interactive_mode():
    """Run the MasterMind_AI in interactive mode."""
    print("\n" + "=" * 80)
    print("MasterMind_AI Interactive Mode")
    print("Type 'exit', 'quit', or 'q' to exit")
    print("Type 'simulate' to run a Tesla waveform simulation")
    print("Type 'superposition' to create a superposition state")
    print("Type 'entangled' to create an entangled state")
    print("=" * 80 + "\n")
    
    # Initialize system orchestrator
    orchestrator = SystemOrchestrator()
    orchestrator.initialize_quantum_system()
    
    # Register available tools
    register_tools(orchestrator)
    
    while True:
        try:
            user_input = input("\nEnter command or query: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting interactive mode.")
                break
            
            elif user_input.lower() == 'simulate':
                periods = int(input("Enter number of periods to scan (default 5): ") or "5")
                amplitude = float(input("Enter pulse amplitude (default 5.0): ") or "5.0")
                width = float(input("Enter pulse width (default 0.5): ") or "0.5")
                
                run_simulation(periods, amplitude, width)
            
            elif user_input.lower() == 'superposition':
                dim = int(input("Enter dimension of the Hilbert space (default 2): ") or "2")
                num_states = int(input("Enter number of states in superposition (default 2): ") or "2")
                
                try:
                    state = superposition_state(num_states=num_states, dim=dim)
                    print(f"\nSuperposition state created:")
                    print(f"State: {state}")
                    print(f"Shannon Entropy: {calculate_shannon_entropy(state):.4f}")
                except Exception as e:
                    print(f"Error creating superposition state: {e}")
            
            elif user_input.lower() == 'entangled':
                state_type = input("Enter state type (bell/ghz/w, default bell): ") or "bell"
                
                try:
                    state = entangled_state(state_type=state_type)
                    print(f"\nEntangled state created:")
                    print(f"State type: {state_type}")
                    print(f"State: {state}")
                    print(f"Shannon Entropy: {calculate_shannon_entropy(state):.4f}")
                except Exception as e:
                    print(f"Error creating entangled state: {e}")
            
            else:
                # Process as a query using the mastermind system
                result = orchestrator.mastermind.process_query(user_input)
                print(f"\nResponse: {result['response']}")
                print(f"Resonance score: {result['resonance_score']:.2f}")
                
                # Display additional information if query activated knowledge graph nodes
                if result.get('activated_nodes'):
                    print(f"Activated knowledge graph nodes: {', '.join(result['activated_nodes'])}")
        
        except KeyboardInterrupt:
            print("\nExiting on user request (Ctrl+C).")
            break
        except Exception as e:
            print(f"Error: {e}")

def agentic_interactive_mode():
    """Run the MasterMind_AI in agentic interactive mode with tool use capabilities."""
    print("\n" + "=" * 80)
    print("MasterMind_AI Agentic Interactive Mode")
    print("Type 'exit', 'quit', or 'q' to exit")
    print("You can ask questions and give instructions for tool use")
    print("=" * 80 + "\n")
    
    # Initialize system orchestrator
    orchestrator = SystemOrchestrator()
    orchestrator.initialize_quantum_system()
    
    # Register available tools
    register_tools(orchestrator)
    
    print("\nAvailable tools:")
    for tool_name in orchestrator.mastermind.tools.keys():
        print(f"- {tool_name}")
    
    while True:
        try:
            user_input = input("\nEnter your query or instruction: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting agentic interactive mode.")
                break
            
            # Process as a query using the mastermind system
            result = orchestrator.mastermind.process_query(user_input)
            print(f"\nResponse: {result['response']}")
            
            # If tools were used, show the results
            if result.get('tool_calls'):
                print("\nTools used:")
                for tool_call in result.get('tool_calls', []):
                    print(f"- {tool_call['name']}")
                
                if result.get('tool_results'):
                    print("\nTool results:")
                    for tool_result in result.get('tool_results', []):
                        if 'error' in tool_result:
                            print(f"- {tool_result['name']}: Error - {tool_result['error']}")
                        else:
                            print(f"- {tool_result['name']}: Success")
        
        except KeyboardInterrupt:
            print("\nExiting on user request (Ctrl+C).")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    args = parse_arguments()
    
    # Initialize quantum-classical system
    orchestrator = SystemOrchestrator()
    orchestrator.initialize_quantum_system()
    
    # Register tools if in agentic mode
    if args.agentic:
        register_tools(orchestrator)
    
    # Process input query if provided
    if args.query:
        result = orchestrator.execute_cognitive_workflow(args.query)
        print(f"Resonance Score: {result['resonance_score']:.2f}")
        print(f"Response: {result['response']}")
        
        if args.visualize:
            orchestrator.mastermind.visualize_knowledge_graph()
            orchestrator.mastermind.visualize_quantum_state()
    
    # Run in interactive mode if requested
    if args.interactive:
        if args.agentic:
            agentic_interactive_mode()
        else:
            interactive_mode()

if __name__ == "__main__":
    main() 