#!/usr/bin/env python3
# demo_mastermind.py
"""
Demo script for the MasterMind_AI system, showcasing its core functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import logging
import argparse
from typing import List, Dict, Any
import json

from code.mastermind_ai import MasterMind_AI
from code.action_registry import action_registry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MasterMind_AI Demo")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--visualize", action="store_true", help="Visualize knowledge graph and resonance")
    parser.add_argument("--query", type=str, nargs='+', help="Initial query to process")
    parser.add_argument("--learn", type=str, nargs='+', 
                       help="Add new knowledge to the system (concept:connections format)")
    parser.add_argument("--learn-file", type=str,
                       help="Path to a JSON file containing concepts to learn")
    return parser.parse_args()

def run_demo_queries(mastermind: MasterMind_AI) -> List[Dict[str, Any]]:
    """Run a series of demo queries and return the results."""
    demo_queries = [
        "Activate ResonantiA for optimal performance",
        "How can machine learning be applied to quantum computing?",
        "Tell me about neural networks and deep learning",
        "What is the relationship between reasoning and planning?",
        "How can I achieve a state of system harmony?",
        "Explain the concept of emergent behavior in complex systems"
    ]
    
    results = []
    for query in demo_queries:
        logger.info(f"Processing query: {query}")
        result = mastermind.process_query(query)
        results.append(result)
        print(f"Query: {query}")
        print(f"Response: {result['response']}")
        print(f"Resonance Score: {result['resonance_score']:.2f}")
        print(f"Activated Nodes: {', '.join(result['activated_nodes'][:3])}{'...' if len(result['activated_nodes']) > 3 else ''}")
        print("=" * 80)
    
    return results

def interactive_mode(mastermind: MasterMind_AI):
    """Enhanced interactive mode with visualization commands"""
    print("\nAvailable Commands:")
    print("kg_visualize - Show knowledge graph visualization")
    print("kg_stats - Show knowledge graph statistics")
    print("kg_search [concept] - Search for concepts in knowledge graph")
    print("exit - Exit interactive mode")
    
    while True:
        try:
            user_input = input("\nMasterMind> ").strip().lower()
            
            if user_input == 'exit':
                break
                
            elif user_input == 'kg_visualize':
                mastermind.visualize_knowledge_graph()
                print("Knowledge graph visualization generated")
                
            elif user_input == 'kg_stats':
                stats = mastermind.get_knowledge_graph_stats()
                print(f"\nKnowledge Graph Statistics:")
                print(f"Nodes: {stats['num_nodes']}")
                print(f"Edges: {stats['num_edges']}")
                print(f"Connected Components: {stats['connected_components']}")
                
            elif user_input.startswith('kg_search '):
                concept = user_input[9:].strip()
                results = mastermind.search_knowledge_graph(concept)
                print(f"\nSearch Results for '{concept}':")
                for node, data in results.items():
                    print(f"- {node} ({data.get('type', 'unknown')})")
                    
            else:
                print("Invalid command. Try: kg_visualize, kg_stats, kg_search, exit")

        except Exception as e:
            print(f"Error: {str(e)}")

def visualize_knowledge_graph(mastermind: MasterMind_AI):
    """Visualize the knowledge graph."""
    plt.figure(figsize=(12, 8))
    
    # Create node colors based on type
    node_colors = []
    for node in mastermind.kg.nodes():
        node_type = mastermind.kg.nodes[node].get('type', 'concept')
        if node_type == 'resonant_state':
            node_colors.append('red')
        elif node_type == 'advanced_concept':
            node_colors.append('purple')
        else:
            node_colors.append('blue')
    
    # Create node sizes based on centrality
    centrality = nx.degree_centrality(mastermind.kg)
    node_sizes = [centrality[node] * 3000 + 100 for node in mastermind.kg.nodes()]
    
    # Draw the graph
    pos = nx.spring_layout(mastermind.kg, seed=42)
    nx.draw_networkx_nodes(mastermind.kg, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(mastermind.kg, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(mastermind.kg, pos, font_size=8)
    
    plt.title("MasterMind_AI Knowledge Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('knowledge_graph.png', dpi=300)
    print("Knowledge graph visualization saved as 'knowledge_graph.png'")
    
    # Plot resonance history if there are queries
    if mastermind.queries_history:
        plt.figure(figsize=(10, 6))
        resonance_history = mastermind.get_resonance_history()
        plt.plot(resonance_history, 'b-o')
        plt.axhline(y=mastermind.resonance_threshold, color='r', linestyle='--', label='Resonance Threshold')
        plt.xlabel('Query Number')
        plt.ylabel('Resonance Score')
        plt.title('MasterMind_AI Resonance History')
        plt.grid(True)
        plt.legend()
        plt.savefig('resonance_history.png', dpi=300)
        print("Resonance history visualization saved as 'resonance_history.png'")

def kg_visualize(mastermind: MasterMind_AI, filename: str = "knowledge_graph.png"):
    """Visualize the knowledge graph using NetworkX and Matplotlib"""
    import matplotlib.pyplot as plt
    import networkx as nx
    
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(mastermind.kg, seed=42)
    nx.draw(mastermind.kg, pos, with_labels=True, node_size=2000, font_size=8)
    plt.title("MasterMind_AI Knowledge Graph")
    plt.savefig(filename, dpi=300)
    print(f"Knowledge graph visualization saved to {filename}")

def main():
    """Main function to run the demo."""
    args = parse_args()
    
    # Convert query list to string if provided
    if args.query:
        args.query = ' '.join(args.query)
    
    # Define example operators for the agent
    operators_example = {
        'default_operator': np.array([[1.0, 0.0, 0.0, 0.0], 
                                      [0.0, 1.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 1.0]]),  # Identity operator
        'explore_operator': np.array([[0.8, 0.1, 0.0, 0.1], 
                                     [-0.1, 0.8, 0.1, 0.0],
                                     [0.0, 0.1, 0.8, 0.1],
                                     [0.1, 0.0, 0.1, 0.8]]),  # Exploration operator
        'resonant_operator': np.array([[1.1, -0.2, 0.1, 0.0], 
                                      [0.3, 0.9, 0.0, 0.1],
                                      [0.1, 0.0, 1.1, -0.2],
                                      [0.0, 0.1, 0.3, 0.9]])   # Resonant operator
    }
    
    # Initialize the MasterMind_AI
    mastermind = MasterMind_AI(
        agent_id="demo_mastermind",
        initial_state=np.array([0.5, 0.5, 0.5, 0.5]),
        operators=operators_example,
        action_registry=action_registry,
        resonance_threshold=0.65
    )
    
    print("=" * 80)
    print("MasterMind_AI Demo Initialized")
    print(f"Knowledge Graph: {len(mastermind.kg.nodes())} nodes, {len(mastermind.kg.edges())} edges")
    print(f"Resonance Threshold: {mastermind.resonance_threshold}")
    print("=" * 80)
    
    # Process initial query if provided
    if args.query:
        result = mastermind.process_query(args.query)
        print(f"\nQuery: {args.query}")
        print(f"Response: {result['response']}")
        print(f"Resonance Score: {result['resonance_score']:.2f}")
    
    # Run demo queries if not in interactive mode
    if not args.interactive:
        results = run_demo_queries(mastermind)
    
    # Run interactive mode if specified
    if args.interactive:
        interactive_mode(mastermind)
    
    # Visualize the knowledge graph if specified
    if args.visualize:
        mastermind.visualize_knowledge_graph()
        print("Generated knowledge graph visualization")
    
    # Process initial query if provided
    if args.query:
        result = mastermind.process_query(args.query)
        print(f"\nQuery: {args.query}")
        print(f"Response: {result['response']}")
        print(f"Resonance Score: {result['resonance_score']:.2f}")
        
        if args.visualize:
            mastermind.visualize_knowledge_graph("post_query_kg.png")
    
    # In main function
    if args.learn:
        learn_text = ' '.join(args.learn)
        # Format should be "ConceptName:Connection1,Connection2"
        parts = learn_text.split(':')
        if len(parts) == 2:
            concept = parts[0].strip()
            connections = [c.strip() for c in parts[1].split(',')]
            mastermind.add_knowledge(concept=concept, connections=connections)
            print(f"Added knowledge: {concept} connected to {', '.join(connections)}")
        else:
            print("Invalid learning format. Use: ConceptName:Connection1,Connection2")

    if args.learn_file:
        try:
            with open(args.learn_file, 'r') as f:
                concepts_data = json.load(f)
            
            for concept_data in concepts_data:
                mastermind.add_knowledge(
                    concept=concept_data['name'],
                    concept_type=concept_data.get('type', 'concept'),
                    connections=concept_data.get('connections', [])
                )
            print(f"Learned {len(concepts_data)} concepts from {args.learn_file}")
        except Exception as e:
            print(f"Error learning from file: {str(e)}")
    
    print("\nMasterMind_AI Demo Completed")

if __name__ == "__main__":
    main() 