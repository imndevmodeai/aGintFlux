#!/usr/bin/env python3
"""
Advanced Learning Demo for MasterMind_AI

This script demonstrates more effective learning in the MasterMind_AI system by:
1. Adding domain-specific knowledge with more meaningful connections
2. Using targeted queries that match the knowledge concepts
3. Building a more connected knowledge graph
4. Improving resonance scoring through supervised learning steps
"""

from code.mastermind_ai import MasterMind_AI
from code.action_registry import action_registry
import numpy as np
import time

def print_separator():
    print("\n" + "=" * 60 + "\n")

# Initialize the MasterMind_AI
mastermind = MasterMind_AI(
    agent_id='advanced_learning',
    initial_state=np.array([0.5, 0.5, 0.5, 0.5]),
    action_registry=action_registry,
    resonance_threshold=0.55  # Lower threshold to start
)

print_separator()
print("STEP 1: Building a Core Knowledge Foundation")
print_separator()

# Add core machine learning concepts with meaningful connections
concepts = [
    {
        "name": "MachineLearning", 
        "type": "domain",
        "connections": ["ResonantiA", "QuantumAlgorithms"]
    },
    {
        "name": "DeepLearning", 
        "type": "domain",
        "connections": ["MachineLearning"]
    },
    {
        "name": "NeuralNetworks", 
        "type": "domain",
        "connections": ["DeepLearning", "MachineLearning"]
    },
    {
        "name": "SupervisedLearning", 
        "type": "learning_type",
        "connections": ["MachineLearning"]
    },
    {
        "name": "QuantumNeuralNetworks", 
        "type": "quantum_ml",
        "connections": ["QuantumAlgorithms", "NeuralNetworks"]
    }
]

# Add each concept to the knowledge graph
for concept in concepts:
    mastermind.add_knowledge(
        concept=concept["name"],
        concept_type=concept["type"],
        connections=concept["connections"]
    )
    print(f"Added concept: {concept['name']} ({concept['type']})")
    print(f"  Connected to: {', '.join(concept['connections'])}")

# Initial knowledge graph stats
print("\nInitial Knowledge Graph:")
stats = mastermind.get_knowledge_graph_stats()
print(f"Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")

# Visualize initial knowledge graph
mastermind.visualize_knowledge_graph('initial_knowledge_graph.png')
print("Initial knowledge graph saved to initial_knowledge_graph.png")

print_separator()
print("STEP 2: Testing Initial Knowledge With Domain-Specific Queries")
print_separator()

# Test queries that specifically mention concepts in our graph
test_queries = [
    "What are neural networks in machine learning?",
    "How does deep learning relate to neural networks?",
    "What is supervised learning?",
    "How can quantum algorithms improve neural networks?"
]

for i, query in enumerate(test_queries):
    print(f"\nQuery {i+1}: {query}")
    result = mastermind.process_query(query)
    print(f"Resonance Score: {result['resonance_score']:.2f}")
    print(f"Activated Nodes: {', '.join(result['activated_nodes']) if result['activated_nodes'] else 'None'}")

print_separator()
print("STEP 3: Enhancing Learning Through Node Connections")
print_separator()

# Add more specific connections to create a richer knowledge graph
connections = [
    ("SupervisedLearning", "NeuralNetworks"),
    ("DeepLearning", "QuantumNeuralNetworks"),
    ("AnalyticalResonance", "SupervisedLearning"),
    ("CreativeFlux", "DeepLearning")
]

for source, target in connections:
    # Check if both nodes exist before adding edge
    if source in mastermind.kg.nodes and target in mastermind.kg.nodes:
        mastermind.kg.add_edge(source, target, type="related_to")
        print(f"Added connection: {source} -> {target}")

# Enhance resonance for better learning
mastermind.orbital_sync_factor = min(1.0, mastermind.orbital_sync_factor + 0.2)
print(f"Enhanced resonance factor to {mastermind.orbital_sync_factor:.2f}")

print_separator()
print("STEP 4: Testing With Enhanced Knowledge")
print_separator()

# Test the same queries to see improved results
for i, query in enumerate(test_queries):
    print(f"\nQuery {i+1}: {query}")
    result = mastermind.process_query(query)
    print(f"Resonance Score: {result['resonance_score']:.2f}")
    print(f"Activated Nodes: {', '.join(result['activated_nodes']) if result['activated_nodes'] else 'None'}")

print_separator()
print("STEP 5: Adding Application Domain Knowledge")
print_separator()

# Add application-specific concepts to expand knowledge
applications = [
    {
        "name": "ComputerVision", 
        "type": "application",
        "connections": ["DeepLearning", "NeuralNetworks"]
    },
    {
        "name": "NaturalLanguageProcessing", 
        "type": "application",
        "connections": ["DeepLearning", "SupervisedLearning"]
    },
    {
        "name": "QuantumImageRecognition", 
        "type": "quantum_application",
        "connections": ["ComputerVision", "QuantumNeuralNetworks"]
    }
]

for app in applications:
    mastermind.add_knowledge(
        concept=app["name"],
        concept_type=app["type"],
        connections=app["connections"]
    )
    print(f"Added application: {app['name']} ({app['type']})")
    print(f"  Connected to: {', '.join(app['connections'])}")

# Final knowledge graph visualization
mastermind.visualize_knowledge_graph('advanced_knowledge_graph.png')
print("\nFinal Knowledge Graph stats:")
stats = mastermind.get_knowledge_graph_stats()
print(f"Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
print("Advanced knowledge graph saved to advanced_knowledge_graph.png")

print_separator()
print("STEP 6: Final Testing With Specific Application Queries")
print_separator()

# Test with application-specific queries
app_queries = [
    "How is computer vision related to deep learning?",
    "What is natural language processing?",
    "How can quantum computing improve image recognition?",
    "What's the relationship between supervised learning and NLP?"
]

for i, query in enumerate(app_queries):
    print(f"\nQuery {i+1}: {query}")
    result = mastermind.process_query(query)
    print(f"Resonance Score: {result['resonance_score']:.2f}")
    print(f"Activated Nodes: {', '.join(result['activated_nodes']) if result['activated_nodes'] else 'None'}")

print_separator()
print("Learning Demo Complete")
print_separator()

# Summary of what we've learned
print("The MasterMind_AI system learns through:")
print("1. Building a connected knowledge graph with domain-specific concepts")
print("2. Creating meaningful relationships between concepts")
print("3. Processing queries related to the knowledge domain")
print("4. Enhancing resonance to improve pattern recognition")
print("5. Adding application-specific knowledge that builds on core concepts")
print("\nThe more connected and domain-relevant the knowledge graph becomes,")
print("the better the system can 'learn' by activating the right nodes and")
print("achieving higher resonance scores for relevant queries.")