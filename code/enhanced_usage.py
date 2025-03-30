"""
Enhanced usage of MasterMind_AI for solving difficult questions with detailed reasoning.
"""

import numpy as np
from code.mastermind_ai import MasterMind_AI
from code.system_orchestrator import SystemOrchestrator
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def solve_difficult_question(question: str, domain: str = "general"):
    """
    Solve a difficult question using MasterMind_AI with enhanced reasoning capabilities.
    
    Args:
        question: The difficult question to solve
        domain: The domain of the question (e.g., "quantum", "ai", "physics")
        
    Returns:
        Dict containing the solution and detailed reasoning
    """
    # 1. Initialize MasterMind_AI with enhanced settings
    mastermind = MasterMind_AI(
        agent_id='enhanced_solver',
        initial_state=np.array([0.5, 0.5, 0.5, 0.5]),
        resonance_threshold=0.75,
        max_thinking_tokens=4000  # Increased for more detailed reasoning
    )
    
    # 2. Add domain-specific knowledge
    domain_concepts = {
        "quantum": [
            {"name": "QuantumComputing", "connections": ["QuantumMechanics", "InformationTheory"]},
            {"name": "QuantumEntanglement", "connections": ["QuantumCorrelation", "NonLocality"]},
            {"name": "QuantumDecoherence", "connections": ["QuantumState", "EnvironmentalInteraction"]}
        ],
        "ai": [
            {"name": "MachineLearning", "connections": ["NeuralNetworks", "DeepLearning"]},
            {"name": "ArtificialIntelligence", "connections": ["CognitiveScience", "Robotics"]},
            {"name": "NaturalLanguageProcessing", "connections": ["Linguistics", "ComputationalLinguistics"]}
        ],
        "general": [
            {"name": "ProblemSolving", "connections": ["Analysis", "Synthesis"]},
            {"name": "CriticalThinking", "connections": ["Logic", "Reasoning"]},
            {"name": "SystemsThinking", "connections": ["Complexity", "Emergence"]}
        ]
    }
    
    # Add relevant domain concepts
    for concept in domain_concepts.get(domain, domain_concepts["general"]):
        mastermind.add_knowledge(
            concept=concept["name"],
            connections=concept["connections"]
        )
    
    # 3. Initialize the system orchestrator with enhanced tools
    orchestrator = SystemOrchestrator(mastermind_ai=mastermind)
    orchestrator.initialize_quantum_system()
    
    # 4. Process the question with enhanced workflow and reasoning
    result = orchestrator.run_agentic_workflow(
        query=question,
        tool_names=[
            'quantum_entanglement_analysis',
            'quantum_field_harmonization',
            'quantum_state_optimization'
        ]
    )
    
    # 5. Generate detailed analysis
    print("\n" + "="*80)
    print(f"Question: {question}")
    print("="*80)
    
    print("\nReasoning Process:")
    print("-"*40)
    print(result.get('thinking', 'No thinking process available'))
    
    print("\nQuantum Metrics:")
    print("-"*40)
    for metric, value in result['quantum_metrics'].items():
        print(f"  {metric}: {value:.2f}")
    
    print("\nActivated Concepts:")
    print("-"*40)
    for node in result.get('activated_nodes', []):
        print(f"  • {node}")
    
    print("\nDetailed Solution:")
    print("-"*40)
    print(result['response'])
    
    print("\nResonance Analysis:")
    print("-"*40)
    print(f"  Resonance Score: {result['resonance_score']:.2f}")
    print(f"  System Harmony: {'High' if result['resonance_score'] >= 0.75 else 'Medium' if result['resonance_score'] >= 0.5 else 'Low'}")
    
    # 6. Visualize the knowledge graph
    mastermind.visualize_knowledge_graph('solution_knowledge_graph.png')
    
    # 7. Save detailed results
    with open('solution_details.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

if __name__ == "__main__":
    # Example usage with different domains
    questions = [
        {
            "question": "How can we optimize a quantum computing algorithm for protein folding while maintaining energy efficiency?",
            "domain": "quantum"
        },
        {
            "question": "What are the key principles of emergent behavior in complex systems?",
            "domain": "general"
        },
        {
            "question": "How can deep learning be applied to natural language processing?",
            "domain": "ai"
        }
    ]
    
    for q in questions:
        print("\n" + "="*80)
        print(f"Processing question in domain: {q['domain']}")
        solution = solve_difficult_question(q['question'], q['domain']) 