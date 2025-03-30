"""
Example usage of MasterMind_AI for solving difficult questions.
This demonstrates the key features and best practices for using the system.
"""

import numpy as np
from code.mastermind_ai import MasterMind_AI
from code.system_orchestrator import SystemOrchestrator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def solve_difficult_question(question: str):
    """
    Solve a difficult question using MasterMind_AI with optimal configuration.
    
    Args:
        question: The difficult question to solve
        
    Returns:
        Dict containing the solution and processing metrics
    """
    # 1. Initialize MasterMind_AI with optimal settings
    mastermind = MasterMind_AI(
        agent_id='problem_solver',
        initial_state=np.array([0.5, 0.5, 0.5, 0.5]),  # Balanced initial state
        resonance_threshold=0.75,  # High threshold for better reasoning
        max_thinking_tokens=2000  # Allow for deep reasoning
    )
    
    # 2. Add relevant domain knowledge
    domain_concepts = [
        {
            "name": "CoreConcept",
            "connections": ["RelatedConcept1", "RelatedConcept2"]
        },
        {
            "name": "ProblemSolving",
            "connections": ["Analysis", "Synthesis"]
        }
    ]
    
    for concept in domain_concepts:
        mastermind.add_knowledge(
            concept=concept["name"],
            connections=concept["connections"]
        )
    
    # 3. Initialize the system orchestrator
    orchestrator = SystemOrchestrator(mastermind_ai=mastermind)
    orchestrator.initialize_quantum_system()
    
    # 4. Process the question with enhanced workflow
    result = orchestrator.run_agentic_workflow(
        query=question,
        tool_names=['quantum_entanglement_analysis', 'quantum_field_harmonization']
    )
    
    # 5. Analyze the results
    print(f"\nQuestion: {question}")
    print(f"\nResonance Score: {result['resonance_score']:.2f}")
    print(f"\nQuantum Metrics:")
    for metric, value in result['quantum_metrics'].items():
        print(f"  {metric}: {value:.2f}")
    
    print(f"\nSolution: {result['response']}")
    
    # 6. Visualize the knowledge graph
    mastermind.visualize_knowledge_graph('solution_knowledge_graph.png')
    
    return result

if __name__ == "__main__":
    # Example usage
    difficult_question = """
    How can we optimize a quantum computing algorithm for protein folding 
    while maintaining energy efficiency and considering quantum decoherence effects?
    """
    
    solution = solve_difficult_question(difficult_question) 