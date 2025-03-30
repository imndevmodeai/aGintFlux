"""
ScalableAgent implementation for the CFP framework.
The agent manages state, operators, and actions within the Comparative Fluxual Processing system.
"""

from typing import Dict, Any, Optional, List, Callable, Union
import numpy as np
import logging
import sys
import matplotlib.pyplot as plt
import networkx as nx
from qiskit.visualization import plot_bloch_multivector
from code.action_registry import action_registry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print(sys.path)  # Make sure your project root is in the path

class ScalableAgent:
    """
    A scalable agent that manages state, operators, and actions within the CFP framework.
    
    The agent can dynamically select operators based on a strategy, maintain state history,
    and execute registered actions.
    
    Attributes:
        agent_id (str): Unique identifier for the agent
        initial_state (np.ndarray): The initial state vector of the agent
        current_state (np.ndarray): The current state vector of the agent
        operators (Dict[str, np.ndarray]): Dictionary of operator matrices
        action_registry (Dict[str, Callable]): Registry of available actions
        workflow_modes (Dict[str, List[str]]): Dictionary of workflow modes
        operator_selection_strategy (Callable): Function to select operators
        current_operator_key (str): Key of the currently selected operator
        state_history (List[np.ndarray]): History of agent states
    """
    
    def __init__(
        self,
        agent_id: str = "default_agent",
        initial_state: Optional[np.ndarray] = None,
        operators: Optional[Dict[str, np.ndarray]] = None,
        action_registry: Optional[Dict[str, Callable]] = None,
        workflow_modes: Optional[Dict[str, List[str]]] = None,
        operator_selection_strategy: Optional[Callable] = None,
        initial_operator_key: Optional[str] = None
    ):
        """
        Initialize the ScalableAgent.
        
        Args:
            agent_id: Unique identifier for the agent
            initial_state: Initial state vector, defaults to [0.5, 0.5]
            operators: Dictionary of operator matrices, defaults to identity operator
            action_registry: Registry of available actions, defaults to empty dict
            workflow_modes: Dictionary of workflow modes, defaults to empty dict
            operator_selection_strategy: Function to select operators, defaults to returning first operator
            initial_operator_key: Key of the initial operator, defaults to first key in operators
        """
        self.agent_id = agent_id
        self.initial_state = initial_state if initial_state is not None else np.array([0.5, 0.5])
        self.current_state = self.initial_state.copy()
        
        # Initialize operators
        self.operators = operators if operators is not None else {
            'default_operator': np.eye(len(self.initial_state))
        }
        
        # Initialize other attributes
        self.action_registry = action_registry if action_registry is not None else {}
        self.workflow_modes = workflow_modes if workflow_modes is not None else {}
        self.operator_selection_strategy = operator_selection_strategy if operator_selection_strategy is not None else self._default_operator_strategy
        
        # Set current operator key
        self.current_operator_key = initial_operator_key if initial_operator_key is not None else next(iter(self.operators))
        
        # Initialize state history
        self.state_history = [self.initial_state.copy()]
        
        # Additional attributes for quantum features
        self.quantum_state = None
        self.orbital_sync_factor = 1.0
        
        logger.info(f"ScalableAgent '{agent_id}' initialized with {len(self.operators)} operators")
    
    def _default_operator_strategy(self, agent):
        """Default strategy for operator selection: return first operator key."""
        return next(iter(self.operators))
    
    def update_state(self, new_state: np.ndarray) -> None:
        """
        Update the agent's current state and append to state history.
        
        Args:
            new_state: The new state vector
        """
        if not isinstance(new_state, np.ndarray):
            new_state = np.array(new_state)
            
        self.current_state = new_state
        self.state_history.append(new_state.copy())
        logger.debug(f"Agent '{self.agent_id}' state updated: {new_state}")
    
    def select_operator(self) -> str:
        """
        Select an operator using the operator selection strategy.
        
        Returns:
            The key of the selected operator
        """
        selected_key = self.operator_selection_strategy(self)
        self.current_operator_key = selected_key
        logger.debug(f"Agent '{self.agent_id}' selected operator: {selected_key}")
        return selected_key
    
    def apply_operator(self, state: Optional[np.ndarray] = None, operator_key: Optional[str] = None) -> np.ndarray:
        """Enhanced operator application with dimension checks"""
        if state is None:
            state = self.current_state
        
        if operator_key is None:
            operator_key = self.current_operator_key
        
        operator = self.operators[operator_key]
        
        # Add dimension compatibility check
        if len(state) != operator.shape[1]:
            # Handle dimension mismatch by padding with zeros
            pad_size = operator.shape[1] - len(state)
            state = np.pad(state, (0, pad_size), mode='constant')
            print(f"Padded state vector with {pad_size} zeros for operator compatibility")
        
        return operator @ state
    
    def get_state_trajectory(self, time_points: np.ndarray) -> np.ndarray:
        """
        Generate a trajectory of states over time points.
        
        Args:
            time_points: Array of time points
        
        Returns:
            Array of states at each time point
        """
        trajectory = []
        current = self.current_state.copy()
        
        for t in time_points:
            # Use a simple model where state evolves based on time and current operator
            transformed = self.apply_operator(current)
            scaled = transformed * (1 + 0.1 * t)  # Simple time scaling
            trajectory.append(scaled)
        
        return np.array(trajectory)
    
    def get_state_history(self) -> List[np.ndarray]:
        """
        Get the complete state history of the agent.
        
        Returns:
            List of state vectors
        """
        return self.state_history
    
    def perform_action(self, action_name: str, **kwargs) -> Any:
        """
        Perform a registered action with the given parameters.
        
        Args:
            action_name: Name of the action to perform
            **kwargs: Parameters to pass to the action
        
        Returns:
            Result of the action
        
        Raises:
            ValueError: If action_name is not in the action_registry
        """
        if action_name not in self.action_registry:
            raise ValueError(f"Action '{action_name}' not found in action registry")
        
        action_function = self.action_registry[action_name]
        
        try:
            logger.info(f"Agent '{self.agent_id}' performing action: {action_name}")
            result = action_function(self, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error performing action '{action_name}': {str(e)}")
            raise
    
    def run_workflow_mode(self, mode_name: str) -> List[Any]:
        """
        Run a sequence of actions defined by a workflow mode.
        
        Args:
            mode_name: Name of the workflow mode to run
        
        Returns:
            List of results from each action in the workflow
        
        Raises:
            ValueError: If mode_name is not in workflow_modes
        """
        if mode_name not in self.workflow_modes:
            raise ValueError(f"Workflow mode '{mode_name}' not found")
        
        action_sequence = self.workflow_modes[mode_name]
        results = []
        
        for action_name in action_sequence:
            result = self.perform_action(action_name)
            results.append(result)
        
        return results
    
    # Methods for quantum resonance features
    def get_parallel_state(self, version: float = 1.0) -> np.ndarray:
        """
        Get a parallel version of the current state.
        
        Args:
            version: Scaling factor for the parallel state
            
        Returns:
            Scaled version of the current state
        """
        return self.current_state * version
    
    def apply_tesla_waveform(self) -> float:
        """
        Apply a Tesla waveform transformation to the quantum state.
        
        Returns:
            Sum of absolute values of the quantum state
        """
        if self.quantum_state is None:
            self.quantum_state = self.current_state.copy()
        
        # Simple implementation that returns the sum of absolute values
        return float(np.sum(np.abs(self.quantum_state)))
    
    def synchronize_orbital_resonance(self) -> None:
        """
        Synchronize the agent with orbital resonance.
        """
        self.orbital_sync_factor = 0.92
        logger.info(f"Agent '{self.agent_id}' synchronized with orbital resonance")
    
    def transmit_power(self, energy: float, target_coordinates: tuple) -> float:
        """
        Simulate power transmission to target coordinates.
        
        Args:
            energy: Energy amount to transmit
            target_coordinates: Target coordinates (lat, lon)
            
        Returns:
            Transmitted energy
        """
        return energy * self.orbital_sync_factor
    
    def recalibrate_resonance(self) -> None:
        """
        Recalibrate the agent's resonance.
        """
        self.orbital_sync_factor = 1.0
        logger.info(f"Agent '{self.agent_id}' resonance recalibrated")
    
    def calculate_quantum_network_flux(self, target_coordinates: tuple) -> float:
        """
        Calculate quantum network flux to target coordinates.
        
        Args:
            target_coordinates: Target coordinates (lat, lon)
            
        Returns:
            Calculated flux
        """
        # Simple implementation that returns a value based on coordinates and sync factor
        lat, lon = target_coordinates
        return np.abs(lat * lon) * self.orbital_sync_factor
    
    def entangled_state_evolution(self, steps: int) -> List[np.ndarray]:
        """
        Simulate quantum-entangled state evolution.
        
        Args:
            steps: Number of evolution steps
            
        Returns:
            List of evolved states
        """
        evolved_states = []
        
        for _ in range(steps):
            # Apply the current operator and a phase factor
            new_state = self.apply_operator()
            phase_factor = np.exp(1j * 0.1)
            phased_state = new_state * phase_factor
            
            # Update state and add to list
            self.update_state(np.real(phased_state))
            evolved_states.append(self.current_state.copy())
        
        return evolved_states

    def visualize_state_trajectory(self, time_points):
        """Visualize the state trajectory over time."""
        import matplotlib.pyplot as plt  # Move import here
        
        trajectory = self.get_state_trajectory(time_points)
        # Visualization code...

    def visualize_quantum_state(self, state: np.ndarray = None, save_path: str = None):
        """Visualize quantum state using Bloch sphere representation
        
        Args:
            state: Optional state vector to visualize. If None, uses the agent's current or quantum state.
            save_path: Optional path to save the visualization. If None, displays interactively.
        """
        import matplotlib.pyplot as plt
        
        # Use provided state, or fall back to agent's internal state
        if state is None:
            if hasattr(self, 'quantum_state') and self.quantum_state is not None:
                state = self.quantum_state
            else:
                state = self.current_state
        
        # Create a simple visualization (since we don't have qiskit's plot_bloch_multivector)
        plt.figure(figsize=(8, 6))
        
        # Convert complex state to real components for simple visualization
        if state.ndim == 1:
            # For state vector, plot real and imaginary parts
            x = np.arange(len(state))
            plt.bar(x, np.real(state), width=0.4, label='Real Part', alpha=0.7)
            plt.bar(x + 0.4, np.imag(state), width=0.4, label='Imaginary Part', alpha=0.7)
            plt.xlabel('State Index')
            plt.ylabel('Amplitude')
            plt.title('Quantum State Visualization')
            plt.legend()
        else:
            # For density matrix, show as heatmap
            plt.imshow(np.abs(state), cmap='viridis')
            plt.colorbar(label='Magnitude')
            plt.title('Density Matrix Visualization')
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.savefig('quantum_state.png', dpi=300)
            plt.close()

def test_mastermind_integration():
    """Test function for MasterMind_AI integration"""
    from code.mastermind_ai import MasterMind_AI
    
    # Initialize MasterMind_AI with a lower threshold for initial learning
    mastermind = MasterMind_AI(
        agent_id='one_pass_learning',
        initial_state=np.array([0.5, 0.5, 0.5, 0.5]),
        resonance_threshold=0.55
    )

    # 1. Add domain concepts with meaningful connections
    concepts = [
        {"name": "CoreConcept1", "connections": ["ResonantiA"]},
        {"name": "RelatedConcept", "connections": ["CoreConcept1"]},
    ]

    for concept in concepts:
        mastermind.add_knowledge(
            concept=concept["name"],
            connections=concept["connections"]
        )

    # 2. Create additional connections for richer graph
    for source, target in [("ConceptA", "ConceptB"), ("ConceptC", "ConceptD")]:
        if source in mastermind.kg.nodes and target in mastermind.kg.nodes:
            mastermind.kg.add_edge(source, target, type="related_to")

    # 3. Enhance resonance for better learning
    mastermind.orbital_sync_factor = min(1.0, mastermind.orbital_sync_factor + 0.2)

    # 4. Process domain-specific queries
    results = []
    for query in ["Query about concept1", "Query about concept2"]:
        results.append(mastermind.process_query(query))

    # 5. Visualize the knowledge graph
    mastermind.visualize_knowledge_graph('learned_knowledge_graph.png')

    result = mastermind.process_query("Your specific domain query")
    print(f"Resonance Score: {result['resonance_score']}")
    print(f"Activated Nodes: {result['activated_nodes']}")

    # Use the new orchestrator
    from code.system_orchestrator import SystemOrchestrator
    orchestrator = SystemOrchestrator()
    orchestrator.mastermind = mastermind  # Use your trained instance
    orchestrator.execute_cognitive_workflow("Query")

    # Add application-specific concepts
    mastermind.add_knowledge("YourSpecializedConcept", 
                           connections=["RelatedConcept1", "RelatedConcept2"])

if __name__ == "__main__":
    test_mastermind_integration()

def quantum_entanglement_analysis_action(
    agent: ScalableAgent, 
    operator: np.ndarray,
    state_series: List[np.ndarray],
    resonance_targets: List[str],
    entanglement_level: int = 9
) -> Dict[str, float]:
    """Action: Perform quantum entanglement analysis with resonance targets."""
    # Simple implementation
    result = {
        'average_orbital_coherence': 0.92,
        'geospheric_resonance': 0.87,
        'quantum_vacuum_entanglement': 0.76 * entanglement_level / 10
    }
    
    # Add metrics for each resonance target
    for target in resonance_targets:
        # Calculate a metric based on the target
        if target == "orbital":
            result[f"{target}_resonance"] = 0.91
        elif target == "geospheric":
            result[f"{target}_resonance"] = 0.85
        elif target == "quantum_vacuum":
            result[f"{target}_resonance"] = 0.78
    
    return result

# Add to action_registry dictionary
action_registry["quantum_entanglement_analysis"] = quantum_entanglement_analysis_action

def comparative_fluxual_processing(system_state_series_1, system_state_series_2):
    # Check that state series have same length
    if len(system_state_series_1) != len(system_state_series_2):
        # If different lengths, pad the shorter one with copies of its last state
        if len(system_state_series_1) < len(system_state_series_2):
            pad_length = len(system_state_series_2) - len(system_state_series_1)
            system_state_series_1 = list(system_state_series_1) + [system_state_series_1[-1]] * pad_length
        else:
            pad_length = len(system_state_series_1) - len(system_state_series_2)
            system_state_series_2 = list(system_state_series_2) + [system_state_series_2[-1]] * pad_length
        
        print(f"Warning: State series had different lengths. Padded the shorter one.")
    
    # Original function continues...

# These example lines were causing the error - removing them or moving to examples section
"""
Example usage:
mastermind.process_query(query)  # Process text query with resonance scoring
mastermind.calculate_resonance_score(query)  # Calculate resonance score
mastermind.visualize_knowledge_graph()  # Create knowledge graph visualization
mastermind.add_knowledge(concept="NewConcept", concept_type="domain", connections=["QuantumAlgorithms", "CRISPR"])  # Add to knowledge graph
mastermind.register_tool(tool)  # Register new tool for agent

orchestrator.initialize_quantum_system()  # Set up the quantum system
orchestrator.execute_cognitive_workflow(query)  # Run full cognitive pipeline

workflow_engine.execute_workflow(workflow_json)  # Run a defined workflow
workflow_engine.load_workflow_from_file(file_path)  # Load workflow from JSON
"""

# 1. Define new action
def quantum_field_harmonization_action(agent: "ScalableAgent", operator: np.ndarray, frequency: float = 7.83) -> Dict[str, float]:
    """Action: Harmonize quantum fields at specified frequency"""
    # Implementation as above...
    return {}  # Placeholder return

# Add to action registry
action_registry["quantum_field_harmonization"] = quantum_field_harmonization_action

# Example section - this code doesn't run on import
"""
Example: Creating and using custom components

# 2. Create and register custom tool
from tools.quantum_tools import QuantumResonanceTool
resonance_tool = QuantumResonanceTool()
mastermind.register_tool(resonance_tool)

# 3. Define custom operator selection strategy
def resonance_based_operator_strategy(agent: "ScalableAgent") -> str:
    \"\"\"Selects operators based on resonance level\"\"\"
    # Implementation as above...
    return selected_operator_key

# 4. Create agent with custom components
mastermind = MasterMind_AI(
    agent_id="quantum_mastermind",
    initial_state=np.array([0.5, 0.5, 0.5, 0.5]),
    operators={
        'exploration_operator': np.array([[0.8, 0.2], [0.2, 0.8]]),  # Encourages exploration
        'balanced_operator': np.array([[0.5, 0.5], [0.5, 0.5]]),     # Balanced approach
        'exploitation_operator': np.array([[0.9, 0.1], [0.1, 0.9]])  # Focuses on exploitation
    },
    action_registry=action_registry,
    operator_selection_strategy=resonance_based_operator_strategy
)

# 5. Load and execute a workflow
workflow_engine = WorkflowEngine(mastermind, action_registry)
workflow = workflow_engine.load_workflow_from_file("quantum_workflow.json")
results = workflow_engine.execute_workflow(workflow)

# 6. Process and visualize results
mastermind.visualize_knowledge_graph()
mastermind.visualize_quantum_state()
"""

# Test code - moved to a function to prevent execution during import
def test_resonance_strategy():
    # Define a sample strategy for testing
    def resonance_based_operator_strategy(agent):
        if agent.orbital_sync_factor > 0.9:
            return 'exploration_operator'
        elif agent.orbital_sync_factor > 0.7:
            return 'balanced_operator'
        else:
            return 'exploitation_operator'
    
    # Test the strategy at different resonance levels
    agent = ScalableAgent(
        agent_id="resonance_agent",
        initial_state=np.array([0.5, 0.5]),
        operators={
            'exploration_operator': np.array([[0.8, 0.2], [0.2, 0.8]]),  # Encourages exploration
            'balanced_operator': np.array([[0.5, 0.5], [0.5, 0.5]]),     # Balanced approach
            'exploitation_operator': np.array([[0.9, 0.1], [0.1, 0.9]])  # Focuses on exploitation
        },
        operator_selection_strategy=resonance_based_operator_strategy
    )

    # Test the strategy at different resonance levels
    agent.orbital_sync_factor = 0.95
    selected_operator = agent.select_operator()
    print(f"High resonance (0.95) selected: {selected_operator}")  # Should print 'exploration_operator'

    agent.orbital_sync_factor = 0.8
    selected_operator = agent.select_operator()
    print(f"Medium resonance (0.8) selected: {selected_operator}")  # Should print 'balanced_operator'

    agent.orbital_sync_factor = 0.6
    selected_operator = agent.select_operator()
    print(f"Low resonance (0.6) selected: {selected_operator}")  # Should print 'exploitation_operator'

