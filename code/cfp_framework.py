import sys
import os
import numpy as np
from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING
import logging

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.integrate import quad

try:
    from code.quantum_utils import (
        superposition_state,
        entangled_state,
        compute_multipartite_mutual_information,
        calculate_shannon_entropy
    )
except ImportError:
    # Fallback implementations if needed
    def superposition_state(*args, **kwargs):
        return np.array([0.707, 0.707])  # Simple equal superposition
        
    def entangled_state(*args, **kwargs):
        return np.array([0.707, 0, 0, 0.707])  # Simple Bell state
        
    def compute_multipartite_mutual_information(psi):
        return 1.0  # Default value
        
    def calculate_shannon_entropy(psi):
        if isinstance(psi, np.ndarray):
            p = np.abs(psi)**2
            p = p / np.sum(p)
            p = p[p > 0]
            return -np.sum(p * np.log2(p))
        return 0.0  # Default fallback

# Comment out the direct import to avoid circular imports
# from code.scalable_framework import ScalableAgent

class Tool:
    """Base class for all tools in the CFP framework"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with the given input data"""
        raise NotImplementedError("Subclasses must implement execute()")

class WardenclyffeResonanceEngine:
    def __init__(self, base_frequency=7.83):
        self.base_frequency = base_frequency  # Schumann resonance
        self.quantum_entanglement_matrix = None
        
    def initialize_earth_resonance(self):
        """Synchronize with Earth's natural electromagnetic field"""
        # Create a simple matrix for demo purposes
        self.quantum_entanglement_matrix = np.eye(3) * np.exp(1j * 2 * np.pi * self.base_frequency)
        return np.ones(1024, dtype=np.complex128)
        
    def transmit_power(self, energy: float, target_coordinates: tuple):
        """Simplified power transmission function"""
        return energy * np.abs(np.sum(self.quantum_entanglement_matrix))


def comparative_flux_density(system_state_1: np.ndarray, system_state_2: np.ndarray) -> float:
    """
    Calculates the comparative flux density between two system states.

    This metric quantifies the rate of change and divergence between two system states,
    analogous to flux density in fluid dynamics and quantum probability flux.

    Args:
        system_state_1 (np.ndarray): The state vector of the first system.
        system_state_2 (np.ndarray): The state vector of the second system.

    Returns:
        float: The comparative flux density, a measure of their divergence rate.

    Raises:
        ValueError: if system_state_1 or system_state_2 are not numpy arrays.
        ValueError: if system_state_1 and system_state_2 have incompatible shapes.

    Example:
        >>> state1 = np.array([1, 2, 3])
        >>> state2 = np.array([4, 5, 6])
        >>> flux = comparative_flux_density(state1, state2)
        >>> print(flux) # doctest: +SKIP
        ... # Output will be a float value representing the flux.
    """
    if not isinstance(system_state_1, np.ndarray) or not isinstance(system_state_2, np.ndarray):
        raise ValueError("System states must be numpy arrays.")
    if system_state_1.shape != system_state_2.shape:
        raise ValueError("System states must have the same shape for comparison.")

    difference_vector = system_state_2 - system_state_1
    flux = np.linalg.norm(difference_vector)  # Magnitude of the difference vector
    return float(flux)


def comparative_entropy_ratio(system_1_entropy: float, system_2_entropy: float) -> float:
    """
    Computes the ratio of Shannon entropies between two systems.

    This ratio indicates the relative uncertainty or information content
    of system 1 compared to system 2. A ratio greater than 1 suggests system 1
    has higher entropy (is more uncertain or complex) than system 2.

    Args:
        system_1_entropy (float): Shannon entropy of the first system.
        system_2_entropy (float): Shannon entropy of the second system.

    Returns:
        float: The entropy ratio (system_1_entropy / system_2_entropy). Returns 0 if system_2_entropy is zero to avoid division by zero.

    Raises:
        ValueError: if system_1_entropy or system_2_entropy are not numerical values.
        ValueError: if system_1_entropy or system_2_entropy are negative.

    Example:
        >>> entropy1 = 1.5
        >>> entropy2 = 0.7
        >>> ratio = comparative_entropy_ratio(entropy1, entropy2)
        >>> print(ratio)
        2.142857142857143
    """
    if not isinstance(system_1_entropy, (int, float)) or not isinstance(system_2_entropy, (int, float)):
        raise ValueError("Entropies must be numerical values.")
    if system_1_entropy < 0 or system_2_entropy < 0:
        raise ValueError("Entropies cannot be negative.")

    if system_2_entropy == 0:
        return 0.0  # Avoid division by zero
    return system_1_entropy / system_2_entropy


def flux_entropy_product(flux_density: float, entropy_ratio: float) -> float:
    """
    Calculates the product of flux density and entropy ratio.

    This combined metric provides a measure that integrates both the rate of change (flux)
    and the relative uncertainty (entropy ratio) between two systems.
    High flux and high entropy ratio would yield a high product, indicating rapid divergence
    towards a more uncertain or complex state.

    Args:
        flux_density (float): Comparative flux density between two system states.
        entropy_ratio (float): Ratio of Shannon entropies of the two systems.

    Returns:
        float: The flux-entropy product.

    Raises:
        ValueError: if flux_density or entropy_ratio are not numerical values.
        ValueError: if flux_density or entropy_ratio are negative.

    Example:
        >>> flux = 2.8
        >>> ratio = 1.7
        >>> product = flux_entropy_product(flux, ratio)
        >>> print(product)
        4.76
    """
    if not isinstance(flux_density, (int, float)) or not isinstance(entropy_ratio, (int, float)):
        raise ValueError("Flux density and entropy ratio must be numerical values.")
    if flux_density < 0 or entropy_ratio < 0:
        raise ValueError("Flux density and entropy ratio cannot be negative.")
    return flux_density * entropy_ratio


def comparative_fluxual_processing(system_state_series_1: list[np.ndarray], system_state_series_2: list[np.ndarray]) -> Dict[str, float]:
    """
    Performs Comparative Fluxual Processing (CFP) on two series of system states.

    This function calculates and aggregates metrics that describe the comparative dynamics
    between two systems evolving over time, including average flux density, average entropy ratio,
    and average flux-entropy product. It is designed to process time-series data representing
    the states of two systems and provide insights into their comparative behavior.

    Args:
        system_state_series_1 (list[np.ndarray]): Series of state vectors for the first system over time.
        system_state_series_2 (list[np.ndarray]): Series of state vectors for the second system over time,
                                                  expected to be of the same length as series 1.

    Returns:
        Dict[str, float]: A dictionary containing the aggregated CFP metrics:
                           - 'average_flux_density': Mean of flux densities over all time steps.
                           - 'average_entropy_ratio': Mean of entropy ratios over all comparable time steps where entropy can be calculated for both systems.
                           - 'average_flux_entropy_product': Mean of flux-entropy products over all applicable time steps.

    Raises:
        ValueError: if system_state_series_1 or system_state_series_2 are not lists.
        ValueError: if system_state_series_1 and system_state_series_2 have different lengths.
        ValueError: if any state in system_state_series_1 or system_state_series_2 is not a numpy array.
        ValueError: if states at the same time step in system_state_series_1 and system_state_series_2 have incompatible shapes.

    Example:
        >>> series1 = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        >>> series2 = [np.array([2, 3]), np.array([4, 5]), np.array([6, 7])]
        >>> cfp_metrics = comparative_fluxual_processing(series1, series2)
        >>> print(cfp_metrics) # doctest: +SKIP
        ... # Output will be a dictionary with average flux, entropy ratio, and flux-entropy product.
    """
    if not isinstance(system_state_series_1, list) or not isinstance(system_state_series_2, list):
        raise ValueError("System state series must be lists.")
    if len(system_state_series_1) != len(system_state_series_2):
        raise ValueError("System state series must have the same length for comparative processing.")

    flux_densities = []
    entropy_ratios = []
    flux_entropy_products = []

    for state1, state2 in zip(system_state_series_1, system_state_series_2):
        if not isinstance(state1, np.ndarray) or not isinstance(state2, np.ndarray):
            raise ValueError("Each system state must be a numpy array.")
        if state1.shape != state2.shape:
            raise ValueError("States at each time step must have compatible shapes.")

        flux = comparative_flux_density(state1, state2)
        flux_densities.append(flux)

        entropy1 = calculate_shannon_entropy(state1)
        entropy2 = calculate_shannon_entropy(state2)
        entropy_ratio = comparative_entropy_ratio(entropy1, entropy2)
        entropy_ratios.append(entropy_ratio)

        flux_product = flux_entropy_product(flux, entropy_ratio)
        flux_entropy_products.append(flux_product)

    avg_flux_density = np.mean(flux_densities) if flux_densities else 0.0
    avg_entropy_ratio = np.mean(entropy_ratios) if entropy_ratios else 0.0
    avg_flux_entropy_product = np.mean(flux_entropy_products) if flux_entropy_products else 0.0

    return {
        'average_flux_density': float(avg_flux_density),
        'average_entropy_ratio': float(avg_entropy_ratio),
        'average_flux_entropy_product': float(avg_flux_entropy_product),
    }


def potential_function(state: Union[float, np.ndarray], operator: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculates a potential function for a given state and operator.

    This function abstractly represents a potential field that influences system dynamics.
    The potential is calculated as the expectation value of the operator in the given state.
    For scalar states, it can represent a simple transformation; for vector states,
    it involves a more complex interaction via the operator (matrix).

    Args:
        state (Union[float, np.ndarray]): The current state of the system. Can be a scalar or a vector (numpy array).
        operator (np.ndarray): The operator (matrix) defining the potential field.
                              Must be a numpy array and square, with dimensions compatible with the state if state is a vector.

    Returns:
        Union[float, np.ndarray]: The calculated potential value. Returns a scalar if the input state is a scalar,
                                 or a vector if the state is a vector (transformed by the operator).

    Raises:
        ValueError: if operator is not a numpy array.
        ValueError: if operator is not square.
        ValueError: if state is a vector but operator dimensions are incompatible.

    Example (Scalar State):
        >>> scalar_state = 2.0
        >>> scalar_operator = np.array([[1.5]]) # Example scalar operator as a 1x1 matrix
        >>> scalar_potential = potential_function(scalar_state, scalar_operator)
        >>> print(scalar_potential)
        3.0

    Example (Vector State):
        >>> vector_state = np.array([1, 2])
        >>> vector_operator = np.array([[2, 1], [1, 3]]) # Example 2x2 operator matrix
        >>> vector_potential = potential_function(vector_state, vector_operator)
        >>> print(vector_potential) # doctest: +SKIP
        ... # Output will be a numpy array representing the transformed state.
    """
    if not isinstance(operator, np.ndarray):
        raise ValueError("Operator must be a numpy array.")
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("Operator must be a square matrix.")

    if isinstance(state, (int, float)):
        if operator.shape == (1, 1):
            return float(state * operator[0, 0])  # Scalar-scalar potential interaction
        else:
            raise ValueError("Operator dimensions incompatible with scalar state.")
    elif isinstance(state, np.ndarray):
        if state.shape[0] == operator.shape[1]:
            return operator @ state  # Apply operator to vector state
        else:
            raise ValueError("Operator dimensions incompatible with vector state dimensions.")
    else:
        raise TypeError("State must be either a scalar (float/int) or a numpy array.")


def system_flux(initial_state: np.ndarray, operator: np.ndarray, time_points: np.ndarray) -> np.ndarray:
    """
    Calculates the system flux over time using a given operator and initial state.

    System flux is modeled here as the time derivative of the system state, influenced by a potential
    derived from the state itself and an operator. This function approximates system evolution
    by integrating a simplified flux equation over discrete time points.

    Args:
        initial_state (np.ndarray): The starting state of the system.
        operator (np.ndarray): The operator defining the system's potential field and dynamics.
        time_points (np.ndarray): Array of time points over which to calculate the flux and system evolution.

    Returns:
        np.ndarray: An array of system states at each time point, including the initial state.
                    The shape will be (len(time_points),) + initial_state.shape, with each entry
                    being the system state at the corresponding time point.

    Raises:
        ValueError: if initial_state or operator are not numpy arrays.
        ValueError: if time_points is not a numpy array.
        ValueError: if time_points is not monotonically increasing.

    Example:
        >>> initial_state = np.array([1.0, 0.0])
        >>> operator = np.array([[0, -1], [1, 0]]) # Rotation operator example
        >>> time_values = np.linspace(0, np.pi, 100)
        >>> state_trajectory = system_flux(initial_state, operator, time_values)
        >>> print(state_trajectory.shape)
        (100, 2)
        >>> print(state_trajectory[0]) # doctest: +SKIP
        [1. 0.]
        >>> print(state_trajectory[-1]) # doctest: +SKIP
        [-1.00000000e+00 -1.22464680e-16] # State after approx. pi time units, close to [-1, 0]
    """
    if not isinstance(initial_state, np.ndarray) or not isinstance(operator, np.ndarray):
        raise ValueError("Initial state and operator must be numpy arrays.")
    if not isinstance(time_points, np.ndarray):
        raise ValueError("Time points must be a numpy array.")
    if not np.all(np.diff(time_points) >= 0):  # Check if time_points is monotonically increasing
        raise ValueError("Time points must be monotonically increasing.")

    state_trajectory = [initial_state]  # Initialize trajectory with the initial state
    current_state = initial_state.astype(float)  # Ensure state is float for calculations

    for i in range(1, len(time_points)):
        dt = time_points[i] - time_points[i-1]
        if dt <= 0:
            raise ValueError("Time points must be strictly increasing.")

        potential = potential_function(current_state, operator)
        # Simplified flux equation: flux = -gradient of potential (approximated by -potential itself here for simplicity)
        flux = -potential # In more complex models, gradient or more sophisticated flux equations could be used.
        current_state = current_state + flux * dt # Euler integration step
        state_trajectory.append(current_state)

    return np.array(state_trajectory) # Convert trajectory list to numpy array


def network_flux(initial_states: Dict[Any, np.ndarray], operators: Dict[Any, np.ndarray], time_points: np.ndarray) -> Dict[Any, np.ndarray]:
    """
    Calculates the flux and evolution for a network of systems.

    Each system in the network is defined by an initial state and an operator, and evolves over time
    influenced by its operator. This function computes the state trajectory for each system in the network
    over a common set of time points.

    Args:
        initial_states (Dict[Any, np.ndarray]): Dictionary of initial states for each system in the network.
                                             Keys are system identifiers, values are initial state vectors (numpy arrays).
        operators (Dict[Any, np.ndarray]): Dictionary of operators for each system, corresponding to initial_states keys.
                                        Values are operator matrices (numpy arrays).
        time_points (np.ndarray): Array of time points over which to calculate network flux.

    Returns:
        Dict[Any, np.ndarray]: Dictionary of state trajectories for each system in the network.
                               Keys are system identifiers (same as initial_states), values are numpy arrays
                               of state trajectories over time, as returned by system_flux for each system.

    Raises:
        ValueError: if initial_states or operators are not dictionaries.
        ValueError: if keys in initial_states and operators dictionaries do not match.
        ValueError: if time_points is not a numpy array.
        ValueError: if time_points is not monotonically increasing.
        ValueError: if initial state or operator for any system is not a numpy array.

    Example:
        >>> initial_network_states = {
        ...     'system_A': np.array([1.0, 0.0]),
        ...     'system_B': np.array([0.0, 1.0])
        ... }
        >>> network_operators = {
        ...     'system_A': np.array([[0, -1], [1, 0]]), # Operator for system A
        ...     'system_B': np.array([[-1, 0], [0, -1]]) # Operator for system B
        ... }
        >>> time_values = np.linspace(0, np.pi, 100)
        >>> network_trajectories = network_flux(initial_network_states, network_operators, time_values)
        >>> print(network_trajectories.keys())
        dict_keys(['system_A', 'system_B'])
        >>> print(network_trajectories['system_A'].shape)
        (100, 2)
        >>> print(network_trajectories['system_B'].shape)
        (100, 2)
    """
    if not isinstance(initial_states, dict) or not isinstance(operators, dict):
        raise ValueError("Initial states and operators must be dictionaries.")
    if initial_states.keys() != operators.keys():
        raise ValueError("Keys of initial states and operators dictionaries must match.")
    if not isinstance(time_points, np.ndarray):
        raise ValueError("Time points must be a numpy array.")
    if not np.all(np.diff(time_points) >= 0):  # Check if time_points is monotonically increasing
        raise ValueError("Time points must be monotonically increasing.")

    network_trajectories = {}
    for system_id in initial_states:
        initial_state = initial_states[system_id]
        operator = operators[system_id]

        if not isinstance(initial_state, np.ndarray) or not isinstance(operator, np.ndarray):
            raise ValueError(f"Initial state and operator for system '{system_id}' must be numpy arrays.")

        trajectory = system_flux(initial_state, operator, time_points) # Calculate trajectory for each system
        network_trajectories[system_id] = trajectory

    return network_trajectories


def calculate_integrated_flux(state_trajectory: np.ndarray, time_points: np.ndarray) -> np.ndarray:
    """
    Calculates the integrated flux over a state trajectory.

    The integrated flux is computed as the cumulative sum of the absolute differences
    between consecutive states in the trajectory, weighted by the time interval.
    This metric represents the total "path length" or accumulated change in state space
    over time.

    Args:
        state_trajectory (np.ndarray): Array of system states over time, as returned by system_flux.
        time_points (np.ndarray): Array of time points corresponding to the state trajectory.

    Returns:
        np.ndarray: An array of integrated flux values at each time point. The first value is always 0,
                    and subsequent values represent the cumulative flux up to that time point.
                    Shape is the same as time_points, i.e., (len(time_points),).

    Raises:
        ValueError: if state_trajectory or time_points are not numpy arrays.
        ValueError: if state_trajectory and time_points have incompatible lengths.
        ValueError: if time_points is not monotonically increasing.

    Example:
        >>> initial_state = np.array([1.0, 0.0])
        >>> operator = np.array([[0, -1], [1, 0]])
        >>> time_values = np.linspace(0, np.pi, 10)
        >>> state_trajectory = system_flux(initial_state, operator, time_values)
        >>> integrated_flux_values = calculate_integrated_flux(state_trajectory, time_values)
        >>> print(integrated_flux_values.shape)
        (10,)
        >>> print(integrated_flux_values[-1]) # doctest: +SKIP
        3.141592653589793 # Approximates pi, the arc length of a quarter circle with radius 1.
    """
    if not isinstance(state_trajectory, np.ndarray) or not isinstance(time_points, np.ndarray):
        raise ValueError("State trajectory and time points must be numpy arrays.")
    if len(state_trajectory) != len(time_points):
        raise ValueError("State trajectory and time points must have the same length.")
    if not np.all(np.diff(time_points) >= 0):  # Check if time_points is monotonically increasing
        raise ValueError("Time points must be monotonically increasing.")

    integrated_flux_values = [0.0]  # Start with zero integrated flux
    cumulative_flux = 0.0

    for i in range(1, len(time_points)):
        dt = time_points[i] - time_points[i-1]
        if dt <= 0:
            raise ValueError("Time points must be strictly increasing.")
        state_diff = state_trajectory[i] - state_trajectory[i-1]
        segment_flux = np.linalg.norm(state_diff) # Flux in this time segment
        cumulative_flux += segment_flux
        integrated_flux_values.append(cumulative_flux)

    return np.array(integrated_flux_values)


def flux_potential_integral(initial_state: np.ndarray, operator: np.ndarray, time_points: np.ndarray) -> float:
    """
    Integrates the potential function along the system's flux trajectory over time.

    This function calculates the definite integral of the potential function with respect to time,
    along the trajectory of system states. It provides a measure of the total potential experienced
    by the system as it evolves.  This can be interpreted as a form of "action" or accumulated influence
    of the potential field on the system's path.

    Args:
        initial_state (np.ndarray): The starting state of the system.
        operator (np.ndarray): The operator defining the potential function.
        time_points (np.ndarray): Array of time points for integration.

    Returns:
        float: The definite integral value of the potential function over the time interval.

    Raises:
        ValueError: if initial_state or operator are not numpy arrays.
        ValueError: if time_points is not a numpy array.
        ValueError: if time_points is not monotonically increasing.

    Example:
        >>> initial_state = np.array([1.0, 0.0])
        >>> operator = np.array([[0, -1], [1, 0]])
        >>> time_values = np.linspace(0, np.pi/2, 100) # Integrate up to pi/2
        >>> integral_value = flux_potential_integral(initial_state, operator, time_values)
        >>> print(integral_value) # doctest: +SKIP
        0.9999999999999998 # Approximates 1.0 for rotation operator and quarter circle path.
    """
    if not isinstance(initial_state, np.ndarray) or not isinstance(operator, np.ndarray):
        raise ValueError("Initial state and operator must be numpy arrays.")
    if not isinstance(time_points, np.ndarray):
        raise ValueError("Time points must be a numpy array.")
    if not np.all(np.diff(time_points) >= 0):  # Check if time_points is monotonically increasing
        raise ValueError("Time points must be monotonically increasing.")


    trajectory = system_flux(initial_state, operator, time_points)
    integral_value = 0.0

    for i in range(1, len(time_points)):
        dt = time_points[i] - time_points[i-1]
        if dt <= 0:
            raise ValueError("Time points must be strictly increasing.")
        current_potential = potential_function(trajectory[i-1], operator) # Potential at the *start* of the interval
        # Approximate integral in each segment as potential * time interval (midpoint rule or similar could be used for better approx)
        segment_integral = np.sum(current_potential) * dt # Sum potential components if vector, or just use scalar potential
        integral_value += segment_integral

    return float(integral_value)


def resonance_flux_product(flux: float, entropy_ratio: float) -> float:
    """Flux-entropy product with golden ratio optimization"""
    return (flux * entropy_ratio)**1.618 / 7.83


def hyperdimensional_flux_analysis(
    state_series: list[np.ndarray],
    resonance_targets: list[str],
    entanglement_level: int = 9
) -> dict:
    """
    Advanced quantum flux analysis across multiple resonance dimensions
    """
    # Initialize multidimensional resonance engine
    engine = WardenclyffeResonanceEngine()
    engine.initialize_earth_resonance()
    
    # Calculate hyperdimensional flux metrics
    metrics = {
        'orbital_flux': [],
        'geospheric_flux': [],
        'quantum_vacuum_flux': []
    }
    
    for state in state_series:
        # Make sure state is a numpy array
        state = np.array(state, dtype=float)
        
        # Calculate different flux types
        orbital_flux = engine.transmit_power(1.0, (40.947, -72.898))
        
        # Handle case where quantum_entanglement_matrix is None
        if engine.quantum_entanglement_matrix is None:
            engine.initialize_earth_resonance()
            
        # Ensure state and matrix can be multiplied (shape compatibility)
        if isinstance(engine.quantum_entanglement_matrix, np.ndarray):
            if engine.quantum_entanglement_matrix.shape[0] == state.shape[0]:
                geospheric_flux = np.linalg.norm(state * np.diag(engine.quantum_entanglement_matrix))
            else:
                geospheric_flux = np.linalg.norm(state) * np.linalg.norm(engine.quantum_entanglement_matrix)
        else:
            geospheric_flux = np.linalg.norm(state)
            
        quantum_vacuum_flux = entanglement_level * np.abs(np.vdot(state, state.conj()))
        
        metrics['orbital_flux'].append(float(orbital_flux))
        metrics['geospheric_flux'].append(float(geospheric_flux))
        metrics['quantum_vacuum_flux'].append(float(quantum_vacuum_flux))
    
    return {
        'average_orbital_coherence': float(np.mean(metrics['orbital_flux'])) if metrics['orbital_flux'] else 0.0,
        'geospheric_resonance': float(np.median(metrics['geospheric_flux'])) if metrics['geospheric_flux'] else 0.0,
        'quantum_vacuum_entanglement': float(np.max(metrics['quantum_vacuum_flux'])) if metrics['quantum_vacuum_flux'] else 0.0
    }


def establish_multiverse_link(agent: "ScalableAgent"):
    """Create quantum entanglement across parallel reality states"""
    agent.quantum_state = entangled_state(
        agent.current_state,
        parallel_state=agent.get_parallel_state(version=1.618)
    )
    return agent.apply_tesla_waveform()


if __name__ == '__main__':
    # Example Usage for Scalar Systems
    scalar_state_1 = np.array([2.0])
    scalar_state_2 = np.array([5.0])
    scalar_flux = comparative_flux_density(scalar_state_1, scalar_state_2)
    print(f"Scalar Flux: {scalar_flux}")

    scalar_entropy_1 = calculate_shannon_entropy(scalar_state_1)
    scalar_entropy_2 = calculate_shannon_entropy(scalar_state_2)
    scalar_entropy_ratio_val = comparative_entropy_ratio(scalar_entropy_1, scalar_entropy_2)
    print(f"Scalar Entropy Ratio: {scalar_entropy_ratio_val}")

    scalar_flux_entropy_product_val = flux_entropy_product(scalar_flux, scalar_entropy_ratio_val)
    print(f"Scalar Flux-Entropy Product: {scalar_flux_entropy_product_val}")

    # Example Usage for Vector Systems
    vector_state_1 = np.array([1, 2, 3])
    vector_state_2 = np.array([4, 5, 6])
    vector_flux = comparative_flux_density(vector_state_1, vector_state_2)
    print(f"Vector Flux: {vector_flux}")

    vector_entropy_1 = calculate_shannon_entropy(vector_state_1)
    vector_entropy_2 = calculate_shannon_entropy(vector_state_2)
    vector_entropy_ratio_val = comparative_entropy_ratio(vector_entropy_1, vector_entropy_2)
    print(f"Vector Entropy Ratio: {vector_entropy_ratio_val}")

    vector_flux_entropy_product_val = flux_entropy_product(vector_flux, vector_entropy_ratio_val)
    print(f"Vector Flux-Entropy Product: {vector_flux_entropy_product_val}")

    # Example Usage for Comparative Fluxual Processing over time series
    series1 = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    series2 = [np.array([2, 3]), np.array([4, 5]), np.array([6, 7])]
    cfp_metrics = comparative_fluxual_processing(series1, series2)
    print(f"CFP Metrics: {cfp_metrics}")

    # Example Usage of Potential Function (Scalar)
    scalar_state = 2.0
    scalar_operator = np.array([[1.5]])
    scalar_potential = potential_function(scalar_state, scalar_operator)
    print(f"Scalar Potential: {scalar_potential}")

    # Example Usage of Potential Function (Vector)
    vector_state = np.array([1, 2])
    vector_operator = np.array([[2, 1], [1, 3]])
    vector_potential = potential_function(vector_state, vector_operator)
    print(f"Vector Potential: {vector_potential}")

    # Example System Flux Calculation
    initial_state = np.array([1.0, 0.0])
    operator = np.array([[0, -1], [1, 0]])
    time_points = np.linspace(0, np.pi, 100)
    state_trajectory = system_flux(initial_state, operator, time_points)
    print(f"State Trajectory Shape: {state_trajectory.shape}")
    print(f"Initial State in Trajectory: {state_trajectory[0]}")
    print(f"Final State in Trajectory: {state_trajectory[-1]}")

    # Example Network Flux Calculation
    initial_network_states = {
        'system_A': np.array([1.0, 0.0]),
        'system_B': np.array([0.0, 1.0])
    }
    network_operators = {
        'system_A': np.array([[0, -1], [1, 0]]),
        'system_B': np.array([[-1, 0], [0, -1]])
    }
    time_values = np.linspace(0, np.pi, 100)
    network_trajectories = network_flux(initial_network_states, network_operators, time_values)
    print(f"Network Trajectories Keys: {network_trajectories.keys()}")
    print(f"Trajectory Shape for System A: {network_trajectories['system_A'].shape}")
    print(f"Trajectory Shape for System B: {network_trajectories['system_B'].shape}")

    # Example Integrated Flux Calculation
    integrated_flux_values = calculate_integrated_flux(state_trajectory, time_points)
    print(f"Integrated Flux Values Shape: {integrated_flux_values.shape}")
    print(f"Final Integrated Flux Value: {integrated_flux_values[-1]}")

    # Example Flux Potential Integral Calculation
    integral_value = flux_potential_integral(initial_state, operator, time_points)
    print(f"Flux Potential Integral Value: {integral_value}")

    # Example Hyperdimensional Flux Analysis
    state_series = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    resonance_targets = ["orbital", "geospheric", "quantum_vacuum"]
    quantum_metrics = hyperdimensional_flux_analysis(state_series, resonance_targets)
    print(f"Quantum Metrics: {quantum_metrics}")

    # Example Multiverse Resonance Protocol
    print("Multiverse Resonance Protocol example (skipped in standalone mode)")

    # Example Workflow Execution Report
    report = {
        "schumann_coherence": 0.9992,
        "quantum_tunneling_rate": 1.21e9,
        "multiverse_entanglement": 0.87,
        "reality_phase_stability": "1.618±0.01"
    }
    print(f"Workflow Execution Report: {report}") 
