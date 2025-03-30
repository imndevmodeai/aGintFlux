import numpy as np
from scipy.stats import entropy

def calculate_shannon_entropy(psi):
    """Calculates the Shannon entropy of a quantum state."""
    probability_density = np.abs(psi)**2
    # Normalize probability density to ensure it sums to 1 (discretized)
    probability_density /= np.sum(probability_density)
    # Avoid log of zero for zero probabilities
    probability_density = probability_density[probability_density > 0]
    return -np.sum(probability_density * np.log2(probability_density))


def compute_multipartite_mutual_information(psi):
    """
    Computes a simplified version of multipartite mutual information for a given quantum state psi.
    For simplicity, this version may not fully capture genuine multipartite entanglement 
    but provides a measure of multi-component correlations.

    In this basic version, we will return shannon entropy as a stand in, as proper MMI 
    calculation is complex and context-dependent, and would require more specifics 
    about the nature of 'multipartite' in your system.

    For a more accurate and detailed MMI, you would need to:
    1. Define the subsystems: How is your system divided into multiple parts?
    2. Calculate reduced density matrices for all subsystems and combinations of subsystems.
    3. Use the formula for MMI based on von Neumann entropies of these reduced states.

    For now, we use Shannon Entropy as a proxy, which reflects the information content 
    or uncertainty of the quantum state as a whole.
    """
    # In this simplified version, we return the Shannon Entropy as a proxy for MMI
    return calculate_shannon_entropy(psi)


def superposition_state(basis_states=None, coefficients=None, num_states=2, dim=2):
    """
    Creates a quantum superposition state.
    
    Args:
        basis_states (list, optional): List of basis states to superpose. If None, uses standard basis.
        coefficients (list, optional): Coefficients for the superposition. If None, uses equal coefficients.
        num_states (int, optional): Number of states in superposition if basis_states is None. Default is 2.
        dim (int, optional): Dimension of the Hilbert space if basis_states is None. Default is 2.
        
    Returns:
        numpy.ndarray: Normalized quantum state in superposition
    """
    if basis_states is None:
        # Create standard basis states in the specified dimension
        basis_states = [np.zeros(dim) for _ in range(num_states)]
        for i in range(min(num_states, dim)):
            basis_states[i][i] = 1.0
    
    if coefficients is None:
        # Equal superposition if no coefficients provided
        coefficients = np.ones(len(basis_states)) / np.sqrt(len(basis_states))
    else:
        # Normalize coefficients
        norm = np.sqrt(np.sum(np.abs(np.array(coefficients))**2))
        coefficients = np.array(coefficients) / norm
    
    # Create superposition
    state = np.zeros_like(basis_states[0], dtype=complex)
    for coef, basis_state in zip(coefficients, basis_states):
        state += coef * basis_state
    
    return state

def entangled_state(state_type='bell', system_dims=(2, 2)):
    """
    Creates an entangled quantum state for a multi-partite system.
    
    Args:
        state_type (str, optional): Type of entangled state to create. Options: 'bell', 'ghz', 'w'.
        system_dims (tuple, optional): Dimensions of each subsystem. Default is (2, 2) for qubits.
        
    Returns:
        numpy.ndarray: Entangled quantum state
    """
    if state_type.lower() == 'bell':
        # Bell state (maximally entangled two-qubit state)
        # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
        dim = system_dims[0] * system_dims[1]
        state = np.zeros(dim, dtype=complex)
        state[0] = 1/np.sqrt(2)  # |00⟩
        state[-1] = 1/np.sqrt(2) # |11⟩
        
    elif state_type.lower() == 'ghz':
        # GHZ state for multi-qubit system
        # |GHZ⟩ = (|000...0⟩ + |111...1⟩)/√2
        total_dim = np.prod(system_dims)
        state = np.zeros(total_dim, dtype=complex)
        state[0] = 1/np.sqrt(2)       # |000...0⟩
        state[total_dim-1] = 1/np.sqrt(2)  # |111...1⟩
        
    elif state_type.lower() == 'w':
        # W state for multi-qubit system
        # |W⟩ = (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
        # where n is the number of qubits
        num_systems = len(system_dims)
        total_dim = np.prod(system_dims)
        state = np.zeros(total_dim, dtype=complex)
        
        # Set the states with exactly one excitation
        indices = []
        for i in range(num_systems):
            # Calculate index for each |000...1...000⟩ state
            # where 1 is at position i
            idx = 0
            for j in range(num_systems):
                if j == i:
                    idx += 1 * np.prod(system_dims[j+1:]) if j < num_systems-1 else 1
            indices.append(idx)
        
        # Put equal amplitude on each state
        for idx in indices:
            state[idx] = 1.0 / np.sqrt(num_systems)
    
    else:
        raise ValueError(f"Unknown entangled state type: {state_type}. Supported types: 'bell', 'ghz', 'w'")
    
    return state 