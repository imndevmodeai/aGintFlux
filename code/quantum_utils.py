# quantum_utils.py
from typing import List, TYPE_CHECKING
import numpy as np
from scipy.linalg import null_space, orth
from sklearn.decomposition import PCA
from scipy import signal

if TYPE_CHECKING:
    from code.scalable_framework import ScalableAgent


def superposition_state(n_dimensions: int, indices: List[int]) -> np.ndarray:
    """
    Generates a superposition state vector in a Hilbert space of n dimensions.

    This function creates a quantum state that is a superposition of basis states.
    The superposition is defined by specifying the indices of the basis states to be included.
    The resulting state vector is normalized to ensure it represents a valid quantum state.

    Args:
        n_dimensions (int): The dimensionality of the Hilbert space. Must be a positive integer.
        indices (List[int]): A list of indices (integers) of the basis states to be superposed.
                             Indices must be within the range [0, n_dimensions-1].

    Returns:
        np.ndarray: A numpy array representing the superposition state vector.
                    It will be a complex vector of shape (n_dimensions,).

    Raises:
        ValueError: if n_dimensions is not a positive integer.
        ValueError: if any index in indices is out of the valid range [0, n_dimensions-1].
        TypeError: if indices is not a list of integers.

    Example:
        >>> state = superposition_state(4, [1, 3])
        >>> print(state)
        [0.  0.70710678+0.j 0.  0.70710678+0.j]
        >>> np.isclose(np.linalg.norm(state), 1.0) # Check normalization
        True
    """
    if not isinstance(n_dimensions, int) or n_dimensions <= 0:
        raise ValueError("n_dimensions must be a positive integer.")
    if not isinstance(indices, list) or not all(isinstance(idx, int) for idx in indices):
        raise TypeError("indices must be a list of integers.")
    for idx in indices:
        if not 0 <= idx < n_dimensions:
            raise ValueError(f"Index {idx} out of dimension range [0, {n_dimensions-1}].")

    state_vector = np.zeros(n_dimensions, dtype=complex)
    for index in indices:
        state_vector[index] = 1.0

    normalized_state = state_vector / np.linalg.norm(state_vector)
    return normalized_state


def entangled_state(n_qubits: int) -> np.ndarray:
    """
    Generates a maximally entangled state vector for n qubits (Bell state generalization).

    For n qubits, this creates an equal superposition of all basis states where all qubits are 0
    and all qubits are 1, generalizing the Bell state concept to multiple qubits.
    The state vector is normalized.

    Args:
        n_qubits (int): The number of qubits for the entangled state. Must be a positive integer.

    Returns:
        np.ndarray: A numpy array representing the entangled state vector.
                    It will be a complex vector of shape (2**n_qubits,).

    Raises:
        ValueError: if n_qubits is not a positive integer.

    Example:
        >>> state = entangled_state(2) # 2-qubit entangled state (Bell state)
        >>> print(state)
        [0.70710678+0.j 0. +0.j 0. +0.j 0.70710678+0.j]
        >>> np.isclose(np.linalg.norm(state), 1.0) # Check normalization
        True
    """
    if not isinstance(n_qubits, int) or n_qubits <= 0:
        raise ValueError("n_qubits must be a positive integer.")

    dimension = 2**n_qubits
    state_vector = np.zeros(dimension, dtype=complex)

    state_vector[0] = 1.0  # |00...0> state
    state_vector[-1] = 1.0 # |11...1> state

    normalized_state = state_vector / np.linalg.norm(state_vector)
    return normalized_state


def compute_multipartite_mutual_information(state_vector: np.ndarray, num_partitions: int) -> float:
    """
    Computes the multipartite mutual information (MMI) for a given quantum state vector.

    MMI quantifies the total amount of correlation (both classical and quantum) in a multipartite system.
    It is calculated based on the von Neumann entropy of the total system and its partitions.
    This function assumes an equal partitioning of the system into num_partitions subsystems.

    Args:
        state_vector (np.ndarray): The quantum state vector of the multipartite system.
                                    Must be a numpy array of shape (N,) where N is the dimension
                                    of the Hilbert space (e.g., N = 2**n for n qubits).
        num_partitions (int): The number of equal partitions to divide the system into.
                              Must be a positive integer less than or equal to log2(N) if N is a power of 2,
                              or a reasonable divisor otherwise.

    Returns:
        float: The multipartite mutual information value.

    Raises:
        ValueError: if state_vector is not a numpy array.
        ValueError: if num_partitions is not a positive integer.
        ValueError: if num_partitions is greater than the number of subsystems implied by state_vector dimension.

    Example:
        >>> state = entangled_state(3) # 3-qubit entangled state
        >>> mmi_value = compute_multipartite_mutual_information(state, 3) # 3 partitions (qubits)
        >>> print(mmi_value) # doctest: +SKIP
        ... # Output will be a float value representing the MMI.
    """
    if not isinstance(state_vector, np.ndarray):
        raise ValueError("state_vector must be a numpy array.")
    if not isinstance(num_partitions, int) or num_partitions <= 0:
        raise ValueError("num_partitions must be a positive integer.")
    if num_partitions > np.log2(state_vector.shape[0]):
        raise ValueError("num_partitions cannot exceed the number of subsystems implied by state_vector dimension.")

    total_entropy = calculate_von_neumann_entropy(state_vector)
    partition_entropy_sum = 0.0

    dimension = state_vector.shape[0]
    if np.log2(dimension).is_integer(): # Check if dimension is power of 2 (e.g., for qubits)
        num_qubits = int(np.log2(dimension))
        partition_size = num_qubits // num_partitions # Integer division for equal partitions

        for i in range(num_partitions):
            start_qubit_index = i * partition_size
            end_qubit_index = (i + 1) * partition_size
            reduced_state_vector = reduce_state_vector(state_vector, list(range(start_qubit_index, end_qubit_index)), num_qubits) # Trace out qubits *not* in partition
            partition_entropy = calculate_von_neumann_entropy(reduced_state_vector)
            partition_entropy_sum += partition_entropy
    else: # For non-qubit systems, assume equal partitioning of state vector indices
        partition_size = dimension // num_partitions
        for i in range(num_partitions):
            start_index = i * partition_size
            end_index = (i + 1) * partition_size
            partition_state_vector = normalize_vector(state_vector[start_index:end_index]) # Extract and normalize partition state
            partition_entropy = calculate_shannon_entropy(partition_state_vector) # Use Shannon entropy for general vectors
            partition_entropy_sum += partition_entropy


    mmi = partition_entropy_sum - total_entropy
    return float(max(0, mmi)) # MMI should be non-negative; clamp at 0 to avoid potential numerical issues


def calculate_von_neumann_entropy(state_vector: np.ndarray) -> float:
    """
    Calculates the von Neumann entropy of a quantum state vector.

    Von Neumann entropy is the quantum analogue of Shannon entropy, measuring the
    uncertainty or mixedness of a quantum state. It is computed from the eigenvalues
    of the density matrix of the state.

    Args:
        state_vector (np.ndarray): The quantum state vector. Must be a numpy array of shape (N,).

    Returns:
        float: The von Neumann entropy value (non-negative).

    Raises:
        ValueError: if state_vector is not a numpy array.

    Example:
        >>> state = entangled_state(2) # Bell state (maximally entangled)
        >>> entropy = calculate_von_neumann_entropy(state)
        >>> print(entropy) # doctest: +SKIP
        0.6931471805599453  # Expected entropy for Bell state is ln(2)
    """
    if not isinstance(state_vector, np.ndarray):
        raise ValueError("state_vector must be a numpy array.")

    density_matrix = np.outer(state_vector, np.conjugate(state_vector)) # Compute density matrix
    eigenvalues = np.linalg.eigvalsh(density_matrix) # Eigenvalues; use eigvalsh for Hermitian matrix

    von_neumann_entropy = 0.0
    for eigenvalue in eigenvalues:
        if eigenvalue > 0: # Avoid log(0)
            von_neumann_entropy -= eigenvalue * np.log(eigenvalue) # -Tr(rho log rho)

    return float(np.real(von_neumann_entropy)) # Return real part; entropy should be real


def reduce_state_vector(full_state_vector: np.ndarray, qubit_indices_to_trace_out: List[int], total_qubits: int) -> np.ndarray:
    """
    Reduces a multipartite quantum state vector by tracing out specified qubits.

    This function performs a partial trace operation on a quantum state vector representing a system of qubits.
    It calculates the reduced density matrix by tracing out (ignoring) the qubits specified by qubit_indices_to_trace_out.
    The resulting reduced state vector represents the quantum state of the remaining subsystem.

    Args:
        full_state_vector (np.ndarray): The complete quantum state vector of the multipartite system.
                                        Must be a numpy array of shape (2**total_qubits,).
        qubit_indices_to_trace_out (List[int]): List of qubit indices (0-indexed) to be traced out.
                                                 Must be within the range [0, total_qubits-1].
        total_qubits (int): The total number of qubits in the full system.

    Returns:
        np.ndarray: The reduced state vector after tracing out the specified qubits.
                    Shape will be (2**(total_qubits - len(qubit_indices_to_trace_out)),).

    Raises:
        ValueError: if full_state_vector is not a numpy array.
        ValueError: if qubit_indices_to_trace_out is not a list of integers.
        ValueError: if any index in qubit_indices_to_trace_out is out of range.
        ValueError: if total_qubits is not consistent with the dimension of full_state_vector.

    Example:
        >>> bell_state = entangled_state(2) # 2-qubit Bell state
        >>> reduced_state = reduce_state_vector(bell_state, [0], 2) # Trace out qubit 0
        >>> print(reduced_state) # doctest: +SKIP
        [0.5+0.j 0.5+0.j] # Reduced state for qubit 1 after tracing out qubit 0
        >>> np.isclose(np.linalg.norm(reduced_state), 1.0) # Check normalization
        True
    """
    if not isinstance(full_state_vector, np.ndarray):
        raise ValueError("full_state_vector must be a numpy array.")
    if not isinstance(qubit_indices_to_trace_out, list) or not all(isinstance(idx, int) for idx in qubit_indices_to_trace_out):
        raise ValueError("qubit_indices_to_trace_out must be a list of integers.")
    for idx in qubit_indices_to_trace_out:
        if not 0 <= idx < total_qubits:
            raise ValueError(f"Qubit index {idx} out of range [0, {total_qubits-1}].")
    if full_state_vector.shape != (2**total_qubits,):
        raise ValueError("Shape of full_state_vector not consistent with total_qubits.")

    reduced_dimension = 2**(total_qubits - len(qubit_indices_to_trace_out))
    reduced_state_vector = np.zeros(reduced_dimension, dtype=complex)
    num_basis_states = 2**total_qubits

    for reduced_state_index in range(reduced_dimension):
        for traced_out_state_index in range(2**len(qubit_indices_to_trace_out)):
            coefficient_sum = 0.0 + 0.0j
            for full_state_index in range(num_basis_states):
                bitstring = format(full_state_index, '0{}b'.format(total_qubits)) # Binary string for full state index
                reduced_bitstring_part = ""
                traced_bitstring_part = ""

                # Construct bitstrings for reduced and traced-out parts based on indices
                reduced_bit_index = 0
                traced_bit_index = 0
                for qubit_index in range(total_qubits):
                    bit = bitstring[qubit_index]
                    if qubit_index not in qubit_indices_to_trace_out:
                        reduced_bitstring_part += bit
                    else:
                        traced_bitstring_part += bit

                if int(reduced_bitstring_part, 2) == reduced_state_index and int(traced_bitstring_part, 2) == traced_out_state_index:
                    coefficient_sum += full_state_vector[full_state_index] # Sum coefficients for matching basis states

            reduced_state_vector[reduced_state_index] = coefficient_sum # Assign summed coefficient to reduced state


    return normalize_vector(reduced_state_vector)


def calculate_shannon_entropy(probabilities):
    """
    Calculate the Shannon entropy of a probability distribution.
    
    Args:
        probabilities: Array of probability values
        
    Returns:
        Shannon entropy value
    """
    # Normalize the probabilities if they don't sum to 1
    prob_sum = np.sum(np.abs(probabilities))
    if prob_sum > 0 and abs(prob_sum - 1.0) > 1e-10:
        probabilities = np.abs(probabilities) / prob_sum
    
    # Ensure probabilities are valid
    if abs(np.sum(probabilities) - 1.0) > 1e-10:
        raise ValueError("Probabilities must sum to approximately 1.") # Allow for small floating point errors
        
    # Calculate entropy with a small epsilon to avoid log(0)
    epsilon = 1e-10
    entropy = -np.sum(probabilities * np.log2(probabilities + epsilon))
    return entropy


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalizes a vector to have unit norm (length).

    Args:
        vector (np.ndarray): The vector to normalize. Must be a numpy array.

    Returns:
        np.ndarray: The normalized vector.

    Raises:
        ValueError: if vector is not a numpy array.
        ValueError: if the norm of the vector is zero (cannot normalize a zero vector).

    Example:
        >>> vec = np.array([3, 4])
        >>> normalized_vec = normalize_vector(vec)
        >>> print(normalized_vec)
        [0.6 0.8]
        >>> np.linalg.norm(normalized_vec) # Check unit norm
        1.0
    """
    if not isinstance(vector, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def orthogonal_complement(basis: np.ndarray) -> np.ndarray:
    """
    Computes the orthogonal complement of a subspace spanned by the given basis vectors.

    Uses QR decomposition for robustness. If the input basis is empty, returns a basis for the full space.

    Args:
        basis (np.ndarray): Matrix whose columns form the basis of the subspace.
                            Shape (N, K) where N is the dimension of the vector space,
                            and K is the number of basis vectors (K <= N).

    Returns:
        np.ndarray: Matrix whose columns form an orthonormal basis for the orthogonal complement subspace.
                    Shape (N, L) where L = N - rank(basis). If basis is empty, returns identity matrix.

    Raises:
        ValueError: if basis is not a numpy array.
        ValueError: if basis is not 2-dimensional.

    Example:
        >>> basis_vectors = np.array([[1, 0], [1, 1], [0, 1]]).T # Basis for a subspace of R^2
        >>> complement_basis = orthogonal_complement(basis_vectors)
        >>> print(complement_basis) # doctest: +SKIP
        [] # In R^2, subspace spanned by (1,0), (1,1), (0,1) is the whole R^2; complement is empty.

        >>> basis_z_axis = np.array([[0, 0, 1]]).T # Basis for z-axis in R^3
        >>> complement_basis_xy = orthogonal_complement(basis_z_axis) # Basis for xy-plane (orthogonal complement)
        >>> print(complement_basis_xy) # doctest: +SKIP
        [[ 1.  0.]
         [ 0.  1.]
         [ 0.  0.]] # Basis for xy-plane (x and y axes)
    """
    if not isinstance(basis, np.ndarray):
        raise ValueError("Basis must be a numpy array.")
    if basis.ndim != 2:
        raise ValueError("Basis must be a 2-dimensional array (matrix).")

    if basis.size == 0: # Handle empty basis case: return basis for the full space (identity matrix)
        n_dim = basis.shape[0] if basis.shape[0] > 0 else basis.shape[1] if basis.shape[1] > 0 else 0
        if n_dim == 0:
            return np.array([])
        return np.eye(n_dim)


    Q, R = np.linalg.qr(basis) # Perform QR decomposition
    rank = np.linalg.matrix_rank(R) # Rank of basis is rank of R

    if rank >= basis.shape[0]: # Basis already spans the full space (or more); complement is empty
        return np.array([]) # Return empty array to indicate no complement basis


    complement_basis = Q[:, rank:] # Orthonormal basis for orthogonal complement is remaining columns of Q
    return complement_basis


def project_to_subspace(vector: np.ndarray, subspace_basis: np.ndarray) -> np.ndarray:
    """
    Projects a vector onto the subspace spanned by the columns of subspace_basis.

    Args:
        vector (np.ndarray): The vector to be projected. Shape (N,).
        subspace_basis (np.ndarray): Matrix whose columns form an orthonormal basis for the subspace. Shape (N, K).

    Returns:
        np.ndarray: The projection of the vector onto the subspace. Shape (N,).

    Raises:
        ValueError: if vector or subspace_basis are not numpy arrays.
        ValueError: if vector is not 1-dimensional.
        ValueError: if subspace_basis is not 2-dimensional.
        ValueError: if dimensions of vector and subspace_basis are incompatible.

    Example:
        >>> subspace_basis_xy = np.eye(3)[:, :2] # Orthonormal basis for xy-plane in R^3
        >>> vector_in_R3 = np.array([1, 2, 3])
        >>> projection_xy = project_to_subspace(vector_in_R3, subspace_basis_xy)
        >>> print(projection_xy)
        [1. 2. 0.] # Projection onto xy-plane; z-component becomes 0.
    """
    if not isinstance(vector, np.ndarray) or not isinstance(subspace_basis, np.ndarray):
        raise ValueError("Vector and subspace_basis must be numpy arrays.")
    if vector.ndim != 1:
        raise ValueError("Vector must be 1-dimensional.")
    if subspace_basis.ndim != 2:
        raise ValueError("Subspace_basis must be 2-dimensional.")
    if vector.shape[0] != subspace_basis.shape[0]:
        raise ValueError("Incompatible dimensions between vector and subspace_basis.")


    projection = subspace_basis @ (subspace_basis.T @ vector) # Projection formula using orthonormal basis
    return projection


def reshape_state_vector(state_vector: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """
    Reshapes a flattened state vector back to its original multi-dimensional shape.

    This is particularly useful for state vectors that were initially multi-dimensional,
    such as those representing multiple qubits or qudits, and were flattened for processing.

    Args:
        state_vector (np.ndarray): The flattened state vector (1-dimensional numpy array).
        original_shape (tuple[int, ...]): The original shape of the state vector before flattening.

    Returns:
        np.ndarray: The reshaped state vector with the original multi-dimensional shape.

    Raises:
        ValueError: if state_vector is not a numpy array.
        ValueError: if original_shape is not a tuple.
        ValueError: if the total size of the reshaped array does not match the size of the flattened vector.

    Example:
        >>> flattened_state = np.array([1, 2, 3, 4, 5, 6])
        >>> original_shape = (2, 3)
        >>> reshaped_state = reshape_state_vector(flattened_state, original_shape)
        >>> print(reshaped_state)
        [[1 2 3]
         [4 5 6]]
        >>> reshaped_state.shape
        (2, 3)
    """
    if not isinstance(state_vector, np.ndarray):
        raise ValueError("state_vector must be a numpy array.")
    if not isinstance(original_shape, tuple):
        raise ValueError("original_shape must be a tuple.")

    if np.prod(original_shape) != state_vector.size:
        raise ValueError("Total size of reshaped array must match flattened vector size.")

    return state_vector.reshape(original_shape)


def hyper_entangled_state(n_qubits: int, schumann_phase: float) -> np.ndarray:
    """Create entanglement state with Schumann resonance phase alignment"""
    state = entangled_state(n_qubits)
    return state * np.exp(1j * schumann_phase * np.pi/4)


def quantum_state_evolution(initial_state: np.ndarray, operator: np.ndarray, steps: int = 10) -> List[np.ndarray]:
    """
    Evolve a quantum state through time using the given operator.
    
    Args:
        initial_state: Initial quantum state vector
        operator: Evolution operator matrix
        steps: Number of evolution steps
        
    Returns:
        List of evolved state vectors at each time step
    """
    if not isinstance(initial_state, np.ndarray) or not isinstance(operator, np.ndarray):
        raise ValueError("Both initial_state and operator must be numpy arrays")
    
    if operator.shape[0] != operator.shape[1] or operator.shape[0] != len(initial_state):
        raise ValueError(f"Operator shape {operator.shape} incompatible with state dimension {len(initial_state)}")
    
    evolution = [initial_state.copy()]
    current_state = initial_state.copy()
    
    for _ in range(steps):
        # Apply operator to evolve the state
        current_state = operator @ current_state
        # Normalize the state vector
        current_state = current_state / np.linalg.norm(current_state)
        evolution.append(current_state.copy())
    
    return evolution


class QuantumResonanceValidator:
    def __init__(self, agent: "ScalableAgent"):
        self.agent = agent
        self.reference_frequency = 7.83
        
    def validate_orbital_sync(self):
        """Verify orbital resonance synchronization stability"""
        trajectory = self.agent.get_state_trajectory(np.linspace(0, 1, 100))
        freq, coherence = signal.coherence(
            trajectory.real, 
            trajectory.imag,
            fs=1/self.reference_frequency
        )
        return np.max(coherence) > 0.9

    def check_entanglement_fidelity(self):
        """Measure quantum state preservation during transmission"""
        initial_entropy = calculate_von_neumann_entropy(self.agent.initial_state)
        current_entropy = calculate_von_neumann_entropy(self.agent.current_state)
        return abs(initial_entropy - current_entropy) < 0.01


if __name__ == '__main__':
    # Example usage of superposition_state
    superpos_state = superposition_state(4, [0, 2, 3])
    print("Superposition State:\n", superpos_state)

    # Example usage of entangled_state
    entangled_2q_state = entangled_state(2)
    print("\n2-Qubit Entangled State:\n", entangled_2q_state)

    entangled_3q_state = entangled_state(3)
    print("\n3-Qubit Entangled State:\n", entangled_3q_state)

    # Example usage of compute_multipartite_mutual_information
    mmi_2q = compute_multipartite_mutual_information(entangled_state(2), 2)
    print("\nMMI for 2-Qubit Entangled State (2 partitions):", mmi_2q)

    mmi_3q = compute_multipartite_mutual_information(entangled_state(3), 3)
    print("\nMMI for 3-Qubit Entangled State (3 partitions):", mmi_3q)

    # Example usage of calculate_von_neumann_entropy
    vn_entropy_2q = calculate_von_neumann_entropy(entangled_state(2))
    print("\nVon Neumann Entropy for 2-Qubit Entangled State:", vn_entropy_2q)

    vn_entropy_3q = calculate_von_neumann_entropy(entangled_state(3))
    print("Von Neumann Entropy for 3-Qubit Entangled State:", vn_entropy_3q)

    # Example usage of reduce_state_vector
    reduced_bell = reduce_state_vector(entangled_state(2), [0], 2)
    print("\nReduced State of Bell State (Traced out qubit 0):\n", reduced_bell)

    reduced_3q = reduce_state_vector(entangled_state(3), [0, 1], 3)
    print("\nReduced State of 3Q Entangled State (Traced out qubits 0, 1):\n", reduced_3q)

    # Example usage of calculate_shannon_entropy
    fair_coin_probs = np.array([0.5, 0.5])
    shannon_entropy_fair_coin = calculate_shannon_entropy(fair_coin_probs)
    print("\nShannon Entropy for Fair Coin:", shannon_entropy_fair_coin, "bits")

    biased_coin_probs = np.array([0.9, 0.1])
    shannon_entropy_biased_coin = calculate_shannon_entropy(biased_coin_probs)
    print("Shannon Entropy for Biased Coin:", shannon_entropy_biased_coin, "bits")

    # Example usage of normalize_vector
    sample_vector = np.array([1.0, 2.0, 3.0])
    normalized_sample = normalize_vector(sample_vector)
    print("\nNormalized Vector:\n", normalized_sample)
    print("Norm of Normalized Vector:", np.linalg.norm(normalized_sample)) # Should be close to 1

    # Example usage of orthogonal_complement
    basis_example = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 0]]).T
    complement_basis_example = orthogonal_complement(basis_example)
    print("\nOrthogonal Complement Basis:\n", complement_basis_example)

    basis_z = np.array([[0, 0, 1]]).T
    complement_basis_xy = orthogonal_complement(basis_z)
    print("\nOrthogonal Complement of z-axis (xy-plane basis):\n", complement_basis_xy)

    # Example usage of project_to_subspace
    vector_r3 = np.array([4, 5, 6])
    xy_basis = np.eye(3)[:, :2] # xy-plane basis in R^3
    projected_vector_xy = project_to_subspace(vector_r3, subspace_basis_xy)
    print("\nProjected Vector onto xy-plane:\n", projected_vector_xy)

    # Example usage of reshape_state_vector
    flat_state = np.arange(1, 7) # [1 2 3 4 5 6]
    original_shape_example = (2, 3)
    reshaped_state_example = reshape_state_vector(flat_state, original_shape_example)
    print("\nReshaped State Vector:\n", reshaped_state_example)
    print("Reshaped State Vector Shape:", reshaped_state_example.shape)

    # Skip ScalableAgent examples when running standalone
    print("\nScalableAgent examples skipped when running standalone module.") 