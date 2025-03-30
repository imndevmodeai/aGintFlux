import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Quantum utils (assuming this is defined in quantum_utils.py)
def calculate_shannon_entropy(psi):
    """Calculates the Shannon entropy of a quantum state."""
    probability_density = np.abs(psi)**2
    # Normalize probability density to ensure it sums to 1 (discretized)
    probability_density /= np.sum(probability_density)
    # Avoid log of zero for zero probabilities
    probability_density = probability_density[probability_density > 0]
    return -np.sum(probability_density * np.log2(probability_density))


# Constants
hbar = 1.0          # Reduced Planck constant
m = 1.0             # Mass
omega = 1.0         # Angular frequency of the harmonic oscillator
t_max = 20.0        # Maximum time
pulse_amplitude = 5.0 # Amplitude of Gaussian pulses
pulse_width = 0.5     # Width of Gaussian pulses
pulse_periods_to_scan = np.linspace(0.5, 10.0, 50)  # Scan pulse periods from 0.5 to 10.0


# Time points
dt = 0.01
t = np.arange(0, t_max, dt)

# Spatial grid
x = np.linspace(-10, 10, 1000)

# Initial state (ground state of harmonic oscillator)
psi0 = np.exp(-x**2 / (2 * 1**2))
psi0 = psi0 / np.sqrt(np.sum(np.abs(psi0)**2) * (x[1] - x[0]))


# Function to calculate Hamiltonian with pulsed Tesla-inspired waveform
def hamiltonian_pulsed(time, pulse_period):
    pulse_potential = np.zeros_like(x)
    for n in range(int(t_max / pulse_period) + 2):
        pulse_potential += pulse_amplitude * np.exp(-((time - n * pulse_period)**2) / (2 * pulse_width**2))
    potential = 0.5 * m * omega**2 * x**2 + pulse_potential
    return np.diag(potential)


# Lists to store final Shannon Entropy and Energy Expectation vs pulse period
final_entropy_values = []
final_energy_expectation_values = []

# Dictionary to store entropy values versus period
entropy_values_vs_period = {}

# Scan over different pulse periods
for pulse_period in pulse_periods_to_scan:
    print(f"Simulating for pulse period T = {pulse_period:.2f}")
    psi = np.zeros((len(x), len(t)), dtype=np.complex128)
    psi[:, 0] = psi0
    shannon_entropy_over_time = [] # To store entropy at each time step

    for time_index in range(1, len(t)):
        current_time = t[time_index]
        H = hamiltonian_pulsed(current_time, pulse_period)
        U = expm(-1j * H * dt / hbar)
        psi[:, time_index] = np.dot(U, psi[:, time_index-1])
        shannon_entropy_over_time.append(calculate_shannon_entropy(psi[:, time_index]))


    entropy_values_vs_period[pulse_period] = shannon_entropy_over_time # Store entropy values

    # Plotting probability density for the last time step for each period (for visual comparison)
    plt.figure(figsize=(8, 6))
    plt.plot(x, np.abs(psi[:, -1])**2) # Plot final state
    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title(f"Final State Probability Density (Pulse Period T = {pulse_period})")
    plt.show()


# Plot Shannon Entropy evolution for different pulse periods
plt.figure(figsize=(12, 8))
for pulse_period in pulse_periods_to_scan:
    plt.plot(t[1:], entropy_values_vs_period[pulse_period], label=f'Pulse Period T = {pulse_period}')

plt.xlabel("Time")
plt.ylabel("Shannon Entropy")
plt.title("Shannon Entropy Evolution for Different Pulse Periods")
plt.legend()
plt.grid(True)
plt.show() 