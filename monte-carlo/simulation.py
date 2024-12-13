import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm

# Physical constants and parameters
J = 1.0      # Interaction strength
kB = 1.0     # Boltzmann constant
# Problem parameters
L_values = [10, 16, 24, 36]
T_min, T_max, T_step = 0.015, 4.5, 0.015

# Simulation parameters
equil_sweeps = 100000      # Number of sweeps for thermalization
measurement_sweeps = 300000  # Number of sweeps for measurement
measure_interval = 10        # Measure energy, magnetization every 10 sweeps

def initialize_spins(L):
    # Initialize spins randomly: returns a int32 array for Numba compatibility
    return np.random.choice([-1, 1], size=(L, L)).astype(np.int32)

@njit
def energy(spins, L):
    E = 0.0
    for i in range(L):
        for j in range(L):
            S = spins[i, j]
            SR = spins[i, (j+1)%L]     # right neighbor
            SD = spins[(i+1)%L, j]     # down neighbor
            E += -J * S * SR
            E += -J * S * SD
    return E

@njit
def energy_change(spins, L, i, j):
    S = spins[i, j]
    up = spins[(i-1)%L, j]
    down = spins[(i+1)%L, j]
    left = spins[i, (j-1)%L]
    right = spins[i, (j+1)%L]
    neighbors_sum = up + down + left + right
    # ΔE = 2 * J * S * (sum of neighbors)
    return 2.0 * J * S * neighbors_sum

@njit
def metropolis_step(spins, L, T):
    i = np.random.randint(L)
    j = np.random.randint(L)
    dE = energy_change(spins, L, i, j)
    if dE <= 0:
        spins[i, j] = -spins[i, j]
    else:
        if np.random.rand() < np.exp(-dE/(kB*T)):
            spins[i, j] = -spins[i, j]

@njit
def magnetization(spins):
    # Numba-friendly mean calculation
    return spins.mean()

# Data storage
# We'll store results in dictionaries keyed by L and then by T
results = {L: {'T': [], 'E': [], 'C': [], 'M': [], 'chi': []} for L in L_values}

# Main simulation loop
for L in L_values:
    # We'll store arrays and convert T to a float64 explicitly for Numba
    temperatures = np.arange(T_min, T_max + T_step, T_step)
    for T in tqdm(temperatures, desc=f"L={L}"):
        spins = initialize_spins(L)
        
        # Thermalization
        for _ in range(equil_sweeps):
            metropolis_step(spins, L, T)
        
        E_samples = []
        M_samples = []
        
        # Measurement
        for sweep in range(measurement_sweeps):
            metropolis_step(spins, L, T)
            if sweep % measure_interval == 0:
                currE = energy(spins, L)
                currM = spins.mean()  # using .mean() directly here is okay with Numba as well
                E_samples.append(currE)
                M_samples.append(currM)
        
        E_arr = np.array(E_samples)
        M_arr = np.array(M_samples)
        
        E_mean = E_arr.mean()
        E2_mean = (E_arr**2).mean()
        M_mean = M_arr.mean()
        M2_mean = (M_arr**2).mean()

        N = L*L
        C = (E2_mean - E_mean**2) / (N * (T**2)) # specific heat
        chi = (M2_mean - M_mean**2) / (N * T)    # susceptibility

        # Store results
        results[L]['T'].append(T)
        results[L]['E'].append(E_mean/N)   # energy per spin
        results[L]['C'].append(C)
        results[L]['M'].append(M_mean)
        results[L]['chi'].append(chi)
        
for L in L_values:
    # Plot Energy
    plt.figure()
    plt.plot(results[L]['T'], results[L]['E'], marker='o', linestyle='-')
    plt.xlabel('Temperature T')
    plt.ylabel('Energy per spin')
    plt.title(f'L={L} - Energy')

    # Plot Magnetization
    plt.figure()
    plt.plot(results[L]['T'], results[L]['M'], marker='o', linestyle='-')
    plt.xlabel('Temperature T')
    plt.ylabel('Magnetization per spin')
    plt.title(f'L={L} - Magnetization')

    # Plot Specific Heat
    plt.figure()
    plt.plot(results[L]['T'], results[L]['C'], marker='o', linestyle='-')
    plt.xlabel('Temperature T')
    plt.ylabel('Specific Heat C')
    plt.title(f'L={L} - Specific Heat')

    # Plot Susceptibility
    plt.figure()
    plt.plot(results[L]['T'], results[L]['chi'], marker='o', linestyle='-')
    plt.xlabel('Temperature T')
    plt.ylabel('Susceptibility χ')
    plt.title(f'L={L} - Susceptibility')

plt.show()  # Show all plots at the end