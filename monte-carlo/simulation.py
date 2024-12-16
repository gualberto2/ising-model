import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ising_helpers import (
    simulate_thermalization,
    simulate_measurement,
    initialize_spins,
    total_energy,
    total_magnetization,
)

# Physical constants and parameters
J = 1.0      # Interaction strength
kB = 1.0     # Boltzmann constant

# Problem parameters
L_values = [10, 16, 24, 36]
T_min, T_max, T_step = 0.015, 4.5, 0.015

# Simulation parameters
equil_sweeps = 100_000  # Reduced to 1e5 as per guidelines
measurement_sweeps = 300_000  # Number of sweeps for measurement
measure_interval = 10  # Measure energy, magnetization every 10 sweeps

def remove_outliers_iqr(data, lower_percentile=25, upper_percentile=75, multiplier=1.5):
    """
    Remove only the most extreme outliers by using a tighter multiplier. 
    This should preserve the main distribution while removing very rare events 
    that might skew the scale.
    """
    if len(data) == 0:
        return data
    Q1 = np.percentile(data, lower_percentile)
    Q3 = np.percentile(data, upper_percentile)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    print(f"Outlier Removal: {len(data) - len(filtered_data)} outliers removed.")
    return filtered_data

# Data storage
results = {L: {'T': [], 'E': [], 'C': [], 'M': [], 'chi': []} for L in L_values}

# Main simulation loop
for L in L_values:
    N = L * L  # Total number of spins

    critical_temperature = 2.27

    temperatures = np.concatenate([
        np.arange(T_min, critical_temperature - 0.5, T_step),
        np.arange(critical_temperature - 0.5, critical_temperature + 0.5, T_step / 2),
        np.arange(critical_temperature + 0.5, T_max + T_step, T_step)
    ])

    for T in tqdm(temperatures, desc=f"L={L}"):
        # Initialize spins
        spins = initialize_spins(L)

        # Thermalization
        simulate_thermalization(spins, L, T, equil_sweeps + L**2)

        # Measurement
        E_samples, M_samples = simulate_measurement(spins, L, T, measurement_sweeps, measure_interval)

        # Convert lists to NumPy arrays for outlier removal and statistics
        E_arr = np.array(E_samples, dtype=np.float64)
        M_arr = np.array(M_samples, dtype=np.float64)

        # Remove outliers with a higher multiplier
        E_arr = remove_outliers_iqr(E_arr, multiplier=1.5)
        M_arr = remove_outliers_iqr(M_arr, multiplier=1.5)

        # Compute averages
        E_mean = E_arr.mean()
        E2_mean = (E_arr**2).mean()
        M_mean = M_arr.mean()
        M2_mean = (M_arr**2).mean()

        # Specific heat and susceptibility
        C = (E2_mean - E_mean**2) / (N * (T**2))
        chi = (M2_mean - M_mean**2) / (N * T)

        # Store results
        results[L]['T'].append(T)
        results[L]['E'].append(E_mean / N)        # Energy per spin
        results[L]['C'].append(C)
        results[L]['M'].append(M_mean / N)        # Magnetization per spin
        results[L]['chi'].append(chi)

for L in L_values:
    # Plot Energy
    plt.figure()
    plt.plot(results[L]['T'], results[L]['E'], marker='o', linestyle='none')
    plt.xlabel('Temperature T')
    plt.ylabel('Energy per spin')
    plt.title(f'L={L} - Energy')

    # Plot Magnetization
    plt.figure()
    plt.plot(results[L]['T'], np.abs(results[L]['M']), marker='o', linestyle='none')
    plt.xlabel('Temperature T')
    plt.ylabel('Magnetization per spin')
    plt.title(f'L={L} - Magnetization')

    # Plot Specific Heat
    plt.figure()
    plt.plot(results[L]['T'], results[L]['C'], marker='o', linestyle='none')
    plt.xlabel('Temperature T')
    plt.ylabel('Specific Heat C')
    plt.title(f'L={L} - Specific Heat')

    # Plot Susceptibility
    plt.figure()
    plt.plot(results[L]['T'], results[L]['chi'], marker='o', linestyle='none')
    plt.xlabel('Temperature T')
    plt.ylabel('Susceptibility χ')
    plt.title(f'L={L} - Susceptibility')

    # Print average values
    avg_energy = np.mean(results[L]['E'])
    avg_magnetization = np.mean(results[L]['M'])
    avg_specific_heat = np.mean(results[L]['C'])
    avg_susceptibility = np.mean(results[L]['chi'])

    print(f"L={L}:")
    print(f"  Average Energy per site: {avg_energy:.5f}")
    print(f"  Average Magnetization per site: {avg_magnetization:.5f}")
    print(f"  Average Specific Heat: {avg_specific_heat:.5f}")
    print(f"  Average Susceptibility: {avg_susceptibility:.5f}")
    print("-" * 30)

plt.show()  # Show all plots at the end

# Create and print summary table
summary = {}
for L in L_values:
    summary[L] = {
        'avg_energy': np.mean(results[L]['E']),
        'avg_magnetization': np.mean(results[L]['M']),
        'avg_specific_heat': np.mean(results[L]['C']),
        'avg_susceptibility': np.mean(results[L]['chi']),
    }

print(f"{'L':>5} {'<E>':>15} {'<|M|>':>15} {'<C>':>15} {'<χ>':>15}")
for L, stats in summary.items():
    print(f"{L:>5} {stats['avg_energy']:>15.5f} {stats['avg_magnetization']:>15.5f} "
          f"{stats['avg_specific_heat']:>15.5f} {stats['avg_susceptibility']:>15.5f}")
