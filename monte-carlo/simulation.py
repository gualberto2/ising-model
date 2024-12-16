import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.ndimage import gaussian_filter1d

from concurrent.futures import ThreadPoolExecutor, as_completed

from ising_helpers import metropolis_step, total_energy, total_magnetization, simulate_at_temperature

# Physical constants and parameters
J = 1.0      # Interaction strength
kB = 1.0     # Boltzmann constant
# Problem parameters
L_values = [10, 16, 24, 36]
T_min, T_max, T_step = 0.015, 4.5, 0.015

# Simulation parameters
equil_sweeps = 1000000      # Number of sweeps for thermalization
measurement_sweeps = 300000  # Number of sweeps for measurement
measure_interval = 10        # Measure energy, magnetization every 10 sweeps

def remove_outliers_iqr(data, lower_percentile=25, upper_percentile=75, multiplier=1.0):
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
    return data[(data >= lower_bound) & (data <= upper_bound)]

# Data storage
# We'll store results in dictionaries keyed by L and then by T
results = {L: {'T': [], 'E': [], 'C': [], 'M': [], 'chi': []} for L in L_values}

# Main simulation loop with multithreading
for L in L_values:
    N = L * L  # Total number of spins
    critical_temperature = 2.27

    # Faster but innaccurate temperature range
    # temperatures = np.concatenate([
    #     np.arange(T_min, critical_temperature - 0.5, 0.05),  # Coarser step far from Tc
    #     np.arange(critical_temperature - 0.5, critical_temperature + 0.5, 0.01),  # Fine steps near Tc
    #     np.arange(critical_temperature + 0.5, T_max + T_step, 0.05)  # Coarser step again
    # ])

    # Slower but accurate temperature range
    temperatures = np.concatenate([
        np.arange(T_min, critical_temperature - 0.5, T_step),  # Coarser step far from Tc
        np.arange(critical_temperature - 0.5, critical_temperature + 0.5, T_step / 2),  # Fine steps near Tc
        np.arange(critical_temperature + 0.5, T_max + T_step, T_step)  # Coarser step again
    ])

    print(f"Starting simulations for L={L}...")
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(simulate_at_temperature, L, T, J, kB, equil_sweeps, measurement_sweeps, measure_interval) for T in temperatures]

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing L={L}"):
            try:
                T, E, C, M, chi = future.result()
                results[L]['T'].append(T)
                results[L]['E'].append(E)
                results[L]['C'].append(C)
                results[L]['M'].append(M)
                results[L]['chi'].append(chi)
            except Exception as e:
                print(f"Error processing L={L}: {e}")

    # Sort the results
    T_array = np.array(results[L]['T'])
    E_array = np.array(results[L]['E'])
    C_array = np.array(results[L]['C'])
    M_array = np.array(results[L]['M'])
    chi_array = np.array(results[L]['chi'])

    sorted_indices = np.argsort(T_array)

    results[L]['T'] = T_array[sorted_indices]
    results[L]['E'] = E_array[sorted_indices]
    results[L]['C'] = C_array[sorted_indices]
    results[L]['M'] = M_array[sorted_indices]
    results[L]['chi'] = chi_array[sorted_indices]

for L in L_values:
    # Truncate outliers
    E_clean = remove_outliers_iqr(np.array(results[L]['E']), multiplier=0.5)
    M_clean = remove_outliers_iqr(np.array(results[L]['M']), multiplier=0.5)
    C_clean = remove_outliers_iqr(np.array(results[L]['C']), multiplier=0.5)
    chi_clean = remove_outliers_iqr(np.array(results[L]['chi']), multiplier=0.5)
    
    # Dynamic y-limits for truncation
    def get_ylim(data):
        return min(data) * 0.9, max(data) * 1.1  # Adjust 10% margins

    # Plot Energy
    plt.figure()
    plt.plot(results[L]['T'], results[L]['E'], marker='o', linestyle='none')
    plt.ylim(get_ylim(E_clean))
    plt.xlabel('Temperature T')
    plt.ylabel('Energy per spin')
    plt.title(f'L={L} - Energy')

    # Plot Magnetization
    plt.figure()
    plt.plot(results[L]['T'], np.abs(results[L]['M']), marker='o', linestyle='none')
    plt.ylim(get_ylim(M_clean))
    plt.xlabel('Temperature T')
    plt.ylabel('Magnetization per spin')
    plt.title(f'L={L} - Magnetization')

    # Plot Specific Heat
    plt.figure()
    plt.plot(results[L]['T'], results[L]['C'], marker='o', linestyle='none')
    plt.ylim(get_ylim(C_clean))
    plt.xlabel('Temperature T')
    plt.ylabel('Specific Heat C')
    plt.title(f'L={L} - Specific Heat')

    # Plot Susceptibility
    plt.figure()
    plt.plot(results[L]['T'], results[L]['chi'], marker='o', linestyle='none')
    plt.ylim(get_ylim(chi_clean))
    plt.xlabel('Temperature T')
    plt.ylabel('Susceptibility χ')
    plt.title(f'L={L} - Susceptibility')

plt.show()  # Show all plots at the end

# Create summary
summary = {}
for L in L_values:
    summary[L] = {
        'avg_energy': np.mean(results[L]['E']) if len(results[L]['E']) > 0 else np.nan,
        'avg_magnetization': np.mean(results[L]['M']) if len(results[L]['M']) > 0 else np.nan,
        'avg_specific_heat': np.mean(results[L]['C']) if len(results[L]['C']) > 0 else np.nan,
        'avg_susceptibility': np.mean(results[L]['chi']) if len(results[L]['chi']) > 0 else np.nan,
    }

print("Simulation complete. Printing results...\n")

# Print summary table
try:
    print(f"\n{'L':>5} {'<E>':>15} {'<|M|>':>15} {'<C>':>15} {'<χ>':>15}")
    for L in L_values:
        if np.isnan(summary[L]['avg_specific_heat']) or np.isnan(summary[L]['avg_susceptibility']):
            print(f"Warning: NaN detected for L={L}")

        if len(results[L]['T']) > 0:
            # Sort results by temperature to ensure correct output order
            T_sorted = np.array(results[L]['T'])
            E_sorted = np.array(results[L]['E'])
            M_sorted = np.array(results[L]['M'])
            C_sorted = np.array(results[L]['C'])
            chi_sorted = np.array(results[L]['chi'])

            # Smooth specific heat and susceptibility
            C_smoothed = gaussian_filter1d(C_sorted, sigma=1)
            chi_smoothed = gaussian_filter1d(chi_sorted, sigma=1)

            # Compute averages
            avg_energy = np.mean(E_sorted)
            avg_magnetization = np.mean(np.abs(M_sorted))
            avg_specific_heat = np.mean(C_smoothed)
            avg_susceptibility = np.mean(chi_smoothed)

            # Print the summary
            print(f"{L:>5} {avg_energy:>15.5f} {avg_magnetization:>15.5f} {avg_specific_heat:>15.5f} {avg_susceptibility:>15.5f}")
        else:
            print(f"{L:>5} {'N/A':>15} {'N/A':>15} {'N/A':>15} {'N/A':>15}")
except Exception as e:
    print(f"An error occurred while printing results: {e}")

print("Results dictionary contents:")
for L in results:
    print(f"L={L}, Data points: {len(results[L]['T'])}")
