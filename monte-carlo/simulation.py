import numpy as np
import os
import matplotlib
# Use the non-interactive Agg backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import itertools
import argparse
import time

from ising_helpers import (
    simulate_thermalization,
    simulate_measurement,
    initialize_spins,
    total_energy,
    total_magnetization,
    seed_rng_custom, 
)

# Physical constants and parameters
J = 1.0      # Interaction strength
kB = 1.0     # Boltzmann constant

# Problem parameters
L_values = [10, 16, 24, 36]
T_min, T_max, T_step = 0.015, 4.5, 0.015

# Simulation parameters
equil_sweeps = 10000000  # Reduced to 1e7 as per guidelines
measurement_sweeps = 3000000  # Number of sweeps for measurement
measure_interval = 10  # Measure energy, magnetization every 10 sweeps

def remove_outliers_iqr(data, lower_percentile=15, upper_percentile=80, multiplier=1.0):
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
    return filtered_data

def initialize_results():
    return {L: {'T': [], 'E': [], 'C': [], 'M': [], 'chi': []} for L in L_values}

# Worker function for parallel processing
def run_simulation(params):
    L, T, seed = params
    N = L * L  # Total number of spins

    # Seed the RNG uniquely for each process
    seed_rng_custom(seed)

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
    E_arr = remove_outliers_iqr(E_arr, lower_percentile=25, upper_percentile=75, multiplier=1.5)
    M_arr = remove_outliers_iqr(M_arr, lower_percentile=25, upper_percentile=75, multiplier=1.5)


    # Compute averages
    E_mean = E_arr.mean()
    E2_mean = (E_arr**2).mean()
    M_mean = M_arr.mean()
    M2_mean = (M_arr**2).mean()

    # Specific heat and susceptibility
    C = (E2_mean - E_mean**2) / (N * (T**2))
    chi = (M2_mean - M_mean**2) / (N * T)

    # Return the results
    return (L, T, E_mean / N, C, M_mean / N, chi)

def generate_seeds(total_jobs):
    """
    Generate unique seeds for each job to ensure independent RNG sequences.
    """
    base_seed = int(time.time())
    return [base_seed + i for i in range(total_jobs)]

def main():
    # Argument parsing for optional display
    parser = argparse.ArgumentParser(description="Ising Model Simulation")
    parser.add_argument('--save-plots', action='store_true', default=True,
                        help='Save plots to files. (Default: True)')
    args = parser.parse_args()

    save_plots = args.save_plots

    if save_plots:
        print("Plot saving enabled: plots will be saved to 'plots/' directory.")

    # Data storage
    results = initialize_results()

    # Define critical temperature
    critical_temperature = 2.27

    # Prepare all (L, T) pairs
    L_T_pairs = []
    total_jobs = 0
    for L in L_values:
        temperatures = np.concatenate([
            np.arange(T_min, critical_temperature - 0.5, T_step),
            np.arange(critical_temperature - 0.5, critical_temperature + 0.5, T_step / 10),
            np.arange(critical_temperature + 0.5, T_max + T_step, T_step)
        ])
        for T in temperatures:
            L_T_pairs.append( (L, T) )
            total_jobs +=1

    seeds = generate_seeds(total_jobs)
    tasks = list(zip([lt[0] for lt in L_T_pairs], [lt[1] for lt in L_T_pairs], seeds))

    # Determine the number of processes (leave one core free)
    num_cpus = cpu_count()
    num_processes = max(1, num_cpus - 1)

    print(f"Starting simulation with {num_processes} parallel processes...")

    # Initialize the multiprocessing Pool
    with Pool(processes=num_processes) as pool:
        # Use imap_unordered for better performance
        results_list = []
        with tqdm(total=total_jobs, desc="Simulations") as pbar:
            for result in pool.imap_unordered(run_simulation, tasks):
                L, T, E, C, M, chi = result
                results[L]['T'].append(T)
                results[L]['E'].append(E)        # Energy per spin
                results[L]['C'].append(C)
                results[L]['M'].append(M)        # Magnetization per spin
                results[L]['chi'].append(chi)
                pbar.update(1)

    # Create 'plots' directory if saving plots
    if save_plots:
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)

    # Plotting and summary
    for L in L_values:
        # Sort the results by temperature for plotting
        sorted_indices = np.argsort(results[L]['T'])
        sorted_T = np.array(results[L]['T'])[sorted_indices]
        sorted_E = np.array(results[L]['E'])[sorted_indices]
        sorted_M = np.array(results[L]['M'])[sorted_indices]
        sorted_C = np.array(results[L]['C'])[sorted_indices]
        sorted_chi = np.array(results[L]['chi'])[sorted_indices]

        # Plot Energy
        plt.figure()
        plt.plot(sorted_T, sorted_E, marker='o', linestyle='none')
        plt.xlabel('Temperature T')
        plt.ylabel('Energy per spin')
        plt.title(f'L={L} - Energy')
        if save_plots:
            plt.savefig(os.path.join(plots_dir, f'L_{L}_energy.png'), dpi=300)
        plt.close()

        # Plot Magnetization
        plt.figure()
        plt.plot(sorted_T, np.abs(sorted_M), marker='o', linestyle='none')
        plt.xlabel('Temperature T')
        plt.ylabel('Magnetization per spin')
        plt.title(f'L={L} - Magnetization')
        if save_plots:
            plt.savefig(os.path.join(plots_dir, f'L_{L}_magnetization.png'), dpi=300)
        plt.close()

        # Plot Specific Heat
        plt.figure()
        plt.plot(sorted_T, sorted_C, marker='o', linestyle='none')
        plt.xlabel('Temperature T')
        plt.ylabel('Specific Heat C')
        plt.title(f'L={L} - Specific Heat')
        if save_plots:
            plt.savefig(os.path.join(plots_dir, f'L_{L}_specific_heat.png'), dpi=300)
        plt.close()

        # Plot Susceptibility
        plt.figure()
        plt.plot(sorted_T, sorted_chi, marker='o', linestyle='none')
        plt.xlabel('Temperature T')
        plt.ylabel('Susceptibility χ')
        plt.title(f'L={L} - Susceptibility')
        if save_plots:
            plt.savefig(os.path.join(plots_dir, f'L_{L}_susceptibility.png'), dpi=300)
        plt.close()

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

    if save_plots:
        print("Plots have been saved to the 'plots/' directory.")

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

if __name__ == "__main__":
    main()
