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
import csv  # Import the CSV module

from ising_helpers import (
    simulate_thermalization,
    simulate_measurement_c,
    initialize_spins,
    total_energy,
    total_magnetization,
    seed_rng_custom,
)

# Physical constants and parameters
J = 1.0      # Interaction strength
kB = 1.0     # Boltzmann constant

beta_over_nu = 1.75

# Problem parameters
L_values = [10, 16, 24, 36]
T_min, T_max, T_step = 0.015, 4.5, 0.015

# Simulation parameters
equil_sweeps = 10000000  # Reduced to 1e7
measurement_sweeps = 3000000
measure_interval = 10  # Measure every 10 sweeps

def remove_outliers_iqr(data, lower_percentile=15, upper_percentile=80, multiplier=1.0):
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
    N = L * L  # Total spins

    seed_rng_custom(seed)
    spins = initialize_spins(L)

    # Thermalization
    simulate_thermalization(spins, L, T, equil_sweeps + L**2)

    # Measurement
    E_arr, M_arr = simulate_measurement_c(spins, L, T, measurement_sweeps, measure_interval)
    E_samples = E_arr.tolist()
    M_samples = M_arr.tolist()

    # Convert to NumPy arrays
    E_arr = np.array(E_samples, dtype=np.float64)
    M_arr = np.array(M_samples, dtype=np.float64)

    # Remove outliers
    E_arr = remove_outliers_iqr(E_arr, lower_percentile=25, upper_percentile=75, multiplier=1.5)
    M_arr = remove_outliers_iqr(M_arr, lower_percentile=25, upper_percentile=75, multiplier=1.5)

    # Compute averages
    E_mean = E_arr.mean()
    E2_mean = (E_arr**2).mean()
    M_mean = M_arr.mean()
    M2_mean = (M_arr**2).mean()

    C = (E2_mean - E_mean**2) / (N * (T**2))
    chi = (M2_mean - M_mean**2) / (N * T)

    return (L, T, E_mean / N, C, M_mean / N, chi)

def generate_seeds(total_jobs):
    base_seed = int(time.time())
    return [base_seed + i for i in range(total_jobs)]

def main():
    parser = argparse.ArgumentParser(description="Ising Model Simulation")
    parser.add_argument('--save-plots', action='store_true', default=True,
                        help='Save plots to files. (Default: True)')
    parser.add_argument('--save-csv', action='store_true', default=True,
                        help='Save simulation data to a CSV file. (Default: True)')
    parser.add_argument('--csv-path', type=str, default='output/top_results.csv',
                        help='Path to save the top N CSV file. (Default: output/top_results.csv)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top data points to save per system size L. (Default: 10)')
    parser.add_argument('--full-csv-path', type=str, default='output/full_results.csv',
                        help='Path to save the full simulation CSV file. (Default: output/full_results.csv)')
    args = parser.parse_args()

    save_plots = args.save_plots
    save_csv = args.save_csv
    csv_file_path = args.csv_path
    top_n = args.top_n
    full_csv_path = args.full_csv_path

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    if save_plots:
        print("Plot saving enabled: plots will be saved to 'plots/' directory.")
        plots_dir = "output/plots"
        os.makedirs(plots_dir, exist_ok=True)

    if save_csv:
        print(f"CSV saving enabled. Top {top_n} data points will be saved to '{csv_file_path}' and all data to '{full_csv_path}'.")

    results = initialize_results()
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
            L_T_pairs.append((L, T))
            total_jobs += 1

    seeds = generate_seeds(total_jobs)
    tasks = list(zip([lt[0] for lt in L_T_pairs], [lt[1] for lt in L_T_pairs], seeds))

    num_cpus = cpu_count()
    num_processes = max(1, num_cpus - 1)

    print(f"Starting simulation with {num_processes} parallel processes...")
    print(f"Total simulation jobs: {total_jobs}")

    # Run simulations in parallel
    with Pool(processes=num_processes) as pool:
        with tqdm(total=total_jobs, desc="Simulations") as pbar:
            for result in pool.imap_unordered(run_simulation, tasks):
                L, T, E, C, M, chi = result
                results[L]['T'].append(T)
                results[L]['E'].append(E)
                results[L]['C'].append(C)
                results[L]['M'].append(M)
                results[L]['chi'].append(chi)
                pbar.update(1)

    # Arrays for storing T_c(L) and χ(T_c(L))
    Tc_values = []
    chi_Tc_values = []
    L_sizes = []

    # Plotting and summary
    for L in L_values:
        sorted_indices = np.argsort(results[L]['T'])
        sorted_T = np.array(results[L]['T'])[sorted_indices]
        sorted_E = np.array(results[L]['E'])[sorted_indices]
        sorted_M = np.array(results[L]['M'])[sorted_indices]
        sorted_C = np.array(results[L]['C'])[sorted_indices]
        sorted_chi = np.array(results[L]['chi'])[sorted_indices]

        # Find Tc(L) as the T at max χ
        max_chi_index = np.argmax(sorted_chi)
        Tc = sorted_T[max_chi_index]
        chi_at_Tc = sorted_chi[max_chi_index]

        Tc_values.append(Tc)
        chi_Tc_values.append(chi_at_Tc)
        L_sizes.append(L)

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

    # Write all data points to full CSV
    if save_csv:
        print(f"Writing all data points to '{full_csv_path}'...")
        try:
            with open(full_csv_path, mode='w', newline='') as full_csv:
                full_writer = csv.writer(full_csv)
                full_writer.writerow(['L', 'T', 'Energy_per_spin', 'Specific_Heat', 'Magnetization_per_spin', 'Susceptibility'])
                for L in L_values:
                    L_T = results[L]['T']
                    L_E = results[L]['E']
                    L_C = results[L]['C']
                    L_M = results[L]['M']
                    L_chi = results[L]['chi']
                    for t, e, c, m, ch in zip(L_T, L_E, L_C, L_M, L_chi):
                        full_writer.writerow([L, f"{t:.5f}", f"{e:.5f}", f"{c:.5f}", f"{m:.5f}", f"{ch:.5f}"])
            print(f"All data points have been saved to '{full_csv_path}'.")
        except Exception as e:
            print(f"Failed to write all data points to CSV file '{full_csv_path}': {e}")

        # Write top N data points per L
        print(f"Attempting to write top {top_n} data points per L to CSV file '{csv_file_path}'...")
        try:
            with open(csv_file_path, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['L', 'T', 'Energy_per_spin', 'Specific_Heat', 'Magnetization_per_spin', 'Susceptibility'])
                for L in L_values:
                    L_T = results[L]['T']
                    L_E = results[L]['E']
                    L_C = results[L]['C']
                    L_M = results[L]['M']
                    L_chi = results[L]['chi']
                    combined_data = list(zip(L_T, L_E, L_C, L_M, L_chi))
                    sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)
                    top_data = sorted_data[:top_n]
                    for data_point in top_data:
                        T, E, C, M, chi = data_point
                        csv_writer.writerow([L, f"{T:.5f}", f"{E:.5f}", f"{C:.5f}", f"{M:.5f}", f"{chi:.5f}"])
            print(f"Top {top_n} data points per L have been saved to '{csv_file_path}'.")
        except Exception as e:
            print(f"Failed to write top data points to CSV file '{csv_file_path}': {e}")

        print("CSV writing process completed.")

    # --------------------------- Additional Plots ---------------------------
    # Convert to NumPy arrays for convenience
    Tc_values = np.array(Tc_values)
    chi_Tc_values = np.array(chi_Tc_values)
    L_sizes = np.array(L_sizes)

    # Plot T_c(L) vs 1/L
    if save_plots:
        plt.figure()
        plt.plot(1.0/L_sizes, Tc_values, 'o', linestyle='none')
        plt.xlabel('1/L')
        plt.ylabel(r'$T_c(L)$')
        plt.title('T_c(L) vs 1/L')
        plt.savefig(os.path.join(plots_dir, 'Tc_vs_1overL.png'), dpi=300)
        plt.close()

        # Plot χ(T_c(L)) vs L on a log-log scale
        plt.figure()
        plt.loglog(L_sizes, chi_Tc_values, 'o', linestyle='none')
        plt.xlabel('L (log scale)')
        plt.ylabel(r'$\chi(T_c(L))$ (log scale)')
        plt.title('$\chi(T_c(L))$ vs L (log-log)')
        plt.savefig(os.path.join(plots_dir, 'chi_Tc_vs_L_loglog.png'), dpi=300)
        plt.close()

    # ----------------------------------------------------------------------

if __name__ == "__main__":
    main()
