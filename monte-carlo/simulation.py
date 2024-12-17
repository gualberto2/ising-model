import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse
import time
import csv

from ising_helpers import (
    simulate_thermalization,
    simulate_measurement_c,
    initialize_spins,
    total_energy,
    total_magnetization,
    seed_rng_custom,
)

# Physical constants and parameters
J = 1.0
kB = 1.0

beta_over_nu = 1.75

L_values = [10, 16, 24, 36]
T_min, T_max, T_step = 0.015, 4.5, 0.015
critical_temperature = 2.27

# As before, finer steps near Tc
T_values = {}
for L in L_values:
    T_values[L] = np.concatenate([
        np.arange(T_min, critical_temperature - 0.5, T_step),
        np.arange(critical_temperature - 0.5, critical_temperature + 0.5, T_step / 5),
        np.arange(critical_temperature + 0.5, T_max + T_step, T_step)
    ])

equil_sweeps = 20000000
measurement_sweeps = 6000000
measure_interval = 10

def remove_outliers_iqr(data, lower_percentile=25, upper_percentile=75, multiplier=1.5):
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

def run_simulation(params):
    L, T, seed = params
    N = L * L

    seed_rng_custom(seed)
    spins = initialize_spins(L)
    simulate_thermalization(spins, L, T, equil_sweeps + L**2)

    E_arr, M_arr = simulate_measurement_c(spins, L, T, measurement_sweeps, measure_interval)
    E_arr = np.array(E_arr, dtype=np.float64)
    M_arr = np.array(M_arr, dtype=np.float64)

    # Compute averages (no outlier removal here, just raw)
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

    os.makedirs("output", exist_ok=True)

    if save_plots:
        print("Plot saving enabled: plots will be saved to 'output/plots' directory.")
        plots_dir = "output/plots"
        os.makedirs(plots_dir, exist_ok=True)

    if save_csv:
        print(f"CSV saving enabled. Top {top_n} data points will be saved to '{csv_file_path}' and all data to '{full_csv_path}'.")

    results = initialize_results()

    tasks = []
    for L in L_values:
        for T in T_values[L]:
            tasks.append((L, T, 0))
    total_jobs = len(tasks)
    seeds = generate_seeds(total_jobs)
    for i in range(total_jobs):
        L, T, _ = tasks[i]
        tasks[i] = (L, T, seeds[i])

    num_cpus = cpu_count()
    num_processes = max(1, num_cpus - 1)

    print(f"Starting simulation with {num_processes} parallel processes...")
    print(f"Total simulation jobs: {total_jobs}")

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

    # Now filter outliers before plotting C and chi to get cleaner peaks
    for L in L_values:
        sorted_indices = np.argsort(results[L]['T'])
        sorted_T = np.array(results[L]['T'])[sorted_indices]
        sorted_E = np.array(results[L]['E'])[sorted_indices]
        sorted_M = np.array(results[L]['M'])[sorted_indices]
        sorted_C = np.array(results[L]['C'])[sorted_indices]
        sorted_chi = np.array(results[L]['chi'])[sorted_indices]

        filtered_C = remove_outliers_iqr(sorted_C)
        filtered_chi = remove_outliers_iqr(sorted_chi)

        def mask_outliers(original, filtered):
            if len(filtered) == 0:
                return np.ones_like(original, dtype=bool)
            Q1 = np.percentile(original, 25)
            Q3 = np.percentile(original, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (original >= lower_bound) & (original <= upper_bound)

        mask_C = mask_outliers(sorted_C, filtered_C)
        mask_chi = mask_outliers(sorted_chi, filtered_chi)
        combined_mask = mask_C & mask_chi

        T_plot = sorted_T[combined_mask]
        E_plot = sorted_E[combined_mask]
        M_plot = sorted_M[combined_mask]
        C_plot = sorted_C[combined_mask]
        chi_plot = sorted_chi[combined_mask]

        # Plot Energy
        plt.figure()
        plt.plot(T_plot, E_plot, marker='o', linestyle='none')
        plt.xlabel('Temperature T')
        plt.ylabel('Energy per spin')
        plt.title(f'L={L} - Energy')
        if save_plots:
            plt.savefig(os.path.join(plots_dir, f'L_{L}_energy.png'), dpi=300)
        plt.close()

        # Plot Magnetization
        plt.figure()
        plt.plot(T_plot, np.abs(M_plot), marker='o', linestyle='none')
        plt.xlabel('Temperature T')
        plt.ylabel('Magnetization per spin')
        plt.title(f'L={L} - Magnetization')
        if save_plots:
            plt.savefig(os.path.join(plots_dir, f'L_{L}_magnetization.png'), dpi=300)
        plt.close()

        # Plot Specific Heat (with ylim for L=24 or L=36)
        plt.figure()
        plt.plot(T_plot, C_plot, marker='o', linestyle='none')
        plt.xlabel('Temperature T')
        plt.ylabel('Specific Heat C')
        plt.title(f'L={L} - Specific Heat')

        # If L=24 or L=36, apply a y-limit based on 95th percentile
        if L in [24, 36]:
            upper_limit_C = np.percentile(C_plot, 95)
            plt.ylim(0, upper_limit_C)

        if save_plots:
            plt.savefig(os.path.join(plots_dir, f'L_{L}_specific_heat.png'), dpi=300)
        plt.close()

        # Plot Susceptibility (with ylim for L=24 or L=36)
        plt.figure()
        plt.plot(T_plot, chi_plot, marker='o', linestyle='none')
        plt.xlabel('Temperature T')
        plt.ylabel('Susceptibility χ')
        plt.title(f'L={L} - Susceptibility')

        # If L=24 or L=36, apply a y-limit based on 95th percentile
        if L in [24, 36]:
            upper_limit_chi = np.percentile(chi_plot, 95)
            plt.ylim(0, upper_limit_chi)

        if save_plots:
            plt.savefig(os.path.join(plots_dir, f'L_{L}_susceptibility.png'), dpi=300)
        plt.close()

        # Print average values (use unfiltered full sets for averages)
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
        print("Plots have been saved to the 'output/plots/' directory.")

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

if __name__ == "__main__":
    main()
