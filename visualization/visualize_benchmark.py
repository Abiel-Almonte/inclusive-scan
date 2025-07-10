import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def _extract_metrics(output: str, kernel_regex: str):
    kernel_header_regex = rf"{kernel_regex}.*?,"
    k_match = re.search(kernel_header_regex, output)
    if not k_match:
        return None
    
    start = k_match.start()
    window = output[start:start + 4000]

    bw_match = re.search(r"Memory Throughput\s+Gbyte/s\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)", window)
    eff_match = re.search(r"DRAM Throughput\s+%\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)", window)
    dur_match = re.search(r"Duration\s+(us|ms)\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)", window)

    if not (bw_match and eff_match and dur_match):
        return None

    bw = float(bw_match.group(1))
    eff = float(eff_match.group(1))
    unit = dur_match.group(1)
    dur_val = float(dur_match.group(2))
    dur_seconds = dur_val / 1_000_000.0 if unit == 'us' else dur_val / 1_000.0
    
    return {'bw': bw, 'eff': eff, 'time': dur_seconds}

def run_profiler(n, log_file):
    print(f"Running profiler for N = {n}...")
    try:
        with open(log_file, 'w') as f:
            subprocess.run(
                ['./build_and_run.sh', 'profile', 'kernel_best.cuh', str(n)],
                stdout=f, stderr=subprocess.STDOUT, text=True, check=True, timeout=300
            )

        with open(log_file, 'r') as f:
            output = f.read()

        my_res = _extract_metrics(output, r"single_pass_scan_4x")
        cub_main_res = _extract_metrics(output, r"cub::DeviceScanKernel")
        cub_init_res = _extract_metrics(output, r"cub::DeviceScanInitKernel")

        if my_res and cub_main_res and cub_init_res:
            total_cub_time = cub_main_res['time'] + cub_init_res['time']
            if total_cub_time > 0:
                cub_eff_bw = ((cub_main_res['bw'] * cub_main_res['time']) + (cub_init_res['bw'] * cub_init_res['time'])) / total_cub_time
                cub_eff_eff = ((cub_main_res['eff'] * cub_main_res['time']) + (cub_init_res['eff'] * cub_init_res['time'])) / total_cub_time
            else:
                cub_eff_bw = 0
                cub_eff_eff = 0

            return {
                'N': n,
                'My_Kernel_GBs': my_res['bw'],
                'CUB_Kernel_GBs': cub_eff_bw,
                'Speedup': my_res['bw'] / cub_eff_bw if cub_eff_bw > 0 else 0,
                'My_Kernel_Efficiency': my_res['eff'],
                'CUB_Kernel_Efficiency': cub_eff_eff,
            }

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        print(f"Failed to profile N={n}. See {log_file} for details.")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"  -> Subprocess failed with exit code {e.returncode}.")
    except FileNotFoundError:
        print(f"Failed to run for N={n}: build_and_run.sh not found.")
    
    return None

def plot_benchmark(df):
    plt.figure(figsize=(12, 7))
    plt.plot(df['N'], df['My_Kernel_GBs'], marker='o', linestyle='-', label='My Kernel')
    plt.plot(df['N'], df['CUB_Kernel_GBs'], marker='x', linestyle='--', label='NVIDIA CUB')

    plt.xscale('log', base=2)
    plt.xlabel('Problem Size (N)')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('Kernel Performance Comparison')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    
    output_png = 'visualization/performance_chart.png'
    plt.savefig(output_png, dpi=300)
    print(f"Performance chart saved to {output_png}")
    plt.close()

def plot_roofline(df):
    plt.figure(figsize=(12, 7))
    plt.plot(df['N'], df['My_Kernel_Efficiency'], marker='o', linestyle='-', label='My Kernel')
    plt.plot(df['N'], df['CUB_Kernel_Efficiency'], marker='x', linestyle='--', label='NVIDIA CUB')

    plt.xscale('log', base=2)
    plt.xlabel('Problem Size (N)')
    plt.ylabel('DRAM Throughput Efficiency (%)')
    plt.title('Roofline: DRAM Throughput Efficiency')
    plt.ylim(0, 105)
    plt.axhline(100, color='r', linestyle='--', label='Theoretical Maximum (100%)')
    plt.axhline(90, color='g', linestyle='--', label='90% Target')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    
    output_png = 'visualization/roofline_chart.png'
    plt.savefig(output_png, dpi=300)
    print(f"Roofline chart saved to {output_png}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks and generate performance plots.")
    parser.add_argument(
        '--plot-only',
        type=str,
        nargs='?',
        const='visualization/all_results.csv',
        default=None,
        help="Skip profiling and plot directly from a CSV file. If no path is provided, it uses the default path."
    )
    args = parser.parse_args()

    output_csv = 'visualization/all_results.csv'
    log_dir = 'visualization/logs'

    if args.plot_only:
        csv_path = args.plot_only
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found at '{csv_path}'. Please run the profiler first.")
            return
        print(f"Plotting from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        benchmark_sizes = {1 << p for p in range(10, 29)}
        detail_sizes = { (1 << p) - 1 for p in range(10, 18)} | { (1 << p) + 1 for p in range(10, 18)}
        test_sizes = sorted(list(benchmark_sizes | detail_sizes))

        results = []
        for n in test_sizes:
            log_file = os.path.join(log_dir, f"profile_n_{n}.log")
            res = run_profiler(n, log_file)
            if res:
                results.append(res)
        
        if not results:
            print("No benchmark results generated.")
            return

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Unified benchmark results saved to {output_csv}")

    plot_benchmark(df)
    plot_roofline(df)

if __name__ == "__main__":
    main()
