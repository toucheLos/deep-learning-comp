import psutil
import time
import csv
import os
import subprocess
import pandas as pd
import threading
import log_results

def log_resource_usage(duration, interval, output_file='resource_usage.csv'):
    end_time = time.time() + duration
    fieldnames = ['timestamp', 'cpu_usage', 'memory_usage']
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        while time.time() < end_time:
            timestamp = time.time()
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().percent
            writer.writerow({
                'timestamp': timestamp,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage
            })
            time.sleep(interval)

def aggregate_resource_usage(file_path='resource_usage.csv'):
    df = pd.read_csv(file_path)
    avg_cpu_usage = df['cpu_usage'].mean()
    avg_memory_usage = df['memory_usage'].mean()
    return avg_cpu_usage, avg_memory_usage

def run_deep_learning_script(script_path, duration, interval):
    # Start resource monitoring
    resource_monitor_thread = threading.Thread(target=log_resource_usage, args=(duration, interval
    , 'resource_usage.csv'))
    resource_monitor_thread.start()
    
    # Run the deep learning script
    start_time = time.time()
    subprocess.run(['python3', script_path])
    training_time = time.time() - start_time
    
    # Wait for resource monitoring to finish
    resource_monitor_thread.join()
    
    # Aggregate resource usage
    avg_cpu_usage, avg_memory_usage = aggregate_resource_usage()

    # Capture GPU utilization
    gpu_usage = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']).decode('utf-8').strip().split('\n')
    gpu_memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).decode('utf-8').strip().split('\n')

    # Extract the base name of the script (e.g., "resnet_50" from "resnet_50.py")
    script_name = os.path.basename(script_path).replace('.py', '')

    # Collect final metrics
    results = {
        'Batch Size': os.environ.get('BATCH_SIZE', 'unknown'),
        'Learning Rate': os.environ.get('LEARNING_RATE', 'unknown'),
        'Precision': os.environ.get('PRECISION', 'unknown'),
        'GPUs': os.environ.get('GPUS', 'unknown'),
        'Training Time (seconds)': training_time,
        'CPU Usage (%)': avg_cpu_usage,
        'Memory Usage (%)': avg_memory_usage,
        'GPU Usage (%)': ', '.join(gpu_usage),
        'GPU Memory Usage (MiB)': ', '.join(gpu_memory),
        'Energy Consumption (W)': 'N/A'  # Placeholder for power consumption
    }

    # Ensure the 'results' directory exists
    os.makedirs('results', exist_ok=True)

    # Write results to CSV using the utility script
    results_file = f'results/{script_name}_results.csv'
    log_results.write_results_to_csv(results_file, results)
    print(f"Results written to {results_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Deep Learning Script with Resource Monitoring')
    parser.add_argument('--script', type=str, required=True, help='Path to the deep learning script')
    parser.add_argument('--duration', type=int, required=True, help='Duration of resource monitoring in seconds')
    parser.add_argument('--interval', type=int, default=1, help='Interval for resource monitoring in seconds')
    args = parser.parse_args()
    
    run_deep_learning_script(args.script, args.duration, args.interval)
