import sys
import subprocess
import json
import re
import os

def process_log_file(log_file_path):
    with open(log_file_path, 'r') as log_file:
        lines = log_file.readlines()

        lines = lines[5:]

        processed_lines = []

        for line in lines:
            parts = line.strip().split(',')
            if parts and parts[0].isdigit() and int(parts[0]) > 10:
                processed_lines.append(line)
        
        return processed_lines

def main():
    config = {
        'gpus': sys.argv[2],
        'batch_size': sys.argv[3],
        'precision': sys.argv[4],
        'learning_rate': sys.argv[5]
    }

    output = subprocess.check_output([
        sys.executable, 
        sys.argv[1], 
        config['gpus'], 
        config['batch_size'], 
        config['precision'], 
        config['learning_rate']
    ]).decode('utf-8')

    try:
        results_json = re.search(r"\{.*\}", output).group(0).replace("'", '"')
        results_json = results_json.replace("'None'", "null")
        model_results = json.loads(results_json)
        for key, value in model_results.items():
            model_results[key] = str(value).strip('[]')
    except json.JSONDecodeError:
        print("Failed to parse JSON. Output was:")
        print(results_json)
        sys.exit(1)

    gpu_metrics = process_log_file("logfile.txt")

    resource_data = [
        dict(zip(['gpu_util', 'mem_util', 'power_draw'], map(float, line.split(', '))))
        for line in gpu_metrics if float(line.split(', ')[0]) > 0
    ]

    num_entries = len(resource_data)

    avg_gpu_util = sum(item['gpu_util'] for item in resource_data) / num_entries if num_entries > 0 else 0
    avg_mem_util = sum(item['mem_util'] for item in resource_data) / num_entries if num_entries > 0 else 0
    avg_power_draw = sum(item['power_draw'] for item in resource_data) / num_entries if num_entries > 0 else 0

    resource_data = {
        'avg_gpu_util': avg_gpu_util,
        'avg_mem_util': avg_mem_util,
        'avg_power_draw': avg_power_draw
    }

    if os.path.exists(f'{sys.argv[1]}_results.json'):
        with open(f'{sys.argv[1]}_results.json', 'r') as f:
            data = json.load(f)
    else:
        data = []

    data.append({
        'config': config,
        'model_results': model_results,
        'resource_data': resource_data
    })

    with open(f'{sys.argv[1]}_results.json', 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Results logged to {sys.argv[1]}_results.json")

if __name__ == "__main__":
    main()