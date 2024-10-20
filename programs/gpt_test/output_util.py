import os
import csv

def ensure_csv_file_exists(results_file, fieldnames):
    results_dir = os.path.dirname(results_file)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    if not os.path.isfile(results_file):
        with open(results_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

def write_results_to_csv(results_file, results):
    ensure_csv_file_exists(results_file, results.keys())
    with open(results_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())
        writer.writerow(results)
