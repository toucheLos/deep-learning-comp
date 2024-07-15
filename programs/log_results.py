import csv
import os

def ensure_csv_file_exists(results_file, fieldnames):
    if not os.path.isfile(results_file):
        with open(results_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

def write_results_to_csv(results_file, results):
    ensure_csv_file_exists(results_file, results.keys())
    with open(results_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())
        writer.writerow(results)
