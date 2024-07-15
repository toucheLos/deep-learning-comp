import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import os
import time
import log_results
import psutil
import subprocess
import threading

# Function to measure resource usage
def measure_resource_usage(duration=10, interval=1, output_file='resource_usage.csv'):
    script_path = os.path.expanduser('~/dl-comp/programs/monitor_resources.py')
    cmd = f'python3 {script_path} --duration {duration} --interval {interval} --output_file {output_file}'
    subprocess.Popen(cmd, shell=True)

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model
base_model = ResNet50(include_top=False, weights=None, input_shape=(32, 32, 3))
model = Sequential([
    base_model,
    Flatten(),
    Dense(10, activation='softmax')
])

# Compile the model
learning_rate = float(os.environ.get('LEARNING_RATE', 0.001))
batch_size = int(os.environ.get('BATCH_SIZE', 32))
precision = os.environ.get('PRECISION', 'FP32')

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Enable mixed precision if TensorFlow version is 2.x
if tf.__version__.startswith('2'):
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and capture metrics
training_times = []
history = None

try:
    # Start resource monitoring
    resource_monitor_thread = threading.Thread(target=measure_resource_usage, args=(600,))
    resource_monitor_thread.start()
    
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size, validation_data=(x_test, y_test))
    training_time = time.time() - start_time
    training_times.append(training_time)

    # Measure inference time
    start_time = time.time()
    model.predict(x_test)
    inference_time = time.time() - start_time

    # Wait for resource monitoring to finish
    resource_monitor_thread.join()

    # Capture resource utilization
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent

    # Capture GPU utilization
    gpu_usage = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']).decode('utf-8').strip().split('\n')
    gpu_memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).decode('utf-8').strip().split('\n')

    # Collect final metrics
    results = {
        'Batch Size': batch_size,
        'Learning Rate': learning_rate,
        'Precision': precision,
        'GPUs': os.environ.get('GPUS', 'unknown'),
        'Training Time Per Epoch (seconds)': training_times,
        'Inference Time (seconds)': inference_time,
        'Validation Accuracy': history.history.get('val_accuracy', [None])[-1],
        'Validation Loss': history.history.get('val_loss', [None])[-1],
        'CPU Usage (%)': cpu_usage,
        'Memory Usage (%)': memory_usage,
        'GPU Usage (%)': ', '.join(gpu_usage),
        'GPU Memory Usage (MiB)': ', '.join(gpu_memory),
        'Energy Consumption (W)': 'N/A'  # Placeholder for power consumption
    }

except KeyError as e:
    print(f"KeyError: {e} not found in training history. Using default values.")
    # Collect final metrics with default values for missing keys
    results = {
        'Batch Size': batch_size,
        'Learning Rate': learning_rate,
        'Precision': precision,
        'GPUs': os.environ.get('GPUS', 'unknown'),
        'Training Time Per Epoch (seconds)': training_times,
        'Inference Time (seconds)': inference_time,
        'Validation Accuracy': None,
        'Validation Loss': None,
        'CPU Usage (%)': cpu_usage,
        'Memory Usage (%)': memory_usage,
        'GPU Usage (%)': ', '.join(gpu_usage),
        'GPU Memory Usage (MiB)': ', '.join(gpu_memory),
        'Energy Consumption (W)': 'N/A'  # Placeholder for power consumption
    }

except Exception as e:
    print(f"An error occurred: {e}")
    # If a general exception occurs, create an empty results dictionary or handle accordingly
    results = {
        'Batch Size': batch_size,
        'Learning Rate': learning_rate,
        'Precision': precision,
        'GPUs': os.environ.get('GPUS', 'unknown'),
        'Training Time Per Epoch (seconds)': training_times,
        'Inference Time (seconds)': inference_time,
        'Validation Accuracy': None,
        'Validation Loss': None,
        'CPU Usage (%)': cpu_usage,
        'Memory Usage (%)': memory_usage,
        'GPU Usage (%)': ', '.join(gpu_usage),
        'GPU Memory Usage (MiB)': ', '.join(gpu_memory),
        'Energy Consumption (W)': 'N/A'  # Placeholder for power consumption
    }

# Ensure the 'results' directory exists
os.makedirs('results', exist_ok=True)

# Write results to CSV using the utility script
results_file = 'results/resnet50_results.csv'
log_results.write_results_to_csv(results_file, results)
print(f"Results written to {results_file}")
