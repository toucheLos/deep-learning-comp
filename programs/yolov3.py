import tensorflow as tf
from tensorflow.keras.models import Model
<<<<<<< HEAD
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Add, UpSampling2D, Concatenate
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy
import time
import json
import sys

def build_yolov3(input_shape, num_classes):
    # YOLOv3 architecture code goes here
    # This is a simplified placeholder. Implement the full YOLOv3 here.
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

def preprocess_data():
    # Placeholder for data preprocessing
    # Implement actual data loading and preprocessing for object detection
    x_train, y_train = None, None
    x_val, y_val = None, None
    return x_train, y_train, x_val, y_val

def main():
    args = sys.argv
    gpus = int(args[1])
    batch_size = int(args[2])
    precision = args[3]
    learning_rate = float(args[4])

    input_shape = (416, 416, 3)  # Typical input shape for YOLOv3
    num_classes = 80  # Number of classes in your dataset

    # Preprocess data
    x_train, y_train, x_val, y_val = preprocess_data()

    # Define the model
    model = build_yolov3(input_shape, num_classes)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Enable mixed precision if TensorFlow version is 2.x and precision is FP16
    if tf.__version__.startswith('2') and precision == 'FP16':
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        print("Mixed Precision ON")

    model.compile(optimizer=optimizer, loss=[binary_crossentropy, sparse_categorical_crossentropy], metrics=['accuracy'])

    # Train the model and capture metrics
    training_times = []
    history = None

    try:
        start_time = time.time()
        history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size, validation_data=(x_val, y_val), verbose=1)
        training_time = time.time() - start_time
        training_times.append(training_time)

        # Measure inference time
        start_time = time.time()
        model.predict(x_val)
        inference_time = time.time() - start_time

        # Capture results
        results = {
            'acc': str(history.history.get('accuracy')),
            'loss': str(history.history.get('loss')),
            'val_acc': str(history.history.get('val_accuracy')),
            'val_loss': str(history.history.get('val_loss')),
            'training_time': training_times,
            'inference_time': inference_time
        }

        print(results)

    except Exception as e:
        print(f"An error occurred during training: {e}", file=sys.stderr)

    print("YOLOv3 model complete.")

if __name__ == "__main__":
    main()
=======
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import os
import time
import log_results
import psutil
import subprocess

# Function to measure power consumption using powerstat
def measure_energy_consumption():
    try:
        output = subprocess.check_output(['powerstat', '-d', '0.5', '1']).decode('utf-8')
        lines = output.split('\n')
        for line in lines:
            if 'Average' in line:
                # Parse the power consumption in watts from the 'Average' line
                energy = float(line.split()[1])
                return energy
    except Exception as e:
        print(f"Error measuring energy consumption: {e}")
        return None

# Alternative method if powerstat is not available
def measure_energy_consumption_alternative():
    # Implement alternative energy measurement method here
    # For example, using Intel Power Gadget or another tool
    return None

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model
input_tensor = Input(shape=(32, 32, 3))
x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
x = Flatten()(x)
output_tensor = Dense(10, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)

# Compile the model
learning_rate = float(os.environ.get('LEARNING_RATE', 0.001))
batch_size = int(os.environ.get('BATCH_SIZE', 32))
precision = os.environ.get('PRECISION', 'FP32')

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
if precision == 'FP16':
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and capture metrics
training_times = []
history = None

try:
    for epoch in range(10):
        start_time = time.time()
        history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size, validation_data=(x_test, y_test))
        epoch_training_time = time.time() - start_time
        training_times.append(epoch_training_time)

    # Measure inference time
    start_time = time.time()
    model.predict(x_test)
    inference_time = time.time() - start_time

    # Capture resource utilization
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent

    # Capture GPU utilization
    gpu_usage = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']).decode('utf-8').strip().split('\n')
    gpu_memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).decode('utf-8').strip().split('\n')

    # Measure energy consumption
    energy_consumption = measure_energy_consumption()
    if energy_consumption is None:
        energy_consumption = measure_energy_consumption_alternative()

    # Collect final metrics
    results = {
        'Batch Size': batch_size,
        'Learning Rate': learning_rate,
        'Precision': precision,
        'GPUs': os.environ.get('GPUS', 'unknown'),
        'Training Time Per Epoch': training_times,
        'Inference Time': inference_time,
        'Model Accuracy': history.history.get('val_accuracy', [None])[-1],
        'Validation Loss': history.history.get('val_loss', [None])[-1],
        'CPU Usage (%)': cpu_usage,
        'Memory Usage (%)': memory_usage,
        'GPU Usage (%)': ', '.join(gpu_usage),
        'GPU Memory Usage (MiB)': ', '.join(gpu_memory),
        'Energy Consumption (W)': energy_consumption
    }

except KeyError as e:
    print(f"KeyError: {e} not found in training history. Using default values.")
    # Collect final metrics with default values for missing keys
    results = {
        'Batch Size': batch_size,
        'Learning Rate': learning_rate,
        'Precision': precision,
        'GPUs': os.environ.get('GPUS', 'unknown'),
        'Training Time Per Epoch': training_times,
        'Inference Time': inference_time,
        'Model Accuracy': None,
        'Validation Loss': None,
        'CPU Usage (%)': cpu_usage,
        'Memory Usage (%)': memory_usage,
        'GPU Usage (%)': ', '.join(gpu_usage),
        'GPU Memory Usage (MiB)': ', '.join(gpu_memory),
        'Energy Consumption (W)': energy_consumption
    }

except Exception as e:
    print(f"An error occurred: {e}")
    # If a general exception occurs, create an empty results dictionary or handle accordingly
    results = {
        'Batch Size': batch_size,
        'Learning Rate': learning_rate,
        'Precision': precision,
        'GPUs': os.environ.get('GPUS', 'unknown'),
        'Training Time Per Epoch': training_times,
        'Inference Time': inference_time,
        'Model Accuracy': None,
        'Validation Loss': None,
        'CPU Usage (%)': cpu_usage,
        'Memory Usage (%)': memory_usage,
        'GPU Usage (%)': ', '.join(gpu_usage),
        'GPU Memory Usage (MiB)': ', '.join(gpu_memory),
        'Energy Consumption (W)': energy_consumption
    }

# Ensure the 'results' directory exists
os.makedirs('results', exist_ok=True)

# Write results to CSV using the utility script
results_file = 'results/yolov3_results.csv'
log_results.write_results_to_csv(results_file, results)
print(f"Results written to {results_file}")
>>>>>>> origin/main
