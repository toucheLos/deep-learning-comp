<<<<<<< HEAD
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import os
import time
import json
import sys
import numpy as np

def main():
    args = sys.argv
    gpus = int(args[1])
    batch_size = int(args[2])
    precision = args[3]
    learning_rate = float(args[4])

    # Set up the mixed precision policy if FP16 is selected
    if tf.__version__.startswith('2') and precision == 'FP16':
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        print("Mixed Precision ON")

    # Load dataset (using IMDb for simplicity)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=5000)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=128)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=128)

    # Load BERT model and tokenizer
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    loss = SparseCategoricalCrossentropy(from_logits=True)
    metrics = [SparseCategoricalAccuracy('accuracy')]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Train the model and capture metrics
    training_times = []
    history = None

    try:
        start_time = time.time()
        history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size, validation_data=(x_test, y_test), verbose=1)
        training_time = time.time() - start_time
        training_times.append(training_time)

        # Measure inference time
        start_time = time.time()
        model.predict(x_test)
        inference_time = time.time() - start_time

        # Capture results
        results = {
            'acc': str(history.history['accuracy']),
            'loss': str(history.history['loss']),
            'val_acc': str(history.history['val_accuracy']),
            'val_loss': str(history.history['val_loss']),
            'training_time': training_times,
            'inference_time': inference_time
        }

        print(results)

    except Exception as e:
        print(f"An error occurred during training: {e}", file=sys.stderr)

    print("BERT model complete.")

if __name__ == "__main__":
    main()
=======
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
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
base_model = VGG16(include_top=False, weights=None, input_shape=(32, 32, 3))
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
results_file = 'results/vgg16_results.csv'
log_results.write_results_to_csv(results_file, results)
print(f"Results written to {results_file}")
>>>>>>> origin/main
