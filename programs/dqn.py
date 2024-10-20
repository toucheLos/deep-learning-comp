import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
<<<<<<< HEAD
import gym
import os
import time
import json
import sys

def create_dqn_model(input_shape, num_actions):
    model = Sequential([
        Dense(24, input_shape=input_shape, activation='relu'),
        Dense(24, activation='relu'),
        Dense(num_actions, activation='linear')
    ])
    return model

def main():
    args = sys.argv
    gpus = int(args[1])
    batch_size = int(args[2])
    precision = args[3]
    learning_rate = float(args[4])

    # Enable mixed precision if TensorFlow version is 2.x and precision is FP16
    if tf.__version__.startswith('2') and precision == 'FP16':
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        print("Mixed Precision ON")

    env = gym.make('CartPole-v1')
    num_actions = env.action_space.n
    input_shape = (env.observation_space.shape[0],)

    model = create_dqn_model(input_shape, num_actions)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    # Parameters for DQN
    gamma = 0.95    # Discount rate
    epsilon = 1.0   # Exploration rate
    epsilon_min = 0.01
    epsilon_decay = 0.995
    episodes = 500

    # Experience replay memory
    memory = []

    # Store training times
    training_times = []

    try:
        start_time = time.time()

        for e in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, input_shape[0]])
            for time_t in range(500):
                if np.random.rand() <= epsilon:
                    action = np.random.choice(num_actions)
                else:
                    act_values = model.predict(state)
                    action = np.argmax(act_values[0])

                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, input_shape[0]])
                memory.append((state, action, reward, next_state, done))
                state = next_state

                if done:
                    print(f"Episode: {e}/{episodes}, score: {time_t}, e: {epsilon:.2}")
                    break

                if len(memory) > batch_size:
                    minibatch = np.random.choice(len(memory), batch_size, replace=False)
                    for i in minibatch:
                        state_mb, action_mb, reward_mb, next_state_mb, done_mb = memory[i]
                        target = reward_mb
                        if not done_mb:
                            target = (reward_mb + gamma * np.amax(model.predict(next_state_mb)[0]))
                        target_f = model.predict(state_mb)
                        target_f[0][action_mb] = target
                        model.fit(state_mb, target_f, epochs=1, verbose=0)

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

        training_time = time.time() - start_time
        training_times.append(training_time)

        # Since it's a RL task, we measure the average reward as an inference metric
        total_reward = 0
        num_tests = 10
        for _ in range(num_tests):
            state = env.reset()
            state = np.reshape(state, [1, input_shape[0]])
            for time_t in range(500):
                act_values = model.predict(state)
                action = np.argmax(act_values[0])
                next_state, reward, done, _ = env.step(action)
                state = np.reshape(next_state, [1, input_shape[0]])
                total_reward += reward
                if done:
                    break

        avg_reward = total_reward / num_tests

        results = {
            'avg_reward': avg_reward,
            'training_time': training_times
        }

        print(results)

    except Exception as e:
        print(f"An error occurred during training: {e}", file=sys.stderr)

    print("DQN model complete.")

if __name__ == "__main__":
    main()
=======
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

# Define the model
model = Sequential([
    Dense(24, input_dim=4, activation='relu'),
    Dense(24, activation='relu'),
    Dense(2, activation='linear')
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
    mixed_precision.set_global_policy(policy)

model.compile(optimizer=optimizer, loss='mse')

# Dummy data for training
x_train = np.random.random((batch_size, 4))
y_train = np.random.random((batch_size, 2))

# Train the model and capture metrics
training_times = []
history = None

try:
    for epoch in range(10):
        start_time = time.time()
        history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size, validation_split=0.2)
        epoch_training_time = time.time() - start_time
        training_times.append(epoch_training_time)

    # Measure inference time
    start_time = time.time()
    model.predict(x_train)
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
        'Model Accuracy': None,  # Update if applicable
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
results_file = 'results/dqn_results.csv'
log_results.write_results_to_csv(results_file, results)
print(f"Results written to {results_file}")


>>>>>>> origin/main
