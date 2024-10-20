import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
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
