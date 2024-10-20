import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2Model, TFGPT2LMHeadModel
import os
import time
import json
import sys

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

    # Load pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = TFGPT2LMHeadModel.from_pretrained("gpt2")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=model.compute_loss)

    # Sample input text for training (in practice, this should be a dataset)
    inputs = tokenizer(["Once upon a time", "In a galaxy far, far away"], return_tensors="tf", padding=True, truncation=True)
    inputs['labels'] = inputs['input_ids']

    # Train the model and capture metrics
    training_times = []
    history = None

    try:
        start_time = time.time()
        history = model.fit(inputs, epochs=1, batch_size=batch_size, verbose=1)
        training_time = time.time() - start_time
        training_times.append(training_time)

        # Measure inference time
        start_time = time.time()
        _ = model.generate(inputs['input_ids'], max_length=50)
        inference_time = time.time() - start_time

        # Capture results
        results = {
            'loss': str(history.history['loss']),
            'training_time': training_times,
            'inference_time': inference_time
        }

        print(results)

    except Exception as e:
        print(f"An error occurred during training: {e}", file=sys.stderr)

    print("GPT-2 model complete.")

if __name__ == "__main__":
    main()
