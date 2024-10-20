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
