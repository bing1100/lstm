# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
import time
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--update", dest='update_labels', action='store_true',
                        help='Make updates to labels')
    parser.set_defaults(update_labels=False)
    
    return parser

class LSTM(tf.keras.Model):
    def __init__(self):
        super(LSTM, self).__init__()
        self.embedding = tf.keras.layers.Embedding(top_words, 
                                                   embedding_vector_length, 
                                                   input_length=max_review_length)
        self.lstm = tf.keras.layers.LSTM(100)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        return self.dense(x)
        
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

opts = get_argparser().parse_args()
top_words = 5000
max_review_length = 500
embedding_vector_length = 32
num_folds = 10
batch_size = 64

# Instantiate an optimizer.
optimizer = Adam()
# Instantiate a loss function.
loss_fn = BinaryCrossentropy()

# Prepare the metrics.
train_acc_metric = tf.keras.metrics.BinaryAccuracy()
val_acc_metric = tf.keras.metrics.BinaryAccuracy()
test_acc_metric = tf.keras.metrics.BinaryAccuracy()

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
print(X_train.shape)
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

x_val = X_train[-5000:]
y_val = tf.cast(y_train[-5000:], tf.float32)
x_train = X_train[:-5000]
y_train = tf.cast(y_train[:-5000], tf.float32)

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)

# Prepare the test dataset.
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

model = LSTM()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 5
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()
    enum_training_dataset = list(enumerate(train_dataset))
    list_training_dataset = [[step, x_batch_train, y_batch_train, y_batch_train] for step, (x_batch_train, y_batch_train) in enum_training_dataset]

    # Iterate over the batches of the dataset.
    for step, x_batch_train, y_batch_train, updated_y_batch_train in list_training_dataset:
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(updated_y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric.
        train_acc_metric.update_state(y_batch_train, logits)

        # Log every batch.
        print(
            "Training loss (for one batch) at step %d: %.4f"
            % (step, float(loss_value))
        )
        print("Seen so far: %d samples" % ((step + 1) * batch_size))
        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        
        
        if step % 30 == 0 and step != 0:
            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in val_dataset:
                val_logits = model(x_batch_val, training=False)
                # Update val metrics
                val_acc_metric.update_state(y_batch_val, val_logits)
            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()
            print("Validation acc: %.4f" % (float(val_acc),))
            print("Time taken: %.2fs" % (time.time() - start_time))
            
            if opts.update_labels:
                for step1, x_batch_train, y_batch_train, updated_y_batch_train in list_training_dataset:
                    size = x_batch_train.shape[0]
                    logits = model(x_batch_train, training=False)
                    diff = (tf.reshape(updated_y_batch_train,(size,1)) - logits) * val_acc**(20)
                    list_training_dataset[step1][3] = tf.reshape(updated_y_batch_train,(size,1)) - diff
                    
                    if step1 % 50 == 0:
                        print("Updated so far:", step1)
                    
                print("Updating Completed")
        
        if step == 150:
            scores = model.evaluate(X_test, y_test, verbose=0)
            print("Testing Accuracy: %.2f%%" % (scores[1]*100))
            
    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Testing Accuracy: %.2f%%" % (scores[1]*100))