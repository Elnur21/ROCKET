from test import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

def train(X,
          Y,
          X_validation,
          Y_validation,
          kernels,
          num_features,
          num_classes,
          minibatch_size = 256,
          max_epochs = 100,
          patience = 2,           # * 10 minibatches; 2 -> 2 * 10 = 20 minibatches; reset if loss improves
          tranche_size = 2 ** 11,
          cache_size = 2 ** 14):  # as much as possible


    

    # -- model -----------------------------------------------------------------

    model = Model(Dense(num_classes, input_shape=(num_features,))) # logistic / softmax regression
    loss_function = CategoricalCrossentropy()
    optimizer = Adam()
    scheduler = ReduceLROnPlateau(monitor="lr", factor = 0.5, min_lr = 1e-8)

    # -- run -------------------------------------------------------------------

    minibatch_count = 0
    best_validation_loss = np.inf
    stall_count = 0
    stop = False

    num_examples = len(X)
    num_tranches = int(np.ceil(num_examples / tranche_size))

    cache = np.zeros((min(cache_size, num_examples), num_features))
    cache_count = 0

    for epoch in range(max_epochs):

        if epoch > 0 and stop:
            break

        for tranche_index in range(num_tranches):

            if epoch > 0 and stop:
                break

            a = tranche_size * tranche_index
            b = a + tranche_size

            Y_tranche = Y[a:b]

            # if cached, use cached transform; else transform and cache the result
            if b <= cache_count:

                X_tranche_transform = cache[a:b]

            else:

                X_tranche = X[a:b]
                X_tranche = (X_tranche - X_tranche.mean(axis = 1, keepdims = True)) / X_tranche.std(axis = 1, keepdims = True) # normalise time series
                X_tranche_transform = apply_kernels(X_tranche, kernels)

                if epoch == 0 and tranche_index == 0:

                    # per-feature mean and standard deviation (estimated on first tranche)
                    f_mean = X_tranche_transform.mean(0)
                    f_std = X_tranche_transform.std(0) + 1e-8

                    # normalise and transform validation data
                    X_validation = (X_validation - X_validation.mean(axis = 1, keepdims = True)) / X_validation.std(axis = 1, keepdims = True) # normalise time series
                    X_validation_transform = apply_kernels(X_validation, kernels)
                    X_validation_transform = (X_validation_transform - f_mean) / f_std  # normalise transformed features
                    X_validation_transform = tf.convert_to_tensor(X_validation_transform)
                    Y_validation = tf.convert_to_tensor(Y_validation)


                X_tranche_transform = (X_tranche_transform - f_mean) / f_std # normalise transformed features

                if b <= cache_size:

                    cache[a:b] = X_tranche_transform
                    cache_count = b

            X_tranche_transform = tf.convert_to_tensor(X_tranche_transform)
            Y_tranche = tf.convert_to_tensor(Y_tranche)

            minibatches = tf.random.shuffle(tf.range(len(X_tranche_transform)))

            for minibatch_index, minibatch in enumerate(minibatches):

                if epoch > 0 and stop:
                    break

                # abandon undersized minibatches
                if minibatch_index > 0 and len(minibatch) < minibatch_size:
                    break

                # -- training --------------------------------------------------

                optimizer.zero_grad()
                Y_tranche_predictions = model(X_tranche_transform[minibatch])
                training_loss = loss_function(Y_tranche_predictions, Y_tranche[minibatch])
                gradients = tf.GradientTape().gradient(training_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))


                minibatch_count += 1

                if minibatch_count % 10 == 0:

                    Y_validation_predictions = model(X_validation_transform)
                    validation_loss = loss_function(Y_validation_predictions, Y_validation)

                    scheduler.on_epoch_end(epoch, logs={'val_loss': validation_loss})

                    if validation_loss.item() >= best_validation_loss:
                        stall_count += 1
                        if stall_count >= patience:
                            stop = True
                    else:
                        best_validation_loss = validation_loss.item()
                        if not stop:
                            stall_count = 0

    return model, f_mean, f_std


import numpy as np
import pandas as pd

# Function to generate synthetic time series data
def generate_time_series(num_examples, num_features, seq_length):
    X = np.random.randn(num_examples, seq_length, num_features)
    return X

# Generate synthetic data
num_examples = 1000
num_features = 10
seq_length = 50

X = generate_time_series(num_examples, num_features, seq_length)
Y = np.random.randint(0, 2, size=num_examples)  # Example binary classification labels

# Split the data into training and validation sets
X_train, X_valid, Y_train, Y_valid = X[:800], X[800:], Y[:800], Y[800:]

#)

# Assuming 'train' function is defined and model is trained, you can now test the model
# Let's say 'train' function has returned trained model, f_mean, f_std

# Assuming 'model', 'f_mean', and 'f_std' are returned from the train function
# Example of testing the model
X_test = generate_time_series(100, num_features, seq_length)  # Generate test data

trained_model, f_mean, f_std = train(X_train, Y_train, X_valid, Y_valid, 10000, num_features, seq_length)


print(f_mean)