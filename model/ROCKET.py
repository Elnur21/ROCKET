import numpy as np
from numba import jit, prange
import time
from sklearn.linear_model import RidgeClassifierCV

class ROCKET:
    def __init__(self):
        self.compiled=False


    def fit(self,df, num_runs = 10, num_kernels = 10_000):
        training_data = np.hstack((np.array(df[1]).reshape(-1,1),df[0]))
        test_data = np.hstack((np.array(df[3]).reshape(-1,1),df[2]))

        print(f"Performing runs".ljust(80 - 5, "."), end="")
        print("Done.")

        results = np.zeros(num_runs)
        timings = np.zeros([4, num_runs]) # training transform, test transform, training, test

        Y_training, X_training = training_data[:, 0].astype(np.int32), training_data[:, 1:]
        Y_test, X_test = test_data[:, 0].astype(np.int32), test_data[:, 1:]

        for i in range(num_runs):

            input_length = X_training.shape[1]
            kernels = generate_kernels(input_length, num_kernels)

            # -- transform training ------------------------------------------------

            time_a = time.perf_counter()
            X_training_transform = apply_kernels(X_training, kernels)
            time_b = time.perf_counter()
            timings[0, i] = time_b - time_a

            # -- transform test ----------------------------------------------------

            time_a = time.perf_counter()
            X_test_transform = apply_kernels(X_test, kernels)
            time_b = time.perf_counter()
            timings[1, i] = time_b - time_a

            # -- training ----------------------------------------------------------

            time_a = time.perf_counter()
            classifier = RidgeClassifierCV(alphas = 10 ** np.linspace(-3, 3, 10))
            classifier.fit(X_training_transform, Y_training)
            time_b = time.perf_counter()
            timings[2, i] = time_b - time_a

            # -- test --------------------------------------------------------------

            time_a = time.perf_counter()
            results[i] = classifier.score(X_test_transform, Y_test)
            time_b = time.perf_counter()
            timings[3, i] = time_b - time_a

            print(f"RUNNING".center(80, "="))

        return results, timings


    def compile(self, dataset_name, df):
        print(f"{dataset_name}".center(80, "-"))
        print(f"Loading data".ljust(80 - 5, "."), end = "", flush = True)
        training_data = np.hstack((np.array(df[1]).reshape(-1,1),df[0]))
        print("Done.")

        # -- precompile ------------------------------------------------------------

        if not self.compiled:

            print(f"Compiling ROCKET functions (once only)".ljust(80 - 5, "."), end = "", flush = True)

            _ = generate_kernels(100, 10)
            apply_kernels(np.zeros_like(training_data), _)

            print("Done.")
        else:
            print("Already compiled.")


@jit
def generate_kernels(input_length, num_kernels):
    candidate_lengths = np.array((7, 9, 11))
    # initialise kernel parameters
    weights = np.zeros((num_kernels, candidate_lengths.max())) # see note
    lengths = np.zeros(num_kernels, dtype = np.int32) # see note
    biases = np.zeros(num_kernels)
    dilations = np.zeros(num_kernels, dtype = np.int32)
    paddings = np.zeros(num_kernels, dtype = np.int32)
    # note: only the first *lengths[i]* values of *weights[i]* are used
    for i in range(num_kernels):
        length = np.random.choice(candidate_lengths)
        _weights = np.random.normal(0, 1, length)
        bias = np.random.uniform(-1, 1)
        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) // (length - 1)))
        padding = ((length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        weights[i, :length] = _weights - _weights.mean()
        lengths[i], biases[i], dilations[i], paddings[i] = length, bias, dilation, padding
    return weights, lengths, biases, dilations, paddings

@jit(fastmath = True)
def apply_kernel( X, weights, length, bias, dilation, padding):
    # zero padding
    if padding > 0:
        _input_length = len(X)
        _X = np.zeros(_input_length + (2 * padding))
        _X[padding:(padding + _input_length)] = X
        X = _X
    input_length = len(X)
    output_length = input_length - ((length - 1) * dilation)
    _ppv = 0 # "proportion of positive values"
    _max = np.NINF
    for i in range(output_length):
        _sum = bias
        for j in range(length):
            _sum += weights[j] * X[i + (j * dilation)]
        if _sum > 0:
            _ppv += 1
        if _sum > _max:
            _max = _sum
    return _ppv / output_length, _max

@jit(parallel = True, fastmath = True)
def apply_kernels(X, kernels):
    weights, lengths, biases, dilations, paddings = kernels
    num_examples = len(X)
    num_kernels = len(weights)
    # initialise output
    _X = np.zeros((num_examples, num_kernels * 2)) # 2 features per kernel
    for i in prange(num_examples):
        for j in range(num_kernels):
            _X[i, (j * 2):((j * 2) + 2)] = \
            apply_kernel(X[i], weights[j][:lengths[j]], lengths[j], biases[j], dilations[j], paddings[j])
    return _X