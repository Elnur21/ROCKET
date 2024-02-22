import numpy as np
import time
from sklearn.linear_model import RidgeClassifierCV

from utils.kernel import *

class ROCKET:
    def __init__(self):
        self.compiled=False


    def fit(self,df, num_runs = 10, num_kernels = 10_000):
        training_data = np.hstack((np.array(df[1]).reshape(-1,1),df[0]))
        test_data = np.hstack((np.array(df[3]).reshape(-1,1),df[2]))

        print(f"Performing runs".ljust(80 - 5, "."), end="")
        print("Done.")

        results_training = np.zeros(num_runs)
        results_test = np.zeros(num_runs)
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
            results_test[i] = classifier.score(X_test_transform, Y_test)
            results_training[i] = classifier.score(X_training_transform, Y_training)
            time_b = time.perf_counter()
            timings[3, i] = time_b - time_a

            print(f"RUNNING".center(80, "="))

        return results_training, results_test, timings


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
 