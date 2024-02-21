
from utils.helper import *
from utils.constants import *
from model.ROCKET import ROCKET


df = read_dataset("ArrowHead")

model=ROCKET()

model.compile("ArrowHead",df)

for dataset_name in UNIVARIATE_DATASET_NAMES_2018:

    print(f"{dataset_name}".center(80, "-"))
    print(f"Loading data".ljust(80 - 5, "."), end = "", flush = True)
    df = read_dataset(dataset_name)
    print("Done.")
    results, timings = model.fit(df)
    timings_mean = timings.mean(1)
    print(f"{dataset_name} TRAINING FINISHED".center(80, "-"))

    


print(f"ALL TRAINING FINISHED".center(80, "="))
