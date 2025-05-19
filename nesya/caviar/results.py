import os
import json
import numpy as np

results_dir = os.path.realpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "results"
    )
)

print("NeSyA results")

with open(os.path.join(results_dir, "results_nesy.json"), "r") as input_file:
    data = json.load(input_file)


for lr, lr_data in data.items():
    train_f1s = np.array([value["train"] for value in lr_data.values()])
    test_f1s = np.array([value["test"] for value in lr_data.values()])

    best_train_f1s = (-np.sort(-train_f1s))[:8]
    best_test_f1s = (-np.sort(-test_f1s))[:8]

    print("-------------------------------------------------")
    print("num runs: {}".format(len(train_f1s)))
    print("lr: {}".format(lr))
    print("train, mean: {}, std: {}".format(train_f1s.mean(), train_f1s.std()))
    print("test, mean: {}, std: {}".format(test_f1s.mean(), test_f1s.std()))
    print(
        "test best: mean: {}, std: {}".format(
            best_test_f1s.mean(), best_train_f1s.std()
        )
    )


print("------------------------------------")
print("CNN-LSTM")

with open(os.path.join(results_dir, "results_neural.json"), "r") as input_file:
    data = json.load(input_file)


for lr, lr_data in data.items():
    train_f1s = np.array([value["train"] for value in lr_data.values()])
    test_f1s = np.array([value["test"] for value in lr_data.values()])

    best_train_f1s = (-np.sort(-train_f1s))[:8]
    best_test_f1s = (-np.sort(-test_f1s))[:8]

    print("-------------------------------------------------")
    print("num runs: {}".format(len(train_f1s)))
    print("lr: {}".format(lr))
    print("train, mean: {}, std: {}".format(train_f1s.mean(), train_f1s.std()))
    print("test, mean: {}, std: {}".format(test_f1s.mean(), test_f1s.std()))
    print(
        "test best: mean: {}, std: {}".format(
            best_test_f1s.mean(), best_train_f1s.std()
        )
    )

print("------------------------------------")
print("CNN-Transformer")

with open(os.path.join(results_dir, "results_transformer.json"), "r") as input_file:
    data = json.load(input_file)


for lr, lr_data in data.items():
    train_f1s = np.array([value["train"] for value in lr_data.values()])
    test_f1s = np.array([value["test"] for value in lr_data.values()])

    best_train_f1s = (-np.sort(-train_f1s))[:8]
    best_test_f1s = (-np.sort(-test_f1s))[:8]

    print("-------------------------------------------------")
    print("num runs: {}".format(len(train_f1s)))
    print("lr: {}".format(lr))
    print("train, mean: {}, std: {}".format(train_f1s.mean(), train_f1s.std()))
    print("test, mean: {}, std: {}".format(test_f1s.mean(), test_f1s.std()))
    print(
        "test best: mean: {}, std: {}".format(
            best_test_f1s.mean(), best_train_f1s.std()
        )
    )
