import os
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "results"
)

with open(os.path.join(results_dir, "performance_results.json"), "r") as input_file:
    performance_data = json.load(input_file)

formula_map = {
    "G((tired | blocked) -> WX(!fast))": 1,
    "G((!(fast) | !(gesture_ok)) -> WX((blocked | tired) U (fast | gesture_ok)))": 2,
    "G((battery_low | night | blocked | tired) -> WX(!fast)) & G(fast & WX(fast) -> WX(WX(battery_low)))": 3,
}
#
for formula, formula_data in performance_data.items():
    for sequence_length, sequence_length_data in formula_data.items():
        fuzzy_data, nesya_data = [
            seed_data["fuzzy"]["metrics"][-1]["test_accuracy"]
            for seed_data in sequence_length_data.values()
        ], [
            seed_data["nesya"]["metrics"][-1]["test_accuracy"]
            for seed_data in sequence_length_data.values()
        ]

        fuzzy_time, nesya_time = [
            seed_data["fuzzy"]["time"]
            for seed_data in sequence_length_data.values()
            if "time" in seed_data["fuzzy"]
        ], [
            seed_data["nesya"]["time"]
            for seed_data in sequence_length_data.values()
            if "time" in seed_data["nesya"]
        ]

        data_list = []
        for seed, methods in sequence_length_data.items():
            for method, method_data in methods.items():
                for epoch, metric in enumerate(method_data["metrics"]):
                    data_list.append(
                        {
                            "Seed": seed,
                            "Method": method,
                            "Epoch": epoch,
                            "Loss": metric["loss"],
                        }
                    )

        df = pd.DataFrame(data_list)
        #
        #         sns.lineplot(
        #             data=df,
        #             x="Epoch",
        #             y="Loss",
        #             hue="Method",
        #             errorbar=("ci", 95),
        #             estimator="mean",
        #         )
        #
        #         plt.xlabel("Epoch number")
        #         plt.ylabel("Loss")
        #         plt.legend()
        #         plt.title(
        #             "NeSyA vs Fuzzy for formula: {} and sequence length {}".format(
        #                 formula_map[formula], sequence_length
        #             )
        #         )
        #         plt.savefig(
        #             os.path.join(
        #                 results_dir,
        #                 "1b_accuracy",
        #                 "{}_{}.png".format(formula_map[formula], sequence_length),
        #             )
        #         )
        #         plt.close()
        #
        print(
            "Formula: {}, Sequence Length: {}, [{}] NeSyA accuracy: {} ± {}, [{}] Fuzzy accuracy: {} ± {}, n-time: {:.2f}, f-time: {:.2f}".format(
                formula,
                sequence_length,
                len(nesya_data),
                np.mean(nesya_data),
                np.std(nesya_data),
                len(fuzzy_data),
                np.mean(fuzzy_data),
                np.std(fuzzy_data),
                np.mean(nesya_time) / 60,
                np.mean(fuzzy_time) / 60,
            )
        )

legend_labels = {
    f"{formula} DeepStochLog": f"DeepStochLog (P{index})"
    for formula, index in formula_map.items()
}
legend_labels.update(
    {f"{formula} NeSyA": f"NeSyA (P{index})" for formula, index in formula_map.items()}
)

with open(os.path.join(results_dir, "timing_results.json"), "r") as input_file:
    timing_data = json.load(input_file)

for formula, formula_data in timing_data.items():
    method_acceptance = {}
    for method, method_data in formula_data.items():
        for sequence_length, sequence_length_data in method_data.items():
            method_acceptance.setdefault(method, []).append(
                sequence_length_data["acceptance_prob"]
            )

    if not all(
        [
            torch.isclose(torch.tensor(m1), torch.tensor(m2)).all()
            for m1, m2 in zip(*method_acceptance.values())
        ]
    ):
        print("tensors are not close")


key_map = {"nesya": "NeSyA", "deepstochlog": "DeepStochLog"}
relevant_results = {}
for i, (formula, formula_results) in enumerate(timing_data.items()):
    for key, value in formula_results.items():
        if key == "deepproblog" or key == "compilation_time":
            continue
        evaluation_times = []
        for sequence_length, sequence_length_data in value.items():
            if (
                sequence_length == "compilation_time"
                or sequence_length_data == "timeout"
                or sequence_length == "1"
                or int(sequence_length) > 40
                or int(sequence_length) < 15
            ):
                continue
            relevant_results.setdefault(int(sequence_length), {})[
                formula + " " + key_map[key]
            ] = sequence_length_data["evaluation_time"]

df = pd.DataFrame.from_dict(relevant_results, orient="index")
df = df.reset_index().rename(columns={"index": "x"})
df_melted = df.melt(id_vars="x", var_name="variable", value_name="value")


unique_formulas = sorted(
    set(
        v.replace(" DeepStochLog", "").replace(" NeSyA", "")
        for v in df_melted["variable"].unique()
    )
)
palette = sns.color_palette("Dark2", len(unique_formulas))
color_map = {
    f"{formula} DeepStochLog": color for formula, color in zip(unique_formulas, palette)
}

for key, value in color_map.copy().items():
    color_map.update(
        {f"{formula} NeSyA": color for formula, color in zip(unique_formulas, palette)}
    )

# plt.figure(figsize=(10, 8))  # You can adjust this as needed
# sns.set_theme(rc={"figure.figsize": (6, 6)})
sns.lineplot(
    data=df_melted,
    x="x",
    y="value",
    hue="variable",
    marker="o",
    style="variable",
    dashes={
        key: "" if key.endswith("NeSyA") else (2, 2)
        for key in df_melted["variable"].unique()
    },
    palette=color_map,
)
plt.xlim(14.5, 40.5)
plt.yscale("log")
plt.ylim(10**-2, 10**3)
plt.grid(True, which="both", linestyle="-", linewidth=1)
plt.xlabel("Sequence Length", fontsize=13)
plt.ylabel("Update time (s)", fontsize=13)
plt.tick_params(axis="x", labelsize=11)
plt.tick_params(axis="y", labelsize=11)

# Custom legend
handles, labels = plt.gca().get_legend_handles_labels()
handles, labels = zip(*sorted(zip(handles, labels), key=lambda item: item[1][-5:]))
new_labels = [legend_labels.get(label, label) for label in labels]
plt.legend(
    handles,
    new_labels,
    loc="upper left",
    frameon=True,
    fontsize=10,
)


plt.tight_layout()
plt.savefig(os.path.join(results_dir, "scalability.png"))
