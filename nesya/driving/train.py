import os
import nnf
import time
import json
import torch
import random
import torchmetrics
from torch.utils.data import DataLoader
from deepfa.utils import parse_sapienza_to_fa
from nesya.driving.data.dataset import SubsymbolicTrajectoriesDriving
from nesya.driving.fuzzy import UmiliFSA, final_states_loss, not_final_states_loss

results_dir = os.path.realpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "results"
    )
)

print("writing in results directory: {}".format(results_dir))

formulae = [
    "G((tired | blocked) -> WX(!fast))",
    "G((!(fast) | !(gesture_ok)) -> WX((blocked | tired) U (fast | gesture_ok)))",
    "G((battery_low | night | blocked | tired) -> WX(!fast)) & G(fast & WX(fast) -> WX(WX(battery_low)))",
]


class CNNSymbolGrounder(torch.nn.Module):
    def __init__(self, num_symbols: int):
        super().__init__()
        self.conv_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 16, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(144 + (240 * (num_symbols - 1)), 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_symbols),
            torch.nn.Sigmoid(),
        )

    def forward(self, input):
        return self.classifier(self.conv_encoder(input))


results = {}

for formula in formulae:
    for sequence_length in (10, 20, 30):
        for random_seed in (5, 10, 20, 30, 40):
            for do_fuzzy in (True, False):
                print(
                    "training for formula: {}, sequence_length: {}, seed: {}, fuzzy: {}".format(
                        formula, sequence_length, random_seed, do_fuzzy
                    )
                )
                start_time = time.time()

                random.seed(random_seed)
                torch.manual_seed(random_seed)

                deepfa = parse_sapienza_to_fa(formula)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                train_dataset, test_dataset = SubsymbolicTrajectoriesDriving(
                    deepfa, sequence_length=sequence_length, double_check=formula
                ), SubsymbolicTrajectoriesDriving(
                    deepfa, sequence_length=sequence_length, double_check=formula
                )

                train_dataloader, test_dataloader = DataLoader(
                    train_dataset, batch_size=16, shuffle=True
                ), DataLoader(train_dataset, batch_size=16)

                network = CNNSymbolGrounder(len(deepfa.symbols))
                network.to(device)
                optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
                num_epochs = 100

                symbols = list(deepfa.symbols)
                symbol_map = {
                    symbol: "c{}".format(i) for i, symbol in enumerate(symbols)
                }

                fuzzy_formula = formula
                for key, value in symbol_map.items():
                    fuzzy_formula = fuzzy_formula.replace(key, value)

                if do_fuzzy:
                    deepfa = UmiliFSA(fuzzy_formula, number_of_symbols=len(symbols))

                for epoch_num in range(num_epochs):
                    epoch_losses, epoch_accuracies = [], []
                    for batch_images, batch_labels, _ in train_dataloader:
                        optimizer.zero_grad()

                        (batch_images, batch_labels) = batch_images.to(
                            device
                        ), batch_labels.to(device)

                        nn_predictions = network(
                            batch_images.reshape(-1, *batch_images.shape[-3:])
                        ).reshape(*batch_images.shape[:-3], -1)

                        weights = {
                            symbol: symbol_probs
                            for symbol, symbol_probs in zip(
                                train_dataloader.dataset.fa_symbols,
                                nn_predictions.permute(2, 0, 1),
                            )
                        }

                        def labelling_function(var: nnf.Var | str) -> torch.Tensor:

                            if isinstance(var, str):
                                return weights[var]

                            return (
                                weights[str(var.name)]
                                if var.true
                                else 1 - weights[str(var.name)]
                            )

                        if not do_fuzzy:
                            acceptance_score = deepfa.forward(labelling_function)
                            acceptance_score = acceptance_score.clip(1e-32, 1 - 1e-32)
                            loss = torch.nn.functional.binary_cross_entropy(
                                acceptance_score,
                                batch_labels.squeeze(-1).float(),
                            )
                        else:
                            batch_losses = torch.zeros(batch_images.shape[0]).to(device)
                            for element in range(batch_images.shape[0]):
                                symbol_probs = torch.zeros(
                                    (sequence_length, len(symbols))
                                ).to(device)
                                for key, value in symbol_map.items():
                                    symbol_probs[:, int(value[1])] = weights[key][
                                        element
                                    ]

                                final_state_predictions = deepfa.forward(symbol_probs)  # type: ignore
                                batch_losses[element] = (
                                    final_states_loss(
                                        deepfa.final_states, final_state_predictions  # type: ignore
                                    )
                                    if batch_labels[element].item()
                                    else not_final_states_loss(
                                        deepfa.final_states, final_state_predictions  # type: ignore
                                    )
                                )

                            loss = batch_losses.mean()

                        loss.backward()
                        optimizer.step()
                        epoch_losses.append(loss.item())
                        if not do_fuzzy:
                            epoch_accuracies.append(
                                torchmetrics.functional.accuracy(
                                    acceptance_score,  # type: ignore
                                    batch_labels.squeeze(-1),
                                    task="binary",
                                )
                            )
                        else:
                            predictions = []
                            for element in range(batch_images.shape[0]):
                                predictions.append(
                                    deepfa.original_dfa.accepts(
                                        [
                                            {
                                                value: (
                                                    weights[key][element][i] > 0.5
                                                ).item()
                                                for key, value in symbol_map.items()
                                            }
                                            for i in range(sequence_length)
                                        ]
                                    )
                                )
                            epoch_accuracies.append(
                                torchmetrics.functional.accuracy(
                                    torch.tensor(predictions).float().to(device),
                                    batch_labels.squeeze(-1),
                                    task="binary",
                                )
                            )

                    test_accuracies = []
                    for batch_images, batch_labels, _ in test_dataloader:

                        (batch_images, batch_labels) = batch_images.to(
                            device
                        ), batch_labels.to(device)

                        nn_predictions = network(
                            batch_images.reshape(-1, *batch_images.shape[-3:])
                        ).reshape(*batch_images.shape[:-3], -1)

                        weights = {
                            symbol: symbol_probs
                            for symbol, symbol_probs in zip(
                                train_dataloader.dataset.fa_symbols,
                                nn_predictions.permute(2, 0, 1),
                            )
                        }

                        def labelling_function(var: nnf.Var | str) -> torch.Tensor:

                            if isinstance(var, str):
                                return weights[var]

                            return (
                                weights[str(var.name)]
                                if var.true
                                else 1 - weights[str(var.name)]
                            )

                        if not do_fuzzy:
                            acceptance_score = deepfa.forward(labelling_function)
                            test_accuracies.append(
                                torchmetrics.functional.accuracy(
                                    acceptance_score,  # type: ignore
                                    batch_labels.squeeze(-1),
                                    task="binary",
                                )
                            )
                        else:
                            predictions = []
                            for element in range(batch_images.shape[0]):
                                predictions.append(
                                    deepfa.original_dfa.accepts(  # type: ignore
                                        [
                                            {
                                                value: (
                                                    weights[key][element][i] > 0.5
                                                ).item()
                                                for key, value in symbol_map.items()
                                            }
                                            for i in range(sequence_length)
                                        ]
                                    )
                                )
                            test_accuracies.append(
                                torchmetrics.functional.accuracy(
                                    torch.tensor(predictions).float().to(device),
                                    batch_labels.squeeze(-1),
                                    task="binary",
                                )
                            )

                    results.setdefault(formula, {}).setdefault(
                        sequence_length, {}
                    ).setdefault(random_seed, {}).setdefault(
                        "fuzzy" if do_fuzzy else "nesya", {}
                    ).setdefault(
                        "metrics", []
                    ).append(
                        {
                            "loss": torch.tensor(epoch_losses).mean().item(),
                            "test_accuracy": torch.tensor(test_accuracies)
                            .mean()
                            .item(),
                        }
                    )

                    print(
                        "epoch [{}]/[{}], loss: {:.2f}, test-accuracy: {:.2f}".format(
                            epoch_num + 1,
                            num_epochs,
                            torch.tensor(epoch_losses).mean().item(),
                            torch.tensor(test_accuracies).mean().item(),
                        )
                    )

                    with open(
                        os.path.join(results_dir, "performance_results_2.json"), "w"
                    ) as output_file:
                        json.dump(results, output_file, indent=2)

                results[formula][sequence_length][random_seed][
                    "fuzzy" if do_fuzzy else "nesya"
                ]["time"] = (time.time() - start_time)
