import os
import json
import torch
import random
import contextlib
from nesya.caviar.neural.neural import TransformerCNN
from sklearn.metrics import classification_report, f1_score
from nesya.caviar.utils.caviar_vision import load_caviar_vision

train_data, test_data = load_caviar_vision()
device = torch.device("cuda:1")

results_dir = os.path.realpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "results"
    )
)
print("running on result dir: {}".format(results_dir))


def forward_data(model, data, with_grad: bool = True):

    bb1, bb2 = (
        torch.stack(
            [
                torch.tensor(frame["objects"][0]["bounding_box"])
                for frame in data["history"]
            ]
        )
        .permute(0, 3, 1, 2)
        .to(device)
        / 255,
        torch.stack(
            [
                torch.tensor(frame["objects"][1]["bounding_box"])
                for frame in data["history"]
            ]
        )
        .permute(0, 3, 1, 2)
        .to(device)
        / 255,
    )

    with contextlib.nullcontext() if with_grad else torch.no_grad():

        model_predictions = model(
            bb1,
            bb2,
            (
                torch.tensor([frame["distance"] for frame in video_data["history"]]).to(
                    device
                )
                < 25
            ).float(),
        )

    return model_predictions


label2int = {"no_event": 0, "moving": 1, "interacting": 2}

results = {}
# for lr in (0.001, 0.0001, 0.00001):
for lr in (0.00001,):
    results[lr] = {}
    for random_seed in (
        12,
        42,
        55,
        65,
        70,
        120,
        140,
        150,
        200,
        220,
        300,
        350,
        400,
        450,
        600,
        650,
        720,
        760,
        800,
        850,
        910,
        930,
        940,
        950,
        960,
        970,
        980,
        990,
        1000,
        1100,
    ):
        results[lr][random_seed] = {}
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        model = TransformerCNN(3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_weights = model.state_dict()

        print("training for lr: {} and seed: {}".format(lr, random_seed))
        current_epoch, loss_per_epoch, patience = 1, [], 10
        while True:
            loss_epochs = []
            train_predictions, train_labels = [], []
            test_predictions, test_labels = [], []
            for video_data in train_data.values():
                optimizer.zero_grad()
                labels = torch.tensor(
                    [
                        (label2int[(frame["silver_event"])])
                        for frame in video_data["history"]
                    ]
                ).to(device)

                model_predictions = forward_data(model, video_data)
                train_labels.extend(labels.cpu().tolist())
                train_predictions.extend(
                    model_predictions.argmax(dim=-1).cpu().tolist()
                )
                loss = torch.nn.functional.cross_entropy(model_predictions, labels)
                loss.backward()

                optimizer.step()
                loss_epochs.append(loss.cpu().item())

            test_loss_epochs = []
            for video_data in test_data.values():
                labels = torch.tensor(
                    [
                        (label2int[(frame["silver_event"])])
                        for frame in video_data["history"]
                    ]
                ).to(device)

                model_predictions = forward_data(model, video_data, with_grad=False)
                loss = torch.nn.functional.cross_entropy(model_predictions, labels)

                test_labels.extend(labels.cpu().tolist())
                test_predictions.extend(model_predictions.argmax(dim=-1).cpu().tolist())
                test_loss_epochs.append(loss.cpu().item())

            print(
                "epoch: {}, loss: {:.2f}, [test] loss: {:.2f}, f1: {:.2f}, [test] f1: {:.2f}".format(
                    current_epoch,
                    torch.tensor(loss_epochs).mean().item(),
                    torch.tensor(test_loss_epochs).mean().item(),
                    f1_score(train_labels, train_predictions, average="macro"),
                    f1_score(test_labels, test_predictions, average="macro"),
                )
            )
            current_epoch += 1
            loss_per_epoch.append(
                round(float(torch.tensor(loss_epochs).mean().item()), 2)
            )

            if current_epoch > patience + 1 and all(
                past_loss >= min(loss_per_epoch[:-patience])
                for past_loss in loss_per_epoch[-patience:]
            ):
                print("early stopping on epoch {}".format(current_epoch))
                break

            if (loss_per_epoch[-1] <= past_loss for past_loss in loss_per_epoch[:-1]):
                best_weights = model.state_dict()

        results[lr][random_seed] = {
            "train": f1_score(train_labels, train_predictions, average="macro"),
            "test": f1_score(test_labels, test_predictions, average="macro"),
        }

        test_predictions, test_labels = [], []
        for video_data in test_data.values():
            labels = torch.tensor(
                [
                    (label2int[(frame["silver_event"])])
                    for frame in video_data["history"]
                ]
            ).to(device)

            state_probs = forward_data(model, video_data, with_grad=False)
            test_labels.extend(labels.cpu().tolist())
            test_predictions.extend(state_probs.argmax(dim=-1).cpu().tolist())

        print(classification_report(test_labels, test_predictions))

        with open(
            os.path.join(results_dir, "results_transformer_multi.json"), "w"
        ) as output_file:
            json.dump(results, output_file, indent=2)
