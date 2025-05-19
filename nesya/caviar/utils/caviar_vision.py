import os
import nnf
import cv2
import torch
import warnings
import itertools
import torchvision
import numpy as np
from collections import Counter
from deepfa.automaton import DeepFA
from nesya.caviar.utils.caviar_utils import cache_caviar_raw, parse_caviar_dataset


def load_vision(
    data,
    root_dir: str,
    debug: bool,
    debug_dir: str,
    remove_keys: list[str] = [
        "fomdgt2:2",
        "fomdgt2:3",
        "lb1gt:1",
        "lb2gt:0",
        "lb2gt:1",
        "lbgt:0",
        "lbpugt:0",
        "lbpugt:1",
    ],
):
    new_data = {}
    if debug and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    for key, value in data.items():
        if key in remove_keys:
            continue
        video_file = value["video_file"]
        cap = cv2.VideoCapture(os.path.join(root_dir, video_file))

        if not cap.isOpened():
            warnings.warn(
                "Could not open video file: {} in {}. Skipping ...".format(
                    video_file, root_dir
                )
            )
            continue

        frames = []
        while cap.isOpened():
            succeed, frame = cap.read()
            if succeed:
                frames.append(frame)
            else:
                break

        display_frames = []

        new_frames = []
        for complex_event_frame in value["history"]:
            current_image = frames[complex_event_frame["frame_id"]]
            new_objects = []
            for object in complex_event_frame["objects"]:
                x, y, w, h = (
                    object["box_xcenter"],
                    object["box_ycenter"],
                    object["box_width"],
                    object["box_height"],
                )
                if debug:
                    cv2.rectangle(
                        current_image,
                        (x - w // 2, y - h // 2),
                        (x + w // 2, y + h // 2),
                        (255, 0, 0),
                        4,
                    )
                object_bounding_box = cv2.resize(
                    current_image[y - h // 2 : y + h // 2, x - w // 2 : x + w // 2, :],
                    (80, 80),
                    interpolation=cv2.INTER_LINEAR,
                )
                new_objects.append(object | {"bounding_box": object_bounding_box})

            new_frames.append(complex_event_frame | {"objects": new_objects})

            if debug:
                display_frames.append(current_image)

        if debug:
            torchvision.io.write_video(
                os.path.join(debug_dir, key + ".mp4"),
                torch.Tensor(np.array(display_frames)).int(),
                25,
            )

        new_data[key] = data[key] | {"history": new_frames}

    return {key: value for key, value in new_data.items() if key not in remove_keys}


def generate_constraint(constrained_vars: list[nnf.Var]):
    return nnf.And(
        [
            nnf.And(
                [
                    (~v1 | ~v2)
                    for v1, v2 in itertools.combinations(constrained_vars, r=2)
                ]
            ),
            nnf.Or(constrained_vars),
        ],
    )


(
    p1_walking,
    p1_running,
    p1_active,
    p1_inactive,
    p2_walking,
    p2_running,
    p2_active,
    p2_inactive,
    close_p1_p2,
) = (
    nnf.Var(var_name)
    for var_name in [
        "p1_walking",
        "p1_running",
        "p1_active",
        "p1_inactive",
        "p2_walking",
        "p2_running",
        "p2_active",
        "p2_inactive",
        "close_p1_p2",
    ]
)

constraint_1, constraint_2 = generate_constraint(
    [p1_walking, p1_running, p1_active, p1_inactive]
), generate_constraint([p2_walking, p2_running, p2_active, p2_inactive])

t12 = p1_walking & p2_walking & close_p1_p2
t13 = close_p1_p2 & (p1_active | p1_inactive) & (p2_active | p2_inactive)

t21 = (
    (~close_p1_p2 & (p1_walking | p2_walking))
    | (p1_active & (p2_active | p2_inactive))
    | (p1_inactive & p2_active)
    | p1_running
    | p2_running
)

t31 = ~close_p1_p2 & (p1_walking | p2_walking) | p1_running | p2_running

t11 = t12.negate() & t13.negate() & constraint_1 & constraint_2
t22 = t21.negate() & constraint_1 & constraint_2
t33 = t31.negate() & constraint_1 & constraint_2

t12 &= constraint_1 & constraint_2
t13 &= constraint_1 & constraint_2
t21 &= constraint_1 & constraint_2
t31 &= constraint_1 & constraint_2

transitions = {
    0: {
        0: t11,
        1: t12,
        2: t13,
    },
    1: {0: t21, 1: t22},
    2: {0: t31, 2: t33},
}

deepfa = DeepFA(transitions, 0, {1})

int2label = {0: "no_event", 1: "moving", 2: "interacting"}


def load_caviar_vision(
    debug: bool = False,
    debug_dir: str = os.path.expanduser("~/.cache/caviar_raw/debug"),
    test_files: list[str] = [
        "fomdgt2:0",
        "fra2gt:0",
        "lb1gt:0",
        "wk2gt:0",
        "mwt1gt:0",
    ],
    complex_events: list[str] = ["moving", "interacting"],
):
    root_dir = cache_caviar_raw()
    vision_data = load_vision(
        parse_caviar_dataset(root_dir), root_dir, debug, debug_dir
    )
    filtered_data = {
        key: value
        for key, value in vision_data.items()
        if any(
            event in set([frame["complex_event"] for frame in value["history"]])
            for event in complex_events
        )
    }

    simple_event2id = {"walking": 0, "running": 1, "active": 2, "inactive": 3}

    for key, video_data in filtered_data.items():
        weights = (
            {
                key: value
                for key, value in zip(
                    ("p1_walking", "p1_running", "p1_active", "p1_inactive"),
                    torch.nn.functional.one_hot(
                        torch.tensor(
                            [
                                simple_event2id[frame["objects"][0]["simple_event"]]
                                for frame in video_data["history"]
                            ]
                        ),
                        num_classes=4,
                    ).T,
                )
            }
            | {
                key: value
                for key, value in zip(
                    ("p2_walking", "p2_running", "p2_active", "p2_inactive"),
                    torch.nn.functional.one_hot(
                        torch.tensor(
                            [
                                simple_event2id[frame["objects"][1]["simple_event"]]
                                for frame in video_data["history"]
                            ]
                        ),
                        num_classes=4,
                    ).T,
                )
            }
            | {
                "close_p1_p2": torch.tensor(
                    [frame["distance"] < 25 for frame in video_data["history"]]
                ).int()
            }
        )

        def labelling_function(var: nnf.Var) -> torch.Tensor:
            # This is very much like the standard labelling function
            # introduced above but we always give the value 1 for
            # negative literals. It's just the way it is.
            if str(var.name).startswith("p"):
                return (
                    weights[str(var.name)]
                    if var.true
                    else torch.ones_like(weights[str(var.name)])
                )

            return (
                weights[str(var.name)] if var.true else 1 - weights[str(var.name)]
            ).squeeze(-1)

        state_probs = deepfa.forward(labelling_function, accumulate=True).squeeze(1)
        for i, frame in enumerate(video_data["history"]):
            frame["silver_event"] = int2label[state_probs.argmax(dim=-1)[i].item()]

        print("video file: {}, test: [{}]".format(key, key in test_files))

        print(
            "true label support: {}".format(
                Counter(
                    [
                        (
                            frame["complex_event"]
                            if frame["complex_event"] in ("moving", "interacting")
                            else "no_event"
                        )
                        for frame in video_data["history"]
                    ]
                )
            ),
        )

        print(
            "silver label support: {}".format(
                Counter([(frame["silver_event"]) for frame in video_data["history"]])
            ),
        )

        print("----------------------------------------------------")

    train_set, test_set = {
        key: value for key, value in filtered_data.items() if key not in test_files
    }, {key: value for key, value in filtered_data.items() if key in test_files}

    print(
        "train true label distribution: {}".format(
            Counter(
                [
                    (
                        frame["complex_event"]
                        if frame["complex_event"] in ("moving", "interacting")
                        else "no_event"
                    )
                    for video_data in train_set.values()
                    for frame in video_data["history"]
                ]
            )
        )
    )

    print(
        "test true label distribution: {}".format(
            Counter(
                [
                    (
                        frame["complex_event"]
                        if frame["complex_event"] in ("moving", "interacting")
                        else "no_event"
                    )
                    for video_data in test_set.values()
                    for frame in video_data["history"]
                ]
            )
        )
    )

    print(
        "train silver label distribution: {}".format(
            Counter(
                [
                    frame["silver_event"]
                    for video_data in train_set.values()
                    for frame in video_data["history"]
                ]
            )
        )
    )

    print(
        "test silver label distribution: {}".format(
            Counter(
                [
                    frame["silver_event"]
                    for video_data in test_set.values()
                    for frame in video_data["history"]
                ]
            )
        )
    )

    return train_set, test_set


if __name__ == "__main__":
    train_data, test_data = load_caviar_vision(debug=True)
    print()
