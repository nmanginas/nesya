import os
import torch
import torchvision
from PIL import Image
from typing import Optional
from deepfa.automaton import DeepFA
from torch.utils.data import Dataset
from flloat.parser.ltlf import LTLfParser
from deepfa.utils import generate_random_sequence


# Synthetically generate trajectories for a DeepFA pattern.
# Any pattern can be defined but must be over the symbols:
# { tired, blocked, fast, night, gesture_ok, battery_low }
# of a subset of them.
class SubsymbolicTrajectoriesDriving(Dataset):
    def __init__(
        self,
        fa: DeepFA,
        num_trajectories_per_class: int = 100,
        sequence_length: int = 10,
        double_check: Optional[str] = None,
    ):
        self.fa = fa
        self.num_trajectories_per_class = num_trajectories_per_class
        self.sequence_length = sequence_length

        if double_check is not None:
            parser = LTLfParser()
            self.double_check_sfa = parser(double_check).to_automaton()
        else:
            self.double_check_sfa = None

        self.trajectories = self.generate_subsymbolic_trajectories()
        self.fa_symbols = list(sorted(fa.symbols))

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_trajectory, current_label, symbol_labels = self.trajectories[index]
        current_images, current_symbols = [], []
        for symbol in self.fa_symbols:
            current_images.append(torch.stack(current_trajectory[symbol]))
            current_symbols.append(torch.tensor(symbol_labels[symbol]))

        return (
            torch.cat(current_images, dim=-2),
            torch.tensor([current_label]).int(),
            torch.stack(current_symbols),
        )

    def generate_subsymbolic_trajectories(self):
        assets_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets")

        load_image = lambda name: torchvision.transforms.Compose(
            (
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((40, 40)),
                torchvision.transforms.GaussianBlur(3),
            )
        )(Image.open(os.path.join(assets_dir, name)).convert("RGB"))

        image_map = {
            "fast": {
                True: load_image("racing_car.png"),
                False: load_image("snail.png"),
            },
            "tired": {True: load_image("sleeping.png"), False: load_image("smile.png")},
            "blocked": {
                True: load_image("construction.png"),
                False: load_image("motorway.png"),
            },
            "gesture_ok": {
                True: load_image("gesture_ok.png"),
                False: load_image("gesture_danger.png"),
            },
            "night": {True: load_image("moon.png"), False: load_image("sun.png")},
            "battery_low": {
                True: load_image("battery_empty.png"),
                False: load_image("battery_full.png"),
            },
        }

        trajectories = []
        for _ in range(self.num_trajectories_per_class):
            positive_trajectory = generate_random_sequence(
                self.fa, sequence_length=self.sequence_length
            )
            trajectories.append(
                (
                    {
                        key: [image_map[key][element] for element in boolean_mask]
                        for key, boolean_mask in positive_trajectory.items()
                    },
                    1,
                    positive_trajectory,
                )
            )
            negative_trajectory = generate_random_sequence(
                self.fa, sequence_length=self.sequence_length, accepting=False
            )
            trajectories.append(
                (
                    {
                        key: [image_map[key][element] for element in boolean_mask]
                        for key, boolean_mask in negative_trajectory.items()
                    },
                    0,
                    negative_trajectory,
                )
            )
            if self.double_check_sfa is not None:
                trajectory_to_list = lambda traj: [
                    {
                        symbol: value
                        for symbol, value in zip(traj.keys(), current_valuation)
                    }
                    for current_valuation in zip(*traj.values())
                ]
                if not self.double_check_sfa.accepts(
                    trajectory_to_list(positive_trajectory)
                ) or self.double_check_sfa.accepts(
                    trajectory_to_list(negative_trajectory)
                ):
                    print(
                        """Double check failed. Sequence generation may be problematic. Data generated will contain 
                        one noisy trajectory, i.e. that is mislabelled"""
                    )

        return trajectories
