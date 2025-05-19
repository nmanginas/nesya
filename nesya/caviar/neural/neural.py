import torch


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 5, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 64, 5, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 64, 5, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

    def forward(self, images: torch.Tensor):
        return self.encoder(images).reshape(images.shape[0], -1)


class LstmCNN(torch.nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0):
        super().__init__()
        self.encoder = CNN()
        self.lstm = torch.nn.LSTM(64 * 2 + 1, 128, batch_first=True, dropout=dropout)
        self.classification_projection = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes),
        )

    def forward(
        self, p1_bbs: torch.Tensor, p2_bbs: torch.Tensor, distances: torch.Tensor
    ) -> torch.Tensor:
        person_1_cnn_outputs = self.encoder(p1_bbs)
        person_2_cnn_outputs = self.encoder(p2_bbs)
        lstm_input = torch.cat(
            (
                torch.cat((person_1_cnn_outputs, person_2_cnn_outputs), dim=-1),
                distances.unsqueeze(-1),
            ),
            dim=-1,
        )
        outputs, _ = self.lstm(lstm_input)

        return self.classification_projection(outputs)


class TransformerCNN(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.encoder = CNN()
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=64 * 2 + 1, nhead=3)
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=4
        )
        self.classification_projection = torch.nn.Sequential(
            torch.nn.Linear(64 * 2 + 1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes),
        )

    def forward(
        self, p1_bbs: torch.Tensor, p2_bbs: torch.Tensor, distances: torch.Tensor
    ) -> torch.Tensor:
        person_1_cnn_outputs = self.encoder(p1_bbs)
        person_2_cnn_outputs = self.encoder(p2_bbs)
        lstm_input = torch.cat(
            (
                torch.cat((person_1_cnn_outputs, person_2_cnn_outputs), dim=-1),
                distances.unsqueeze(-1),
            ),
            dim=-1,
        )
        outputs = self.transformer_encoder(lstm_input)

        return self.classification_projection(outputs)


class CaviarNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_backbone = CNN()
        self.projection = torch.nn.Linear(64, 4)

    def forward(
        self, p1_bbs: torch.Tensor, p2_bbs: torch.Tensor, distances: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        p1_predictions, p2_predictions = self.projection(
            self.cnn_backbone(p1_bbs)
        ).softmax(-1), self.projection(self.cnn_backbone(p2_bbs)).softmax(-1)

        return {
            "p1": p1_predictions,
            "p2": p2_predictions,
            "close(p1, p2)": (distances < 25).float().unsqueeze(-1),
        }
