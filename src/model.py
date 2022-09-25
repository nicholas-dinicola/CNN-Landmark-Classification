import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(
            self,
            num_classes: int = 1000,
            dropout: float = 0.7,
            input_channel: int = 3
    ) -> None:
        super().__init__()

        self.backbone = nn.Sequential()
        output_channel = 16

        for _ in range(5):
            output_channel = output_channel * 2
            self.backbone.append(nn.Sequential(
                nn.Conv2d(input_channel, output_channel, 3, 1, 1),
                nn.BatchNorm2d(output_channel),
                nn.MaxPool2d((2, 2)),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            input_channel = output_channel

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_channel * 7 * 7, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        backbone_out = self.backbone(x)
        head_out = self.head(backbone_out)
        return head_out


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):
    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"


if __name__ == "__main__":
    net = MyModel()
    x = torch.randn(1, 3, 224, 224)
    x = net(x)
    print(x.shape)
