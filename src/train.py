#!/usr/bin/env python3
""" Script to train model to classify melanoma. """

import pathlib
import random

import torch
import numpy as np

from src import dataset, model

_DATA_DIR = pathlib.Path("~/datasets/melanoma10k").expanduser()
_SAVE_DIR = pathlib.Path("~/runs/melanoma10k").expanduser()


def train(
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    skin_model,
    optimizer,
    loss_fn,
) -> None:
    highest_acc = 0
    losses = []
    for epoch in range(20):
        for img, label in train_loader:
            skin_model.train()
            optimizer.zero_grad()

            img = img.float().cuda()
            label = label.cuda()

            out = skin_model(img)
            loss = loss_fn(out, label.cuda())
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {np.mean(losses):.5}")

        num_right = 0
        total = 0
        for img, label in eval_loader:
            skin_model.eval()
            img = img.float().cuda()
            out = skin_model(img).cpu()

            _, predicted = torch.max(out.data, 1)
            total += label.size(0)
            num_right += (predicted == label).sum().item()

        print(f"Epoch {epoch}, Accuracy: {num_right / total:.2}")

        if highest_acc < num_right / total:
            highest_acc = num_right / total

            torch.save(
                skin_model.state_dict(), _SAVE_DIR / f"model-{highest_acc:.3}.pt"
            )
            print(
                f"Saving model with highest accuracy {highest_acc:.3} to {_SAVE_DIR}."
            )


if __name__ == "__main__":

    torch.random.manual_seed(42)
    random.seed(42)

    _SAVE_DIR.mkdir(exist_ok=True, parents=True)

    train_loader = torch.utils.data.DataLoader(
        dataset.LesionDataset(_DATA_DIR / "train"), batch_size=32, pin_memory=True
    )

    eval_loader = torch.utils.data.DataLoader(
        dataset.LesionDataset(_DATA_DIR / "eval"), batch_size=32, pin_memory=True
    )

    # instantiate model
    test_model = model.SkinModel(len(dataset._DATA_CLASSES))
    # create the optimizer
    optimizer = torch.optim.SGD(
        test_model.parameters(), lr=1e-2, weight_decay=1e-4, momentum=0.9, nesterov=True
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    test_model.cuda()
    train(train_loader, eval_loader, test_model, optimizer, loss_fn)
