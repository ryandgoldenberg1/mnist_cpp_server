import argparse
import os

import torch
import torch.nn as nn
from torchvision import datasets, transforms


def create_model():
    return nn.Sequential(
        nn.Conv2d(1, 16, (5, 5)),
        nn.BatchNorm2d(16),
        nn.MaxPool2d((2, 2)),
        nn.ReLU(),
        nn.Conv2d(16, 16, (5, 5)),
        nn.BatchNorm2d(16),
        nn.MaxPool2d((2, 2)),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(256, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )


def train(model, train_ds, test_ds, batch_size, learning_rate, num_epochs):
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(1, num_epochs+1):
        tr_correct = 0
        tr_total = 0.
        tr_loss = 0.
        for data, target in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            tr_correct += out.argmax(dim=-1).eq(target).sum()
            tr_total += data.shape[0]
            tr_loss += loss.item() * data.shape[0]

        test_correct = 0.
        test_total = 0.
        model.eval()
        for data, target in test_loader:
            out = model(data)
            test_correct += out.argmax(dim=-1).eq(target).sum()
            test_total += data.shape[0]
        model.train()

        test_acc = test_correct / test_total
        tr_avg_loss = tr_loss / tr_total
        tr_acc = tr_correct / tr_total
        print(f'[Epoch {epoch}] train_loss: {tr_avg_loss:0.04f} | train_acc: {tr_acc:0.04f}'
              f' | test_acc: {test_acc:0.04f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--output_path', default='model.pt')
    parser.add_argument('--dataset_root', default='~/datasets/pytorch')
    parser.add_argument('--download', action='store_true', default=False)
    args = parser.parse_args()
    print(args.__dict__)

    train_ds = datasets.MNIST(args.dataset_root, download=args.download, train=True, transform=transforms.ToTensor())
    test_ds = datasets.MNIST(args.dataset_root, download=args.download, train=False, transform=transforms.ToTensor())
    model = create_model()
    print(model)
    train(model, train_ds=train_ds, test_ds=test_ds, batch_size=args.batch_size, learning_rate=args.learning_rate,
          num_epochs=args.num_epochs)

    model.eval()
    torch_script_model = torch.jit.script(model)
    if os.path.dirname(args.output_path):
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch_script_model.save(args.output_path)
    print('Saved torch script model to:', args.output_path)


if __name__ == '__main__':
    main()
