import os
import tqdm
import numpy as np

import torch
import torchvision
from torchvision import transforms

from nets.nn import resnet50
from utils.loss import yoloLoss
from utils.dataset import Dataset

import argparse
import re

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = args.data_dir
    num_epochs = args.epoch
    batch_size = args.batch_size
    lr = args.lr

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # -------------------------
    # MODEL (with pretrained=True)
    # -------------------------
    net = resnet50(pretrained=True).to(device)

    # pretrained checkpoint (optional)
    if args.pre_weights is not None:
        ckpt = torch.load(f'./weights/{args.pre_weights}')
        net.load_state_dict(ckpt['state_dict'], strict=False)
        epoch_start = int(args.pre_weights.split('_')[1].split('.')[0]) + 1
    else:
        epoch_start = 1

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    # LOSS
    criterion = yoloLoss().to(device)

    # OPTIMIZER
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4
    )

    # DATASET
    with open('./Dataset/train.txt') as f:
        train_names = f.readlines()
    train_dataset = Dataset(root, train_names, train=True, transform=[transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    with open('./Dataset/test.txt') as f:
        test_names = f.readlines()
    test_dataset = Dataset(root, test_names, train=False, transform=[transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size//2, shuffle=False)

    # -------------------------
    # TRAINING
    # -------------------------
    for epoch in range(epoch_start, num_epochs):

        net.train()
        total_loss = 0

        # lr schedule
        if epoch == 30: lr = 1e-4
        if epoch == 40: lr = 1e-5
        for g in optimizer.param_groups: g['lr'] = lr

        pbar = tqdm.tqdm(train_loader)
        for images, target in pbar:
            images, target = images.to(device), target.to(device)

            pred = net(images)
            loss = criterion(pred, target.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_description(f"Epoch {epoch} Loss {total_loss/len(pbar):.4f}")

        # -------------------------
        # VALIDATION
        # -------------------------
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for images, target in test_loader:
                images, target = images.to(device), target.to(device)
                loss = criterion(net(images), target)
                val_loss += loss.item()

        print(f"VAL LOSS = {val_loss / len(test_loader):.4f}")

        if epoch % 5 == 0:
            torch.save({'state_dict': net.state_dict()}, f'./weights/yolov1_{epoch:04d}.pth')

    torch.save({'state_dict': net.state_dict()}, './weights/yolov1_final.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_dir", type=str, default="./Dataset")
    parser.add_argument("--pre_weights", type=str)
    args = parser.parse_args()
    main(args)