import os
import re
import tqdm
import numpy as np
import argparse
import torch

from torchvision import transforms

from nets.nn import resnet50
from utils.loss import yoloLoss
from utils.dataset import Dataset


# ============================================================
#  Main Training Script
# ============================================================
def main(args):

    # --------------------------------------------------
    #  Device 설정
    # --------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE:", device)

    root = args.data_dir
    num_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr

    # --------------------------------------------------
    #  Seed 설정
    # --------------------------------------------------
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --------------------------------------------------
    #  모델 로딩 (pretrained or resume)
    # --------------------------------------------------
    if args.pre_weights is not None:
        # Resume mode
        pattern = r'yolov1_([0-9]+)'
        strs = args.pre_weights.split('.')[-2]
        fname = strs.split('/')[-1]

        epoch_str = re.search(pattern, fname).group(1)
        epoch_start = int(epoch_str) + 1

        net = resnet50()
        checkpoint = torch.load(f'./weights/{args.pre_weights}')
        net.load_state_dict(checkpoint['state_dict'])

        print(f"Loaded pretrained weights: {args.pre_weights}, start epoch {epoch_start}")

    else:
        # Train from scratch using ImageNet pretrain
        net = resnet50(pretrained=True)
        epoch_start = 1
        print("Loaded ImageNet pretrained backbone")

    net = net.to(device)

    print("NUMBER OF CUDA DEVICES:", torch.cuda.device_count())

    # --------------------------------------------------
    #  Loss, Optimizer
    # --------------------------------------------------
    criterion = yoloLoss().to(device)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    # Parameter grouping (different LR for some layers)
    params = []
    params_dict = dict(net.named_parameters())

    for key, value in params_dict.items():
        if key.startswith('features'):
            params += [{'params': [value], 'lr': learning_rate * 7}]
        else:
            params += [{'params': [value], 'lr': learning_rate}]

    optimizer = torch.optim.SGD(
        params,
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4
    )

    # --------------------------------------------------
    #  Dataset Load
    # --------------------------------------------------
    with open('./Dataset/train.txt') as f:
        train_names = f.readlines()

    train_dataset = Dataset(
        root, train_names, train=True,
        transform=[transforms.ToTensor()]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    with open('./Dataset/test.txt') as f:
        test_names = f.readlines()

    test_dataset = Dataset(
        root, test_names, train=False,
        transform=[transforms.ToTensor()]
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size // 2,
        shuffle=False,
        num_workers=2
    )

    print(f'NUMBER OF DATA SAMPLES: {len(train_dataset)}')
    print(f'BATCH SIZE: {batch_size}')

    # --------------------------------------------------
    #  Training Loop
    # --------------------------------------------------
    for epoch in range(epoch_start, num_epochs + 1):

        # ---- LR Scheduler ----
        if epoch <= 5:
            learning_rate = 1e-4
        elif epoch <= 30:
            learning_rate = 5e-4   # base LR
        elif epoch <= 45:
            learning_rate = 3e-4
        elif epoch <= 55:
            learning_rate = 1e-4
        else:
            learning_rate = 5e-5

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        # --------------------------------------------------
        #  TRAIN
        # --------------------------------------------------
        net.train()
        total_loss = 0.0

        print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (images, target) in progress_bar:
            images = images.to(device)
            target = target.to(device)

            pred = net(images)

            optimizer.zero_grad()
            loss = criterion(pred, target.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            gpu_mem = (
                '%.3gG' % (torch.cuda.memory_reserved() / 1E9)
                if torch.cuda.is_available() else '0G'
            )

            s = ('%10s' + '%10.4g' + '%10s') % (
                f'{epoch}/{num_epochs}',
                total_loss / (i + 1),
                gpu_mem
            )
            progress_bar.set_description(s)

        # --------------------------------------------------
        #  VALIDATION
        # --------------------------------------------------
        net.eval()
        validation_loss = 0.0

        with torch.no_grad():
            progress_bar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
            for i, (images, target) in progress_bar:
                images = images.to(device)
                target = target.to(device)

                prediction = net(images)
                loss = criterion(prediction, target)

                validation_loss += loss.item()

        validation_loss /= len(test_loader)
        print(f'Validation_Loss: {validation_loss:07.3f}')

        # --------------------------------------------------
        #  SAVE CHECKPOINT
        # --------------------------------------------------
        if (epoch % 10) == 0:
            save = {'state_dict': net.state_dict()}
            save_name = f'./weights/yolov1_{epoch:04d}.pth'
            torch.save(save, save_name)
            print(f"Saved checkpoint: {save_name}")

    # --------------------------------------------------
    #  FINAL SAVE
    # --------------------------------------------------
    save = {'state_dict': net.state_dict()}
    torch.save(save, './weights/yolov1_final.pth')
    print("Saved final model: yolov1_final.pth")


# ============================================================
#  MAIN ENTRY
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--data_dir", type=str, default='./Dataset')
    parser.add_argument("--pre_weights", type=str, help="pretrained weight")
    parser.add_argument("--save_dir", type=str, default="./weights")
    parser.add_argument("--img_size", type=int, default=448)

    args = parser.parse_args()

    # resume training example
    # args.pre_weights = 'yolov1_0050.pth'

    main(args)
