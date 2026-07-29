#!/usr/bin/env python3
#
# Image classifier training script optimized for NVIDIA Jetson.
# Based on the PyTorch ImageNet training example.
#

import argparse
import datetime
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from voc import VOCDataset
from nuswide import NUSWideDataset
from reshape import reshape_model


model_names = sorted(
    name for name in models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(models.__dict__[name])
)


parser = argparse.ArgumentParser(
    description="PyTorch Image Classifier Training"
)

parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--dataset-type",
    type=str,
    default="folder",
    choices=["folder", "nuswide", "voc"],
    help="dataset type (default: folder)",
)
parser.add_argument(
    "--multi-label",
    action="store_true",
    help="multi-label model",
)
parser.add_argument(
    "--multi-label-threshold",
    type=float,
    default=0.5,
    help="confidence threshold for multi-label accuracy",
)
parser.add_argument(
    "--model-dir",
    type=str,
    default="models",
    help="checkpoint output directory (default: models)",
)
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet18",
    choices=model_names,
    help="model architecture (default: resnet18)",
)
parser.add_argument(
    "--resolution",
    default=224,
    type=int,
    metavar="N",
    help="input image resolution (default: 224)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=2,
    type=int,
    metavar="N",
    help="number of DataLoader workers (default: 2)",
)
parser.add_argument(
    "--epochs",
    default=35,
    type=int,
    metavar="N",
    help="number of epochs (default: 35)",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual starting epoch",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=64,
    type=int,
    metavar="N",
    help="mini-batch size (default: 64)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    dest="lr",
    help="initial learning rate (default: 0.1)",
)
parser.add_argument(
    "--momentum",
    default=0.9,
    type=float,
    metavar="M",
    help="SGD momentum (default: 0.9)",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    dest="weight_decay",
    help="weight decay (default: 1e-4)",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="checkpoint path to resume from",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate on the validation set",
)
parser.add_argument(
    "--pretrained",
    dest="pretrained",
    action="store_true",
    default=True,
    help="use a pretrained model",
)
parser.add_argument(
    "--disable-amp",
    action="store_true",
    help="disable CUDA mixed precision",
)
parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="random seed",
)
parser.add_argument(
    "--gpu",
    default=0,
    type=int,
    help="GPU ID to use (default: 0)",
)

args = parser.parse_args()


tensorboard = SummaryWriter(
    log_dir=os.path.join(
        args.model_dir,
        "tensorboard",
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
)
print(
    "To start TensorBoard, run: "
    f"tensorboard --logdir={os.path.join(args.model_dir, 'tensorboard')}"
)

best_accuracy = 0.0


def main(args):
    global best_accuracy

    if args.workers < 0:
        raise ValueError("--workers must be 0 or greater")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This script expects a CUDA GPU.")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn(
            "Deterministic mode can make training considerably slower."
        )
    else:
        cudnn.benchmark = True

    torch.cuda.set_device(args.gpu)
    print(
        f"=> using GPU {args.gpu} "
        f"({torch.cuda.get_device_name(args.gpu)})"
    )

    amp_enabled = not args.disable_amp
    print(f"=> mixed precision: {'enabled' if amp_enabled else 'disabled'}")
    print(f"=> batch size: {args.batch_size}")
    print(f"=> DataLoader workers: {args.workers}")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        normalize,
    ])

    if args.dataset_type == "folder":
        train_dataset = datasets.ImageFolder(
            os.path.join(args.data, "train"),
            train_transforms,
        )
        val_dataset = datasets.ImageFolder(
            os.path.join(args.data, "val"),
            val_transforms,
        )
    elif args.dataset_type == "nuswide":
        train_dataset = NUSWideDataset(
            args.data,
            "trainval",
            train_transforms,
        )
        val_dataset = NUSWideDataset(
            args.data,
            "test",
            val_transforms,
        )
    elif args.dataset_type == "voc":
        train_dataset = VOCDataset(
            args.data,
            "trainval",
            train_transforms,
        )
        val_dataset = VOCDataset(
            args.data,
            "val",
            val_transforms,
        )
    else:
        raise ValueError(f"unsupported dataset type: {args.dataset_type}")

    if args.dataset_type in ("nuswide", "voc") and not args.multi_label:
        raise ValueError(
            "NUS-WIDE and VOC must be run with --multi-label"
        )

    print(
        f"=> dataset classes: {len(train_dataset.classes)} "
        f"{train_dataset.classes}"
    )

    loader_settings = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_settings,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_settings,
    )

    if args.pretrained:
        print(f"=> using pretrained model '{args.arch}'")
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print(f"=> creating model '{args.arch}'")
        model = models.__dict__[args.arch]()

    model = reshape_model(
        model,
        args.arch,
        len(train_dataset.classes),
    )

    if args.multi_label:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    model = model.cuda(args.gpu)
    criterion = criterion.cuda(args.gpu)

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(
                args.resume,
                map_location=f"cuda:{args.gpu}",
            )

            args.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            if amp_enabled and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])

            if "best_accuracy" in checkpoint:
                best_accuracy = float(checkpoint["best_accuracy"])
            elif "accuracy" in checkpoint:
                best_accuracy = float(
                    checkpoint["accuracy"].get("val", 0.0)
                )

            print(
                f"=> loaded checkpoint '{args.resume}' "
                f"at epoch {checkpoint['epoch']}"
            )
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    if args.evaluate:
        validate(
            val_loader,
            model,
            criterion,
            0,
            amp_enabled,
        )
        tensorboard.close()
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train_loss, train_acc = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            scaler,
            amp_enabled,
        )

        val_loss, val_acc = validate(
            val_loader,
            model,
            criterion,
            epoch,
            amp_enabled,
        )

        is_best = val_acc > best_accuracy
        best_accuracy = max(val_acc, best_accuracy)

        print(f"=> Epoch {epoch}")
        print(f"  * Train Loss     {train_loss:.4e}")
        print(f"  * Train Accuracy {train_acc:.4f}")
        print(f"  * Val Loss       {val_loss:.4e}")
        print(
            f"  * Val Accuracy   {val_acc:.4f}"
            f"{'*' if is_best else ''}"
        )

        save_checkpoint({
            "epoch": epoch,
            "arch": args.arch,
            "resolution": args.resolution,
            "classes": train_dataset.classes,
            "num_classes": len(train_dataset.classes),
            "multi_label": args.multi_label,
            "state_dict": model.state_dict(),
            "accuracy": {
                "train": train_acc,
                "val": val_acc,
            },
            "best_accuracy": best_accuracy,
            "loss": {
                "train": train_loss,
                "val": val_loss,
            },
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        }, is_best)

    tensorboard.close()


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    scaler,
    amp_enabled,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    acc = AverageMeter("Accuracy", ":7.3f")

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc],
        prefix=f"Epoch: [{epoch}]",
    )

    model.train()

    epoch_start = time.time()
    end = epoch_start

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output = model(images)
            loss = criterion(output, target)

        losses.update(loss.item(), images.size(0))
        acc.update(accuracy(output, target), images.size(0))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            progress.display(i)

    print(
        f"Epoch: [{epoch}] completed, elapsed time "
        f"{time.time() - epoch_start:6.3f} seconds"
    )

    tensorboard.add_scalar("Loss/train", losses.avg, epoch)
    tensorboard.add_scalar("Accuracy/train", acc.avg, epoch)

    return losses.avg, acc.avg


def validate(
    val_loader,
    model,
    criterion,
    epoch,
    amp_enabled,
):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    acc = AverageMeter("Accuracy", ":7.3f")

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, acc],
        prefix="Val:   ",
    )

    model.eval()

    with torch.no_grad():
        end = time.time()

        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(images)
                loss = criterion(output, target)

            losses.update(loss.item(), images.size(0))
            acc.update(accuracy(output, target), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == len(val_loader) - 1:
                progress.display(i)

    tensorboard.add_scalar("Loss/val", losses.avg, epoch)
    tensorboard.add_scalar("Accuracy/val", acc.avg, epoch)

    return losses.avg, acc.avg


def save_checkpoint(
    state,
    is_best,
    filename="checkpoint.pth.tar",
    best_filename="model_best.pth.tar",
    labels_filename="labels.txt",
):
    model_dir = os.path.expanduser(args.model_dir)
    os.makedirs(model_dir, exist_ok=True)

    filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, best_filename)
    labels_filename = os.path.join(model_dir, labels_filename)

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, best_filename)
        print(f"saved best model to: {best_filename}")
    else:
        print(f"saved checkpoint to: {filename}")

    if state["epoch"] == 0:
        with open(labels_filename, "w", encoding="utf-8") as file:
            for label in state["classes"]:
                file.write(f"{label}\n")
        print(f"saved class labels to: {labels_filename}")


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print(f"=> learning rate: {lr:.6f}")


def accuracy(output, target):
    with torch.no_grad():
        if args.multi_label:
            probabilities = torch.sigmoid(output)
            predictions = probabilities >= args.multi_label_threshold
            correct = predictions == target.bool()
        else:
            predictions = output.argmax(dim=1)
            correct = predictions == target

        return correct.float().mean().item() * 100.0


class AverageMeter:
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("  ".join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main(args)