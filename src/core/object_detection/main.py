"""

source https://github.com/keshik6/KITTI-2d-object-detection

"""

from __future__ import division

import argparse
import csv
import os
import warnings

import torch
from torch.utils.data import DataLoader

from dataset import KITTI2D
from model import Darknet
from train_model import train_model

warnings.filterwarnings("ignore")


def main(
    train_path,
    val_path,
    labels_path,
    weights_path,
    preload_weights_file,
    output_path,
    yolo_config_file,
    fraction=1,
    learning_rate=1e-3,
    weight_decay=1e-4,
    batch_size=8,
    epochs=20,
    freeze_struct=[True, 5],
):
    """
        This is the point of entry to the neural network program.
        All the training history will be saved as a csv in the output path

        Args
            train_path (string): Directory containing the training images
            val_path (string):: Directory containing the val images
            labels_path (string):: Directory containing the yolo format labels for data
            weights_path (string):: Directory containing the weights (new weights for this program will also be added here)
            preload_weights_file (string): Name of preload weights file
            output_path (string): Directory to store the training history outputs as csv
            yolo_config_file (string): file path of yolo configuration file
            fraction (float): fraction of data to use for training
            learning_rate (float): initial learning rate
            weight_decay (float): weight decay value
            batch_size (int): batch_size for both training and validation
            epochs (int): maximum number of epochs to train the model
            freeze_struct (list): [bool, int] indicating whether to freeze the Darknet backbone and until which epoch should it be frozen

        Returns
            None

    """

    # Set up checkpoints path
    checkpoints_path = weights_path

    # Set up env variables and create required directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    # Set up cuda
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Available device = ", device)

    # Create model and load pretrained darknet weights
    model = Darknet(yolo_config_file)
    print("Loading imagenet weights to darknet")
    model.load_weights(os.path.join(weights_path, preload_weights_file))
    model.to(device)
    # print(model)

    # Create datasets
    train_dataset = KITTI2D(train_path, labels_path, fraction=fraction, train=True)
    valid_dataset = KITTI2D(val_path, labels_path, fraction=fraction, train=False)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Create optimizers
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)

    # Create log csv files
    train_log_file = open(
        os.path.join(output_path, "train_results.csv"), "w", newline=""
    )
    valid_log_file = open(
        os.path.join(output_path, "valid_results.csv"), "w", newline=""
    )
    train_csv = csv.writer(train_log_file)
    valid_csv = csv.writer(valid_log_file)

    print("Starting to train yolov3 model...")

    # Train model here
    train_model(
        model,
        device,
        optimizer,
        lr_scheduler,
        train_dataloader,
        valid_dataloader,
        train_csv,
        valid_csv,
        weights_path,
        max_epochs=epochs,
        tensor_type=torch.cuda.FloatTensor,
        update_gradient_samples=1,
        freeze_darknet=freeze_struct[0],
        freeze_epoch=freeze_struct[1],
    )

    # Close the log files
    train_log_file.close()
    valid_log_file.close()

    print("Training completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="neural network training")

    parser.add_argument(
        "--train_path", type=str, metavar="PATH", help="path to train images"
    )
    parser.add_argument(
        "--val_path", type=str, metavar="PATH", help="path to val images",
    )
    parser.add_argument(
        "--labels_path", type=str, metavar="PATH", help="path to yolo labels"
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        metavar="PATH",
        help="path to pretrained imeagenet weights",
    )
    parser.add_argument(
        "--preload_weights_file", type=str, help="name of pretrained weight file"
    )
    parser.add_argument(
        "--output_path", type=str, metavar="PATH", help="path to save output files"
    )
    parser.add_argument(
        "--yolo_config_file", type=str, metavar="PATH", help="path to yolo config"
    )
    args = parser.parse_args()
    args = vars(args)
    main(**args)
