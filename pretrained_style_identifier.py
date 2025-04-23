# CMU 16-726 Learning-Based Image Synthesis / Spring 2024, Final Project
import argparse
import os

import imageio
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import numpy as np

import utils
from dataloader import get_data_loader
from models import StyleIdentifier

SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def print_model(m):
    """Prints model information for the generators and discriminators."""
    print("                 MODEL                ")
    print("---------------------------------------")
    print(m)
    print("---------------------------------------")


def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("style_identifier_iter") and f.endswith(".pkl")
    ]
    if not checkpoints:
        return None, 0
    checkpoints.sort(key=lambda x: int(x.split("iter")[1].split(".pkl")[0]))
    latest = checkpoints[-1]
    iteration = int(latest.split("iter")[1].split(".pkl")[0])
    return os.path.join(checkpoint_dir, latest), iteration


def create_model(opts):
    """Builds the generators and discriminators."""
    model = models.resnet18(pretrained=True)

    # Freeze all pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head
    model.fc = nn.Linear(model.fc.in_features, 31)

    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()
        print("Models moved to GPU.")

    # # 5. Loss + Optimizer
    # style_counts = {
    #     "가르마": 12510,
    #     "기타남자스타일": 11242,
    #     "기타레이어드": 48240,
    #     "기타여자스타일": 2964,
    #     "남자일반숏": 12577,
    #     "댄디": 7798,
    #     "루프": 2544,
    #     "리젠트": 9837,
    #     "리프": 7276,
    #     "미스티": 6161,
    #     "바디": 20810,
    #     "베이비": 3076,
    #     "보니": 15469,
    #     "보브": 20811,
    #     "빌드": 23617,
    #     "소프트투블럭댄디": 7446,
    #     "숏단발": 13614,
    #     "쉐도우": 10298,
    #     "쉼표": 2401,
    #     "스핀스왈로": 4936,
    #     "시스루댄디": 4981,
    #     "애즈": 8143,
    #     "에어": 29418,
    #     "여자일반숏": 5527,
    #     "원랭스": 34040,
    #     "원블럭댄디": 3869,
    #     "테슬": 2926,
    #     "포마드": 7970,
    #     "플리츠": 15370,
    #     "허쉬": 34040,
    #     "히피": 13455,
    # }

    # STYLE_CLASSES = sorted(set( [
    #     "가르마",
    #     "기타남자스타일",
    #     "기타레이어드",
    #     "기타여자스타일",
    #     "남자일반숏",
    #     "댄디",
    #     "루프",
    #     "리젠트",
    #     "리프",
    #     "미스티",
    #     "바디",
    #     "베이비",
    #     "보니",
    #     "보브",
    #     "빌드",
    #     "소프트투블럭댄디",
    #     "숏단발",
    #     "쉐도우",
    #     "쉼표",
    #     "스핀스왈로",
    #     "시스루댄디",
    #     "애즈",
    #     "에어",
    #     "여자일반숏",
    #     "원랭스",
    #     "원블럭댄디",
    #     "테슬",
    #     "포마드",
    #     "플리츠",
    #     "허쉬",
    #     "히피",
    # ]))
    # STYLE_TO_IDX = {style: i for i, style in enumerate(STYLE_CLASSES)} 
    # class_counts = torch.tensor([style_counts[style] for style in STYLE_CLASSES], dtype=torch.float)

    # class_weights = 1.0 / (class_counts.float() + 1e-6) 
    optimizer = optim.Adam(model.fc.parameters(), lr=opts.lr)  # Only train the head
 
    print_model(model)

    checkpoint_path, resume_iter = get_latest_checkpoint(opts.checkpoint_dir)
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint_data = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        opts.resume_iter = resume_iter
    else:
        opts.resume_iter = 0
        print(f"Checkpoints not found. Starting from scratch.")

    return model, optimizer#, criterion


def checkpoint(iteration, model, optimizer, opts):
    """Save model and optimizer"""
    path = os.path.join(opts.checkpoint_dir, "style_identifier_iter%d.pkl" % iteration)
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


# probably want update to dataloader_images dataloader_labels <--- @EMILY
def training_loop(dataloader_X, validation_loader, opts):
    """Runs the training loop.
    * Saves checkpoint every opts.checkpoint_every iterations
    * Saves generated samples every opts.sample_every iterations
    """
    # Create generators and discriminators
    model, optimizer = create_model(opts)

    params = list(model.parameters())

    iter_X = iter(dataloader_X)

    # Get some fixed data for sampling test loss?
    # that allow us to inspect the model's performance.
    pair = next(iter_X)
    fixed_X = (utils.to_var(pair[0]), utils.to_var(pair[1]))
    iter_per_epoch = len(dataloader_X)

    for iteration in range(opts.resume_iter + 1, opts.train_iters + 1):

        # Reset data_iter for each epoch
        if iteration % iter_per_epoch == 0:
            iter_X = iter(dataloader_X)

        images_X = next(iter_X)
        images_X, labels = (utils.to_var(images_X[0]), utils.to_var(images_X[1]))
        # print(images_X.device, labels.device)


        # TRAIN THE DISCRIMINATORS
        # 1. Compute the discriminator losses on real images
        out = model(images_X)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(out, labels.float())
        # loss = torch.mean((model(images_X) - labels) ** 2)

        # sum up the losses and update D_X and
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # plot the losses in tensorboard
        logger.add_scalar("labelTraining", loss, iteration)

        # Print the log info
        if iteration % opts.log_step == 0:
            print(
                "Iteration [{:5d}/{:5d}] | loss: {:6.4f}".format(
                    iteration,
                    opts.train_iters,
                    loss.item(),
                )
            )
        # Save the model parameters
        if iteration % opts.checkpoint_every == 0:
            checkpoint(iteration, model, optimizer, opts)

            model.eval()
            with torch.no_grad():
                val_loss_total = 0.0
                val_batches = 0

                for val_images, val_labels in validation_loader:
                    val_images = utils.to_var(val_images)
                    val_labels = utils.to_var(val_labels)

                    val_outputs = model(val_images)
                    loss_fn = nn.BCEWithLogitsLoss()
                    val_loss = loss_fn(out, labels.float()) 

                    val_loss_total += val_loss.item()
                    val_batches += 1

                avg_val_loss = val_loss_total / val_batches
                print("Validation | loss: {:6.4f}".format(avg_val_loss))
                logger.add_scalar("labelValidation", avg_val_loss, iteration)

            model.train()


def main(opts):
    """Loads the data and starts the training loop."""
    # Create dataloaders for images w/ labels
    dataloader_X, validation_loader = get_data_loader(opts.X, opts=opts)

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    # Start training
    training_loop(dataloader_X, validation_loader, opts)


def print_opts(opts):
    """Prints the values of all command-line arguments."""
    print("=" * 80)
    print("Opts".center(80))
    print("-" * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print("{:>30}: {:<30}".format(key, opts.__dict__[key]).center(80))
    print("=" * 80)


def create_parser():
    """Creates a parser for command-line arguments."""
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--disc", type=str, default="dc")  # or 'patch'
    parser.add_argument("--gen", type=str, default="cycle")
    parser.add_argument("--g_conv_dim", type=int, default=32)
    parser.add_argument("--d_conv_dim", type=int, default=32)
    parser.add_argument("--norm", type=str, default="instance")
    parser.add_argument("--use_cycle_consistency_loss", action="store_true")
    parser.add_argument("--init_zero_weights", action="store_true")
    parser.add_argument("--init_type", type=str, default="naive")

    # Training hyper-parameters
    parser.add_argument(
        "--train_iters", type=int, default=50000
    )  # 55670 samples, but pretrained model so we cut down a bit
    parser.add_argument("--resume_iter", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001) # also lower cus finetuning
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda_cycle", type=float, default=10)

    # Data sources
    parser.add_argument("--X", type=str, default="..")
    parser.add_argument("--ext", type=str, default="*.png")
    parser.add_argument("--use_diffaug", action="store_true")
    parser.add_argument("--data_preprocess", type=str, default="vanilla")

    # Saving directories and checkpoint/sample iterations
    parser.add_argument("--checkpoint_dir", default="checkpoints_pretrained_style_id")
    parser.add_argument("--sample_dir", type=str, default="pretrained")
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=100)
    parser.add_argument("--checkpoint_every", type=int, default=800)

    parser.add_argument("--gpu", type=str, default="0")

    return parser


if __name__ == "__main__":
    parser = create_parser()
    opts = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu
    opts.sample_dir = os.path.join(
        "output/", opts.sample_dir, "%s_%g" % (opts.X.split("/")[0], opts.lambda_cycle)
    )
    opts.sample_dir += "%s_%s_%s_%s_%s" % (
        opts.data_preprocess,
        opts.norm,
        opts.disc,
        opts.gen,
        opts.init_type,
    )
    if opts.use_cycle_consistency_loss:
        opts.sample_dir += "_cycle"
    if opts.use_diffaug:
        opts.sample_dir += "_diffaug"

    if os.path.exists(opts.sample_dir):
        cmd = "rm %s/*" % opts.sample_dir
        os.system(cmd)

    logger = SummaryWriter(opts.sample_dir)

    print_opts(opts)
    main(opts)
