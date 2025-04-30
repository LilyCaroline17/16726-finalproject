# CMU 16-726 Learning-Based Image Synthesis / Spring 2024, Assignment 3
#
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip
# This is the main training file for the second part of the assignment.
#
# Usage:
# ======
#    To train with the default hyperparamters:
#       python cycle_gan.py
#
#    To train with cycle consistency loss:
#       python cycle_gan.py --use_cycle_consistency_loss

#   To train with pertrained:
#   python style_GAN.py --iden_checkpoint_dir checkpoints_pretrained_style_id --iden "pretrained"
#
#
#    For optional experimentation:
#    -----------------------------
#    If you have a powerful computer (ideally with a GPU),
#    then you can obtain better results by
#    increasing the number of filters used in the generator
#    and/or discriminator, as follows:
#      python cycle_gan.py --g_conv_dim=64 --d_conv_dim=64

import argparse
import os

import math
import re

import imageio
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import clip

import utils
from dataloader import get_data_loader, get_all_data_loader, STYLE_CLASSES
from models import Generator, Discriminator, StyleIdentifier

# from diff_augment import DiffAugment

policy = "color,translation,cutout"


SEED = 2

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def print_models(G, D, style_iden):
    """Prints model information for the generators and discriminators."""
    print("                 G                ")
    print("---------------------------------------")
    print(G)
    print("---------------------------------------")

    print("                 D                ")
    print("---------------------------------------")
    print(D)
    print("---------------------------------------")

    print("                  style_identifier                  ")
    print("---------------------------------------")
    print(style_iden)
    print("---------------------------------------")


def create_model(opts):
    """Builds the generators and discriminators."""
    G = Generator(conv_dim=opts.g_conv_dim, norm=opts.norm)
    D = Discriminator(conv_dim=opts.d_conv_dim, norm=opts.norm)
    g_optimizer = optim.Adam(G.parameters(), opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(D.parameters(), opts.lr, [opts.beta1, opts.beta2])

    # Check if checkpoint for G/D exists
    latest_ckpt_dir = None
    if os.path.exists(opts.checkpoint_dir):
        checkpoints = sorted(
            [d for d in os.listdir(opts.checkpoint_dir) if d.endswith("itr")]
        )
        if len(checkpoints) > 0:
            latest_ckpt_dir = os.path.join(
                opts.checkpoint_dir, checkpoints[-1]
            )  # Use the latest one

    if latest_ckpt_dir:
        print(f"Loading models from checkpoint: {latest_ckpt_dir}")
        G.load_state_dict(torch.load(os.path.join(latest_ckpt_dir, "G.pkl")))
        D.load_state_dict(torch.load(os.path.join(latest_ckpt_dir, "D.pkl")))
    else:
        print("No checkpoint found, initializing new models.")

    # style_models = {"ours":StyleIdentifier, "pretrained":models.resnet18(pretrained=True)}
    style_iden = None
    if opts.iden == "clip":
        model, preprocess = clip.load("ViT-B/32")

        if torch.cuda.is_available():
            G.cuda()
            D.cuda()
            model.cuda()
            preprocess.cuda()
            print("Models moved to GPU.")

        style_iden = (model, preprocess)

        # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
        # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

        print_models(G, D, style_iden)
        return G, D, style_iden, g_optimizer, d_optimizer

    elif opts.iden == "ours":
        style_iden = StyleIdentifier(conv_dim=opts.d_conv_dim, norm=opts.norm)
    elif opts.iden == "pretrained":
        style_iden = models.resnet18(pretrained=True)
        # Freeze all pretrained layers
        for param in style_iden.parameters():
            param.requires_grad = False

        # Replace the classifier head
        style_iden.fc = nn.Linear(style_iden.fc.in_features, 31)

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        style_iden.cuda()
        print("Models moved to GPU.")

    if os.path.isdir(opts.iden_checkpoint_dir):
        # Find all checkpoints like style_identifier_iterXXXXXX.pkl
        checkpoints = [
            f
            for f in os.listdir(opts.iden_checkpoint_dir)
            if f.startswith("style_identifier_iter") and f.endswith(".pkl")
        ]

        if len(checkpoints) == 0:
            raise FileNotFoundError(
                f"No style identifier checkpoints found in {opts.iden_checkpoint_dir}"
            )

        # Extract iteration numbers
        def get_iter_num(filename):
            match = re.search(r"iter(\d+)", filename)
            return int(match.group(1)) if match else -1

        checkpoints.sort(key=get_iter_num)
        latest_ckpt = checkpoints[-1]
        latest_ckpt_path = os.path.join(opts.iden_checkpoint_dir, latest_ckpt)

        print(f"Loading Style Identifier from {latest_ckpt_path}")
        style_iden.load_state_dict(torch.load(latest_ckpt_path)["model_state_dict"])
    else:
        raise FileNotFoundError(
            f"Style identifier directory not found: {opts.iden_checkpoint_dir}"
        )

    print_models(G, D, style_iden)

    return G, D, style_iden, g_optimizer, d_optimizer


def checkpoint(iteration, G, D, g_optimizer, d_optimizer, opts):
    """Save generators, discriminators, and optimizers"""
    directory = os.path.join(opts.checkpoint_dir, "%ditr" % iteration)
    if not os.path.exists(directory):
        os.mkdir(directory)
    G_path = os.path.join(opts.checkpoint_dir, "%ditr/G.pkl" % iteration)
    D_path = os.path.join(opts.checkpoint_dir, "%ditr/D.pkl" % iteration)
    g_opt_path = os.path.join(opts.checkpoint_dir, "%ditr/g_optimizer.pkl" % iteration)
    d_opt_path = os.path.join(opts.checkpoint_dir, "%ditr/d_optimizer.pkl" % iteration)

    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)
    torch.save(g_optimizer.state_dict(), g_opt_path)
    torch.save(d_optimizer.state_dict(), d_opt_path)


def make_grid(images, nrow=4):
    """
    Arrange images into a grid.

    Args:
        images (Tensor): shape [N, C, H, W]
        nrow (int): number of images per row

    Returns:
        grid (Tensor): [C, H_total, W_total]
    """
    N, C, H, W = images.size()
    ncol = (N + nrow - 1) // nrow  # number of rows needed

    # Add padding if necessary
    pad = nrow * ncol - N
    if pad > 0:
        padding = torch.zeros((pad, C, H, W), device=images.device)
        images = torch.cat([images, padding], dim=0)

    images = images.reshape(ncol, nrow, C, H, W)
    images = images.permute(2, 0, 3, 1, 4)  # C, ncol, H, nrow, W
    images = images.reshape(C, ncol * H, nrow * W)
    # images = images.permute(1, 2, 0).cpu().numpy() # H, W, C
    images = utils.to_data(images)
    return images.transpose(1, 2, 0)


# have maybe 2 ppl images, have them try on all other 30 hair styles
def save_samples(iteration, fixed_X, G, opts, style_iden):
    """Saves samples from generator"""

    image, label = (
        utils.to_var(fixed_X[0])[0].unsqueeze(0),
        utils.to_var(fixed_X[1])[0].unsqueeze(0),
    )
    image_expanded, new_label, _ = diff_labels_for_generation(label, image)

    new_imgs = G(image_expanded, new_label)

    all_images = torch.cat([image, new_imgs], dim=0)

    merged = make_grid(all_images, int(math.sqrt(31)))
    path = os.path.join(opts.sample_dir, "sample-{:06d}.png".format(iteration))
    merged = np.uint8(255 * (merged + 1) / 2)
    imageio.imwrite(path, merged)
    print("Saved {}".format(path))


def diff_labels_for_generation(labels, images):
    batchsize = labels.shape[0]
    # Repeat each image 30 times (31 total styles, want to try other 30)
    images_expanded = (
        images.unsqueeze(1).repeat(1, 30, 1, 1, 1).view(-1, *images.shape[1:])
    )
    # (batch_size * num_styles, C, H, W)

    # Possible labels
    all_labels = utils.to_var(
        torch.eye(31)
    )  # Step 3: For each label, pick the 30 rows not corresponding to ground truth
    true_indices = labels.argmax(dim=1)  # (batch_size,)

    # Now, for each sample, we need all labels except the true one
    mask = utils.to_var(torch.ones((batchsize, 31), dtype=torch.bool))
    mask[utils.to_var(torch.arange(batchsize)), true_indices] = False
    # mask.scatter_(1, true_indices.unsqueeze(1), False)  # set true index to False (mask it out)

    new_labels = all_labels.repeat(batchsize, 1, 1)  # (batch_size, 31, 31)
    new_labels = new_labels[mask]  # Select the rows where the mask is True
    new_labels = new_labels.view(batchsize * 30, 31)  # (batch_size * 30, 31)

    # # Step 4: Expand the mask to match the shape of `all_labels`
    # expanded_mask = mask.unsqueeze(2).expand(opts.batch_size, 31, 31)  # (batch_size, 31, 31)

    # # Step 5: Apply the mask to select the correct labels and reshape
    # new_labels = all_labels.unsqueeze(0).expand(opts.batch_size, 31, 31)  # (batch_size, 31, 31)
    # new_labels = new_labels[expanded_mask].view(opts.batch_size * 30, 31)  # (batch_size * 30, 31)

    # Step 6: Repeat the original labels
    orig_labels = (
        labels.unsqueeze(1).expand(-1, 30, -1).reshape(batchsize * 30, 31)
    )  # (batch_size * 30, 31)

    return images_expanded, new_labels, orig_labels


def training_loop(dataloader_X, opts):
    """Runs the training loop.
    * Saves checkpoint every opts.checkpoint_every iterations
    * Saves generated samples every opts.sample_every iterations
    """
    # Create generators and discriminators
    G, D, style_iden, g_optimizer, d_optimizer = create_model(opts)
    preprocess = None
    text_features = None

    if isinstance(style_iden, tuple):
        style_iden, preprocess = style_iden
        text_tokens = utils.to_var(clip.tokenize(STYLE_CLASSES))
        text_features = style_iden.encode_text(text_tokens)

    iter_X = iter(dataloader_X)
    iter_per_epoch = len(dataloader_X)

    # Get some fixed data for sampling.
    # These are images that are held constant throughout training,
    # that allow us to inspect the model's performance.
    fixed_X = next(iter_X)

    for iteration in range(1, opts.train_iters + 1):

        # Reset data_iter for each epoch
        if iteration % iter_per_epoch == 0:
            iter_X = iter(dataloader_X)

        images = next(iter_X)
        images, labels = (utils.to_var(images[0]), utils.to_var(images[1]))
        # image, label = (utils.to_var(fixed_X[0])[0].unsqueeze(0), utils.to_var(fixed_X[1])[0].unsqueeze(0))

        if iteration == 1:
            print(
                torch.sigmoid(style_iden(images[0].unsqueeze(0))),
                labels[0].unsqueeze(0),
            )

        # TRAIN THE DISCRIMINATORS
        # 1. Compute the discriminator losses on real images
        d_real_loss = torch.mean((D(images) - 1) ** 2)

        # 2. Generate domain-X-like images based on real images in domain Y
        fake_images = G(images, labels)

        # 3. Compute the loss for D_X
        d_fake_loss = torch.mean((D(fake_images)) ** 2)

        # sum up the losses and update D_X and D_Y
        d_optimizer.zero_grad()
        d_total_loss = d_real_loss + d_fake_loss
        d_total_loss.backward()
        d_optimizer.step()

        # plot the losses in tensorboard
        logger.add_scalar("D/real", d_real_loss, iteration)
        logger.add_scalar("D/fake", d_fake_loss, iteration)

        # TRAIN THE GENERATORS

        # 1. Creating inputs for the generator
        images_expanded, new_labels, orig_labels = diff_labels_for_generation(
            labels, images
        )

        # 2. Generate fake images based on real images
        fake_images = G(images_expanded, new_labels)

        # 3. Compute the generator loss based on domain X
        g_loss = torch.mean((D(fake_images) - 1) ** 2)
        logger.add_scalar("G/fake", g_loss, iteration)

        if opts.use_cycle_consistency_loss:
            # 4. Compute the cycle consistency loss (the reconstruction loss)
            cycle_consistency_loss = torch.mean(
                torch.abs(images_expanded - G(fake_images, orig_labels))
            )

            g_loss += opts.lambda_cycle * cycle_consistency_loss
            logger.add_scalar(
                "G/cycle", opts.lambda_cycle * cycle_consistency_loss, iteration
            )

        if opts.use_style_loss:
            # 5. Compute the style loss (style alignment loss)
            with torch.no_grad():
                style_loss = 0
                if opts.iden == "clip":
                    imgencoded = utils.to_var(
                        preprocess(fake_images)
                    )  # UH, might not be right
                    image_features = style_iden.encode_image(imgencoded)
                    logits_per_image, logits_per_text = style_iden(
                        image_features, text_features
                    )
                    target_labels = torch.argmax(new_labels, dim=1) 
                    predicted_labels = (logits_per_image).argmax(dim=1)
                    style_loss = (predicted_labels == target_labels).float().mean()

                elif opts.iden == "ours":
                    ...
                elif opts.iden == "pretrained": 
                    target_labels = torch.argmax(new_labels, dim=1) 
                    predicted_labels = (style_iden(fake_images)).argmax(dim=1)
                    style_loss = (predicted_labels == target_labels).float().mean()

            g_loss += opts.lambda_style * style_loss
            logger.add_scalar("G/style", opts.lambda_style * style_loss, iteration)

        # backprop the aggregated g losses and update G_XtoY and G_YtoX
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Print the log info
        if iteration % opts.log_step == 0:
            print(
                "Iteration [{:5d}/{:5d}] | d_real_loss: {:6.4f} | "
                "d_fake_loss: {:6.4f} | d_total_loss: {:6.4f} |  g_loss: {:6.4f}".format(
                    iteration,
                    opts.train_iters,
                    d_real_loss.item(),
                    d_fake_loss.item(),
                    d_total_loss.item(),
                    g_loss.item(),
                )
            )

        # Save the generated samples
        if iteration % opts.sample_every == 0:
            save_samples(iteration, fixed_X, G, opts, style_iden)

        # Save the model parameters
        if iteration % opts.checkpoint_every == 0:
            checkpoint(iteration, G, D, g_optimizer, d_optimizer, opts)


def main(opts):
    """Loads the data and starts the training loop."""
    # Create  dataloaders for images from the two domains X and Y
    dataloader_X = get_all_data_loader(opts.X, opts=opts)

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    # Start training
    training_loop(dataloader_X, opts)


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
    parser.add_argument("--iden", type=str, default="ours")
    parser.add_argument("--g_conv_dim", type=int, default=32)
    parser.add_argument("--d_conv_dim", type=int, default=32)
    parser.add_argument("--norm", type=str, default="instance")
    parser.add_argument("--use_cycle_consistency_loss", action="store_true")
    parser.add_argument("--use_style_loss", action="store_true")
    parser.add_argument("--init_zero_weights", action="store_true")
    parser.add_argument("--init_type", type=str, default="naive")

    # Training hyper-parameters
    parser.add_argument("--train_iters", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda_cycle", type=float, default=10)
    parser.add_argument("--lambda_style", type=float, default=10)

    # Data sources
    parser.add_argument("--X", type=str, default="..")
    parser.add_argument("--ext", type=str, default="*.png")
    parser.add_argument("--use_diffaug", action="store_true")
    parser.add_argument("--data_preprocess", type=str, default="vanilla")

    # Saving directories and checkpoint/sample iterations
    parser.add_argument("--checkpoint_dir", default="checkpoints_stylegan")
    parser.add_argument("--iden_checkpoint_dir", default="checkpoints_cyclegan")
    parser.add_argument("--sample_dir", type=str, default="cyclegan")
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=100)
    parser.add_argument("--checkpoint_every", type=int, default=1000)

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
