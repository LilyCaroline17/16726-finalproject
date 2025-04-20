# CMU 16-726 Learning-Based Image Synthesis / Spring 2024, Final Project
#
# The code base is based on Cycle-GAN from HW 3
#
# Usage:
# ======
#    To train with the default hyperparamters:
#       python style_identifier.py
#
#    To train with cycle consistency loss:
#       python style_identifier.py --use_cycle_consistency_loss

import argparse
import os

import imageio
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import numpy as np

import utils
from data_loader import get_data_loader
from models import StyleIdentifier
from diff_augment import DiffAugment
policy = 'color,translation,cutout'


SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def print_model(m):
    """Prints model information for the generators and discriminators.
    """
    print("                 MODEL                ")
    print("---------------------------------------")
    print(m)
    print("---------------------------------------")


def create_model(opts):
    """Builds the generators and discriminators.
    """ 
    model = StyleIdentifier(conv_dim=opts.g_conv_dim, norm=opts.norm)
    print_model(model)

    if torch.cuda.is_available():
        model.cuda()
        print('Models moved to GPU.')

    return model


def checkpoint(iteration, model, opts):
    """Save generators G_LtoX, G_XtoY and discriminators D_X, D_L."""
    path = os.path.join(
        opts.checkpoint_dir, 'style_identifier_iter%d.pkl' % iteration
    )
    torch.save(model.state_dict(), path)

# probably want update to dataloader_images dataloader_labels <--- @EMILY
def training_loop(dataloader_X, dataloader_L, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """
    # Create generators and discriminators
    model = create_model(opts)

    params = list(model.parameters()) 

    # Create optimizers for the generators and discriminators
    optimizer = optim.Adam(params, opts.lr, [opts.beta1, opts.beta2]) 

    iter_X = iter(dataloader_X)
    iter_L = iter(dataloader_L)

    # Get some fixed data for sampling test loss?
    # that allow us to inspect the model's performance.
    fixed_X = utils.to_var(next(iter_X))
    fixed_L = utils.to_var(next(iter_L))

    iter_per_epoch = min(len(iter_X), len(iter_L))

    for iteration in range(1, opts.train_iters + 1):

        # Reset data_iter for each epoch
        if iteration % iter_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_L = iter(dataloader_L)

        images_X = next(iter_X)
        images_X = utils.to_var(images_X)

        lables = next(iter_L)
        lables = utils.to_var(lables)

        # TRAIN THE DISCRIMINATORS
        # 1. Compute the discriminator losses on real images
        loss = torch.mean(model(images_X) - lables) 

        # sum up the losses and update D_X and D_L
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # plot the losses in tensorboard
        logger.add_scalar('lableTraining', loss, iteration)

        # Print the log info
        if iteration % opts.log_step == 0:
            print(
                'Iteration [{:5d}/{:5d}] | loss: {:6.4f}'.format(
                    iteration, opts.train_iters,loss.item(),
                )
            )

        # Save the model parameters
        if iteration % opts.checkpoint_every == 0:
            checkpoint(iteration, model, opts)


def main(opts):
    """Loads the data and starts the training loop."""
    # Create  dataloaders for images from the two domains X and Y
    dataloader_X = get_data_loader(opts.X, opts=opts)
    dataloader_L = get_data_loader(opts.Y, opts=opts)

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    # Start training
    training_loop(dataloader_X, dataloader_L, opts)


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--disc', type=str, default='dc')  # or 'patch'
    parser.add_argument('--gen', type=str, default='cycle')
    parser.add_argument('--g_conv_dim', type=int, default=32)
    parser.add_argument('--d_conv_dim', type=int, default=32)
    parser.add_argument('--norm', type=str, default='instance')
    parser.add_argument('--use_cycle_consistency_loss', action='store_true')
    parser.add_argument('--init_zero_weights', action='store_true')
    parser.add_argument('--init_type', type=str, default='naive')

    # Training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lambda_cycle', type=float, default=10)

    # Data sources
    parser.add_argument('--X', type=str, default='cat/grumpifyAprocessed')
    parser.add_argument('--Y', type=str, default='cat/grumpifyBprocessed')
    parser.add_argument('--ext', type=str, default='*.png')
    parser.add_argument('--use_diffaug', action='store_true')
    parser.add_argument('--data_preprocess', type=str, default='vanilla')

    # Saving directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', default='checkpoints_cyclegan')
    parser.add_argument('--sample_dir', type=str, default='cyclegan')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=100)
    parser.add_argument('--checkpoint_every', type=int, default=800)

    parser.add_argument('--gpu', type=str, default='0')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
    opts.sample_dir = os.path.join(
        'output/', opts.sample_dir,
        '%s_%g' % (opts.X.split('/')[0], opts.lambda_cycle)
    )
    opts.sample_dir += '%s_%s_%s_%s_%s' % (
        opts.data_preprocess, opts.norm, opts.disc, opts.gen, opts.init_type
    )
    if opts.use_cycle_consistency_loss:
        opts.sample_dir += '_cycle'
    if opts.use_diffaug:
        opts.sample_dir += '_diffaug'

    if os.path.exists(opts.sample_dir):
        cmd = 'rm %s/*' % opts.sample_dir
        os.system(cmd)

    logger = SummaryWriter(opts.sample_dir)

    print_opts(opts)
    main(opts)
