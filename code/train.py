"""
Usage:
    train.py [options]

Options:
    -h
    --local                         Run the code locally [default: False]
    --debug                         Enable debug routines. [default: False]
    --model-to-train=NAME           Model to train [default: mnist]
    --log-interval=INT              Number of batches to print statistics [default: 1]
    --max-iterations=INT            [default: -1] Number of max iterations. Use when training on CPU to stop early.

    --epochs=INT                    Number of epochs to run [default: 5]
    --batch-size=INT                [default: 128]
"""
import time
from time import sleep

from cox.store import Store
from docopt import docopt
from tqdm import trange
from tqdm.autonotebook import tqdm

from os import path

from dpu_utils.utils import run_and_debug
import torch
from torch.utils.data import DataLoader

from utils import initialize, get_mnist_dl, AverageMeter
import constants as consts
from models import MnistClassifier


def test_tqdm():
    print("test_tqdm")
    iterator = trange(100)
    for i in iterator:
        sleep(0.1)
        iterator.set_description(f"{i}")
        iterator.refresh()


def run(args):
    if args['--local'] == True:
        args['--base-path'] = '/Users/andrei/Google Drive/_Facultate/MPhil Cambridge/Dissertation/project'
    else:
        args['--base-path'] = '/content/drive/My Drive/_Facultate/MPhil Cambridge/Dissertation/project'

    initialize(args, seed=0)

    OUT_DIR = path.join(args['--base-path'], 'logs')
    store = Store(OUT_DIR)

    if args['--model-to-train'] == 'mnist':
        model = MnistClassifier()
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_dl = get_mnist_dl(args, train=True)
        valid_dl = get_mnist_dl(args, train=False)

        train_model(args, model, optim, train_dl, valid_dl, store)


def train_model(args, model, optim, train_dl: DataLoader, valid_dl: DataLoader, store: Store, attack=None, ratio: int = 0):
    """
    Generic training routine, which is flexible to allow both standard and adversarial training.
    """
    start_time = time.time()

    # Initial setup
    store.add_table(consts.LOGS_TABLE, consts.LOGS_SCHEMA)
    store.add_table(consts.ARGS_TABLE, consts.ARGS_SCHEMA)
    args_info = {
        'epochs': args['--epochs'],
        'batch_size': args['--batch-size'],
        'model': 'mnist'
    }

    # store[consts.ARGS_TABLE].append_row(args_info)

    for epoch in range(args['--epochs']):
        # Train for one epoch
        train_acc, train_loss = _internal_loop(args, True, model, optim, train_dl, epoch, store)

        # Evaluate on validation
        with torch.no_grad():
            valid_acc, valid_loss = _internal_loop(args, False, model, None, valid_dl, epoch, store)

        # Log
        log_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'train_error_rate': 1 - train_acc,
            'valid_error_rate': 1 - valid_acc,
            'valid_adv_error_rate': -1,
            'time': time.time() - start_time
        }

        # store[consts.LOGS_TABLE].append_row(log_info)

    return model


def _internal_loop(args, is_train, model, optim, loader: DataLoader, epoch, store: Store):
    """
    *Internal function used by train_model or eval_model*

    Runs a single epoch of either training or evaluation.

    Arguments:
    - args: arguments
    - is_train (True, False) - tells whether we are in training or evaluation mode
    - model: model
    - optim: optimizer
    - loader: data loader
    - store: cox.Store

    Returns average accuracy and loss
    """
    accuracy = AverageMeter()
    losses = AverageMeter()

    # Initial setup
    model = model.train() if is_train else model.eval()
    criteria = torch.nn.CrossEntropyLoss()

    loop_type = "Train" if is_train else "Valid"

    iterator = tqdm(iter(loader), total=len(loader))
    for i, (xb, yb) in enumerate(iterator):
        y_hat = model(xb)  # (batch_size, num_classes)
        loss = criteria(y_hat, yb)  # (tensor of one value)

        # Metrics
        pred_class = y_hat.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        acc = pred_class.eq(yb.view_as(pred_class)).sum().item()

        losses.update(loss / y_hat.shape[0])
        accuracy.update(acc / y_hat.shape[0])

        # Backward pass
        if is_train:
            optim.zero_grad()
            loss.backward()
            optim.step()

        # ITERATOR
        description = 'Epoch: %d ' % (epoch)
        iterator.set_description(description)
        iterator.set_postfix_str("%s loss: %.4f, %s accuracy: %.4f" % (loop_type, losses.avg, loop_type, accuracy.avg))

        # Tensorboard
        store.tensorboard.add_scalar(f'{loop_type}_loss', losses.val, i)
        store.tensorboard.add_scalar(f'{loop_type}_accuracy', accuracy.val, i)

    return accuracy.avg, losses.avg


if __name__ == "__main__":
    args = docopt(__doc__)

    run_and_debug(lambda: run(args), args["--debug"])
