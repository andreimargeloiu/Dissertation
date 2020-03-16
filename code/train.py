"""
Usage:
    train.py [options]

Options:
    -h
    --local                         Run the code locally [default: False]
    --debug                         Enable debug routines. [default: False]
    --model-to-train=NAME           Model to train [default: mnist]
    --log-interval=INT              Number of batches to print statistics [default: 1]
    # --base-path=NAME                Path to the project folder [default: .]
    --max-iterations=INT            [default: -1] Number of max iterations. Use when training on CPU to stop early.

    --epochs=INT                    Number of epochs to run [default: 5]
    --batch-size=INT                [default: 128]
"""
from docopt import docopt
from dpu_utils.utils import run_and_debug
import torch
import torch.nn.functional as F
from utils import initialize
from models import MnistClassifier

from utils import get_train_loader, get_test_loader


def run(args):
    if args['--local'] == True:
        args['--base-path'] = '/Users/andrei/Google Drive/_Facultate/MPhil Cambridge/Dissertation/project'
    else:
        args['--base-path'] = '/content/drive/My Drive/_Facultate/MPhil Cambridge/Dissertation/project'


    print(args)
    config = initialize(args, seed=0)

    if args['--model-to-train'] == 'mnist':
        train_mnist(args, config)


def train_mnist(args, config):
    config.logger.info("Training MNIST classifier.")

    model = MnistClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args['--epochs']):
        train_loader = get_train_loader(args)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            loss = F.nll_loss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args['--log-interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    test(args, model, config)


def test(args, model, config):
    test_loss = 0
    correct = 0
    model.eval()

    with torch.no_grad():
        test_loader = get_test_loader(args)

        for data, target in test_loader:
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    args = docopt(__doc__)

    run_and_debug(lambda: run(args), args["--debug"])
