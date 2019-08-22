import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Net
from load_dataset import LocalDataset, DataPreprocess


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batchsize', '-tb', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='Number of GPU in each mini-batch')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', '-sm', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()
    root = os.path.dirname(os.path.realpath(__file__)) + '/../../datasets/mnist/pytorch/'

    if args.gpu == 1:
        print("=====================================")
        print("# train model with a GPU")
        print('# minibatch-size: {0}'.format(args.batchsize))
        print('# epochs: {0}'.format(args.epochs))
        print('# learning rate: {0}'.format(args.lr))
        print("=====================================")

        try:
            device = torch.device('cuda:0' if torch.cuda.is_available()
                                   else print("cuda is not available"))
        except Exception as e:
            raise str(e)

    torch.manual_seed(args.seed)

    DataPreprocess(root).convert_to_img()

    train_data = LocalDataset(root + 'train.txt')
    test_data = LocalDataset(root + 'test.txt')

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=args.batchsize,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=args.batchsize,
                                              shuffle=False)
    model = Net()
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), os.path.dirname(os.path.realpath(__file__)) + "/mnist_net.pt")


if __name__ == '__main__':
    main()
