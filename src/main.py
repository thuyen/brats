import argparse
import os
import shutil
import time
import logging
from types import MethodType

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.models as models
from torch.utils.data import DataLoader

from model import Model
from data import ImageList, MemTuple, PEDataLoader


#model_names = sorted(name for name in models.__dict__
#    if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch DeepMedic Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=20, type=int,
                    metavar='N', help='mini-batch size (default: 20)')
parser.add_argument('-g', '--gpu', default='0', type=str,
                    metavar='N', help='mini-batch size (default: 0)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

best_loss = float('inf')


log_file = os.path.join("train_log.txt")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename=log_file)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logging.getLogger('').addHandler(console)

def main():
    global args, best_loss
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # create model
    model = Model().cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    ckpts = 'ckpts'
    if not os.path.exists(ckpts): os.makedirs(ckpts)

    # Data loading code
    args.arch = 'deepmedic'
    train_dir = args.data
    valid_dir = args.data

    train_list = 'train_list.txt'
    valid_list = 'valid_list.txt'

    # The loader will get 1000 patches from 50 subjects for each subepoch
    train_loader = PEDataLoader(
        ImageList(train_list, root=train_dir, split='train', sample_size=10),
        batch_size=50, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    valid_loader = PEDataLoader(
        ImageList(valid_list, root=valid_dir, split='valid', sample_size=10),
        batch_size=50, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        loss = validate(valid_loader, model, criterion)
        print('Validation loss', loss)
        return

    logging.info('-------------- New training session, LR = %f ----------------' % (args.lr, ))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        valid_loss = validate(valid_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        file_name = os.path.join(ckpts, 'model_epoch_%d.tar' % (epoch + 1, ))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
        }, is_best, filename=file_name)

        msg = 'Epoch: {0:02d} Train loss {1:.4f} Valid loss {2:.4f}'.format(epoch+1, train_loss, valid_loss)
        logging.info(msg)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for i, data in enumerate(train_loader):
        loader = DataLoader(
                MemTuple(data),
                batch_size=args.batch_size, shuffle=False,
                num_workers=2, pin_memory=True)
        for datum in loader:
            x1, x2, target = [torch.autograd.Variable(v.cuda())
                    for v in datum[:-1]]

            # compute output
            output = model((x1, x2)) # nx5x9x9x9
            output = output.view(-1, 5, 9**3).permute(0, 2, 1).contiguous()
            output = output.view(-1, 5)
            target = target.view(-1)

            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.data[0], target.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return losses.avg


def validate(valid_loader, model, criterion):
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, data in enumerate(valid_loader):
        loader = DataLoader(
                MemTuple(data),
                batch_size=args.batch_size, shuffle=False,
                num_workers=2, pin_memory=True)
        for datum in loader:
            x1, x2, target = [torch.autograd.Variable(v.cuda(), volatile=True)
                    for v in datum[:-1]]

            # compute output
            output = model((x1, x2)) # nx5x9x9x9
            output = output.view(-1, 5, 9**3).permute(0, 2, 1).contiguous()
            output = output.view(-1, 5)
            target = target.view(-1)

            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.data[0], target.size(0))

    return losses.avg



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
