"""
2D convolution (HR-CNN) based model training script
"""
from hr_cnn import HrCNN
import pure_dataset
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)


def train(train_loader, extractor_model, criterion, extractor_optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode

    extractor_model.train()

    end = time.time()
    for i, (net_input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        net_input = net_input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = extractor_model(net_input)
        loss = criterion(output.squeeze(), target).cuda()

        # compute gradient and do SGD step
        extractor_optimizer.zero_grad()

        loss.backward()

        extractor_optimizer.step()

        loss = loss.float()
        # measure accuracy and record loss
        losses.update(loss.item(), net_input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
        with open('train_log.csv', 'a') as log:
            log.write("{}, {}, {}, {}\n".format(losses.val, losses.avg, top1.val, top1.avg))


def validate(val_loader, extractor_model, criterion):
    print('validation start ')
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    extractor_model.eval()

    end = time.time()
    for i, (net_input, target) in enumerate(val_loader):

        net_input = net_input.cuda(non_blocking=True)

        # target = target.squeeze()
        # target = target.cuda(non_blocking=True)
        target = torch.median(target).cuda(non_blocking=True)
        print(net_input.shape)

        # compute output
        with torch.no_grad():
            output = extractor_model(net_input)

        loss = criterion(output.squeeze(), target)
        output = output.float()
        print(output, target)
        loss = loss.float()

        # measure accuracy and record loss
        losses.update(loss.item(), net_input.size(0))
        # top1.update(prec1.item(), net_input1.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    with open('test_log.csv', 'a') as log:
        log.write("{}, {}\n".format(losses.avg, top1.avg))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


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
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    seq_list = []
    end_indexes = []

    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print("initialize model...")
    extractor_model = HrCNN(3)

    extractor_model = torch.nn.DataParallel(extractor_model)
    extractor_model.cuda()

    resume = 'save_temp/extractor_checkpoint_selfattention_1.tar'
    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    seqlen = 1
    seq_dir = 'E:/Datasets_PULSE/set_all/'
    pulse = pure_dataset.PulseDataset("transfer_train.txt", seq_dir,
                                      transform=transforms.ToTensor())

    pulse_test = pure_dataset.PulseDataset("seq_test.txt", seq_dir,
                                           transform=transforms.ToTensor())

    # fig = plt.figure()
    # for i in range(len(pulse)):
    #     sample = pulse[i]
    #
    #     print(i, sample[0].shape, sample[1].shape)
    #
    #     ax = plt.subplot(1, 4, i + 1)
    #     print(sample[0])
    #     plt.imshow((sample[0].permute(1,2,0)))
    #     plt.tight_layout()
    #     ax.set_title('Sample #{}'.format(i))
    #     ax.axis('off')
    #
    #     if i == 3:
    #         plt.show()
    #         break

    train_loader = torch.utils.data.DataLoader(
        pulse,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        pulse_test,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss()
    criterion = criterion.cuda()

    extractor_optimizer = torch.optim.SGD(extractor_model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # extractor_optimizer = torch.optim.Adam(extractor_model.parameters(), 0.0001,
    #                                       weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, extractor_model, criterion)
    print('starting training...')
    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(extractor_optimizer, epoch)

        # train for one epoch
        train(train_loader, extractor_model,  criterion, extractor_optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, extractor_model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': extractor_model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'extractor_checkpoint_selfattention2_{}.tar'.format(epoch)))
