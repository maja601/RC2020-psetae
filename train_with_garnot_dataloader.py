import argparse
import time
import shutil
import pickle as pkl

from torch.utils.data import DataLoader, random_split

import torch

from losses.focalloss import FocalLoss
from models.pse_tae_with_garnot_infos import PSE_TAE
# from datasets.pixelsettoy_dataset import PixelSetToyDataset
# from datasets.pixelset_dataset_19 import PixelSetDataset
# from datasets.pixelset_dataset import PixelSetDataset
from datasets.pixelset_dataset_garnot import PixelSetData

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--set_size", type=int, default=64, help="size of the pixel set")
parser.add_argument("--gamma", type=float, default=1., help="focal loss:  focusing parameter")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--print_freq", type=int, default=53, help="print frequency (default: 53)")

"""
Structure From https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""

best_acc1 = 0


def main():
    global best_acc1
    opt = parser.parse_args()
    print(opt)

    # Tensorboard Summary Writer
    # current_run = 'runs/psetae_' + str(int(time.time()))
    # writer = SummaryWriter(current_run)

    # CUDA
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # Initialize Spatial and Temporal Attention Encoder
    pse_tae = PSE_TAE(device).to(device)

    # Configure data loader
    datapath = '/home/maja/ssd/rc2020dataset/pixelset/'
    metapath = '/home/maja/ssd/rc2020dataset/pixelset/META/'
    mean_std = pkl.load(open('/home/maja/ssd/rc2020dataset/pixelset/S2-2017-T31TFM-meanstd.pkl', 'rb'))
    extra = 'geomfeat'
    dataset = PixelSetData(datapath, labels='label_44class', npixel=64,
                           sub_classes=[1, 3, 4, 5, 6, 8, 9, 12, 13, 14, 16, 18, 19, 23, 28, 31, 33, 34, 36, 39],
                           norm=mean_std,
                           extra_feature=extra)
    fold_len = int(len(dataset) / 5)
    n_train = len(dataset) - 2 * fold_len
    train_set, val_set, test_set = random_split(dataset,
                                                (n_train, fold_len, fold_len),
                                                generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False)

    # Loss function
    focal_loss = FocalLoss(opt.gamma).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(pse_tae.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    for epoch in range(opt.n_epochs):
        # ----------
        #  Training
        # ----------

        train(train_loader,
              pse_tae,
              focal_loss,
              optimizer,
              epoch,
              opt,
              device)

        # -----------
        #  Validation
        # -----------

        acc1 = validate(val_loader,
                        pse_tae,
                        focal_loss,
                        opt,
                        device)

        # -----------
        #  Remember best acc@1 and save checkpoint
        # -----------
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'best_acc1': best_acc1,
        #         'optimizer' : optimizer.state_dict(),
        #     }, is_best)


def train(train_loader,
          pse_tae,
          focal_loss,
          optimizer,
          epoch,
          opt,
          device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    # pse_time = AverageMeter('PSE', ':6.3f')
    # tae_time = AverageMeter('TAE', ':6.3f')
    # decode_time = AverageMeter('Decode', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # -------------------------
    #  Put Models in Train Mode
    # -------------------------
    pse_tae.train()

    end = time.time()

    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)

        # Get training data
        data = batch['data'].to(device)
        label = batch['label'].to(device)
        geom = batch['geom'].to(device)
        # pixels_in_parcel = batch['pixels_in_parcel'].to(device)
        mask = batch['mask'].to(device)

        # ---------------------------------
        #  Train Everything together
        # ---------------------------------
        optimizer.zero_grad()
        output = pse_tae(data, geom, 0, mask)
        _, prediction = torch.max(output.data, 1)

        # ---------------------------------
        #  Loss
        # ---------------------------------

        # Focal Loss between output and target
        loss = focal_loss(output.to(device), label)

        # ---------------------------------
        #  Record Stats
        # ---------------------------------
        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output, label, topk=(1, 2))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        top5.update(acc5[0], data.size(0))

        # ---------------------------------
        #  Gradient & SGD step
        # ---------------------------------
        loss.backward()
        optimizer.step()

        # ---------------------------------
        #  Time
        # ---------------------------------
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            progress.display(i)

        # writer.add_scalar('train_loss', loss.item(), epoch * len(train_loader) + i)


def validate(val_loader, pse_tae, focal_loss, opt, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # -------------------------
    #  Put Models in Eval Mode
    # -------------------------
    pse_tae.eval()

    with torch.no_grad():
        end = time.time()
        for i, val_batch in enumerate(val_loader):
            # Get validation data
            data_val = val_batch['data'].to(device)
            label_val = val_batch['label'].to(device)
            geom_val = val_batch['geom'].to(device)
            # pixels_in_parcel_val = val_batch['pixels_in_parcel'].to(device)
            mask_val = val_batch['mask'].to(device)

            # -------------------------
            #  Compute Predictions
            # -------------------------
            output = pse_tae(data_val, geom_val, 0, mask_val)
            loss = focal_loss(output.to(device), label_val)
            _, prediction = torch.max(output.data, 1)

            # ---------------------------------
            #  Record Stats
            # ---------------------------------
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, label_val, topk=(1, 5))
            losses.update(loss.item(), data_val.size(0))
            top1.update(acc1[0], data_val.size(0))
            top5.update(acc5[0], data_val.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        with open('log.txt', 'a') as the_file:
            the_file.write('\t'.join(entries) + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
