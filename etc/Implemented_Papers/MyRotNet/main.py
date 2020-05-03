from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import numpy as np
import shutil
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, models, transforms

import tensorflow as tf
import torch.nn.functional as F

from ImageDataLoader import SimpleImageLoader
from models import Res18, Res50, Dense121, Res18_basic
#
# from pytorch_metric_learning import miners
# from pytorch_metric_learning import losses as lossfunc
import glob

import nsml
from nsml import DATASET_PATH, IS_ON_NSML

NUM_CLASSES = 265
if not IS_ON_NSML:
    DATASET_PATH = 'fashion_demo'

def top_n_accuracy_score(y_true, y_prob, n=5, normalize=True):
    num_obs, num_labels = y_prob.shape
    idx = num_labels - n - 1
    counter = 0
    argsorted = np.argsort(y_prob, axis=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx+1:]:
            counter += 1
    if normalize:
        return counter * 1.0 / num_obs
    else:
        return counter

# accuracy for rotation
def accuracy_rot(target, pred):
    #print("accuracy")
    #print(target,pred)
    pred = np.argmax(pred,axis=1)
    result = (target == pred).astype(int)
    #print(pred,result)
    return np.mean(result)

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

def adjust_learning_rate(opts, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opts.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, final_epoch):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, opts.lambda_u * linear_rampup(epoch, final_epoch)

def criterion_rot(pred_rot, targets_rot):
    #print("criterion")
    #print(pred_rot, targets_rot)
    loss = -torch.mean(torch.sum(F.log_softmax(pred_rot, dim=1) * targets_rot, dim=1))
    #loss = torch.mean((torch.softmax(pred_rot,dim=1) - targets_rot)**2)
    return loss

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def split_ids(path, ratio):
    train_ids = np.arange(50000)
    val_ids = np.arange(50000,60061)

    return train_ids, val_ids


### NSML functions
def _infer(model, root_path, test_loader=None):
    if test_loader is None:
        test_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(root_path, 'test',
                               transform=transforms.Compose([
                                   transforms.Resize(opts.imResize),
                                   transforms.CenterCrop(opts.imsize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                               ])), batch_size=opts.batchsize, shuffle=False, num_workers=4, pin_memory=True)
        print('loaded {} test images'.format(len(test_loader.dataset)))

    outputs = []
    s_t = time.time()
    for idx, image in enumerate(test_loader):
        if torch.cuda.is_available():
            image = image.cuda()
        _, probs = model(image)
        output = torch.argmax(probs, dim=1)
        output = output.detach().cpu().numpy()
        outputs.append(output)

    outputs = np.concatenate(outputs)
    return outputs

def bind_nsml(model):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = model.state_dict()
        torch.save(state, os.path.join(dir_name, 'model.pt'))
        print('saved')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pt'))
        model.load_state_dict(state)
        print('loaded')

    def infer(root_path):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer)


######################################################################
# Options
######################################################################
parser = argparse.ArgumentParser(description='Sample Product200K Training')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N', help='number of start epoch (default: 1)')
parser.add_argument('--epochs', type=int, default=250, metavar='N', help='number of epochs to train (default: 200)')

# basic settings
parser.add_argument('--name',default='Res18baseMM', type=str, help='output model name')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--batchsize', default=50, type=int, help='batchsize')
parser.add_argument('--seed', type=int, default=123, help='random seed')

# basic hyper-parameters
parser.add_argument('--momentum', type=float, default=0.9, metavar='LR', help=' ')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 5e-5)')
parser.add_argument('--imResize', default=256, type=int, help='')
parser.add_argument('--imsize', default=224, type=int, help='')
parser.add_argument('--lossXent', type=float, default=1, help='lossWeight for Xent')

# arguments for logging and backup
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='logging training status')
parser.add_argument('--save_epoch', type=int, default=50, help='saving epoch interval')

# hyper-parameters for mix-match
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=150, type=float)
parser.add_argument('--T', default=0.5, type=float)

### DO NOT MODIFY THIS BLOCK ###
# arguments for nsml
parser.add_argument('--pause', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
################################

def main():
    global opts
    opts = parser.parse_args()
    opts.cuda = 0

    # Set GPU
    seed = opts.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(torch.cuda.device_count())

    ###Hyperparameter printing
    print("Hyperparameters. lr: {}, batchsize: {}, alpha: {}, lambda_u: {}, T: {}".format(opts.lr, opts.batchsize, opts.alpha, opts.lambda_u, opts.T))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_ids
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        opts.cuda = 1
        print("Currently using GPU {}".format(opts.gpu_ids))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")


    # Set model
    model = Res50(NUM_CLASSES)
    model.eval()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    if use_gpu:
        model.cuda()

    ### DO NOT MODIFY THIS BLOCK ###
    if IS_ON_NSML:
        bind_nsml(model)
        if opts.pause:
            nsml.paused(scope=locals())
    ################################

    if opts.mode == 'train':
        model.train()
        # Set dataloader
        train_ids, val_ids = split_ids(os.path.join(DATASET_PATH, 'train/train_label'), 0.2)
        print('found {} train, {} validation images'.format(len(train_ids), len(val_ids)))
        train_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'train', train_ids,
                              transform=transforms.Compose([
                                  transforms.Resize(opts.imResize),
                                  transforms.RandomResizedCrop(opts.imsize),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                                batch_size=opts.batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        print('train_loader done')


        validation_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'val', val_ids,
                               transform=transforms.Compose([
                                   transforms.Resize(opts.imResize),
                                   transforms.CenterCrop(opts.imsize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                               batch_size=opts.batchsize, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        print('validation_loader done')

        # Set optimizer
        optimizer = optim.Adam(model.parameters(), lr=opts.lr)

        # INSTANTIATE LOSS CLASS
        train_criterion = SemiLoss()

        # INSTANTIATE STEP LEARNING SCHEDULER CLASS
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[50, 150], gamma=0.1)

        print("Title: {}".format("Rotation pretext for pretraining(Yongalls)"))
        print("Purpose: {}".format("Test wheter pretrained model by rotation pretext works well for mixmatch classification task"))
        print("Environments")
        print("Model: {}".format("Resnet 50"))
        print("Hyperparameters: batchsize {}, lr {}, epoch {}, lambdau {}".format(opts.batchsize, opts.lr, opts.epochs, "None"))
        print("Optimizer: {}, Scheduler: {}".format("Adam", "MultiStepLR ( [50,150], 0.1 )"))
        print("Other necessary Hyperparameters: {}={}".format("lambda_rot",1))
        print("Details: {}".format("Loss is calculated by cross entropy"))
        print("Etc: {}".format("Changes from original code: Res18_basic -> Res50, rotation task"))


        # Train and Validation
        best_acc = -1
        for epoch in range(opts.start_epoch, opts.epochs + 1):
            print('start training')
            loss, avg_top1 = train(opts, train_loader, model, train_criterion, optimizer, epoch, use_gpu)
            scheduler.step()

            print('start validation')
            acc_top1 = validation(opts, validation_loader, model, epoch, use_gpu)
            is_best = acc_top1 > best_acc
            best_acc = max(acc_top1, best_acc)
            nsml.report(summary=True, train_loss= loss, train_acc_top1 = avg_top1, val_acc_top1= acc_top1, step=epoch)
            if is_best:
                print('saving best checkpoint...')
                if IS_ON_NSML:
                    nsml.save(opts.name + '_best')
                else:
                    torch.save(model.state_dict(), os.path.join('runs', opts.name + '_best'))
            if (epoch + 1) % opts.save_epoch == 0:
                if IS_ON_NSML:
                    nsml.save(opts.name + '_e{}'.format(epoch))
                else:
                    torch.save(model.state_dict(), os.path.join('runs', opts.name + '_e{}'.format(epoch)))


def train(opts, train_loader, model, criterion, optimizer, epoch, use_gpu):
    losses = AverageMeter()
    
    acc_top1 = AverageMeter()
    avg_loss = 0.0
    avg_top1 = 0.0

    model.train()

    nCnt =0
    labeled_train_iter = iter(train_loader)

    for batch_idx in range(len(train_loader)):
        try:
            data = labeled_train_iter.next()
            inputs, targets = data
        except:
            labeled_train_iter = iter(train_loader)
            data = labeled_train_iter.next()
            inputs, targets = data


        batch_size = inputs.size(0)
        # Transform label to one-hot
        targets_org = targets
        targets = torch.zeros(batch_size, 4).scatter_(1, targets.view(-1,1), 1)

        if use_gpu :
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        optimizer.zero_grad()

        _, pred = model(inputs)

        loss = criterion_rot(pred, targets)

        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            _, pred_ = model(inputs)

        acc_top1b = accuracy_rot(targets_org.data.cpu().numpy(), pred_.data.cpu().numpy())*100

        acc_top1.update(torch.as_tensor(acc_top1b), inputs.size(0))

        avg_loss += loss.item()
        avg_top1 += acc_top1b

        if batch_idx % opts.log_interval == 0:
            print('Train Epoch:{} [{}/{}] Loss:{:.4f}({:.4f}) Top-1:{:.2f}%({:.2f}%) '.format(
                epoch, batch_idx *inputs.size(0), len(train_loader.dataset), losses.val, losses.avg, acc_top1.val, acc_top1.avg ))

        nCnt += 1

    avg_loss =  float(avg_loss/nCnt)
    avg_top1 = float(avg_top1/nCnt)

    return  avg_loss, avg_top1


def validation(opts, validation_loader, model, epoch, use_gpu):
    model.eval()
    avg_top1= 0.0
    nCnt =0
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader):
            inputs, labels = data
            if use_gpu :
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            nCnt +=1
            _, preds = model(inputs)

            acc_top1 = accuracy_rot(labels.data.cpu().numpy(), preds.data.cpu().numpy())*100
            avg_top1 += acc_top1

        avg_top1 = float(avg_top1/nCnt)

        print('Test Epoch:{} Top1_acc_val:{:.2f}% '.format(epoch, avg_top1))
    return avg_top1



if __name__ == '__main__':
    main()
