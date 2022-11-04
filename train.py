import argparse
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torchvision import datasets, transforms

from utils.cutout import Cutout

from models.resnet import ResNet18
from models import resnet_nobn
from models.wideresnet import WideResNet
from models.mlp import MLP
from models.create_model import create_model
import wandb
import warmup_scheduler

import asdfghjkl as asdl
from asdfghjkl import FISHER_EXACT, FISHER_MC, FISHER_EMP
from asdfghjkl import SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG
from asdfghjkl.precondition import Shampoo,ShampooHyperParams
from models.factory import create_model_fractal

import pandas as pd
import os
import timm

model_options = ['mlp','resnet18','resnet18_nobn', 'wideresnet']
dataset_options = ['MNIST','CIFAR10', 'CIFAR100', 'svhn']

OPTIM_SGD = 'sgd'
OPTIM_ADAM = 'adamw'
OPTIM_KFAC_MC = 'kfac_mc'
OPTIM_KFAC_EMP = 'kfac_emp'
OPTIM_SKFAC_MC = 'skfac_mc'
OPTIM_SKFAC_EMP= 'skfac_emp'
OPTIM_SMW_NGD = 'smw_ng'
OPTIM_FULL_PSGD = 'full_psgd'
OPTIM_KRON_PSGD = 'psgd'
OPTIM_NEWTON = 'newton'
OPTIM_ABS_NEWTON = 'abs_newton'
OPTiM_KBFGS = 'kbfgs'
OPTIM_SHAMPOO='shampoo'

def main():
    total_train_time = 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train(epoch)
        total_train_time += time.time() - start
        val(epoch)
        test(epoch)

    print(f'total_train_time: {total_train_time:.2f}s')
    print(f'avg_epoch_time: {total_train_time / args.epochs:.2f}s')
    print(f'avg_step_time: {total_train_time / args.epochs / num_steps_per_epoch * 1000:.2f}ms')
    if args.wandb:
        wandb.run.summary['total_train_time'] = total_train_time
        wandb.run.summary['avg_epoch_time'] = total_train_time / args.epochs
        wandb.run.summary['avg_step_time'] = total_train_time / args.epochs / num_steps_per_epoch

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    if args.wandb:
        log = {'epoch': epoch,
               'iteration': epoch * num_steps_per_epoch,
               'test_loss': test_loss,
               'test_accuracy': test_accuracy}
        wandb.log(log)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))

def val(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    test_accuracy = 100. * correct / len(val_loader.dataset)
    if args.wandb:
        log = {'epoch': epoch,
               'iteration': epoch * num_steps_per_epoch,
               'val_loss': test_loss,
               'val_accuracy': test_accuracy}
        wandb.log(log)
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), test_accuracy))

def train(epoch):
    model.train()
    for batch_idx, (x, t) in enumerate(train_loader):
        x, t = x.to(device), t.to(device)
        optimizer.zero_grad(set_to_none=True)

        # y = model(x)
        # loss = F.cross_entropy(y, t)
        # loss.backward()

        dummy_y = grad_maker.setup_model_call(model, x)
        loss_func=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        grad_maker.setup_loss_call(loss_func, dummy_y, t)

        y, loss = grad_maker.forward_and_backward()
        if args.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.kl_clip)
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            if args.wandb:
                log = {'epoch': epoch,
                       'iteration': (epoch - 1) * num_steps_per_epoch + batch_idx + 1,
                       'train_loss': float(loss),
                       'learning_rate': optimizer.param_groups[0]['lr']}
                wandb.log(log)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / num_steps_per_epoch, float(loss)))

        scheduler.step()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='CIFAR10',
                        choices=dataset_options)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--warmup', default=0, type=int, metavar='N', help='number of warmup epochs')
    parser.add_argument('--data_augmentation', action='store_false', default=True,
                        help='augment data by flipping and cropping')
    parser.add_argument('--cutout', action='store_true', default=True,
                        help='apply cutout')
    parser.add_argument('--mixup', type=float, default=0)
    parser.add_argument('--cutmix', type=float, default=0)

    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=16,
                        help='length of the holes')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument('--train_size', type=int, default=45056)
    parser.add_argument('--img_size', type=int, default=32)

    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_ratio', type=float, default=0,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--optim', default=OPTIM_KRON_PSGD, choices=[OPTIM_KFAC_EMP,OPTIM_KFAC_MC,OPTIM_SKFAC_EMP,OPTIM_SKFAC_MC, OPTIM_SMW_NGD, OPTIM_KRON_PSGD,OPTIM_SHAMPOO,OPTIM_SGD,OPTIM_ADAM])
    parser.add_argument('--damping', type=float, default=1e-2)
    parser.add_argument('--swift', action='store_true', default=True)

    parser.add_argument('--nesterov', action='store_true', default=False)

    parser.add_argument('--gradient_clipping', action='store_true', default=True)
    parser.add_argument('--kl_clip', type=float, default=10,
                        help='kl_clip')

    parser.add_argument('--thratio', type=float, default=2)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--power_iter', type=int, default=10)

    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument('--int_exp', type=float, default=1.2)

    parser.add_argument('--width', type=int, default=2048)
    parser.add_argument('--depth', type=int, default=3)

    parser.add_argument('--log_interval', type=int, default=1,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--wandb', action='store_false', default=True)

    parser.add_argument('--sd', default=0.1, type=float, help='rate of stochastic depth')
    parser.add_argument('--is_LSA',default=True, action='store_false', help='Locality Self-Attention')
    parser.add_argument('--is_SPT',default=True, action='store_false', help='Shifted Patch Tokenization')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_connect', type=float, default=None, metavar='PCT',
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    parser.add_argument('--drop_path', type=float, default=None, metavar='PCT',
                        help='Drop path rate (default: None)')
    parser.add_argument('--drop_block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')
    parser.add_argument('--pretrained_path', default='./pretrain/exfractal_21k_tiny.pth.tar', type=str, metavar='PATH',
                    help='Load model from local pretrained checkpoint')
    parser.add_argument('--pretrained', type=bool, default=True,
                        help='whether pre trained')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    cudnn.benchmark = True  # Should make training should go faster for large models
    print(args)

    config = vars(args).copy()
    if args.wandb:
        wandb.init(config=config,
                   entity=os.environ.get('WANDB_ENTITY', None),
                   project=os.environ.get('WANDB_PROJECT', None),
                   )

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda')
    
    interval=args.interval

    if args.dataset == 'CIFAR10':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        train_transform = transforms.Compose([])
        if args.img_size != 32:
            train_transform.transforms.append(transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),)
        else:
            train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)
        if args.cutout:
            train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

        test_transform = transforms.Compose([])
        if args.img_size != 32:
            test_transform.transforms.append(transforms.Resize((args.img_size,args.img_size)),)
        test_transform.transforms.append(transforms.ToTensor())
        test_transform.transforms.append(normalize)

        num_classes = 10
        train_dataset = datasets.CIFAR10(root='data/',
                                         train=True,
                                         download=True,
                                         transform=train_transform,)
        val_dataset = datasets.CIFAR10(root='data/',
                                         train=True,
                                         download=True,
                                         transform=test_transform,)
        test_dataset = datasets.CIFAR10(root='data/',
                                        train=False,
                                        transform=test_transform,
                                        download=True)

        train_dataset, _ = torch.utils.data.random_split(train_dataset, [args.train_size, len(train_dataset)-args.train_size], generator=torch.Generator().manual_seed(args.seed))
        _, val_dataset = torch.utils.data.random_split(val_dataset, [args.train_size, len(val_dataset)-args.train_size], generator=torch.Generator().manual_seed(args.seed))

    elif args.dataset == 'MNIST':
        train_transform = transforms.Compose([])
        if args.data_augmentation:
            train_transform.transforms.append(transforms.RandomAffine([-15,15], scale=(0.8, 1.2)))
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(transforms.Normalize((0.1307,), (0.3081,)))

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        num_classes = 10
        train_dataset = datasets.MNIST(root='data/',
                                         train=True,
                                         download=True,
                                         transform=train_transform,)
        val_dataset = datasets.MNIST(root='data/',
                                         train=True,
                                         download=True,
                                         transform=test_transform,)
        test_dataset = datasets.MNIST(root='data/',
                                        train=False,
                                        transform=test_transform,
                                        download=True)

        train_dataset, _ = torch.utils.data.random_split(train_dataset, [49152, 10848], generator=torch.Generator().manual_seed(args.seed))
        _, val_dataset = torch.utils.data.random_split(val_dataset, [49152, 10848], generator=torch.Generator().manual_seed(args.seed))

    
    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.num_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.num_workers)
    num_steps_per_epoch = len(train_loader)
    
    if args.model == 'mlp':
        model = MLP(n_hid=args.width,depth=args.depth)
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif args.model == 'wideresnet':
        model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                             dropRate=0.3)
    elif args.model in ['vit','cait','pit','t2t','swin']:
        model = create_model(img_size=32,n_classes=10,args=args)
    elif args.model == 'vit_tiny_imagenet':
        model = timm.create_model('vit_tiny_patch16_224',pretrained=args.pretrained,num_classes=num_classes)
    elif args.model == 'mixer_base_fractal':
        model = timm.create_model('mixer_b16_224',pretrained=args.pretrained,num_classes=num_classes)

    model = model.cuda()

    if args.optim == OPTIM_ADAM:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == OPTIM_SHAMPOO:
        config = ShampooHyperParams(weight_decay=args.weight_decay,preconditioning_compute_steps=interval,statistics_compute_steps=interval,nesterov=args.nesterov)
        optimizer = Shampoo(model.parameters(),lr=args.lr,momentum=args.momentum,hyperparams=config)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay,nesterov=args.nesterov)

    if args.optim == OPTIM_KFAC_MC:
        config = asdl.NaturalGradientConfig(data_size=args.batch_size,
                                            fisher_type=FISHER_MC,
                                            damping=args.damping,
                                            interval=args.interval,
                                            ignore_modules=[nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.LayerNorm],
                                            ema_decay=args.ema_decay)
        grad_maker = asdl.KfacGradientMaker(model, config,swift=False)
    elif args.optim == OPTIM_KFAC_EMP:
        config = asdl.NaturalGradientConfig(data_size=args.batch_size,
                                            fisher_type=FISHER_EMP,
                                            damping=args.damping,
                                            interval=args.interval,
                                            ignore_modules=[nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.LayerNorm],
                                            ema_decay=args.ema_decay)
        grad_maker = asdl.KfacGradientMaker(model, config,swift=False)
    elif args.optim == OPTIM_SKFAC_MC:
        config = asdl.NaturalGradientConfig(data_size=args.batch_size,
                                            fisher_type=FISHER_MC,
                                            damping=args.damping,
                                            intervalr=args.interval,
                                            ignore_modules=[nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.LayerNorm])
        grad_maker = asdl.KfacGradientMaker(model, config,swift=True)
    elif args.optim == OPTIM_SMW_NGD:
        config = asdl.SmwEmpNaturalGradientConfig(data_size=args.batch_size,
                                                  damping=args.damping)
        grad_maker = asdl.SmwEmpNaturalGradientMaker(model, config)
    elif args.optim == OPTIM_FULL_PSGD:
        config = asdl.PsgdGradientConfig(upd_precond_interval=args.interval)
        grad_maker = asdl.PsgdGradientMaker(model,config)
    elif args.optim == OPTIM_KRON_PSGD:
        config = asdl.PsgdGradientConfig(upd_precond_interval=args.interval)
        grad_maker = asdl.KronPsgdGradientMaker(model,config)
    elif args.optim == OPTIM_NEWTON:
        config = asdl.NewtonGradientConfig(damping=args.damping)
        grad_maker = asdl.NewtonGradientMaker(model, config)
    elif args.optim == OPTIM_ABS_NEWTON:
        config = asdl.NewtonGradientConfig(damping=args.damping, absolute=True)
        grad_maker = asdl.NewtonGradientMaker(model, config)
    elif args.optim == OPTiM_KBFGS:
        config = asdl.KronBfgsGradientConfig(data_size=args.batch_size,
                                             damping=args.damping)
        grad_maker = asdl.KronBfgsGradientMaker(model, config)
    else:
        grad_maker = asdl.GradientMaker(model)

    if args.warmup > 0:
        base_scheduler=CosineAnnealingLR(optimizer, T_max=args.epochs*num_steps_per_epoch,eta_min=args.lr*args.lr_ratio)
        scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=args.warmup*num_steps_per_epoch, after_scheduler=base_scheduler)
    if args.warmup==0:
        scheduler=CosineAnnealingLR(optimizer, T_max=args.epochs*num_steps_per_epoch,eta_min=args.lr*args.lr_ratio)

    torch.cuda.synchronize()
    try:
        main()
        max_memory = torch.cuda.max_memory_allocated()
    except RuntimeError as err:
        if 'CUDA out of memory' in str(err):
            print(err)
            max_memory = -1  # OOM
        else:
            raise RuntimeError(err)

    print(f'cuda_max_memory: {max_memory/float(1<<30):.2f}GB')
    if args.wandb:
        wandb.run.summary['cuda_max_memory'] = max_memory



