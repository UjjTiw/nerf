# optimizer
import torch
from torch.optim import SGD, Adam
from .optimizers import RAdam, Ranger
# scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, LambdaLR
from .warmup_scheduler import GradualWarmupScheduler

from .visualization import *

def get_optimizer(hparams, models):
    eps = 1e-8
    parameters = [param for model in models for param in model.parameters()]

    optimizers = {
        'sgd': SGD(parameters, lr=hparams.lr, momentum=hparams.momentum, weight_decay=hparams.weight_decay),
        'adam': Adam(parameters, lr=hparams.lr, eps=eps, weight_decay=hparams.weight_decay),
        'radam': RAdam(parameters, lr=hparams.lr, eps=eps, weight_decay=hparams.weight_decay),
        'ranger': Ranger(parameters, lr=hparams.lr, eps=eps, weight_decay=hparams.weight_decay)
    }

    if hparams.optimizer not in optimizers:
        raise ValueError('Optimizer not recognized!')
    
    return optimizers[hparams.optimizer]

def get_scheduler(hparams, optimizer):
    eps = 1e-8
    schedulers = {
        'steplr': MultiStepLR(optimizer, milestones=hparams.decay_step, gamma=hparams.decay_gamma),
        'cosine': CosineAnnealingLR(optimizer, T_max=hparams.num_epochs, eta_min=eps),
        'poly': LambdaLR(optimizer, lambda epoch: (1 - epoch / hparams.num_epochs) ** hparams.poly_exp)
    }

    if hparams.lr_scheduler not in schedulers:
        raise ValueError('Scheduler not recognized!')

    scheduler = schedulers[hparams.lr_scheduler]

    if hparams.warmup_epochs > 0 and hparams.optimizer not in ['radam', 'ranger']:
        scheduler = GradualWarmupScheduler(
            optimizer, 
            multiplier=hparams.warmup_multiplier, 
            total_epoch=hparams.warmup_epochs, 
            after_scheduler=scheduler
        )

    return scheduler

def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']

def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint = checkpoint.get('state_dict', checkpoint)
    
    checkpoint_ = {
        k[len(model_name) + 1:]: v for k, v in checkpoint.items()
        if k.startswith(model_name) and not any(k[len(model_name) + 1:].startswith(prefix) for prefix in prefixes_to_ignore)
    }

    for k in checkpoint:
        if k.startswith(model_name) and any(k[len(model_name) + 1:].startswith(prefix) for prefix in prefixes_to_ignore):
            print('ignore', k)

    return checkpoint_

def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)
