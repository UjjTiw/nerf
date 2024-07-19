import os
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models.nerf import Embedding, NeRF
from models.rendering import render_rays
from datasets import dataset_dict
from utils import get_optimizer, get_scheduler, get_learning_rate
from losses import loss_dict
from metrics import psnr, visualize_depth

class NeRFSystem(LightningModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.loss_func = loss_dict[config['loss_type']]()
        
        self.embeddings = nn.ModuleList([
            Embedding(3, 10),  # xyz embedding
            Embedding(3, 4)    # direction embedding
        ])
        
        self.nerf_models = nn.ModuleList([NeRF()])
        if config['N_importance'] > 0:
            self.nerf_models.append(NeRF())
    
    def forward(self, rays: torch.Tensor) -> Dict[str, torch.Tensor]:
        return render_rays(
            self.nerf_models, self.embeddings, rays,
            self.hparams.N_samples, self.hparams.use_disp,
            self.hparams.perturb, self.hparams.noise_std,
            self.hparams.N_importance, self.hparams.chunk,
            self.train_dataset.white_back
        )
    
    def setup(self, stage: str):
        dataset_cls = dataset_dict[self.hparams.dataset_name]
        dataset_kwargs = {
            'root_dir': self.hparams.root_dir,
            'img_wh': tuple(self.hparams.img_wh)
        }
        if self.hparams.dataset_name == 'llff':
            dataset_kwargs.update({
                'spheric_poses': self.hparams.spheric_poses,
                'val_num': self.hparams.num_gpus
            })
        self.train_dataset = dataset_cls(split='train', **dataset_kwargs)
        self.val_dataset = dataset_cls(split='val', **dataset_kwargs)
    
    def configure_optimizers(self):
        optimizer = get_optimizer(self.hparams, self.nerf_models)
        scheduler = get_scheduler(self.hparams, optimizer)
        return [optimizer], [scheduler]
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, shuffle=True, num_workers=4,
                          batch_size=self.hparams.batch_size, pin_memory=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, shuffle=False, num_workers=4,
                          batch_size=1, pin_memory=True)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        rays, rgbs = batch['rays'], batch['rgbs']
        results = self(rays)
        loss = self.loss_func(results, rgbs)
        
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        psnr_value = psnr(results[f'rgb_{typ}'], rgbs)
        
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/psnr', psnr_value, prog_bar=True)
        self.log('lr', get_learning_rate(self.optimizers()))
        
        return {'loss': loss}
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        rays, rgbs = batch['rays'].squeeze(), batch['rgbs'].squeeze()
        results = self(rays)
        loss = self.loss_func(results, rgbs)
        
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        psnr_value = psnr(results[f'rgb_{typ}'], rgbs)
        
        if batch_idx == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W))
            self.logger.experiment.add_images('val/GT_pred_depth',
                                              torch.stack([img_gt, img, depth]),
                                              self.global_step)
        
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/psnr', psnr_value, prog_bar=True)
        
        return {'val_loss': loss, 'val_psnr': psnr_value}

def main(hparams: Dict):
    system = NeRFSystem(hparams)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(f'ckpts/{hparams["exp_name"]}'),
        filename='{epoch:d}',
        monitor='val/loss',
        mode='min',
        save_top_k=5
    )
    
    logger = TensorBoardLogger("logs", name=hparams['exp_name'])
    
    trainer = Trainer(
        max_epochs=hparams['num_epochs'],
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=hparams.get('ckpt_path'),
        logger=logger,
        gpus=hparams['num_gpus'],
        strategy='ddp' if hparams['num_gpus'] > 1 else None,
        num_sanity_val_steps=1,
        benchmark=True,
        profiler="simple" if hparams['num_gpus'] == 1 else None
    )
    
    trainer.fit(system)

if __name__ == '__main__':
    from opt import get_opts
    hparams = get_opts()
    main(hparams)