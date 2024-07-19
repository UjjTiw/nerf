import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from opt import get_opts
from datasets import dataset_dict
from models.nerf import Embedding, NeRF
from models.rendering import render_rays
from utils import get_optimizer, get_scheduler, get_learning_rate, visualize_depth
from losses import loss_dict
from metrics import psnr

class NeRFSystem(pl.LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.save_hyperparameters(hparams)
        
        # Loss function
        self.loss = loss_dict[hparams.loss_type]()
        
        # Embeddings
        self.embedding_xyz = Embedding(3, 10)  # 10 is the default number
        self.embedding_dir = Embedding(3, 4)   # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        # Models
        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

    def decode_batch(self, batch):
        rays = batch['rays']  # (B, 8)
        rgbs = batch['rgbs']  # (B, 3)
        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = render_rays(
                self.models,
                self.embeddings,
                rays[i:i+self.hparams.chunk],
                self.hparams.N_samples,
                self.hparams.use_disp,
                self.hparams.perturb,
                self.hparams.noise_std,
                self.hparams.N_importance,
                self.hparams.chunk,  # chunk size is effective in val mode
                self.train_dataset.white_back
            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir, 'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, optimizer)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, num_workers=4, batch_size=self.hparams.batch_size, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, num_workers=4, batch_size=1, pin_memory=True)

    def training_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        results = self(rays)
        loss = self.loss(results, rgbs)

        log = {'train/loss': loss, 'lr': get_learning_rate(self.optimizers())}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        results = self(rays)
        loss = self.loss(results, rgbs)

        log = {'val_loss': loss}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu().permute(2, 0, 1)  # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W))  # (3, H, W)
            stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth', stack, self.global_step)

        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        self.log_dict(log)
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr)

if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('ckpts', hparams.exp_name),
        filename='{epoch:d}',
        monitor='val/loss',
        mode='min',
        save_top_k=5
    )

    logger = TensorBoardLogger(
        save_dir="logs",
        name=hparams.exp_name,
        log_graph=True
    )

    trainer = pl.Trainer(
    max_epochs=hparams.num_epochs,
    callbacks=[checkpoint_callback],
    logger=logger,
    accelerator='gpu' if hparams.num_gpus > 0 else 'cpu',
    devices=hparams.num_gpus if hparams.num_gpus > 0 else 1,
    num_sanity_val_steps=1,
    benchmark=True,
    profiler="simple" if hparams.num_gpus == 1 else None
)




    trainer.fit(system, ckpt_path=hparams.ckpt_path if hparams.ckpt_path else None)
