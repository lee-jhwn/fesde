import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader

import utils
from data_utils import EEGAudioLoader, EEGAudioCollate

from EEGModule import EEGModule
# from eeg_enc.engine.trainer import trainer

from tqdm import tqdm
import wandb
from losses import eeg_loss
import os

USE_WANDB=False

def main():
    hps = utils.get_hparams()


    if USE_WANDB:
        wandb.login()
        wandb.init(project='fesde', name='pretrain_eeg_module')

    train_dataset = EEGAudioLoader(hps.data.training_files, hps.data)
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=True,
                            sampler=None,
                            batch_size=32,
                            pin_memory=True,
                            collate_fn=EEGAudioCollate())
    
    val_dataset = EEGAudioLoader(hps.data.validation_files_both, hps.data)
    val_loader = DataLoader(val_dataset, num_workers=8, shuffle=False,
                            sampler=None,
                            batch_size=32,
                            pin_memory=True,
                            collate_fn=EEGAudioCollate())

    eeg_module = EEGModule(
    n_layers_cnn=hps.model.eeg_module.n_layers_cnn,
    use_s4=hps.model.eeg_module.use_s4,
    n_layers_s4=hps.model.eeg_module.n_layers_s4,
    embedding_size=hps.model.inter_channels,
    is_mask=False,
    in_channels=hps.model.eeg_module.in_channels
  )
    
    eeg_module.to('cuda')
    optim = torch.optim.AdamW(eeg_module.parameters(), 10*hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)


    for epoch in range(1,300):
        eeg_module.train()

        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
            x, x_lengths = x.cuda(0, non_blocking=True), x_lengths.cuda(0, non_blocking=True)
            spec, spec_lengths = spec.cuda(0, non_blocking=True), spec_lengths.cuda(0, non_blocking=True)
            y, y_lengths = y.cuda(0, non_blocking=True), y_lengths.cuda(0, non_blocking=True)
            
            x, x_mask_output, mid_output, ecog_decoder_output = eeg_module(x)
            

            loss_eeg_enc = eeg_loss(x, ecog_decoder_output, x_lengths)

            optim.zero_grad()
            loss_eeg_enc.backward()
            optim.step()
            if USE_WANDB:
                wandb.log({'train_loss': loss_eeg_enc})
        
        eeg_module.eval()
        val_loss_total = []
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(val_loader):
            x, x_lengths = x.cuda(0, non_blocking=True), x_lengths.cuda(0, non_blocking=True)
            spec, spec_lengths = spec.cuda(0, non_blocking=True), spec_lengths.cuda(0, non_blocking=True)
            y, y_lengths = y.cuda(0, non_blocking=True), y_lengths.cuda(0, non_blocking=True)
            
            x, x_mask_output, mid_output, eeg_decoder_output = eeg_module(x)
            

            loss_eeg_module = eeg_loss(x, eeg_decoder_output, x_lengths)

            val_loss_total.append(loss_eeg_module.cpu().item())

        val_loss_total = np.mean(val_loss_total)

        if USE_WANDB:
            wandb.log({'val_loss': val_loss_total})

        utils.save_checkpoint(eeg_module, optim, 10*hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "pretrained_E_{}.pth".format(epoch)))

        eeg_module.train()
   


if __name__=='__main__':
    main()
