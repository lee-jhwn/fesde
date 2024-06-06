import os
import json
import argparse
import itertools
import math
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy.io.wavfile import write


import commons
import utils
from data_utils import EEGAudioLoader, EEGAudioCollate
from models import SpeechDecoder, MultiPeriodDiscriminator
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss,
  eeg_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from EEGModule import EEGModule

import wandb

torch.backends.cudnn.benchmark = True
global_step = 0


import warnings
warnings.filterwarnings('ignore')



def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  hps = utils.get_hparams()

  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '7000'

  hps = utils.get_hparams()
  run(rank=0, n_gpus=1, hps=hps) # using only single GPU


def run(rank, n_gpus, hps):
  global global_step

  if hps.wandb:
    wandb.login()
    wandb.init(project='fesde', name=hps.run_name)

  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = EEGAudioLoader(hps.data.training_files, hps.data)
  collate_fn = EEGAudioCollate()
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=True, pin_memory=True,
      collate_fn=collate_fn, batch_size=hps.train.batch_size)
  if rank == 0:
    eval_loader_both = DataLoader(EEGAudioLoader(hps.data.validation_files_both, hps.data), num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
    eval_loader_audio = DataLoader(EEGAudioLoader(hps.data.validation_files_audio, hps.data), num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
    eval_loader_subject = DataLoader(EEGAudioLoader(hps.data.validation_files_subject, hps.data), num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
    
    eval_loaders = [eval_loader_both, eval_loader_audio, eval_loader_subject]
    
  
  eeg_module = EEGModule(
    n_layers_cnn=hps.model.eeg_module.n_layers_cnn,
    use_s4=hps.model.eeg_module.use_s4,
    n_layers_s4=hps.model.eeg_module.n_layers_s4,
    embedding_size=hps.model.inter_channels,
    is_mask=False,
    in_channels=hps.model.eeg_module.in_channels
  ).cuda(rank)

  net_g = SpeechDecoder(
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model).cuda(rank)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_eeg = torch.optim.AdamW(
      eeg_module.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
  net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
  eeg_module = DDP(eeg_module, device_ids=[rank], find_unused_parameters=True)

  if hps.train.pretrained_audio:
    print('loading pretrained audio generator from ' + hps.train.pretrained_audio)
    pretrained_audio_ckpt = torch.load(hps.train.pretrained_audio, map_location='cpu')
    pretrained_stated_dict = pretrained_audio_ckpt['model']
    new_state_dict = {}
    for k, v in net_g.module.state_dict().items():
      if k.split('.')[0] != 'enc_proj':
        new_state_dict[k] = pretrained_stated_dict[k]
      else:
        new_state_dict[k] = v
    net_g.module.load_state_dict(new_state_dict)
        
    print('finished loading pretrained audio generator')
  


    if hps.train.freeze_modules:
      if "dec" in hps.train.freeze_modules:
        for param in net_g.module.dec.parameters():
          param.requires_grad = False
      if "enc_q" in hps.train.freeze_modules:
        for param in net_g.module.enc_q.parameters():
          param.requires_grad = False
      if "dp" in hps.train.freeze_modules:
        for param in net_g.module.dp.parameters():
          param.requires_grad = False
      if "flow" in hps.train.freeze_modules:
        for param in net_g.module.flow.parameters():
          param.requires_grad = False
      if "eeg" in hps.train.freeze_modules:
        for param in eeg_module.module.parameters():
          param.requires_grad = False
          print('freezing eeg module')
    
  if hps.train.pretrained_eeg:
    print('loading pretrained eeg module', hps.train.pretrained_eeg)
    _ = utils.load_checkpoint(hps.train.pretrained_eeg, eeg_module, None)
    if "eeg" in hps.train.freeze_modules:
      for param in eeg_module.module.parameters():
        param.requires_grad = False
        print('freezing eeg module')
  

  epoch_str = 1
  global_step = 0

  if True: # continue from past run
    try:
      _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
      _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
      _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "E_*.pth"), eeg_module, optim_eeg)
      global_step = (epoch_str - 1) * len(train_loader)
    except:
      pass

  
  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_eeg = torch.optim.lr_scheduler.ExponentialLR(optim_eeg, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=False)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d, eeg_module], [optim_g, optim_d, optim_eeg], [scheduler_g, scheduler_d, scheduler_eeg], scaler, [train_loader, eval_loaders], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d, eeg_module], [optim_g, optim_d, optim_eeg], [scheduler_g, scheduler_d, scheduler_eeg], scaler, [train_loader, None], None, None)
    scheduler_g.step()
    scheduler_d.step()
    scheduler_eeg.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  net_g, net_d, eeg_module = nets
  optim_g, optim_d, optim_eeg = optims
  scheduler_g, scheduler_d, scheduler_eeg = schedulers
  train_loader = loaders[0]
  if writers is not None:
    writer, writer_eval = writers

  global global_step

  net_g.train()
  net_d.train()
  eeg_module.train()



  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)

    with autocast(enabled=False):
      x, x_mask_output, mid_output, eeg_decoder_out = eeg_module(x)
      mid_output_lengths = x_lengths.clone() * mid_output.size(2) / x.size(2)
      mid_output_lengths = mid_output_lengths.long()

      y_hat, _, _, ids_slice, x_mask, z_mask,\
      (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(mid_output.detach(), mid_output_lengths, spec, spec_lengths)

      mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax)
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate, 
          hps.data.hop_length, 
          hps.data.win_length, 
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )

      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

      # Discriminator
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast(enabled=False):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      with autocast(enabled=False):
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel  + loss_kl

        loss_eeg_module = eeg_loss(x, eeg_decoder_out, x_lengths)

    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)


    optim_eeg.zero_grad()
    if "eeg" not in hps.train.freeze_modules:
      scaler.scale(loss_eeg_module).backward()
    scaler.unscale_(optim_eeg)
    grad_norm_eeg_enc = commons.clip_grad_value_(eeg_module.parameters(), None)
    scaler.step(optim_eeg)

    scaler.update()

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g, "loss/eeg/total":loss_eeg_module
                       }
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = { 
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)
        
        if hps.wandb:
          wandb.log(scalar_dict)
          image_dict_wandb = {k:wandb.Image(v, caption=k) for k,v in image_dict.items()}
          wandb.log(image_dict_wandb)

      if global_step % hps.train.eval_interval == 0:
        for val_type, eval_loader in list(zip(['both', 'audio', 'subject'], loaders[1])):
          evaluate(hps, [net_g, eeg_module], eval_loader, writer_eval, val_type)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
        utils.save_checkpoint(eeg_module, optim_eeg, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "E_{}.pth".format(global_step)))

    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, net_g, eval_loader, writer_eval, val_type):
    generator, eeg_module = net_g
    generator.eval()
    eeg_module.eval()

    with torch.no_grad():
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)

        # remove else
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        break

      x, x_mask_output, mid_output, eeg_decoder_out = eeg_module(x)
      mid_output_lengths = x_lengths.clone() * mid_output.size(2) / x.size(2)
      mid_output_lengths = mid_output_lengths.long()

      y_hat, _, mask, *_ = generator.module.infer(mid_output, mid_output_lengths, max_len=1000)

      y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

      mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
    image_dict = {
      f"{val_type}/gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
      f"{val_type}/audio": y_hat[0,:,:y_hat_lengths[0]].cpu().float().numpy()
    }

    if global_step == 0:
      image_dict.update({f"{val_type}/gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({f"{val_type}/gt/audio": y[0,:,:y_lengths[0]].cpu().float().numpy()})


    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )

    if hps.wandb:
      image_dict_wandb = {k:wandb.Image(v, caption=k) for k,v in image_dict.items()}
      audio_dict_wandb = {k:wandb.Audio(v[0], caption=k, sample_rate=hps.data.sampling_rate) for k,v in audio_dict.items()}
      wandb.log(image_dict_wandb)
      wandb.log(audio_dict_wandb)


    generator.train()
    eeg_module.train()

                           
if __name__ == "__main__":
  main()
