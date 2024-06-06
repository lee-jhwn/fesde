import os
import utils
import argparse
import torch
from data_utils import (
  EEGAudioLoader,
  EEGAudioCollate
)
from torch.utils.data import DataLoader
from EEGModule import EEGModule
from models import SpeechDecoder
from scipy.io.wavfile import write

torch.manual_seed(777)

OUTPUT_DIR = './logs'

def synthesize(eeg_enc, audio_gen, eval_loader, suffix, hps, args):

    eeg_enc.eval()
    audio_gen.eval()

    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(eval_loader):
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)

            x, x_mask_output, mid_output, eeg_decoder_output = eeg_enc(x)


            mid_output_lengths = x_lengths.clone() * mid_output.size(2) / x.size(2)
            mid_output_lengths = mid_output_lengths.long()

            y_hat, _, mask, *_ = audio_gen.infer(mid_output, mid_output_lengths, max_len=1000, noise_scale=1)
            
            os.makedirs(os.path.join(OUTPUT_DIR, args.run_name, 'synthesized', str(args.checkpoint_idx), suffix), exist_ok=True)

            write(os.path.join(OUTPUT_DIR, args.run_name, 'synthesized', str(args.checkpoint_idx), suffix, f'{batch_idx}_gt.wav'), hps.data.sampling_rate, y[0,:,:y_hat.size(2)][0].cpu().float().numpy())
            write(os.path.join(OUTPUT_DIR, args.run_name, 'synthesized', str(args.checkpoint_idx), suffix, f'{batch_idx}_syn.wav'), hps.data.sampling_rate, y_hat[0,:,:y_hat.size(2)][0].cpu().float().numpy())


    return


def run(args):

    config_dir = os.path.join('./logs', args.run_name, 'config.json')
    hps = utils.get_hparams_from_file(config_dir)
    print(hps)
    
    
    collate_fn = EEGAudioCollate()

    eval_loader_both = DataLoader(EEGAudioLoader(hps.data.validation_files_both, hps.data), num_workers=1, shuffle=False,
        batch_size=1, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
    eval_loader_audio = DataLoader(EEGAudioLoader(hps.data.validation_files_audio, hps.data), num_workers=1, shuffle=False,
        batch_size=1, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
    eval_loader_subject = DataLoader(EEGAudioLoader(hps.data.validation_files_subject, hps.data), num_workers=1, shuffle=False,
        batch_size=1, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
    
    eval_loaders = [eval_loader_both, eval_loader_audio, eval_loader_subject]


    eeg_module = EEGModule(
    n_layers_cnn=hps.model.eeg_module.n_layers_cnn,
    use_s4=hps.model.eeg_module.use_s4,
    n_layers_s4=hps.model.eeg_module.n_layers_s4,
    embedding_size=hps.model.inter_channels,
    is_mask=False,
    in_channels=hps.model.eeg_module.in_channels
    ).cuda()

    net_g = SpeechDecoder(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()

    utils.load_checkpoint(os.path.join('./logs', args.run_name, f"G_{args.checkpoint_idx}.pth"), net_g, None)
    utils.load_checkpoint(os.path.join('./logs', args.run_name, f"E_{args.checkpoint_idx}.pth"), eeg_module, None)


    net_g.eval()
    eeg_module.eval()

    for val_type, eval_loader in list(zip(['both', 'audio', 'subject'], eval_loaders)):
        synthesize(eeg_module, net_g, eval_loader, val_type, hps, args)

    return



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="fesde")
    parser.add_argument('--checkpoint_idx', default=0)
    args = parser.parse_args()

    run(args)
