import time
import os
import random
import numpy as np
import torch
import torch.utils.data

import commons 
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_eeg, get_hparams

EEG_CHANNEL_LIST = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31', 'B32', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status']
EEG_CHANNEL_DICT = {k:i for i,k in enumerate(EEG_CHANNEL_LIST)}


class EEGAudioLoader(torch.utils.data.Dataset):
    """
        loads EEG and audio pairs
    """
    def __init__(self, audiopaths_and_eeg, hparams):
        self.audiopaths_and_eeg = load_filepaths_and_eeg(audiopaths_and_eeg)
        self.data_root_dir = hparams.data_root_dir
        self.max_wav_value  = hparams.max_wav_value
        self.sampling_rate  = hparams.sampling_rate
        self.filter_length  = hparams.filter_length 
        self.hop_length     = hparams.hop_length 
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate 

        try:
            self.eeg_channels_select = hparams.eeg_channels
            self.eeg_channels_select = [EEG_CHANNEL_DICT[ch] for ch in self.eeg_channels_select]
            print(f'{len(self.eeg_channels_select)} channels selected:{hparams.eeg_channels}')
        except:
            self.eeg_channels_select = 'all'

        try:
            self.dataset_type = hparams.dataset_type
        except:
            self.dataset_type = 'n400'


        random.seed(1234)
        random.shuffle(self.audiopaths_and_eeg)

        self._get_audio_length()



    def _get_audio_length(self):

        lengths = []
        for audiopath in self.audiopaths_and_eeg:
            audiopath = audiopath.split('||')[0]
            lengths.append(os.path.getsize(os.path.join(self.data_root_dir,'2022N400_Jan_cp', 'stimuli_22k', audiopath.split('-_-')[-1])+'_22k.wav') // (2 * self.hop_length))

        self.lengths = lengths

    def get_audio_eeg_pair(self, audiopath_and_eeg):
        datapath, text = audiopath_and_eeg.split('||')
        audiopath = os.path.join(self.data_root_dir,'2022N400_Jan_cp', 'stimuli_22k', datapath.split('-_-')[-1])+'_22k.wav'
        eeg_path = os.path.join(self.data_root_dir, '2022N400_Epoched', datapath)+'.npy'
        
        eeg_data = self.get_eeg(eeg_path)
        spec, wav = self.get_audio(audiopath)
        return (eeg_data, spec, wav, text)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(audio_norm, self.filter_length,
            self.sampling_rate, self.hop_length, self.win_length,
            center=False)
        spec = torch.squeeze(spec, 0)
        return spec, audio_norm
    
    def get_eeg(self, filename):
        eeg = np.load(filename)
        eeg = torch.FloatTensor(eeg) # T x Ch

        if self.eeg_channels_select == 'all':
            eeg = eeg[:128]
        else:
            eeg = eeg[self.eeg_channels_select]
        eeg = torch.nn.functional.normalize(eeg, p=2, dim=1) # time-wise normalization

        return eeg


    def __getitem__(self, index):
        return self.get_audio_eeg_pair(self.audiopaths_and_eeg[index])

    def __len__(self):
        return len(self.audiopaths_and_eeg)


class EEGAudioCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized eeg and audio
        PARAMS
        ------
        batch: [eeg, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), # eeg, spec, wav
            dim=0, descending=True)

        max_eeg_len = max([x[0].size(1) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])


        eeg_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))


        eeg_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_eeg_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

        eeg_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            eeg = row[0]
            eeg_padded[i,:,:eeg.size(1)] = eeg
            eeg_lengths[i] = eeg.size(1)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

        if self.return_ids:
            return eeg_padded, eeg_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, ids_sorted_decreasing
        return eeg_padded, eeg_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths





if __name__=='__main__':
    hps = get_hparams()
    temp_dataset = EEGAudioLoader(hps.data.validation_files_subject, hps.data)
