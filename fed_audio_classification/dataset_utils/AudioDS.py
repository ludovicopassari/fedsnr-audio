from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
import pandas as pd
import torch
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from logger_config import get_logger

logger = get_logger(__name__)

class AudioDS(Dataset):
    def __init__(self, data_path, folds, sample_rate,  partition_id, metadata_filename , max_duration=4,training=False, aug=True, num_aug=1, aug_prob = 0.1):
        self._data_path = Path(data_path)
        self._folds = folds
        self.partition_id = partition_id
        self.metadata_filename = metadata_filename
        self._sample_rate = sample_rate
        self._max_duration = max_duration
        self._train = training
        self._target_len = sample_rate * max_duration
        self._metadata = self._load_metadata()
        self._augmentation = aug
        self._max_augmentation= num_aug
        self.augumentation_prob = aug_prob

        #Feature extraction utilizzando spettrogrammi Mel
        n_fft = 2048
        win_length = 512
        hop_length = 256
        n_mels = 160
        
        self.mel_transform= torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="htk",
            f_max= self._sample_rate /2,
            f_min=0
            
        )
        #Augmentation spettrogramma Mel
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=6)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=int(0.15 * n_mels))



    def __len__(self):
        return len(self._metadata)
    
    def __getitem__(self,idx):
        row = self._metadata.iloc[idx]
        label = row['classID']

        #Costruisce il path
        audio_full_path = self._data_path / f"fold{row['fold']}" / row['slice_file_name']

        #carico l'audio dal disco in memoria
        raw_waveform, raw_sample_rate = self._load_sample(audio_full_path)
        waveform = raw_waveform
        

        #ricampiona il segnale audio ad un sample rate target
        if raw_sample_rate != self._sample_rate:
            waveform = self._resample_waveform(waveform=raw_waveform, orig_freq=raw_sample_rate, new_freq= self._sample_rate)

        if self._target_len != waveform.shape[1]:
            waveform = self._fix_lenght(waveform)
    
        if self._train and self._augmentation and torch.rand(1).item() < 0.4:
            waveform = self._apply_balanced_augmentations(waveform=waveform)
 

        waveform = self._normalize_with_soft_clipping(waveform=waveform)

        spectrograms = []
        for channel in range(waveform.shape[0]):
            channel_waveform = waveform[channel:channel+1]  # Mantieni la dimensione batch
            raw_spectrogram = self.mel_transform(channel_waveform)
            spec = librosa.power_to_db(raw_spectrogram.squeeze().numpy())

            #spec = 10.0 * torch.log10(raw_spectrogram + 1e-10)  # evita .numpy() #per utilizzare solo torch

            # Normalizzazione per canale
            spec_mean = spec.mean()
            spec_std = spec.std()
            spec = (spec - spec_mean) / (spec_std + 1e-8)
            
            spectrograms.append(torch.tensor(spec, dtype=torch.float32))
        
        # Stack dei canali: forma finale [channels, n_mels, time_steps]
        spec = torch.stack(spectrograms)
        spec = torch.nan_to_num(spec, nan=0.0, posinf=1.0, neginf=0.0)

        #applica augmentation allo spettrogramma per ogni canale
        if self._augmentation and self._train and torch.rand(1).item() < 0.4:
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)
            
        return spec, label 
    
    def _apply_balanced_augmentations(self, waveform):

        available_augs = ['time_stretch', 'pitch_shift', 'time_shift', 'noise']
        

        # Seleziona casualmente il numero di augmentations da applicare
        num_augs = torch.randint(1, self._max_augmentation + 1, (1,)).item()
        selected_augs = torch.randperm(len(available_augs))[:num_augs].tolist()
        
        # Applica le augmentations nell'ordine ottimale
        augmentation_order = ['time_stretch', 'pitch_shift', 'time_shift', 'noise']
        
        for aug_name in augmentation_order:
            if available_augs.index(aug_name) in selected_augs:
                if aug_name == 'time_stretch':
                    rate = float(torch.empty(1).uniform_(0.8, 1.2).item())
                    waveform = self._apply_time_stretch_stereo(waveform, rate)
                elif aug_name == 'pitch_shift':
                    waveform = self._apply_pitch_shift_stereo(waveform)
                elif aug_name == 'time_shift':
                    max_shift = int(0.2 * self._target_len)
                    waveform = self.random_shift(waveform, max_shift)
                elif aug_name == 'noise':
                    snr_db = torch.randint(low=15, high=30, size=(1,)).item()
                    waveform = self._add_noise_gaussian(waveform, snr_db)
        
        return waveform
    
    def _normalize_with_soft_clipping(self, waveform):
        #Normalizza con soft clipping per evitare saturazione
        normalized_channels = []
        
        for channel in range(waveform.shape[0]):
            channel_data = waveform[channel]
            
            # Calcola RMS per normalizzazione più stabile
            rms = torch.sqrt(torch.mean(channel_data**2))
            if rms > 0:
                # Normalizza per RMS invece che per peak
                channel_data = channel_data / (rms * 3.0)  # Factor per headroom
            
            # Soft clipping usando tanh per evitare distorsioni dure
            channel_data = torch.tanh(channel_data)
            
            normalized_channels.append(channel_data)
        
        return torch.stack(normalized_channels) 
    
    def _apply_time_stretch_stereo(self, waveform, rate):
        #Applica time stretch a audio stereo
        stretched_channels = []
        for channel in range(waveform.shape[0]):
            channel_np = waveform[channel].numpy()
            stretched_np = librosa.effects.time_stretch(channel_np, rate=rate)
            stretched_channels.append(torch.tensor(stretched_np))
        
        result = torch.stack(stretched_channels)
        return self._fix_lenght(result)
    
    def _apply_pitch_shift_stereo(self, waveform):
        """Applica pitch shift a audio stereo"""
        shifted_channels = []
        n_steps = int(torch.randint(-2, 3, (1,)).item())
        
        for channel in range(waveform.shape[0]):
            channel_np = waveform[channel].numpy()
            shifted_np = librosa.effects.pitch_shift(channel_np, sr=self._sample_rate, n_steps=n_steps)
            shifted_channels.append(torch.tensor(shifted_np))
        
        return torch.stack(shifted_channels)
    

    """ def random_shift(self, waveform, max_shift):
        
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if shift == 0:
            return waveform  # niente shift
        
        channels, samples = waveform.shape

        if shift > 0:
            # shift a destra
            pad = torch.zeros((channels, shift), device=waveform.device)
            waveform = torch.cat([pad, waveform[:, :-shift]], dim=1)
        else:  # shift < 0
            # shift a sinistra
            pad = torch.zeros((channels, -shift), device=waveform.device)
            waveform = torch.cat([waveform[:, -shift:], pad], dim=1)
        
        return waveform """
    
    def random_shift(self, waveform, max_shift):
            """Applica uno shift temporale casuale al waveform stereo con padding circolare."""
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            if shift == 0:
                return waveform  # niente shift

            # torch.roll applica automaticamente il wrap-around sui campioni
            waveform = torch.roll(waveform, shifts=shift, dims=1)
            
            return waveform
        
    def _add_noise_gaussian(self, speech, snr_db):
        """Aggiunge rumore gaussiano a ogni canale separatamente"""
        noisy_channels = []
        
        for channel in range(speech.shape[0]):
            channel_speech = speech[channel:channel+1]  # Mantieni dimensione
            
            # Generiamo rumore casuale
            noise = torch.randn_like(channel_speech)

            # Calcoliamo potenza segnale e rumore
            power_speech = channel_speech.pow(2).mean()
            power_noise = noise.pow(2).mean()

            # Calcoliamo il fattore di scala del rumore per ottenere SNR desiderato
            snr = 10 ** (snr_db / 10)
            scale = torch.sqrt(power_speech / (snr * power_noise))
            noise_scaled = noise * scale

            # Sommiamo rumore al segnale
            noisy_channel = channel_speech + noise_scaled
            noisy_channels.append(noisy_channel.squeeze(0))
        
        return torch.stack(noisy_channels)


    def _load_sample(self, path):
        samples, sr = torchaudio.load(path) #[channels, samples]
        
        # Se è mono, duplica il canale per renderlo stereo
        if samples.shape[0] == 1:
            samples = samples.repeat(2, 1)  # Duplica il canale mono per creare stereo
        
        # Se ha più di 2 canali, prendi solo i primi 2
        elif samples.shape[0] > 2:
            samples = samples[:2, :]
    
        return samples, sr

    def _resample_waveform(self, waveform, orig_freq, new_freq):
        """Ricampiona ogni canale separatamente"""
        resampled_channels = []
        resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq, resampling_method='sinc_interp_kaiser')
        
        for channel in range(waveform.shape[0]):
            resampled_channel = resampler(waveform[channel:channel+1])
            resampled_channels.append(resampled_channel.squeeze(0))
        
        return torch.stack(resampled_channels)
    
    def _fix_lenght(self, waveform):
        """Aggiusta la lunghezza per audio stereo"""
        num_samples = waveform.shape[1]

        if num_samples > self._target_len:
            if self._train:
                start = torch.randint(0, num_samples - self._target_len + 1, (1,)).item()
            else:
                start = (num_samples - self._target_len) // 2
            waveform = waveform[:, start:start + self._target_len]
        elif num_samples < self._target_len:
            # Pad con zeri
            pad_len = self._target_len - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            
        return waveform

    def _process_waveform(self, waveform):
        pass

    def _load_metadata(self):
        metadata_file_path = self._data_path / self.metadata_filename
        
        
        df = pd.read_csv(metadata_file_path)

        df = df[
            df['fold'].isin(self._folds) & 
            (df['partition_id'] == self.partition_id)
        ]

        

        return df