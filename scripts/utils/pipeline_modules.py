import torch
import torch.nn.functional as F

from utils.stft import STFT


class NetFeeder(object):
    def __init__(self, device, win_size=320, hop_size=160):
        self.eps = torch.finfo(torch.float32).eps
        # Return STFT object to device
        self.stft = STFT(win_size, hop_size).to(device)

    # Feed the spectra of input mixture and speech signals to the network.
    # I should probably modify this part to enable multi-channel input.
    def __call__(self, mix, sph):
        real_mix, imag_mix = self.stft.stft(mix)
        feat = torch.stack([real_mix, imag_mix], dim=1)
        
        real_sph, imag_sph = self.stft.stft(sph)
        lbl = torch.stack([real_sph, imag_sph], dim=1)

        return feat, lbl


class Resynthesizer(object):
    def __init__(self, device, win_size=320, hop_size=160):
        # Return STFT object to device
        self.stft = STFT(win_size, hop_size).to(device)

    # Create audio samples from estimated spectrum
    def __call__(self, est, mix):
        sph_est = self.stft.istft(est)
        sph_est = F.pad(sph_est, [0, mix.shape[1]-sph_est.shape[1]])

        return sph_est
