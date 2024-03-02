import torch
import torch.nn as nn
import torch.nn.functional as F

class STFT(nn.Module):
    def __init__(self, win_size=320, hop_size=160):
        super(STFT, self).__init__()

        self.win_size = win_size
        self.hop_size = hop_size

        # Create a Hann window
        self.win = torch.hann_window(self.win_size, 
                                     periodic=True, 
                                     dtype=torch.float32)

    def stft(self, sig):
        # Use PyTorch's built-in function to compute the STFT
        spec = torch.stft(sig, n_fft=self.win_size, 
                          hop_length=self.hop_size, 
                          window=self.win, 
                          return_complex=True)

        # Separate the real and imaginary parts, then change the order of 
        # the dimensions
        spec_r = spec.real.transpose(-1, -2).contiguous()
        spec_i = spec.imag.transpose(-1, -2).contiguous()

        return spec_r, spec_i

    def istft(self, est):
       
        # Rearrange the dimensions of the tensor to what pytorch expects
        est = est.permute(0, 3, 2, 1)

        # Use PyTorch's built-in function to compute the inverse STFT
        sig = torch.istft(est, 
                          n_fft=self.win_size, 
                          hop_length=self.hop_size, 
                          window=self.win,
                          onesided=True) # Because only positive 
                                         # frequencies are used

        return sig