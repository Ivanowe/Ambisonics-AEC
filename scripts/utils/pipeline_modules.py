import torch
import torch.nn.functional as F

from utils.stft import STFT


class NetFeeder(object):
    def __init__(self, device, win_size=320, hop_size=160):
        self.eps = torch.finfo(torch.float32).eps
        # Return STFT object to device
        self.stft = STFT(win_size, hop_size).to(device)

    # Now supports multi-channel input and output
    def __call__(self, mix, sph):
        # Initialize lists to store the "features" (stft of mix) and 
        # "labels" (stft of sph)
        feat_list = []
        lbl_list = []
        # Iterate for each channel in the input mixture and target speech each
        # TODO: Look into parallelizing these loops eventually
        
        for i in range(mix.shape[1]):       
            real_mix, imag_mix = self.stft.stft(mix[:, i, :])
            feat = torch.cat([real_mix, imag_mix], dim=0)
            feat_list.append(feat)

        for i in range(sph.shape[1]):
            real_sph, imag_sph = self.stft.stft(sph[:, i, :])
            lbl = torch.cat([real_sph, imag_sph], dim=0)
            lbl_list.append(lbl)

        feat = torch.cat(feat_list, dim=0) # Interleave real and imaginary parts  
        feat = feat.unsqueeze(0) 
        lbl = torch.cat(lbl_list, dim=0) 
        lbl = lbl.unsqueeze(0)  

        return feat, lbl


class Resynthesizer(object):
    def __init__(self, device, win_size=320, hop_size=160):
        # Return STFT object to device
        self.stft = STFT(win_size, hop_size).to(device)

    # Create audio samples from estimated spectrum
    # Multichannel support implemented for future-proofing
    def __call__(self, est, mix):
        sph_est_list = []
        for i in range(0, est.shape[1], 2):
            est_i = est[:, i:i+2, :, :]
            sph_est = self.stft.istft(est_i)
            #sph_est = F.pad(sph_est, [0, mix.shape[2]-sph_est.shape[1]])
            sph_est_list.append(sph_est)

        sph_est = torch.stack(sph_est_list, dim=1)
        return sph_est
