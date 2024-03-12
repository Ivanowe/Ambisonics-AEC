import torch
import numpy as np

class LossFunction(object):
    #old version kept for reference
    # def __call__(self, est, lbl, loss_mask, n_frames):
    #     est *= loss_mask
    #     lbl *= loss_mask

    #     n_feats = est.shape[-1]

    #     loss = torch.sum((est - lbl)**2) / float(sum(n_frames) * n_feats * 2)
        
    #     return loss
    
    def __call__(self, est, lbl, loss_mask, n_out_channels, n_frames):
        est *= loss_mask
        lbl *= loss_mask
        
        n_feats = est.shape[-1]
      
        # output channels are interleaved real and imaginary parts of the 
        # spectrum. The real part is therefore at even indeces and the imaginary 
        # part is at the odd indeces.
        est_real = est[:, ::2, :, :]
        est_imag = est[:, 1::2, :, :]
        lbl_real = lbl[:, ::2, :, :]
        lbl_imag = lbl[:, 1::2, :, :]
        
        loss = (0.5 * torch.sum(torch.abs(est_real)  
                                - torch.abs(lbl_real))**2 
                + 0.5 * torch.sum((torch.abs(est_imag) 
                                   - torch.abs(lbl_imag))**2))
        
        loss = loss / float(sum(n_frames) * n_feats * 2)
            
        return loss

