import torch
import numpy as np



class LossFunction(object):
    # old version kept for reference
    # def __call__(self, est, lbl, loss_mask, n_frames):
    #     est *= loss_mask
    #     lbl *= loss_mask

    #     n_feats = est.shape[-1]

    #     loss = torch.sum((est - lbl)**2) / float(sum(n_frames) * n_feats * 2)
        
    #     return loss
    
     def __call__(self, est, lbl, loss_mask, n_out_channels):
        est *= loss_mask
        lbl *= loss_mask
        
        # output channels are interleaved real and imaginary parts of the 
        # spectrum. The real part is therefore at even indeces and the imaginary 
        # part is at the odd indeces.
        est_real = torch.stack([est[:, i :, :] for i in range(0, n_out_channels, 2)], dim=1)
        est_imag = torch.stack([est[:, i :, :] for i in range(1, n_out_channels, 2)], dim=1)
        
        lbl_real = torch.stack([lbl[:, i :, :] for i in range(0, n_out_channels, 2)], dim=1)
        lbl_imag = torch.stack([lbl[:, i :, :] for i in range(1, n_out_channels, 2)], dim=1)
        
        loss = (0.5 * torch.sum(np.absolute(est_real) - np.absolute(lbl_real))**2 
                + 0.5 * torch.sum((np.absolute(est_imag) - np.absolute(lbl_imag))**2))
 
        
        
        return loss

