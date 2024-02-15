# Configs as dictionaries
exp_conf = {
    'in_norm': False, # normalize the input audio
    'sample_rate': 16000,
    'win_len': 0.020, # window length (sec)
    'hop_len': 0.010,  # window shift (sec)
}
amb_aec_conf = {
    # normalize the input audio?
    'in_norm': False,
    'sample_rate': 16000,
    # window length (sec)
    'win_len': 0.020, 
    # window shift (sec)
    # shorter hop -> 75% overlap
    'hop_len': 0.005,  
    # number of input channels
    'n_in_channels': 5, 
    # number of output channels, future-proofing for multi-channel output
    'n_out_channels': 1 
}
