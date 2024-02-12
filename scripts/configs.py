exp_conf = {
    'in_norm': False, # normalize the input audio
    'sample_rate': 16000,
    'win_len': 0.020, # window length (sec)
    'hop_len': 0.010,  # window shift (sec)
}
amb_aec_conf = {
    'in_norm': False, # normalize the input audio
    'sample_rate': 16000,
    'win_len': 0.020, # window length (sec)
    # shorter hop -> 75% overlap
    'hop_len': 0.005,  # window shift (sec)
    'num_channels': 5 # number of input channels
}
