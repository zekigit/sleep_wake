subjects = ('P8', 'P9', 'P14', 'P15', 'P5')

IDs = ('s1', 's2', 's3', 's4', 's5')

conditions = ['wake', 'sleep']

conds_x_subj = {'s1': ('wake', 'sleep'),
                's2': ('wake', 'sleep'),  # also 'transition'
                's3': ('wake', 'sleep'),
                's4': ('wake', 'sleep'),
                's5': ('wake', 'sleep')}


study_path = '/Users/lpen/Documents/wake_sleep/study'
data_path = study_path + 'data/'

scalings = {'eeg': 150e-6}

reject = {'eeg': 150e6}

epoch_info = {'tmin': 0, 'tmax': 0.5, 'reject': reject}

refs = ('bip', 'avg')

lengths_str = ['250ms', '500ms', '1s', '2s', '4s', '8s', '16s']

lengths_nr = [0.25, 0.5, 1, 2, 4, 8, 16]

bad_channels_scalp = {'s5': dict(wake=[1, 9, 30, 32, 35, 45, 46, 54, 58, 176, 199, 218, 219], sleep=[31, 37, 45, 94, 103, 111, 176, 199, 218, 219])}