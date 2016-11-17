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
results_path = study_path + 'results/'
figures_path = study_path + 'figures/'
ch_info_path = study_path + 'info/'


scalings = {'eeg': 150e-6}

reject = {'eeg': 150e6}

epoch_info = {'tmin': 0, 'tmax': 0.5, 'reject': reject}

