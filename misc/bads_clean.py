# np.set_printoptions(threshold=sys.maxsize)
import matplotlib
import matplotlib.pyplot as plt
import mne

matplotlib.get_backend()
matplotlib.rcsetup.all_backends
matplotlib.use('Qt5Agg')
matplotlib.get_backend()
import autoreject

# load raw
raw = mne.io.read_raw_edf(r"/run/media/****/prj/phase2/eog artifact articles/00008512_s009_t001.edf", preload=True)
print(raw.info['ch_names'][20:])
print(raw.info)
# raw.plot()

# add annotations to raw
import pandas as pd

pandatse = pd.read_csv("/run/media/****/project/edf/dev/01_tcp_ar/085/00008512/s009_2012_07_04/00008512_s009_t001.tse",
                       delim_whitespace=True,
                       index_col=False)
pandatse.rename(columns={'version': 'start', '=': 'end', 'tse_v1.0.0': 'event'}, inplace=True)
onset = []
duration = []
description = []
for i in range(pandatse.shape[0]):
    onset.append(pandatse.iloc[i]['start'])
    duration.append(pandatse.iloc[i]['end'] - pandatse.iloc[i]['start'])
    description.append(pandatse.iloc[i]['event'])
annot = mne.Annotations(onset=onset,  # in seconds
                        duration=duration,  # in seconds, too
                        description=description)
print(annot)
raw.set_annotations(annot)

# Change channel names to correspond with 'standard_1020' montage, and append that montage to loaded raw
mont_1020 = mne.channels.make_standard_montage('standard_1020')
print(raw.info['ch_names'])
kept_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'T1',
                 'T2', 'Fz', 'Cz', 'Pz', 'A1', 'A2']
ind = [i for (i, channel) in enumerate(mont_1020.ch_names) if channel in kept_channels]
kept_channel_digs = [mont_1020.dig[x + 3] for x in ind]
mont_1020_new = mont_1020.copy()
mont_1020_new.ch_names = [mont_1020.ch_names[x] for x in ind]
mont_1020_new.dig = mont_1020.dig[0:3] + kept_channel_digs
print(len(kept_channels))
print(raw.info['nchan'])
mappingdict = {'EEG F4-REF': 'F4', 'EEG F3-REF': 'F3', 'EEG FP1-REF': 'Fp1', 'EEG FP2-REF': 'Fp2', 'EEG C3-REF': 'C3',
               'EEG C4-REF': 'C4',
               'EEG P3-REF': 'P3', 'EEG P4-REF': 'P4', 'EEG O1-REF': 'O1', 'EEG O2-REF': 'O2', 'EEG A1-REF': 'A1',
               'EEG F7-REF': 'F7', 'EEG F8-REF': 'F8', 'EEG T3-REF': 'T3', 'EEG T4-REF': 'T4', 'EEG T5-REF': 'T5',
               'EEG T6-REF': 'T6', 'EEG T1-REF': 'T1', 'EEG T2-REF': 'T2', 'EEG FZ-REF': 'Fz', 'EEG CZ-REF': 'Cz',
               'EEG PZ-REF': 'Pz', 'EEG A2-REF': 'A2'}
raw.rename_channels(mappingdict)
print(raw.info['nchan'])
raw.drop_channels(
    ['EEG EKG1-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', 'EEG 31-REF', 'EEG 32-REF', 'IBI',
     'BURSTS', 'SUPPR'])
raw.drop_channels(['T1', 'T2', 'PHOTIC-REF'])
raw.set_montage(mont_1020_new)
# check new raw
print(raw.info['ch_names'])
montage = raw.get_montage()
print(raw.info['nchan'])
montage.plot()

##########################
# autoreject
raw.filter(l_freq=1, h_freq=None)  # hpf
epochs = mne.make_fixed_length_epochs(raw, duration=10, preload=True)
ar = autoreject.AutoReject(n_interpolate=[1, 2, 21], random_state=24123, n_jobs=8, verbose=True)
ar.fit(epochs)  # fit on a few epochs to save time
epochs_ar, reject_log = ar.transform(epochs, return_log=True)

################### PLOTS #########
# all original epochs
epochs.plot(scalings=dict(eeg=100e-6))
epochs.average().detrend().plot_joint()
# rejected epochs *
epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))
epochs[[i for i, isbad in enumerate(reject_log.bad_epochs) if isbad == True]].average().plot_joint()
# original-rejected *
epochs_ar.plot(scalings=dict(eeg=100e-6))
epochs_ar.average().detrend().plot_joint()
# reject box
reject_log.plot('horizontal')
# evoked
evoked_bad = epochs[reject_log.bad_epochs].average()
plt.figure()
plt.plot(evoked_bad.times, evoked_bad.data.T * 1e6, 'r', zorder=-1)
epochs_ar.average().plot(axes=plt.gca())
####################################

# ica (????)
ica = mne.preprocessing.ICA(random_state=13412)
ica.fit(epochs[~reject_log.bad_epochs])
ica.plot_components()
ica.plot_sources(raw)
ica.exclude = [1]
ica.plot_overlay(epochs_ar.average(), exclude=ica.exclude)
epochs_applied = epochs.copy()
ica.apply(epochs_applied, exclude=ica.exclude)
