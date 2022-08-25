import arabic_reshaper
import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from bidi.algorithm import get_display
from nptdms import TdmsFile
from sklearn import preprocessing
import pandas as pd


matplotlib.use('Qt5Agg')

tdms_file = TdmsFile.read("/run/media/****/Desktop/edfs/AmirhoseinJ1_Filtered.tdms")
# tdms_file = TdmsFile.read("sample2_kh_Filtered.tdms")

# tdms_file.groups()[0].channels()[5].properties
times = tdms_file.groups()[0].channels()[5].time_track()

channels_dict = {}
for channel in tdms_file.groups()[0].channels():
    # if channel[:].astype("int").all() != 0:
    channels_dict[channel.name] = channel[:]

all_array_list = []
for array in channels_dict.values():
    all_array_list.append(array * 10e-9)

fs = 250.0
info = mne.create_info(ch_names=list(channels_dict.keys()), ch_types="eeg", sfreq=fs)
raw = mne.io.RawArray(all_array_list, info=info)

# Define EEG bands
eeg_bands = {'Delta': (0, 4),
             'Theta': (4, 8),
             'Alpha': (8, 12),
             'Beta': (12, 30),
             'Gamma': (30, 45)}

# raw.filter(0,45)

ffts_channels = []
ffts_freqs = []
for i, channel in enumerate(raw):
    ffts_channels.append((np.fft.rfft(preprocessing.scale(raw[i][0][0][10 * 250:40 * 250]))))
    ffts_freqs.append(np.fft.rfftfreq(len(preprocessing.scale(raw[i][0][0][10 * 250:40 * 250])), 1 / fs))
    if i == 12:
        break

bands_all = []
for fft_data, fft_freqs in zip(ffts_channels, ffts_freqs):
    waves_separated = {}
    for band in eeg_bands:
        freq_ix_band = np.where((fft_freqs >= eeg_bands[band][0]) &
                                (fft_freqs <= eeg_bands[band][1]))[0]
        filter_band = np.zeros(len(fft_data))
        filter_band[freq_ix_band] = 1
        fft_band = fft_data * filter_band
        waves_separated[band] = preprocessing.scale(np.fft.irfft(fft_band))
    bands_all.append(waves_separated)


# Plot each band in time domain
def bands_plot(bands_dict, begin=10 * 250, truncate=20 * 250, fs=fs, chan=2):
    fig, axs = plt.subplots(6,figsize=(16, 9))

    fig.tight_layout()

    times = np.arange(0, len(list(bands_dict.values())[0])) / fs

    # fig.suptitle('EEG Frequency Ranges')

    axs[0].plot(times[begin:truncate], preprocessing.scale(raw[chan][0][0][1 * begin:truncate]))

    axs[0].set_title('Raw Channel (' + str(list(channels_dict.keys())[chan]) + ")")

    # axs[0].set_ylim(-7.5*1e-7,7.5*1e-7)

    for i, band_wave in enumerate(bands_dict.values()):
        axs[i + 1].plot(times[begin:20 * 250], band_wave[begin:20 * 250])
        axs[i + 1].set_title(list(eeg_bands.keys())[i])

    # max_amp_waves_separated = np.max(list(bands_dict.values()))

    for i in range(4):
        # axs[i+2].set_ylim([-max_amp_waves_separated/5000, max_amp_waves_separated/5000])
        axs[i].set_ylim([-4, 4])
    axs[4].set_ylim([-4, 4])
    axs[5].set_ylim([-4, 4])
    # axs[0].set_facecolor('#fceab1')
    fig.savefig('toGIT/SVG_figures/4-bands_time_domain.svg')
    plt.show()

    plt.xlabel(get_display(arabic_reshaper.reshape("زمان (ثانیه)")), font="B Nazanin", fontsize=16, position=(1, 25),
               horizontalalignment='right')
    axs[0].set_ylabel(get_display(arabic_reshaper.reshape("اندازه نرمال شده (واریانس واحد)")), font="B Nazanin",
                      fontsize=14, position=(25, 1), verticalalignment="center",
                      horizontalalignment='right', labelpad=10)


# matplotlib.rcParams['figure.figsize'] = (10.2, 5.8)

bands_plot(bands_all[2])

# with help from https://stackoverflow.com/questions/63995409/wrong-values-calculating-fft-with-eeg-bands-using-numpy
# Take the mean of the fft amplitude for each EEG band
eeg_bands_mean = dict()
for band in eeg_bands:
    freq_ix = np.where((fft_freqs >= eeg_bands[band][0]) &
                       (fft_freqs <= eeg_bands[band][1]))[0]
    eeg_bands_mean[band] = np.abs(np.mean(fft_data[freq_ix]))

# Plot the mean amplitude of each band
df = pd.DataFrame(columns=['band', 'val'])
df['band'] = list(eeg_bands.keys())[0:]
df['val'] = [eeg_bands_mean[band] for band in list(eeg_bands)[0:]]
ax = df.plot.bar(x='band', y='val', legend=False)
ax.set_xlabel("EEG band")
ax.set_ylabel("Mean band Amplitude")
plt.show()
plt.figure(figsize=(16, 9))
plt.savefig('toGIT/SVG_figures/4-bands_barplot.svg')