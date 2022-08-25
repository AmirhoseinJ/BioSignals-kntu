import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import mne
import numpy as np
import scipy.signal
from nptdms import TdmsFile

matplotlib.use('Qt5Agg')

tdms_file = TdmsFile.read("AmirhoseinJ1_Filtered.tdms")
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

chnames_raw = raw.info["ch_names"]

chnames_montage = [
    "Fp1-F5",
    "F5-Fc5",
    "Fc5-C5",
    "Fp2-F6",
    "F6-Fc6",
    "Fp1-F3",
    "Fp2-F4"
]

montaged_channels_arrays = []
for ch_montage in chnames_montage:
    two_channels = ch_montage.split("-")
    current_montage = raw[two_channels[1]][0][0] - raw[two_channels[0]][0][0]
    montaged_channels_arrays.append(current_montage)

raw_montaged = mne.io.RawArray(np.asarray(montaged_channels_arrays),
                               info=mne.create_info(ch_names=chnames_montage, ch_types="eeg", sfreq=250))

left_chains_gaze = raw_montaged.copy().filter(1, 30).pick(["Fp1-F5", "F5-Fc5"])

butterworth_N = 5
butterworth_wn = 0.04
numerator, denumerator = scipy.signal.butter(butterworth_N, butterworth_wn)

farsi_time = get_display(arabic_reshaper.reshape("زمان (ثانیه)"))
farsi_amp = get_display(arabic_reshaper.reshape("اندازه (ولت)"))
farsi_gaze_left = get_display(arabic_reshaper.reshape("حرکت چشم به چپ"))
farsi_gaze_right = get_display(arabic_reshaper.reshape("حرکت چشم به راست"))

range = (100, 110)
with plt.style.context("seaborn"):
    with plt.rc_context(
            {'lines.linewidth': 1.2, 'lines.linestyle': '-', 'font.family': ['Roboto'], 'legend.frameon': True}):
        font = matplotlib.font_manager.FontProperties(family='B Nazanin',
                                                      weight='bold',
                                                      style='normal', size=10)
        fig, axs = plt.subplots(2, squeeze=False, figsize=(16, 9))
        fig.suptitle(get_display(arabic_reshaper.reshape("مقایسه دو زنجیره نیم‌کره چپ در حرکت چشم به سمت چپ")),
                     fontfamily="B Nazanin", fontsize=22, color="#2d1754", x=0.511, y=0.99)

        # fig.tight_layout()

        # First Channel
        filtered = scipy.signal.filtfilt(numerator, denumerator,
                                         left_chains_gaze[0][0][0][range[0] * 250:range[1] * 250], method="gust")
        times = np.arange(range[0], range[1], 1 / fs)

        axs.flat[0].plot(times, filtered, color='black')

        axs.flat[0].set_title("Fp1-F5", fontsize=18, fontweight="bold", color="#414242", pad=20)

        axs.flat[0].set_xlabel(farsi_time, fontfamily="B Nazanin", fontsize=14, labelpad=15, color="#056385")
        axs.flat[0].set_ylabel(farsi_amp, fontfamily="B Nazanin", fontsize=14, labelpad=10, color="#056385")

        # axs.flat[0].set_ylim([-2.5*1e-6,2.5*1e-6])

        axs.flat[0].axvspan(104.53, 105.03, color='#b2d9d5', alpha=0.4, label=farsi_gaze_left)
        # axs.flat[0].axvspan(31.9, 32.6, color='#b2d9d5', alpha=0.4)
        # axs.flat[0].axvspan(29.4, 30.3, color='#b2d9d5', alpha=0.4)
        # axs.flat[0].axvspan(33.02, 33.62, color='#b2d9d5', alpha=0.4)
        axs.flat[0].legend(prop=font, facecolor='white', framealpha=1)
        axs.flat[0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
        # axs.flat[0].annotate('local max', xy=(33,1.5*1e-5),  xycoords='data',
        #     xytext=(0.8, 0.95), textcoords='axes fraction',arrowprops=dict(facecolor='black', shrink=0.05),
        #     horizontalalignment='right', verticalalignment='top',)

        axs.flat[0].annotate('KNTU BSP Lab: Author, 24M', xy=(range[0], 0), xytext=(1, 0), xycoords='data',
                             textcoords='axes fraction',
                             horizontalalignment='right', verticalalignment='bottom', fontsize=8)

        # Second Channel
        filtered = scipy.signal.filtfilt(numerator, denumerator,
                                         left_chains_gaze[1][0][0][range[0] * 250:range[1] * 250], method="gust")

        axs.flat[1].plot(times, filtered, color='black')

        axs.flat[1].set_title("F5-Fc5", fontsize=18, fontweight="bold", color="#414242", pad=20)

        axs.flat[1].set_xlabel(farsi_time, fontfamily="B Nazanin", fontsize=14, labelpad=15, color="#056385")
        axs.flat[1].set_ylabel(farsi_amp, fontfamily="B Nazanin", fontsize=14, labelpad=10, color="#056385")

        # axs.flat[1].set_ylim([-2.5 * 1e-6, 2.5 * 1e-6])

        axs.flat[1].axvspan(104.45, 105.08, color='#b2d9d5', alpha=0.4, label=farsi_gaze_left)
        # axs.flat[1].axvspan(33.12, 33.63, color='#b2d9d5', alpha=0.4)
        # axs.flat[1].axvspan(29.69, 30.12, color='#b2d9d5', alpha=0.4)
        # axs.flat[1].axvspan(25.98, 26.37, color='#b2d9d5', alpha=0.4)
        axs.flat[1].legend(prop=font, facecolor='white', framealpha=1)
        axs.flat[1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))

        axs.flat[1].annotate('KNTU BSP Lab: Author, 24M', xy=(range[0], 0), xytext=(1, 0), xycoords='data',
                             textcoords='axes fraction',
                             horizontalalignment='right', verticalalignment='bottom', fontsize=8)

        fig.tight_layout()
        # https://stackoverflow.com/questions/26084231/draw-a-separator-or-lines-between-subplots
        # Get the bounding boxes of the axes including text decorations
        r = fig.canvas.get_renderer()
        get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
        bboxes = np.array(list(map(get_bbox, axs.flat)), mtrans.Bbox).reshape(axs.shape)

        # Get the minimum and maximum extent, get the coordinate half-way between those
        ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(axs.shape).max(axis=1)
        ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(axs.shape).min(axis=1)
        ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)

        # Draw a horizontal lines at those coordinates
        for y in ys:
            line = plt.Line2D([0, 1], [y, y], transform=fig.transFigure, color="black", linestyle="dotted")
            fig.add_artist(line)


fig.savefig('toGIT/SVG_figures/4-gaze_left.svg')