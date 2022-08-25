import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import mne
import numpy as np
import scipy.signal

matplotlib.use('Qt5Agg')

raw = mne.io.read_raw_edf("/run/media/****/prj/phase2/eog artifact articles/00008512_s009_t001.edf", preload=True)
fs = raw.info['sfreq']  # Sampling rate

chnames_raw = raw.info["ch_names"]
chnames_montage = [
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "A1-T3",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "T4-A2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2"
]

montaged_channels_arrays = []
for ch_montage in chnames_montage:
    two_channels = ch_montage.split("-")
    current_montage = raw["EEG " + two_channels[1] + "-REF"][0][0] - raw["EEG " + two_channels[0] + "-REF"][0][0]
    montaged_channels_arrays.append(current_montage)

raw_montaged = mne.io.RawArray(np.asarray(montaged_channels_arrays),
                               info=mne.create_info(ch_names=chnames_montage, ch_types="eeg", sfreq=250))

left_chains_blink = raw_montaged.copy().filter(1, 30).pick(["FP1-F3", "F3-C3"])

butterworth_N = 5
butterworth_wn = 0.04
numerator, denumerator = scipy.signal.butter(butterworth_N, butterworth_wn)

farsi_time = get_display(arabic_reshaper.reshape("زمان (ثانیه)"))
farsi_amp = get_display(arabic_reshaper.reshape("اندازه (ولت)"))
farsi_probable_blink = get_display(arabic_reshaper.reshape("پلک احتمالی"))

range = (0, 10)
with plt.style.context("seaborn"):
    with plt.rc_context(
            {'lines.linewidth': 1.2, 'lines.linestyle': '-', 'font.family': ['Roboto'], 'legend.frameon': True}):
        font = matplotlib.font_manager.FontProperties(family='B Nazanin',
                                                      weight='bold',
                                                      style='normal', size=10)
        fig, axs = plt.subplots(2, squeeze=False, figsize=(16, 9))
        fig.suptitle(get_display(arabic_reshaper.reshape("مقایسه دو زنجیره نیم‌کره چپ در هنگام پلک زدن")),
                     fontfamily="B Nazanin", fontsize=22, color="#2d1754",x=0.521, y=1.00)
        # fig.tight_layout()
        filtered = scipy.signal.filtfilt(numerator, denumerator, left_chains_blink[0][0][0], method="gust")
        times = np.arange(0, len(filtered[range[0] * 250:range[1] * 250]), 1) / fs

        axs.flat[0].plot(times, filtered[range[0] * 250:range[1] * 250], color='black')

        axs.flat[0].set_title("Fp1-F3", fontsize=18, fontweight="bold", color="#414242", pad=20)

        axs.flat[0].set_xlabel(farsi_time, fontfamily="B Nazanin", fontsize=14, labelpad=15, color="#056385")
        axs.flat[0].set_ylabel(farsi_amp, fontfamily="B Nazanin", fontsize=14, labelpad=10, color="#056385")

        axs.flat[0].set_ylim([-2.5 * 1e-5, 2.5 * 1e-5])

        axs.flat[0].axvspan(26 - 25, 26.6 - 25, color='#b2d9d5', alpha=0.4)
        axs.flat[0].axvspan(31.9 - 25, 32.6 - 25, color='#b2d9d5', alpha=0.4)
        axs.flat[0].axvspan(29.4 - 25, 30.3 - 25, color='#b2d9d5', alpha=0.4)
        axs.flat[0].axvspan(33.02 - 25, 33.62 - 25, color='#b2d9d5', alpha=0.4, label=farsi_probable_blink)

        axs.flat[0].legend(prop=font, facecolor='white', framealpha=1)

        # def y_fmt(x, y):
        #     return '{:2.2e}'.format(x).replace('e', 'x10^')
        #
        #
        # axs.flat[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(y_fmt))
        axs.flat[0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(
            useMathText=True))  # axs.flat[0].annotate('local max', xy=(33,1.5*1e-5),  xycoords='data',
        # axs.flat[0].set_yticks(np.arange(filtered.min(), filtered.max()))

        axs.flat[0].annotate('TUH Seizure: 00008512_s009, 61M', xy=(0, 0), xytext=(1, 0), xycoords='data',
                             textcoords='axes fraction',
                             horizontalalignment='right', verticalalignment='bottom', fontsize=8)

        # Next Channel
        filtered = scipy.signal.filtfilt(numerator, denumerator, left_chains_blink[1][0][0], method="gust")

        axs.flat[1].plot(times, filtered[range[0] * 250:range[1] * 250], color='black')

        axs.flat[1].set_title("F3-C3", fontsize=18, fontweight="bold", color="#414242", pad=20)

        axs.flat[1].set_xlabel(farsi_time, fontfamily="B Nazanin", fontsize=14, labelpad=15, color="#056385")
        axs.flat[1].set_ylabel(farsi_amp, fontfamily="B Nazanin", fontsize=14, labelpad=10, color="#056385")

        axs.flat[1].set_ylim([-2.5 * 1e-5, 2.5 * 1e-5])

        axs.flat[1].axvspan(32.24 - 25, 32.58 - 25, color='#b2d9d5', alpha=0.4)
        axs.flat[1].axvspan(33.12 - 25, 33.63 - 25, color='#b2d9d5', alpha=0.4)
        axs.flat[1].axvspan(29.69 - 25, 30.12 - 25, color='#b2d9d5', alpha=0.4)
        axs.flat[1].axvspan(25.98 - 25, 26.37 - 25, color='#b2d9d5', alpha=0.4, label=farsi_probable_blink)
        axs.flat[1].legend(prop=font, facecolor='white', framealpha=1)
        axs.flat[1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))

        axs.flat[1].annotate('TUH Seizure: 00008512_s009, 61M', xy=(0, 0), xytext=(1, 0), xycoords='data',
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



fig.savefig('toGIT/SVG_figures/4-blink.svg')