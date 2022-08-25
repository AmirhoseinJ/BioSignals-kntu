import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np


class EEGHandler:
    def __init__(self, fs=250, type="edf",
                 edf_path="00008512_s009_t001.edf",
                 tdms_path="AmirhoseinJ1_Filtered.tdms", tdms_check_zero=False,
                 tdms_fs=250.0, tdms_scale=1):
        self.edf_path = edf_path
        self.tdms_path = tdms_path
        self.tdms_check_zero = tdms_check_zero
        self.tdms_fs = tdms_fs
        self.tdms_scale = tdms_scale
        self.fs = fs
        self.type = type

    def create_raw(self):
        import mne
        if self.type == 'edf':
            print("Loading EDF file...")
            self.raw = mne.io.read_raw_edf(self.edf_path, preload=True)
        elif self.type == "tdms":
            from nptdms import TdmsFile
            print("Loading TDMS file...")
            tdms_file = TdmsFile.read(self.tdms_path)
            channels_dict = {}
            for channel in tdms_file.groups()[0].channels():
                if self.tdms_check_zero:
                    if channel[:].astype("int").all() != 0:
                        channels_dict[channel.name] = channel[:]
                else:
                    channels_dict[channel.name] = channel[:]
            all_array_list = []
            for array in channels_dict.values():
                all_array_list.append(array * self.tdms_scale)
            info = mne.create_info(ch_names=list(channels_dict.keys()), ch_types="eeg", sfreq=self.tdms_fs)
            self.raw = mne.io.RawArray(all_array_list, info=info)
        else:
            print("Warning: Invalid options. Raw object not created.")
            self.raw = None
        return self.raw

    def create_montage(self, chnames_montage=None):
        import mne
        # if self.type == "edf":
        #     convention = "TUH"
        if self.type == "edf":
            if chnames_montage is None:
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
                current_montage = self.raw["EEG " + two_channels[1] + "-REF"][0][0] - \
                                  self.raw["EEG " + two_channels[0] + "-REF"][0][0]
                montaged_channels_arrays.append(current_montage)
            self.raw_montaged = mne.io.RawArray(np.asarray(montaged_channels_arrays),
                                                info=mne.create_info(ch_names=chnames_montage, ch_types="eeg",
                                                                     sfreq=self.fs))
        elif self.type == "tdms":
            if chnames_montage is None:
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
                current_montage = self.raw[two_channels[1]][0][0] - self.raw[two_channels[0]][0][0]
                montaged_channels_arrays.append(current_montage)
            self.raw_montaged = mne.io.RawArray(np.asarray(montaged_channels_arrays),
                                                info=mne.create_info(ch_names=chnames_montage, ch_types="eeg",
                                                                     sfreq=self.tdms_fs))
        else:
            print("Warning: Invalid options. Raw_montaged object not created.")
            self.raw_montaged = None
        return self.raw_montaged

    @staticmethod
    def farsi_reshape(string):
        import arabic_reshaper
        from bidi.algorithm import get_display
        string_reshaped = get_display(arabic_reshaper.reshape(string))
        return string_reshaped

    def filter_butterworth(self, channels=["Fp1"], range=None, butterworth_N=5, butterworth_wn=0.05,
                           use_montaged=False):
        import scipy.signal
        if use_montaged:
            if len(channels) == 1:
                numerator, denumerator = scipy.signal.butter(butterworth_N, butterworth_wn, output='ba')
                if range is not None:
                    butterworth_filtered = scipy.signal.filtfilt(numerator, denumerator,
                                                                 self.raw_montaged[channels[0]][0][0][
                                                                 range[0] * self.fs:range[
                                                                                        1] * self.fs],
                                                                 method="gust")
                else:
                    butterworth_filtered = scipy.signal.filtfilt(numerator, denumerator,
                                                                 self.raw_montaged[channels[0]][0][0],
                                                                 method="gust")
                return butterworth_filtered
            else:
                filtered_list = []
                for channel in channels:
                    numerator, denumerator = scipy.signal.butter(butterworth_N, butterworth_wn, output='ba')
                    if range is not None:
                        butterworth_filtered = scipy.signal.filtfilt(numerator, denumerator,
                                                                     self.raw_montaged[channel][0][0][
                                                                     range[0] * self.fs:range[
                                                                                            1] * self.fs],
                                                                     method="gust")
                    else:
                        butterworth_filtered = scipy.signal.filtfilt(numerator, denumerator,
                                                                     self.raw_montaged[channel][0][0],
                                                                     method="gust")
                    filtered_list.append(butterworth_filtered)
                return filtered_list
        else:
            if len(channels) == 1:
                numerator, denumerator = scipy.signal.butter(butterworth_N, butterworth_wn, output='ba')
                if range is not None:
                    butterworth_filtered = scipy.signal.filtfilt(numerator, denumerator, self.raw[channels[0]][0][0][
                                                                                         range[0] * self.fs:range[
                                                                                                                1] * self.fs],
                                                                 method="gust")
                else:
                    butterworth_filtered = scipy.signal.filtfilt(numerator, denumerator, self.raw[channels[0]][0][0],
                                                                 method="gust")
                return butterworth_filtered
            else:
                filtered_list = []
                for channel in channels:
                    numerator, denumerator = scipy.signal.butter(butterworth_N, butterworth_wn, output='ba')
                    if range is not None:
                        butterworth_filtered = scipy.signal.filtfilt(numerator, denumerator, self.raw[channel][0][0][
                                                                                             range[0] * self.fs:range[
                                                                                                                    1] * self.fs],
                                                                     method="gust")
                    else:
                        butterworth_filtered = scipy.signal.filtfilt(numerator, denumerator, self.raw[channel][0][0],
                                                                     method="gust")
                    filtered_list.append(butterworth_filtered)
                return filtered_list

    @staticmethod
    def beauty_plot(arrays, times, titles, xlabel=farsi_reshape("زمان (ثانیه)"), ylabel=farsi_reshape("اندازه (ولت)"),
                    xlims=None, ylims=None, data_label="label", style="seaborn", linewidth=1.2, main_font='Roboto',
                    legend_frame=True, line_color='black',
                    title_fontsize=16, title_fontweight="bold", title_color="#2d1754", title_pad=20,
                    label_color="#056385", legend_label="", suptitle="",
                    xlabel_fontsize=14, ylabel_fontsize=14, xlabel_pad=15, ylabel_pad=10, label_font="B Nazanin",
                    hightlight_dict=None, highlight_color="#b2d9d5", highlight_alpha=0.4
                    , sup_fontfamily="Roboto", sup_fontsize=24, sup_color="black", sup_y=0.9, savefname=None):

        with plt.style.context(style):
            with plt.rc_context({'lines.linewidth': linewidth, 'lines.linestyle': '-', 'font.family': [main_font],
                                 'legend.frameon': legend_frame}):
                matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
                font = matplotlib.font_manager.FontProperties(family='B Nazanin',
                                                              weight='bold',
                                                              style='normal', size=10)
                fig, axs = plt.subplots(len(arrays), squeeze=False, figsize=(16, 9))
                fig.suptitle(suptitle, fontfamily=sup_fontfamily, fontsize=sup_fontsize, color=sup_color, y=sup_y)
                for i, array in enumerate(arrays):
                    axs.flat[i].plot(times[i], arrays[i], color=line_color)
                    axs.flat[i].set_title(titles[i], fontsize=title_fontsize, fontweight=title_fontweight,
                                          color=title_color, pad=title_pad)
                    axs.flat[i].set_xlabel(xlabel, fontfamily=label_font, fontsize=xlabel_fontsize, labelpad=xlabel_pad,
                                           color=label_color)
                    axs.flat[i].set_ylabel(ylabel, fontfamily=label_font, fontsize=ylabel_fontsize, labelpad=ylabel_pad,
                                           color=label_color)
                    if ylims is not None:
                        axs.flat[i].set_ylim(ylims[i])
                    if xlims is not None:
                        axs.flat[i].set_xlim(xlims[i])
                    axs.flat[i].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
                    axs.flat[i].annotate(data_label, xy=(times[i][0], 0), xytext=(1, 0), xycoords='data',
                                         textcoords='axes fraction',
                                         horizontalalignment='right', verticalalignment='bottom', fontsize=8)
                    if hightlight_dict is not None:
                        for j, coors in enumerate(hightlight_dict.values()):
                            axs.flat[j].axvspan(coors[0], coors[1], color=highlight_color, alpha=highlight_alpha,
                                                label=legend_label)
                        axs.flat[i].legend(prop=font, facecolor='white', framealpha=1)
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
                if savefname is not None:
                    fig.savefig(savefname)

    class coordinates_saver():
        def __init__(self, times, array):
            self.times = times
            self.array = array
            self.coord_dict = {}
            self.coord_list = []
            self.curr_coord = 0

        def onclick(self, event):
            global ix
            ix = event.xdata
            print('x = %1.2f' % (ix))
            self.coord_list.append(ix)
            if len(self.coord_list) == 2:
                print(self.coord_list)
                self.coord_dict[self.curr_coord] = self.coord_list
                print(self.coord_dict)
                self.curr_coord += 1
                self.coord_list = []

        def coord_plot(self):
            self.coord_list = []
            self.coord_dict = {}
            self.curr_coord = 0
            fig, ax = plt.subplots(1)
            cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
            ax.plot(self.times, self.array)
            plt.show(block=True)
            return self.coord_dict
