import mne
import numpy as np
import scipy.signal
import matplotlib
import time
import matplotlib.pyplot as plt
import time

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.signal

matplotlib.use('Qt5Agg')


### Load

def power_indicate(array, band, fs=100.0):
    eeg_bands = {'delta': (0, 4),
                 'theta': (4, 8),
                 'alpha': (8, 12),
                 'beta': (12, 30),
                 'gamma': (30, 45)}

    # adopted from yasa docs
    freqs, psd = scipy.signal.welch(array, fs, return_onesided=True)
    psd = psd.reshape(1, -1)
    low, high = eeg_bands[band.lower()][0], eeg_bands[band.lower()][1]
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    idx_band = idx_band.reshape(1, -1)

    from scipy.integrate import simps
    freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25
    # Compute the absolute power by approximating the area under the curve
    theta_power = simps(psd[idx_band], dx=freq_res)
    # print('Absolute ' + str(band) + ' power: %.18f uV^2' % theta_power)
    # Relative delta power (expressed as a percentage of total power)
    total_power = simps(psd, dx=freq_res)
    tetha_relative_power = theta_power / total_power
    print('Relative ' + str(band) + ' power to total power: %.5f' % tetha_relative_power)
    return tetha_relative_power


raw_psg = mne.io.read_raw_edf('SC4021E0-PSG.edf', stim_channel='Event marker',
                              misc=['Temp rectal'], preload=True)
annotations = mne.read_annotations('SC4021EH-Hypnogram.edf')
raw_psg.set_annotations(annotations)
annotations_df = annotations.to_data_frame()
# matplotlib.rcParams['figure.figsize'] = (200, 100)
raw_psg.plot()
plt.tight_layout()

# may try to autocrop
# stage_w_column = annotations_df.loc[annotations_df["description"] == "Sleep stage W"]
# print(stage_w_column)

cropfirst = 20000
raw_psg.crop(cropfirst, include_tmax=False)
raw_psg.crop(0, 30000)

stage_1_column = annotations_df.loc[annotations_df["description"] == "Sleep stage 1"]
stage_2_column = annotations_df.loc[annotations_df["description"] == "Sleep stage 2"]
stage_3_column = annotations_df.loc[annotations_df["description"] == "Sleep stage 3"]
stage_4_column = annotations_df.loc[annotations_df["description"] == "Sleep stage 4"]
stage_awake_column = annotations_df.loc[annotations_df["description"] == "Sleep stage W"]
stage_rem_column = annotations_df.loc[annotations_df["description"] == "Sleep stage R"]

stage_1_raws = []
for i in range(len(stage_1_column)):
    onset_rawadj = time.mktime(stage_1_column.iloc[i]['onset'].timetuple()) - time.mktime(
        annotations_df.iloc[0]['onset'].timetuple())
    duration = stage_1_column.iloc[i]['duration']
    curr_raw = raw_psg.copy().crop(onset_rawadj - cropfirst, onset_rawadj - cropfirst + duration)
    stage_1_raws.append(curr_raw)

stage_2_raws = []
for i in range(len(stage_2_column)):
    onset_rawadj = time.mktime(stage_2_column.iloc[i]['onset'].timetuple()) - time.mktime(
        annotations_df.iloc[0]['onset'].timetuple())
    duration = stage_2_column.iloc[i]['duration']
    curr_raw = raw_psg.copy().crop(onset_rawadj - cropfirst, onset_rawadj - cropfirst + duration)
    stage_2_raws.append(curr_raw)

stage_3_raws = []
for i in range(len(stage_3_column)):
    onset_rawadj = time.mktime(stage_3_column.iloc[i]['onset'].timetuple()) - time.mktime(
        annotations_df.iloc[0]['onset'].timetuple())
    duration = stage_3_column.iloc[i]['duration']
    curr_raw = raw_psg.copy().crop(onset_rawadj - cropfirst, onset_rawadj - cropfirst + duration)
    stage_3_raws.append(curr_raw)

stage_4_raws = []
for i in range(len(stage_4_column)):
    onset_rawadj = time.mktime(stage_4_column.iloc[i]['onset'].timetuple()) - time.mktime(
        annotations_df.iloc[0]['onset'].timetuple())
    duration = stage_4_column.iloc[i]['duration']
    curr_raw = raw_psg.copy().crop(onset_rawadj - cropfirst, onset_rawadj - cropfirst + duration)
    stage_4_raws.append(curr_raw)

stage_rem_raws = []
for i in range(len(stage_rem_column)):
    onset_rawadj = time.mktime(stage_rem_column.iloc[i]['onset'].timetuple()) - time.mktime(
        annotations_df.iloc[0]['onset'].timetuple())
    duration = stage_rem_column.iloc[i]['duration']
    curr_raw = raw_psg.copy().crop(onset_rawadj - cropfirst, onset_rawadj - cropfirst + duration)
    stage_rem_raws.append(curr_raw)

awake_arr = raw_psg[0][0][0][:1800 * 100]

#### Relative Bands

print("Awake:")
awake_rel_band = power_indicate(awake_arr, 'delta')
print("N1:")
n1_rel_band = power_indicate(stage_1_raws[0][0][0][0], 'delta')
print("N2:")
n2_rel_band = power_indicate(stage_2_raws[0][0][0][0], 'delta')
print("N3:")
n3_rel_band = power_indicate(stage_3_raws[0][0][0][0], 'delta')
print("N4:")
n4_rel_band = power_indicate(stage_4_raws[0][0][0][0], 'delta')
print("REM:")
rem_rel_band = power_indicate(stage_rem_raws[0][0][0][0], 'delta')

### All in one stages and relative band powers

labels = ["Delta/4", "Theta", "Alpha", "Beta", "Gamma"]

rem_rel_bands = []
n1_rel_bands = []
n2_rel_bands = []
n3_rel_bands = []
n4_rel_bands = []
awake_rel_bands = []
for band in labels:
    curr_awake_rel_band = power_indicate(awake_arr, band.lower())
    curr_n1_rel_band = power_indicate(stage_1_raws[0][0][0][0], band.lower())
    curr_n2_rel_band = power_indicate(stage_2_raws[0][0][0][0], band.lower())
    curr_n3_rel_band = power_indicate(stage_3_raws[0][0][0][0], band.lower())
    curr_n4_rel_band = power_indicate(stage_4_raws[0][0][0][0], band.lower())
    curr_rem_rel_band = power_indicate(stage_rem_raws[0][0][0][0], band.lower())
    rem_rel_bands.append(curr_rem_rel_band[0])
    n1_rel_bands.append(curr_n1_rel_band[0])
    n2_rel_bands.append(curr_n2_rel_band[0])
    n3_rel_bands.append(curr_n3_rel_band[0])
    n4_rel_bands.append(curr_n4_rel_band[0])
    awake_rel_bands.append(curr_awake_rel_band[0])
del curr_n4_rel_band
del curr_n3_rel_band
del curr_n2_rel_band
del curr_n1_rel_band
del curr_rem_rel_band
del curr_awake_rel_band

# divide delta by two because it's a lot higher than others
n1_rel_bands[0] = n1_rel_bands[0] / 2
n2_rel_bands[0] = n2_rel_bands[0] / 2
n3_rel_bands[0] = n3_rel_bands[0] / 2
n4_rel_bands[0] = n4_rel_bands[0] / 2
rem_rel_bands[0] = rem_rel_bands[0] / 2
awake_rel_bands[0] = awake_rel_bands[0] / 2
#

### barplot stages and relative band powers

x = np.arange(len(labels))  # the label locations
width = 0.03  # the width of the bars

fig, ax = plt.subplots()
# from matplotlib.ticker import FormatStrFormatter
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
rects1 = ax.bar(x - (5 / 2) * width, awake_rel_bands, width, label='Awake')
rects2 = ax.bar(x - (3 / 2) * width, n1_rel_bands, width, label='N1')
rects3 = ax.bar(x - 0.5 * width, n2_rel_bands, width, label='N2')
rects4 = ax.bar(x + 0.5 * width, n3_rel_bands, width, label='N3')
rects5 = ax.bar(x + 1.5 * width, n4_rel_bands, width, label='N4')
rects6 = ax.bar(x + 2.5 * width, rem_rel_bands, width, label='REM')

# Add some text for labels, title and custom x-axis tick labels, etc.
import arabic_reshaper
from bidi.algorithm import get_display

ax.set_ylabel('Relative Power')
# ax.set_title('Relative Band Powers in Sleep Stages')
ax.set_title(get_display(arabic_reshaper.reshape("توان نسبی باندها بر حسب مرحله‌های خواب")), font="B Nazanin",
             fontsize=16)
ax.set_xticks(x, labels)
ax.legend()

# ax.bar_label(rects1, padding=1)
# ax.bar_label(rects2, padding=1)
# ax.bar_label(rects3, padding=1)
# ax.bar_label(rects4[0], fmt='%.2f', padding=1)
# ax.bar_label(rects5, padding=1)
# ax.bar_label(rects6, padding=1)
# for c in ax.containers: ax.bar_label(c, fmt='%.2f')

fig.tight_layout()

ax.set_facecolor('#fceab1')

### PSD integrated plot

from scipy.signal import welch
import seaborn as sns

freqs_n2, psd_n2 = welch(stage_2_raws[0][0][0][0], fs=100.0)
freqs_n1, psd_n1 = welch(stage_1_raws[0][0][0][0], fs=100.0)
freqs_n3, psd_n3 = welch(stage_3_raws[0][0][0][0], fs=100.0)
freqs_n4, psd_n4 = welch(stage_4_raws[0][0][0][0], fs=100.0)

plt.plot(freqs_n1, psd_n1, color="#e04356", lw=1.5)
plt.plot(freqs_n2, psd_n2, color='green', lw=1.5)
plt.plot(freqs_n3, psd_n3, color='cyan', lw=1.5)
plt.plot(freqs_n4, psd_n4, color='#8868de', lw=1.5)

plt.fill_between(freqs_n1, psd_n1, cmap='Spectral', color="#e04356", alpha=1, label="N1")
plt.fill_between(freqs_n2, psd_n2, cmap='Spectral', color='green', alpha=1, label="N2")
plt.fill_between(freqs_n3, psd_n3, cmap='Spectral', color='cyan', alpha=1, label="N3")
plt.fill_between(freqs_n4, psd_n4, cmap='Spectral', color='#8868de', alpha=1, label="N4")

plt.xlim(0, 50)
plt.yscale('log')
sns.despine()
plt.title(get_display(arabic_reshaper.reshape("چگالی طیفی توان در هر کدام از مرحله‌های خواب")), font="B Nazanin",
          fontsize=16)
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD Logarithmic ($uV^2$/Hz) [dB]')
plt.legend()

### Spectogram

raw_psg.plot()

raw_psg.plot_psd()

array = raw_psg[0][0][0][2800 * 100:3130 * 100 + 330 * 100]

fig, ax = plt.subplots()

cmap = plt.get_cmap('viridis').copy()

pxx, freq, t, cax = ax.specgram(array, Fs=100.0,
                                cmap=cmap,
                                vmax=-100,
                                )

fig.colorbar(cax).set_label("dB", labelpad=15)
ax.set_xlabel(get_display(arabic_reshaper.reshape("زمان (ثانیه)")), font="B Nazanin", fontsize=16,
              horizontalalignment='right', position=(1, 25), labelpad=10)
ax.set_ylabel(get_display(arabic_reshaper.reshape("فرکانس (هرتز)")), font="B Nazanin", fontsize=16, position=(25, 1),
              horizontalalignment='right', verticalalignment='center', labelpad=20)
plt.title(get_display(arabic_reshaper.reshape("طیف‌نگاره مرحله‌های اول و دوم خواب")), font="B Nazanin", fontsize=16)

ax.annotate('', xy=(0, -1 / 2), xytext=(330, -1 / 2), xycoords='data', textcoords='data',
            arrowprops={'arrowstyle': '|-|'}, annotation_clip=False)

ax.annotate('N1', xy=(150, -4), ha='center', va='bottom', annotation_clip=False, weight='bold')

ax.annotate('', xy=(331, -1 / 2), xytext=(660, -1 / 2), xycoords='data', textcoords='data',
            arrowprops={'arrowstyle': '|-|'}, annotation_clip=False)

ax.annotate('N2', xy=(490, -4), ha='center', va='bottom', annotation_clip=False, weight='bold')
ax.set_ylim(0, 25)

plt.tight_layout()


######## spindle
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


matplotlib.use('qtagg')
plt.disconnect()
import arabic_reshaper
from bidi.algorithm import get_display

n2_15s_arr = stage_2_raws[0].copy().filter(1, 30)[0][0][0][45 * 100:55 * 100]
# n2_15s_arr = n2_15s_arr * 1e6
times = np.arange(0, len(n2_15s_arr)) / 100.0

plt.plot(n2_15s_arr)

mask_dict = coordinates_saver(array=n2_15s_arr, times=times).coord_plot()
mask_dict = {0: [3.1191895161290315, 3.2890463709677418], 1: [3.5253689516129025, 3.665685483870967],
             2: [5.2756330645161285, 5.3642540322580645], 3: [5.504570564516129, 5.5858064516129025],
             4: [5.977215725806451, 6.0879919354838705], 5: [6.774804435483871, 6.856040322580645],
             6: [7.380381048387097, 7.5133125], 7: [7.8234858870967745, 7.897336693548386],
             8: [7.993342741935484, 8.133659274193548], 9: [8.71708064516129, 8.77616129032258],
             10: [9.654985887096775, 9.721451612903225], 11: [9.736221774193549, 9.765762096774193],
             12: [9.824842741935484, 9.86176814516129], 13: [1.6052479838709672, 1.5535524193548382]}

mask = np.zeros(len(n2_15s_arr))

for item in list(mask_dict.values()):
    mask[int(item[0] * 100):int(item[1] * 100)] = 1

spindles_highlight = n2_15s_arr * mask
spindles_highlight[spindles_highlight == 0] = np.nan

fig, ax = plt.subplots(1)
ax.plot(times[263:], n2_15s_arr[263:])
ax.plot(times[263:], spindles_highlight[263:], color="red", label="Spindle")
plt.legend()
ax.set_facecolor('#fceab1')
plt.title(get_display(arabic_reshaper.reshape("اسپیندل‌ها در مرحله دوم خواب")), font="B Nazanin", fontsize=16)
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))

ax.set_xlabel(get_display(arabic_reshaper.reshape("زمان (ثانیه)")), font="B Nazanin", fontsize=16,
              horizontalalignment='right', position=(1, 25), labelpad=10)
ax.set_ylabel(get_display(arabic_reshaper.reshape("اندازه (ولت)")), font="B Nazanin", fontsize=16, position=(25, 1),
              horizontalalignment='right', verticalalignment='center', labelpad=20)

plt.clf()
plt.switch_backend("agg")

####### REM and EOG

eog_rem_array = stage_rem_raws[0][2][0][0][:10 * 100]
eog_n1_array = stage_1_raws[0][2][0][0][:10 * 100]
plt.plot(eog_rem_array)
plt.plot(eog_n1_array)

raw_psg.copy().crop(5000, 15000).filter(0, 10).pick([2]).plot()
plt.tight_layout()
stage_2_column

### new plot (https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html)

annotation_desc_2_event_id = {'Sleep stage W': 1,
                              'Sleep stage 1': 2,
                              'Sleep stage 2': 3,
                              'Sleep stage 3': 4,
                              'Sleep stage 4': 5,
                              'Sleep stage R': 6}
events_train, events_id = mne.events_from_annotations(
    raw_psg, event_id=annotation_desc_2_event_id, chunk_duration=5.)
new_labels = {'Awake': 1,
              'N1': 2,
              'N2': 3,
              'N3': 4,
              'N4': 5,
              'REM': 6}
fig, ax = plt.subplots(1)
fig_e = mne.viz.plot_events(events_train, event_id=new_labels,
                            sfreq=raw_psg.info['sfreq'],
                            first_samp=events_train[0, 0], axes=ax)

# keep the color-code for further plotting
stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

ax.set_xlim(55)
# print(fig.legend().get_texts())
fig.set_facecolor('#fceab1')

plt.title(get_display(arabic_reshaper.reshape("رویدادهای کل نمونه (تکه‌های ۵ ثانیه‌ای)")), font="B Nazanin",
          fontsize=16)
# plt.legend()
plt.xlabel(get_display(arabic_reshaper.reshape("زمان (ثانیه)")), font="B Nazanin", fontsize=16,
           horizontalalignment='right', position=(1, 25), labelpad=10)
plt.ylabel(get_display(arabic_reshaper.reshape("شماره رویداد")), font="B Nazanin", fontsize=16, position=(25, 1),
           horizontalalignment='right', verticalalignment='center', labelpad=20)
