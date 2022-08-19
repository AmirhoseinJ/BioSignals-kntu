from pathlib import Path

import numpy as np


def path_list_generator():
    path_list = []
    for path in Path(r'/run/media/****/project/edf/').rglob('*.edf'):
        path_list.append(path)
    print("EDF Files in path: " + str(len(path_list)))
    return path_list


path_list_generator()  # 669 edf fies

# 1385 labels == all events
# 726 featuers... which is which?
# import antropy as ant
#
# arr=np.ones(555)
# ant.katz_fd(arr)


npy_labels = []
npy_features = []
for path in Path(r'/run/media/****/prjfromJupyter/fft+ica/').rglob('*.npy'):
    if "label" in str(path):
        current_nparray = (np.load(str(path), allow_pickle=True))
        npy_labels.append(current_nparray.reshape(len(current_nparray), -1))
    elif "feature" in str(path):
        current_nparray = (np.load(str(path), allow_pickle=True))
        npy_features.append(current_nparray.reshape(len(current_nparray), -1))

labels_array = np.concatenate(
    npy_labels)  # https://stackoverflow.com/questions/28125265/concatenate-numpy-arrays-which-are-elements-of-a-list
features_array = np.concatenate(npy_features)

###############################################

import mne
import numpy as np
# import cupy as cp
import antropy as ant
import pandas as pd
from pathlib import Path

mne.set_log_level(verbose="CRITICAL")
# mne.cuda.init_cuda(verbose=True)
import warnings

warnings.filterwarnings("ignore")
# from numba import jit
import time


def path_list_generator():
    path_list = []
    for path in Path(r'/run/media/****/project/edf/').rglob('*.edf'):
        path_list.append(path)
    print("EDF Files is path: " + str(len(path_list)))
    return path_list


# data features extraction functions
# @jit
def maximum(x):
    return np.max(x, axis=-1)


# @jit
def minimum(x):
    return np.min(x, axis=-1)


# @jit
def whichmin(x):
    return np.argmin(x, axis=-1)


# @jit
def whichmax(x):
    return np.argmax(x, axis=-1)


# @jit
def mean(x):
    return np.mean(x, axis=-1)


# @jit
def median(x):
    return np.median(x, axis=-1)


# @jit
def std(x):
    return np.std(x, axis=-1)


# @jit
def variance(x):
    return np.var(x, axis=-1)


# @jit
def peaktopeak(x):
    return np.ptp(x, axis=-1)


# @jit
def rms(x):
    return np.sqrt(np.mean(x ** 2, axis=-1))


# @jit
def sad(x):  # Sum of absolute differences
    return np.sum(np.abs(np.diff(x, axis=-1)), axis=-1)


from scipy import stats


# @jit
def kurtosis(x):
    return stats.kurtosis(x, axis=-1)


# @jit
def skewness(x):
    return stats.skew(x, axis=-1)


# @jit
def load_features(path_tuple, features_list, labels_list, limiter, loop_count=0):
    chans = ['EEG T4-REF', 'EEG T6-REF', 'EEG P3-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG T5-REF', 'EEG C3-REF',
             'EEG T3-REF', 'EEG F3-REF', 'EEG FP1-REF',
             'EEG F8-REF', 'EEG A1-REF', 'EEG O1-REF', 'EEG A2-REF', 'EEG FP2-REF', 'EEG F4-REF', 'EEG C4-REF',
             'EEG CZ-REF', 'EEG P4-REF']
    for path in path_list_generator()[path_tuple[0]:path_tuple[1]]:
        loopstart_time = time.time()
        # load edf
        mne_edf = mne.io.read_raw_edf(str(path), preload=True)
        mne_edf.pick_channels(chans)
        fsamp = mne_edf.info['sfreq']
        mne_edf.filter(1, 40)
        mne_edf.set_eeg_reference()
        print("loadedf took %s seconds" % (time.time() - loopstart_time))
        # loopstart_time=time.time()
        # # ICA artifact removal
        # ica = mne.preprocessing.ICA(n_components=0.9999, method="picard", max_iter='auto', random_state=random.randint(10,90000))
        # ica.fit(mne_edf)
        # ica.apply(mne_edf)
        # print("ICA Artifacts took %s seconds" % (time.time() - loopstart_time))
        # loopstart_time=time.time()
        # annotations
        tse_file = pd.read_csv(str(path).replace(".edf", ".tse"), delim_whitespace=True, index_col=False)
        tse_file.rename(columns={'version': 'start', '=': 'end', 'tse_v1.0.0': 'event'}, inplace=True)
        onset = []
        duration = []
        description = []
        for i in range(tse_file.shape[0]):
            onset.append(tse_file.iloc[i]['start'])
            duration.append(tse_file.iloc[i]['end'] - tse_file.iloc[i]['start'])
            description.append(tse_file.iloc[i]['event'])

        annot = mne.Annotations(onset=onset,  # in seconds
                                duration=duration,  # in seconds
                                description=description)
        print("Annotations took %s seconds" % (time.time() - loopstart_time))
        loopstart_time = time.time()
        # create current data_list and overall labels_list
        data_list = []

        for i, event in enumerate(annot.description):
            labels_list.append(event)
            data_list.append(mne_edf.get_data()[:, round(annot.onset[i] * fsamp):round(annot.onset[i] * fsamp) + round(
                annot.duration[i] * fsamp)])
        print("data_list.append took %s seconds" % (time.time() - loopstart_time))
        loopstart_time = time.time()
        # features
        for event in data_list:

            f0 = median(event).reshape(-1, 1)
            f1 = mean(event).reshape(-1, 1)
            f2 = maximum(event).reshape(-1, 1)
            f3 = minimum(event).reshape(-1, 1)
            f4 = sad(event).reshape(-1, 1)
            f5 = rms(event).reshape(-1, 1)
            f6 = peaktopeak(event).reshape(-1, 1)
            f7 = kurtosis(event).reshape(-1, 1)
            f8 = skewness(event).reshape(-1, 1)
            f9 = variance(event).reshape(-1, 1)
            f10 = std(event).reshape(-1, 1)
            f11 = ant.num_zerocross(event, axis=-1).reshape(-1, 1)
            f12 = ant.katz_fd(event, axis=-1).reshape(-1, 1)
            f13 = ant.hjorth_params(event, axis=-1)[0].reshape(-1, 1)
            f14 = ant.hjorth_params(event, axis=-1)[1].reshape(-1, 1)
            f15 = np.empty(0)
            for channel in event: f15 = np.append(f15, ant.perm_entropy(channel, normalize=True)); f15 = f15.reshape(-1,
                                                                                                                     1)
            f16 = ant.spectral_entropy(event, sf=fsamp, method='welch', normalize=True).reshape(-1, 1)
            f17 = np.empty(0)
            for channel in event: f17 = np.append(f17, ant.svd_entropy(channel, normalize=True)); f17 = f17.reshape(-1,
                                                                                                                    1)
            f18 = ant.petrosian_fd(event).reshape(-1, 1)
            f19 = np.empty(0)
            for channel in event: f19 = np.append(f19, ant.higuchi_fd(channel)); f19 = f19.reshape(-1, 1)
            f20 = np.empty(0)
            for channel in event: f20 = np.append(f20, ant.detrended_fluctuation(channel)); f20 = f20.reshape(-1, 1)

            # f21_lz = np.empty(0) # removed because of performance issues
            # for channel in event: f21_lz = np.append(f21_lz, ant.lziv_complexity(channel)); f21_lz = f21_lz.reshape(-1, 1)
            #

            # psd features
            fft_event = np.fft.rfft(event)

            f21 = mean(fft_event).real.reshape(-1, 1)
            f22 = mean(fft_event).imag.reshape(-1, 1)
            f23 = median(fft_event).real.reshape(-1, 1)
            f24 = median(fft_event).imag.reshape(-1, 1)
            f25 = variance(fft_event).real.reshape(-1, 1)
            # f25_1 = variance(fft_event).imag.reshape(-1, 1) is empty
            f26 = std(fft_event).real.reshape(-1, 1)
            # f27 = std(fft_event).imag.reshape(-1, 1) is empty
            f28 = skewness(fft_event).real.reshape(-1, 1)
            f29 = skewness(fft_event).imag.reshape(-1, 1)
            f30 = kurtosis(fft_event).real.reshape(-1, 1)
            f31 = kurtosis(fft_event).imag.reshape(-1, 1)

            psd_event = mne.time_frequency.psd_welch(
                mne.io.RawArray(event, info=mne.create_info(ch_names=chans, ch_types="eeg", sfreq=fsamp)),
                picks="all", n_fft=int(len(event[0]) / 2), window='hamming')[0]

            # fpsd1 = median(psd_event).reshape(-1,1)
            # fpsd2 = mean(psd_event).reshape(-1, 1)
            # fpsd3 = maximum(psd_event).reshape(-1, 1)
            # fpsd4 = minimum(psd_event).reshape(-1, 1)
            # fpsd0 = sad(event).reshape(-1, 1)
            # fpsd5 = rms(event).reshape(-1, 1)
            # fpsd6 = peaktopeak(event).reshape(-1, 1)
            # fpsd7 = kurtosis(event).reshape(-1, 1)
            # fpsd8 = skewness(event).reshape(-1, 1)
            # fpsd9 = variance(event).reshape(-1, 1)
            # fpsd10 = std(event).reshape(-1,1)

            fpsd_list = [median(psd_event).reshape(-1, 1), mean(psd_event).reshape(-1, 1),
                         std(psd_event).reshape(-1, 1), maximum(psd_event).reshape(-1, 1),
                         minimum(psd_event).reshape(-1, 1), sad(psd_event).reshape(-1, 1),
                         rms(psd_event).reshape(-1, 1), peaktopeak(psd_event).reshape(-1, 1),
                         kurtosis(psd_event).reshape(-1, 1), skewness(psd_event).reshape(-1, 1),
                         variance(psd_event).reshape(-1, 1)]

            # WARN: 27 REMOVED

            print("Features Calculations took %s seconds" % (time.time() - loopstart_time))
            loopstart_time = time.time()
            features_list.append(np.concatenate((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
                                                 f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f28, f29, f30,
                                                 f31, *[fpsd for fpsd in fpsd_list]), axis=-1))
            print("Features append took %s seconds" % (time.time() - loopstart_time))
            loopstart_time = time.time()
        if (limiter != 0):
            loop_count += 1
        if (limiter != 0 and loop_count == limiter):
            break


##################


fl_test = []
ll_test = []
fl_main = load_features((0, 6), fl_test, ll_test, 0)

features_dict = {}
features_flattened = []
features_str_flattened = []
for evn_no, event in enumerate(fl_test):
    curr_event_str = ll_test[evn_no]
    curr_features_str_list = []
    curr_features_list = []
    for ch_no, channel in enumerate(event):
        curr_channel_str = "ch" + str(ch_no)
        for feat_no, feature in enumerate(channel):
            curr_feature_str = curr_channel_str + "_feat" + str(feat_no)
            curr_features_str_list.append(curr_feature_str)
            curr_features_list.append(feature)
    features_flattened.append(curr_features_list)
    features_str_flattened.append(curr_features_str_list)
    features_dict[evn_no] = curr_features_list

dataframe_test = pd.DataFrame(features_flattened, columns=features_str_flattened[0])

np.set_printoptions(threshold=10000)
dataframe_test.head()

from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Extract sorted F-values
fvals = pd.Series(f_classif(X=dataframe_test, y=ll_test)[0], index=dataframe_test.columns)
features_count = 42

# Average By Channel

from collections import defaultdict

scoring_dict_bychan = defaultdict(list)

curr = 0
curr_ch = 0
curr_list = []

for feat_value in fvals:
    curr_list.append(feat_value)
    curr += 1
    if curr == features_count - 1:
        scoring_dict_bychan["ch" + str(curr_ch)].append(curr_list)
        curr_ch += 1
        curr = 0
        curr_list = []

print(scoring_dict_bychan)

scoring_dict_avg_by_chan = defaultdict(list)

for item in list(scoring_dict_bychan.keys()):
    curr_arr = np.array(scoring_dict_bychan[item])
    curr_arr = curr_arr[~np.isnan(curr_arr)]
    scoring_dict_avg_by_chan[item] = np.mean(curr_arr)

np.array(scoring_dict_bychan[item])

matplotlib.use('Qt5Agg')

# Plot

plt.figure(figsize=(6, 6))
sns.barplot(y=list(scoring_dict_avg_by_chan.keys()), x=list(scoring_dict_avg_by_chan.values()), palette='RdYlGn')
plt.xlabel('F-values')
plt.xticks(rotation=20)
import arabic_reshaper
from bidi.algorithm import get_display

plt.title(get_display(arabic_reshaper.reshape("کانال‌هابه عنوان ویژگی، آزمون اف")), font="B Nazanin", fontsize=16)
# plt.legend()
plt.xlabel(get_display(arabic_reshaper.reshape("امتیاز میانگین")), font="B Nazanin", fontsize=16, labelpad=10)

matplotlib.rcParams['figure.figsize'] = (64, 54)
plt.savefig("SVGfigs/5-chFtest.svg")

# By Features

scoring_dict_byfeat = defaultdict(list)

channels_count = 19
for feat_no in range(features_count):
    feat_curr = []
    for chan_no in range(channels_count):
        feat_curr.append(fvals[chan_no * features_count + feat_no])
    scoring_dict_byfeat["feature" + str(feat_no)] = feat_curr

# Rename feature names to match their corresponding names

corresp_feat_names = [
    'Median',
    'Mean',
    'Maximum',
    'Minimum',
    'SAD',
    'RMS',
    'PtP',
    'Kurtosis',
    'Skewness',
    'Variance',
    'Std',
    'Zerocross',
    'Hatd_fd',
    'Hjorth0',
    'Hjorth1',
    'Permutation Entropy',
    'Spectral Entropy',
    'SVD Entropy',
    'Petrosian_fd',
    'Higuchi_fd',
    'Detrended_fluctuation',
    'fft_Mean_Re',
    'fft_Mean_Im',
    'fft_Median_Re',
    'fft_Median_Im',
    'fft_Variance',
    'fft_Std',
    'fft_Skewness_Re',
    'fft_Skewness_Im',
    'fft_Kurtosis_Re',
    'fft_Kurtosis_Im',
    'psd_Median',
    'psd_Mean',
    'psd_Std',
    'psd_Maximum',
    'psd_Minimum',
    'psd_SAD',
    'psd_RMS',
    'psd_PtP',
    'psd_Kurtosis',
    'psd_Skewness',
    'psd_Variance'
]

for feat_no in range(features_count):
    scoring_dict_byfeat[corresp_feat_names[feat_no]] = scoring_dict_byfeat.pop("feature" + str(feat_no))

# Average

scoring_dict_avg_by_feat = defaultdict(list)

for item in scoring_dict_byfeat.keys():
    scoring_dict_avg_by_feat[item] = np.mean(scoring_dict_byfeat[item]).tolist()

matplotlib.use('Qt5Agg')

# Plot


plt.rcParams['figure.figsize'] = (14, 21)
plt.tight_layout()
plt.rcParams['axes.facecolor'] = 'white'
# plt.tight_layout()
# plt.figure(figsize=(6, 6))
sns.barplot(y=list(scoring_dict_avg_by_feat.keys()), x=list(scoring_dict_avg_by_feat.values()), palette='RdYlGn')
# plt.xlabel('F-values')
plt.xticks(rotation=20)
import arabic_reshaper

sns.despine()
from bidi.algorithm import get_display

plt.title(get_display(arabic_reshaper.reshape("مقایسه ویژگی‌های منتخب بر اساس آزمون اف")), font="B Nazanin",
          fontsize=16)
# plt.legend()
plt.xlabel(get_display(arabic_reshaper.reshape("امتیاز میانگین")), font="B Nazanin", fontsize=16, labelpad=10)
# plt.ylabel(get_display(arabic_reshaper.reshape("شماره رویداد")), font="B Nazanin", fontsize=16 , position=(25,1),horizontalalignment='right', verticalalignment='center',labelpad=20)


plt.savefig("SVGfigs/5-featFtest.svg")

#################################
# from sklearn.feature_selection import SelectKBest
# bestF = SelectKBest(score_func=f_classif, k =4)
# fit = bestF.fit(features_flattened, ll_test)
# print(fit.pvalues_)
# # bestF.score_func(features_flattened, ll_test)


# begin sklearn

from sklearn.preprocessing import LabelEncoder
import random

le = LabelEncoder()
labels_encoded = le.fit_transform(ll_test)
mapping = dict(zip(le.classes_, range(len(le.classes_))))
print(labels_encoded, mapping)

from sklearn.model_selection import train_test_split

input_train, input_test, output_train, output_test = train_test_split(features_flattened, labels_encoded,
                                                                      test_size=0.15,
                                                                      random_state=random.randint(1, 90000))
# import sys
# np.set_printoptions(threshold=sys.maxsize)
# print(output_train)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

grid_params_dict_2 = {
    'clf__C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1, 3, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
clf = LogisticRegression(max_iter=2000)
pipe = Pipeline([('scalar', StandardScaler()), ('clf', clf)])
grid_search = GridSearchCV(
    pipe,
    param_grid=grid_params_dict_2,
    cv=2,
    n_jobs=-1
)
grid_search.fit(input_train, output_train)
print(grid_search.best_score_)
# print(grid_search.best_params_)
# print(grid_search.score(input_test, output_test))
# print(output_test)
