from toGIT.single import EEG_handler
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np


matplotlib.use('Qt5Agg')

# matplotlib.get_cachedir() # remove font cache

############## lpf
fs = 256.0
high_freq = 30

fig, axs = plt.subplots(2,figsize=(16, 16))

fig.suptitle(EEG_handler.EEGHandler.farsi_reshape("فیلتر پایین‌گذر (30 هرتز) باترورث"), fontfamily="B Nazanin",  y=0.92, fontsize=25, color="#2d1754")

for order in [1, 3, 6]:
    numerator, denumerator = scipy.signal.butter(order, high_freq, fs=fs, btype='low')
    w, spect_amp = scipy.signal.freqz(numerator, denumerator, fs=fs, worN=2000)
    axs[1].plot(w, 20*np.log10(abs(spect_amp)), label=EEG_handler.EEGHandler.farsi_reshape("مرتبه") + " = %d" % order)

axs[1].plot([0, 0.5 * fs], [-3, -3], linestyle='--', label=EEG_handler.EEGHandler.farsi_reshape("تضعیف گین"))

axs[1].set_xlabel(EEG_handler.EEGHandler.farsi_reshape("فرکانس (هرتز)"), fontfamily="B Nazanin", fontsize=12)

axs[1].set_ylabel(EEG_handler.EEGHandler.farsi_reshape("گین (لگاریتمی)"), fontfamily="B Nazanin", fontsize=12)

axs[1].grid(True)


font = matplotlib.font_manager.FontProperties(family='B Nazanin',
                                              weight='bold',
                                              style='normal', size=14)

axs[1].legend(prop=font, facecolor='white', framealpha=1)

matplotlib.rc('text', usetex=False)

matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

axs[1].set_ylim(-50,5)

# axs[1].annotate(r'$\mathbf{f_c}$', xy=(high_freq, 0), xytext=(high_freq,-0.55),horizontalalignment='center', verticalalignment='center', fontsize=12,weight='bold')

axs[1].annotate(r'$\mathbf{-3dB}$', xy=(0, np.sqrt(0.5)), xytext=(-9.3,-3),horizontalalignment='center', verticalalignment='center', fontsize=10,weight='bold')


for order in [1, 3, 6]:
    numerator, denumerator = scipy.signal.butter(order, high_freq, fs=fs, btype='low')
    w, spect_amp = scipy.signal.freqz(numerator, denumerator, fs=fs, worN=2000)
    axs[0].plot(w, (abs(spect_amp)), label=EEG_handler.EEGHandler.farsi_reshape("مرتبه") + " = %d" % order)


axs[0].plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], linestyle='--', label=EEG_handler.EEGHandler.farsi_reshape("تضعیف گین"))

ideal = np.linspace(0, 0.5*fs, len(w))

mask = (ideal <=high_freq)

ideal = np.where(mask, 1, 0)

axs[0].plot(w, ideal, linestyle='--', label=EEG_handler.EEGHandler.farsi_reshape("فیلتر ایده‌آل"),color='black')

axs[0].set_xlabel(EEG_handler.EEGHandler.farsi_reshape("فرکانس (هرتز)"), fontfamily="B Nazanin", fontsize=12)

axs[0].set_ylabel(EEG_handler.EEGHandler.farsi_reshape("گین"), fontfamily="B Nazanin", fontsize=12)

axs[0].grid(True)

font = matplotlib.font_manager.FontProperties(family='B Nazanin',
                                              weight='bold',
                                              style='normal', size=14)
axs[0].legend(prop=font, facecolor='white', framealpha=1)

matplotlib.rc('text', usetex=False)

matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

# axs[0].set_ylim(-50,5)

axs[0].annotate(r'$\mathbf{f_c}$', xy=(high_freq, 0), xytext=(high_freq,-0.085),horizontalalignment='center', verticalalignment='center', fontsize=12,weight='bold')

axs[0].annotate(r'$\mathbf{\sqrt{0.5}}$', xy=(0, np.sqrt(0.5)), xytext=(-9.3,np.sqrt(0.5)),horizontalalignment='center', verticalalignment='center', fontsize=10,weight='bold')

fig.savefig('toGIT/SVG_figures/3_lpf-both.svg')




######################### bpf
fs = 256.0
low_freq =40.0
high_freq = 80.0

fig, axs = plt.subplots(2,figsize=(16, 16))

fig.suptitle(EEG_handler.EEGHandler.farsi_reshape("فیلتر میان‌گذر (40 تا 80 هرتز) باترورث"), fontfamily="B Nazanin", y=0.92, fontsize=25, color="#2d1754")

for order in [1, 3, 6]:
    numerator, denumerator = scipy.signal.butter(order, [low_freq, high_freq], fs=fs, btype='band')
    w, spect_amp = scipy.signal.freqz(numerator, denumerator, fs=fs)
    axs[1].plot(w, 20*np.log10((spect_amp)), label=EEG_handler.EEGHandler.farsi_reshape("مرتبه") + " = %d" % order)

axs[1].plot([0, 0.5 * fs], [-3, -3], linestyle='--', label=EEG_handler.EEGHandler.farsi_reshape("تضعیف گین"))

axs[1].set_xlabel(EEG_handler.EEGHandler.farsi_reshape("فرکانس (هرتز)"), fontfamily="B Nazanin", fontsize=12)

axs[1].set_ylabel(EEG_handler.EEGHandler.farsi_reshape("گین (لگاریتمی)"), fontfamily="B Nazanin", fontsize=12)

axs[1].grid(True)


font = matplotlib.font_manager.FontProperties(family='B Nazanin',
                                              weight='bold',
                                              style='normal', size=14)

axs[1].legend(prop=font, facecolor='white', framealpha=1)

matplotlib.rc('text', usetex=False)

matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

axs[1].set_ylim(-50,5)

axs[1].annotate(r'$\mathbf{-3dB}$', xy=(0, np.sqrt(0.5)), xytext=(-9.3,-3),horizontalalignment='center', verticalalignment='center', fontsize=10,weight='bold')


for order in [1, 3, 6]:
    numerator, denumerator = scipy.signal.butter(order, [low_freq, high_freq], fs=fs, btype='band')
    w, spect_amp = scipy.signal.freqz(numerator, denumerator, fs=fs)
    axs[0].plot(w, abs((spect_amp)), label=EEG_handler.EEGHandler.farsi_reshape("مرتبه") + " = %d" % order)

axs[0].plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], linestyle='--', label=EEG_handler.EEGHandler.farsi_reshape("تضعیف گین"))

axs[0].set_xlabel(EEG_handler.EEGHandler.farsi_reshape("فرکانس (هرتز)"), fontfamily="B Nazanin", fontsize=12)

axs[0].set_ylabel(EEG_handler.EEGHandler.farsi_reshape("گین"), fontfamily="B Nazanin", fontsize=12)

axs[0].grid(True)


font = matplotlib.font_manager.FontProperties(family='B Nazanin',
                                              weight='bold',
                                              style='normal', size=14)

axs[0].legend(prop=font, facecolor='white', framealpha=1)

matplotlib.rc('text', usetex=False)

matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

axs[0].annotate(r'$\mathbf{\sqrt{0.5}}$', xy=(0, np.sqrt(0.5)), xytext=(-9.3,np.sqrt(0.5)),horizontalalignment='center', verticalalignment='center', fontsize=10,weight='bold')

fig.savefig('toGIT/SVG_figures/3_bpf-both.svg')


################## hpf
fs = 256.0
low_freq = 30

fig, axs = plt.subplots(2,figsize=(16, 16))

fig.suptitle(EEG_handler.EEGHandler.farsi_reshape("فیلتر بالاگذر (30 هرتز) باترورث"), fontfamily="B Nazanin", y=0.92, fontsize=25, color="#2d1754")

for order in [1, 3, 6]:
    numerator, denumerator = scipy.signal.butter(order, low_freq, fs=fs, btype='high')
    w, spect_amp = scipy.signal.freqz(numerator, denumerator, fs=fs, worN=2000)
    axs[1].plot(w, 20*np.log10(abs(spect_amp)), label=EEG_handler.EEGHandler.farsi_reshape("مرتبه") + " = %d" % order)

axs[1].plot([0, 0.5 * fs], [-3, -3], linestyle='--', label=EEG_handler.EEGHandler.farsi_reshape("تضعیف گین"))

axs[1].set_xlabel(EEG_handler.EEGHandler.farsi_reshape("فرکانس (هرتز)"), fontfamily="B Nazanin", fontsize=12)

axs[1].set_ylabel(EEG_handler.EEGHandler.farsi_reshape("گین (لگاریتمی)"), fontfamily="B Nazanin", fontsize=12)

axs[1].grid(True)


font = matplotlib.font_manager.FontProperties(family='B Nazanin',
                                              weight='bold',
                                              style='normal', size=14)

axs[1].legend(prop=font, facecolor='white', framealpha=1)

matplotlib.rc('text', usetex=False)

matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

axs[1].set_ylim(-50,5)

# axs[1].annotate(r'$\mathbf{f_c}$', xy=(high_freq, 0), xytext=(high_freq,-0.55),horizontalalignment='center', verticalalignment='center', fontsize=12,weight='bold')

axs[1].annotate(r'$\mathbf{-3dB}$', xy=(0, np.sqrt(0.5)), xytext=(-9.3,-3),horizontalalignment='center', verticalalignment='center', fontsize=10,weight='bold')


for order in [1, 3, 6]:
    numerator, denumerator = scipy.signal.butter(order, high_freq, fs=fs, btype='low')
    w, spect_amp = scipy.signal.freqz(numerator, denumerator, fs=fs, worN=2000)
    axs[0].plot(w, (abs(spect_amp)), label=EEG_handler.EEGHandler.farsi_reshape("مرتبه") + " = %d" % order)


axs[0].plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], linestyle='--', label=EEG_handler.EEGHandler.farsi_reshape("تضعیف گین"))

ideal = np.linspace(0, 0.5*fs, len(w))

mask = (ideal <=high_freq)

ideal = np.where(mask, 1, 0)

axs[0].plot(w, ideal, linestyle='--', label=EEG_handler.EEGHandler.farsi_reshape("فیلتر ایده‌آل"),color='black')

axs[0].set_xlabel(EEG_handler.EEGHandler.farsi_reshape("فرکانس (هرتز)"), fontfamily="B Nazanin", fontsize=12)

axs[0].set_ylabel(EEG_handler.EEGHandler.farsi_reshape("گین"), fontfamily="B Nazanin", fontsize=12)

axs[0].grid(True)

font = matplotlib.font_manager.FontProperties(family='B Nazanin',
                                              weight='bold',
                                              style='normal', size=14)
axs[0].legend(prop=font, facecolor='white', framealpha=1)

matplotlib.rc('text', usetex=False)

matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

# axs[0].set_ylim(-50,5)

axs[0].annotate(r'$\mathbf{= f_c}$', xy=(high_freq, 0), xytext=(high_freq,-0.185),horizontalalignment='center', verticalalignment='center', fontsize=12,weight='bold')

axs[0].annotate(r'$\mathbf{\sqrt{0.5}}$', xy=(0, np.sqrt(0.5)), xytext=(-9.3,np.sqrt(0.5)),horizontalalignment='center', verticalalignment='center', fontsize=10,weight='bold')

fig.savefig('toGIT/SVG_figures/3_hpf-both.svg')
