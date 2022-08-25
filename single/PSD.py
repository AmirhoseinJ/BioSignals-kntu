from toGIT.single import EEG_handler
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


raw = EEG_handler.EEGHandler().create_raw().pick(4)

raw_notch50 = raw.copy().notch_filter(60)

fig, axs = plt.subplots(2,figsize=(16, 9))

raw.plot_psd(fmax=110, ax=axs[0])

axs[0].set_title(EEG_handler.EEGHandler.farsi_reshape("چگالی طیفی توان، قبل از اعمال فیلتر ناچ"), fontfamily="B Nazanin", fontsize=22, color="#2d1754", y=1.03)

raw_notch50.plot_psd(fmax=110, ax=axs[1])

axs[1].set_title(EEG_handler.EEGHandler.farsi_reshape("چگالی طیفی توان، بعد از اعمال فیلتر ناچ در 60 هرتز"), fontfamily="B Nazanin", fontsize=22, color="#2d1754", y=1.03)

fig.savefig('toGIT/SVG_figures/3_psd.svg')