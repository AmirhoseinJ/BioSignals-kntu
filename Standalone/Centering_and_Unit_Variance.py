from toGIT.single import EEG_handler
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')
import numpy as np

###### Centering
fig, ax = plt.subplots(1,figsize=(16, 9))

arr1 = np.random.random_sample(50)

centered = (arr1 - np.mean(arr1))

ax.plot(arr1, label=EEG_handler.EEGHandler.farsi_reshape("داده اصلی"))

ax.plot(centered, label=EEG_handler.EEGHandler.farsi_reshape("داده مرکزی‌سازی شده"))
font = matplotlib.font_manager.FontProperties(family='B Nazanin',
                                              weight='bold',
                                              style='normal', size=12)

ax.axhline(0, linestyle="--", color='black')

ax.legend(prop=font, facecolor='white', framealpha=1)

ax.set_xlabel(EEG_handler.EEGHandler.farsi_reshape("شماره نمونه"), font="B Nazanin", fontsize=12)

ax.set_ylabel(EEG_handler.EEGHandler.farsi_reshape("اندازه نمونه"), font="B Nazanin", fontsize=12)

fig.tight_layout()

fig.savefig('toGIT/SVG_figures/3-centering.svg')

###### Unit-V
fig, ax = plt.subplots(1,figsize=(16, 9))

arr1 = np.random.uniform(-1, 8, 50)

centered = (arr1 - np.mean(arr1))

unit_var = centered / np.std(arr1 - np.mean(arr1))

ax.plot(arr1, label=EEG_handler.EEGHandler.farsi_reshape("داده اصلی"))

ax.plot(unit_var, label=EEG_handler.EEGHandler.farsi_reshape("داده نرمال شده (واریانس واحد)"))

font = matplotlib.font_manager.FontProperties(family='B Nazanin',
                                              weight='bold',
                                              style='normal', size=12)

ax.axhline(np.max(unit_var), linestyle="--", color='black', linewidth=0.7)

ax.axhline(np.min(unit_var), linestyle="--", color='black', linewidth=0.7)

ax.legend(prop=font, facecolor='white', framealpha=1)

ax.set_xlabel(EEG_handler.EEGHandler.farsi_reshape("شماره نمونه"), font="B Nazanin", fontsize=12)

ax.set_ylabel(EEG_handler.EEGHandler.farsi_reshape("اندازه نمونه"), font="B Nazanin", fontsize=12)

fig.tight_layout()

fig.savefig('toGIT/SVG_figures/3-unit-V.svg')
