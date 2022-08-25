from toGIT.single import EEG_handler
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Qt5Agg')

array = EEG_handler.EEGHandler().create_raw()[0][0][0][256 * 20:256 * 40]
n = len(array)
smoothed_array = [None] * n
k = 100
for i in range(k, n - k):
    smoothed_array[i - 1] = np.mean(array[i - k:i + k + 1])
# smoothed_signal = [x for x in smoothed_signal if x is not None]

plt.figure(figsize=(16, 9))
plt.plot(array, label=EEG_handler.EEGHandler.farsi_reshape("داده اصلی"), linewidth=1)
plt.plot(smoothed_array, label=EEG_handler.EEGHandler.farsi_reshape("داده هموار شده"), linestyle="-", linewidth=3)

plt.suptitle(EEG_handler.EEGHandler.farsi_reshape("روش میانگین متحرک"), fontfamily="B Nazanin", fontsize=22,
             color="#2d1754", y=0.96)
font = matplotlib.font_manager.FontProperties(family='B Nazanin',
                                              weight='bold',
                                              style='normal', size=12)
plt.legend(prop=font, facecolor='white', framealpha=1)
plt.xlabel(EEG_handler.EEGHandler.farsi_reshape("شماره نمونه"), font="B Nazanin", fontsize=12)
plt.ylabel(EEG_handler.EEGHandler.farsi_reshape("اندازه نمونه"), font="B Nazanin", fontsize=12)
plt.tight_layout()
plt.show()
plt.savefig('toGIT/SVG_figures/3-smooth.svg')


