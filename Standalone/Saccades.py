from toGIT.single import EEG_handler
import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans


matplotlib.use('Qt5Agg')

labels = scipy.io.loadmat(
    '/run/media/****/Users/**/Desktop/prj/EOG_saccades/DATASET/S1/ControlSignal.mat')
# np.set_printoptions(threshold=1000)
labels = labels['ControlSignal'][0]

data = scipy.io.loadmat('/run/media/****/Users/**/Desktop/prj/EOG_saccades/DATASET/S1/EOG.mat')
data = data['EOG']

angles = scipy.io.loadmat(
    '/run/media/****/Users/**/Desktop/prj/EOG_saccades/DATASET/S1/TargetGA.mat')
angles = angles['TargetGA']

# extract ranges for highlighting
pos_1s = np.argwhere(labels == 1)
pos_2s = np.argwhere(labels == 2)
pos_3s = np.argwhere(labels == 3)
ranges_1 = []
ranges_2 = []
ranges_3 = []
start = 0
for i in range(1, len(pos_1s)):
    prev_pos = i - 1
    if pos_1s[i][0] - pos_1s[prev_pos][0] != 1:
        ranges_1.append([pos_1s[start][0], pos_1s[prev_pos][0]])
        start = i
    if i == len(pos_1s) - 1:
        ranges_1.append([pos_1s[start][0], pos_1s[i][0]])
start = 0
for i in range(1, len(pos_2s)):
    prev_pos = i - 1
    if pos_2s[i][0] - pos_2s[prev_pos][0] != 1:
        ranges_2.append([pos_2s[start][0], pos_2s[prev_pos][0]])
        start = i
    if i == len(pos_2s) - 1:
        ranges_2.append([pos_2s[start][0], pos_2s[i][0]])
start = 0
for i in range(1, len(pos_3s)):
    prev_pos = i - 1
    if pos_3s[i][0] - pos_3s[prev_pos][0] != 1:
        ranges_3.append([pos_3s[start][0], pos_3s[prev_pos][0]])
        start = i
    if i == len(pos_3s) - 1:
        ranges_3.append([pos_3s[start][0], pos_3s[i][0]])

# angles
angles_h = []
angles_v = []
for angle in angles:
    if angle[0] != 0.0:
        angles_h.append(angle[0])
    if angle[1] != 0.0:
        angles_v.append(angle[1])

# begin plotting
duration = 256 * 16
times = np.arange(0, len(data[0][:duration])) / 256.0
# titles = ["Horizontal EOG", "Vertical EOG"]
titles = [EEG_handler.EEGHandler.farsi_reshape("کانال افقی"), EEG_handler.EEGHandler.farsi_reshape('کانال عمودی')]
xlabel = EEG_handler.EEGHandler.farsi_reshape("زمان (ثانیه)")
ylabel = EEG_handler.EEGHandler.farsi_reshape("اندازه")
data_label = "Eye movement EOG Data [24] "

fig, axs = plt.subplots(2, squeeze=False, figsize=(16, 9))
# fig.suptitle("EOG", fontfamily="Roboto", fontsize=25, color="purple")
for i in range(2):
    with plt.style.context("seaborn"):
        with plt.rc_context({'lines.linewidth': 1.2, 'lines.linestyle': '-', 'font.family': ["Roboto"],
                             'legend.frameon': True}):
            font = matplotlib.font_manager.FontProperties(family='Roboto',
                                                          weight='bold',
                                                          style='normal', size=10)
            axs.flat[i].plot(times, data[i][:duration], color="black")
            axs.flat[i].set_title(titles[i], fontsize=18, fontweight='bold', color="k",
                                  pad=10, fontfamily="B Nazanin")
            axs.flat[i].set_xlabel(xlabel, fontfamily="B Nazanin", fontsize=15, labelpad=10,
                                   color="black")
            # axs.flat[i].set_ylabel(ylabel, fontfamily="B Nazanin", fontsize=15, labelpad=10,
            #                        color="green")

            # axs.flat[i].set_ylim(ylims[i])
            # axs.flat[i].set_xlim(xlims[i])

            axs.flat[i].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
            axs.flat[i].annotate(data_label, xy=(times[0], 0), xytext=(1, 0), xycoords='data',
                                 textcoords='axes fraction',
                                 horizontalalignment='right', verticalalignment='bottom', fontsize=8)

            axvspan_num = 0
            for range1, range2, range3 in zip(ranges_1, ranges_2, ranges_3):
                angle_annot_point_x = ((range1[1] - range1[0]) / 512) + range1[0] / 256
                print(angle_annot_point_x)
                if not range1[0] < duration:
                    break
                if axvspan_num == 0:
                    axs.flat[i].axvspan(range1[0] / 256, range1[1] / 256, color="red", alpha=0.5,
                                        label="Forward Saccade")
                    axs.flat[i].axvspan(range2[0] / 256, range2[1] / 256, color="yellow", alpha=0.5,
                                        label="Return Saccade")
                    axs.flat[i].axvspan(range3[0] / 256, range3[1] / 256, color="blue", alpha=0.5, label="Blink")
                    axs.flat[i].annotate("angle_H: " + "%.2f" % angles_h[int(axvspan_num)],
                                         xy=(angle_annot_point_x, 450),
                                         horizontalalignment='center', verticalalignment='bottom', fontsize=8)
                    if i == 1:
                        axs.flat[i].annotate("angle_V: " + "%.2f" % angles_v[int(axvspan_num)],
                                             xy=(angle_annot_point_x, 350),
                                             horizontalalignment='center', verticalalignment='bottom', fontsize=8)
                    else:
                        axs.flat[i].annotate("angle_V: " + "%.2f" % angles_v[int(axvspan_num)],
                                             xy=(angle_annot_point_x, 400),
                                             horizontalalignment='center', verticalalignment='bottom', fontsize=8)
                    axvspan_num += 1
                else:
                    # https://stackoverflow.com/questions/44632903/setting-multiple-axvspan-labels-as-one-element-in-legend
                    axs.flat[i].axvspan(range1[0] / 256, range1[1] / 256, color="red", alpha=0.5,
                                        label="_Forward Saccade")
                    axs.flat[i].axvspan(range2[0] / 256, range2[1] / 256, color="yellow", alpha=0.5,
                                        label="_Return Saccade")
                    axs.flat[i].axvspan(range3[0] / 256, range3[1] / 256, color="blue", alpha=0.5, label="_Blink")
                    axs.flat[i].annotate("angle_H: " + "%.2f" % angles_h[int(axvspan_num)],
                                         xy=(angle_annot_point_x, 450),
                                         horizontalalignment='center', verticalalignment='bottom', fontsize=8)
                    if i == 1:
                        axs.flat[i].annotate("angle_V: " + "%.2f" % angles_v[int(axvspan_num)],
                                             xy=(angle_annot_point_x, 350),
                                             horizontalalignment='center', verticalalignment='bottom', fontsize=8)
                    else:
                        axs.flat[i].annotate("angle_V: " + "%.2f" % angles_v[int(axvspan_num)],
                                             xy=(angle_annot_point_x, 400),
                                             horizontalalignment='center', verticalalignment='bottom', fontsize=8)
                    axvspan_num += 1

            axs.flat[i].legend(prop=font, facecolor='white', framealpha=1, loc="upper right")
fig.tight_layout()
# Get the bounding boxes of the axes including text decorations (https://stackoverflow.com/questions/26084231/draw-a-separator-or-lines-between-subplots)
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

fig.savefig('toGIT/SVG_figures/4_saccades.svg')