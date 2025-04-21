import numpy
import pandas
import os
import string
from matplotlib import pyplot as plt
from itertools import product
from matplotlib.transforms import Bbox

plt.rcParams["font.size"] = 8

dpi_figures = 600

EGU = False

observed_scenario = True

if EGU:
    fontSizeAxes = 12
else:
    fontSizeAxes = 10

scenarioDirectory = '../data/scenarios/runs_from_sonic_velocity/kals_model_fit_on_observations/results/'
#scenarioDirectory = '../data/scenarios/runs_from_sonic_velocity/kals_model_fit_on_arti_data/results/'

# no error in streamflow (replaced) and precipitation and temperature copied into the results folder
#scenarioDirectory = "../data/scenarios/runs_from_sonic_velocity/kals_model_fit_on_arti_data_with_error_subf_val_noerror/results/"

#scenarioDirectory = "../data/scenarios/runs_from_sonic_velocity/kals_model_fit_on_arti_data_with_error/results/"  # note that the streamflow for validation is with error

figure_directory = "../figures/"

# only nn scenarios
scenarios = [
    "fit_eva",
    "fit_sno",
    "fit_sub",
    "fit_sne",
    "fit_sue",
    "fit_sus",
    "fit_thr",
    "fit_exp",
]

## all scenarios
#scenarios = ['fit_eva', 'fit_sno', 'fit_sub', 'fit_sne', 'fit_sue', 'fit_sus', 'fit_thr', \
#             'fit_xva', 'fit_xno', 'fit_xub', 'fit_xne', 'fit_xue', 'fit_xus', 'fit_xhr']

## only nn scenarios
scenarios_to_plot = [
    "fit_eva",
    "fit_sno",
    "fit_sub",
    "fit_sne",
    "fit_sue",
    "fit_sus",
    "fit_thr",
    "fit_exp"     # or xhr for observational data
]

if EGU:
    #scenarios = ["fit_sno", "fit_eva", "fit_thr"]
    scenarios = ["fit_thr"]
    scenarios_to_plot = scenarios

def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.4 + pad, 1.4 + pad)

# only exp scenarios
# scenarios_to_plot = ['fit_xva', 'fit_xno', 'fit_xub', 'fit_xne', 'fit_xue', 'fit_xus', 'fit_xhr']

trainingScenarios = ["1", "2", "3", "4"]

# rerun scenarios
numberOfScenarios = 2
aRange = numpy.arange(1, numberOfScenarios + 1)
reRunScenarios = []
for s in aRange:
    reRunScenarios.append(str(s))

folderWithArrays = (
    scenarioDirectory
    + "/"
    + scenarios[0]
    + "/"
    + trainingScenarios[0]
    + "/"
    + reRunScenarios[0]
    + "/arrays"
)
arrayFiles = os.listdir(folderWithArrays)


arrays = []
for arrayFile in arrayFiles:
    arrays.append(arrayFile.split(".")[0])

df = pandas.DataFrame()
pandas.set_option("display.precision", 7)


# load the scenarios, the training scenarios and the rerun scenario numbers
scList = []
tsList = []
rsList = []

for sc, ts, rs in product(scenarios, trainingScenarios, reRunScenarios):
    scList.append(sc)
    tsList.append(int(ts))
    rsList.append(int(rs))
df["sc"] = scList
df["ts"] = tsList
df["rs"] = rsList

for array in arrays:
    arrayContents = []
    for sc, ts, rs in product(scenarios, trainingScenarios, reRunScenarios):
        folder = scenarioDirectory + sc + "/" + ts + "/" + rs + "/arrays/"
        arrayName = folder + array + ".npy"
        arrayContent = numpy.load(arrayName, allow_pickle=True)
        arrayContents.append(arrayContent)
    df[array] = arrayContents

df["lossTrainingValue"] = df["lossTraining"].apply(lambda x: x[-1])
df["lossValidationValue"] = df["lossValidation"].apply(lambda x: x[-1])
df["lossStoppingValue"] = df["lossStopping"].apply(lambda x: x[-1])


# for calculation of nash sutcliffe
# remove first year (366) which was also not used for training
# assumes observed q for validation is the same across scenarios of course
valid_mean_q = (df["valid_ts_OBS"].apply(lambda x: x[366:].mean()))[0]
valid_q = df["valid_ts_OBS"][0][366:]
valid_ss_q_mean = ((valid_q - valid_mean_q) ** 2.0).mean()
df["NSEVal"] = 1.0 - (df["lossValidationValue"] / valid_ss_q_mean)

# colors
green = "#4daf4a"
blue = "#377eb8"
red = "#e41a1c"
purple = "#984ea3"
df["color"] = numpy.where(df["ts"] == 1, purple, "1")
df["color"] = numpy.where(df["ts"] == 2, red, df.color)
df["color"] = numpy.where(df["ts"] == 3, green, df.color)
df["color"] = numpy.where(df["ts"] == 4, blue, df.color)
print(df)

# a = df[ (df['sc'] == 'fit_sub') & (df['ts'] == 1) & df['rs'] == 1]


def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_axes['x'].join(target, ax)
        if sharey:
            target._shared_axes['y'].join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1,:].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:,1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)


###################
# response curves #
###################

fig = plt.figure(dpi=dpi_figures)
gs = fig.add_gridspec(8, 3, hspace=0, wspace=0)
fig, axs = plt.subplots(8, 3, sharex="col", sharey=True)
fig.set_size_inches(8.27, 11.69)
rij = 0
for sc in scenarios_to_plot:
    a = df[(df["sc"] == sc)]
    for index, row in a.iterrows():
        # if row['lossValidationValue'] < 8e-6:
        # if row['lossValidationValue'] < 1e-7:
        #if row["lossTrainingValue"] < 7.3e-6:
        if row["lossTrainingValue"] < 0.0000014:
            axs[rij, 0].plot(
                row["response_eva_x"], row["response_eva_y"], color=row["color"]
            )
            axs[rij, 1].plot(
                row["response_sno_x"], row["response_sno_y"], color=row["color"]
            )
            axs[rij, 2].plot(
                row["response_sub_x"], row["response_sub_y"], color=row["color"]
            )
        axs[rij, 0].set_yticks([0.0, 0.005, 0.010, 0.015])
        axs[rij, 0].set_yticklabels([0.0, 0.005, 0.010, 0.015], size=fontSizeAxes)
        #axs[rij, 0].set_yticklabels([0.0, 0.005, 0.010, 0.015])
    rij += 1
axs[0, 0].set_ylim(0, 0.015)
axs[0, 2].set_xlim(0, 0.4)
axs[7, 0].set_xlabel("temperature (C)", fontsize=fontSizeAxes)
axs[7,0].xaxis.set_tick_params(labelsize=fontSizeAxes)
axs[7,1].xaxis.set_tick_params(labelsize=fontSizeAxes)
axs[7,2].xaxis.set_tick_params(labelsize=fontSizeAxes)
axs[7, 1].set_xlabel("temperature (C)", fontsize=fontSizeAxes)
axs[7, 2].set_xlabel("storage (m)", fontsize=fontSizeAxes)
for i in range(0, 8):
    axs[i, 0].set_ylabel("flux (m/day)", fontsize=fontSizeAxes)
axs[0, 0].set_title("evapotranspiration\nincluding sublimation", fontsize=fontSizeAxes)
axs[0, 1].set_title("snow melt", fontsize=fontSizeAxes)
axs[0, 2].set_title("outflow subsurf. storage", fontsize=fontSizeAxes)
if EGU:
    print('')
else:
    plt.subplots_adjust(wspace=0, hspace=0)
fig.savefig(figure_directory + "response.pdf")
plt.close(fig)

##########################
# modelled vs artificial #
##########################

modelled_tss_list = [
    "valid_ts_eva_f",
    "valid_ts_sno_f",
    "valid_ts_sno_s",
    "valid_ts_sub_f",
    "valid_ts_sub_s",
    "valid_ts_sub_f",
]

if observed_scenario:
    observed_tss_list = [
        "val_art_ts_eva_f",
        "val_art_ts_sno_f",
        "val_art_ts_sno_s",
        "val_art_ts_sub_f",
        "val_art_ts_sub_s",
        "valid_ts_OBS"
    ]
else:
    observed_tss_list = [
        "val_art_ts_eva_f",
        "val_art_ts_sno_f",
        "val_art_ts_sno_s",
        "val_art_ts_sub_f",
        "val_art_ts_sub_s",
        "val_art_ts_sub_f"
    ]

tssVariables = len(observed_tss_list)

# timeseries

def timeseries_plot_by_variable(scenarios, modelledTss, observedTss, start, end):
    fig = plt.figure(dpi=dpi_figures)
    gs = fig.add_gridspec(9, 3, hspace=0, wspace=0)
    #fig, axs = plt.subplots(9, 1, sharex="col", sharey=True)
    fig, axs = plt.subplots(9, 1)
    set_share_axes(axs[1:], sharex=True)
    set_share_axes(axs[1:], sharey=True)
    fig.set_size_inches(8.27, 11.69)
    rij = 1
    # print temperature and precipitation in first row
    # get the first scenario and the first run from that scenario
    firstSc = (df[df["sc"] == scenarios[0]]).iloc[0]
    axs[0].bar(
            firstSc["valid_date"][start:end],
            firstSc["valid_ts_precipitation"][start:end],
            linewidth = 1.5,
            color = "blue"
            )
    axs[0].yaxis.set_tick_params(labelsize=fontSizeAxes)
    axs[0].set(xticklabels=[])
    axTemp = axs[0].twinx()
    axTemp.plot(
            firstSc["valid_date"][start:end],
            firstSc["valid_ts_temperature"][start:end] - 0.93,
            linewidth = 1.0,
            color = "black"
            )
    axTemp.plot(
            firstSc["valid_date"][start:end],
            (firstSc["valid_ts_temperature"][start:end] - 0.93)/100000,
            linestyle = 'dashed',
            linewidth = 0.5,
            color = "black",
            )
    axTemp.yaxis.set_tick_params(labelsize=fontSizeAxes)
    #axs[1]._shared_axes['y'].remove(axs[0])
    # print rest in remaining rows
    for sc in scenarios:
        a = (df[df["sc"] == sc].sort_values(by="lossTrainingValue")).iloc[0]
        axs[rij].plot(
            a["valid_date"][start:end],
            a[observedTss][start:end],
            linewidth=1.0,
            color=green
        )
        axs[rij].plot(
            a["valid_date"][start:end],
            a[modelledTss][start:end],
            linewidth=1.0,
            color="black"
        )
        axs[rij].yaxis.set_tick_params(labelsize=fontSizeAxes)
        axs[rij].locator_params(axis='y', nbins=2)
        axs[rij].xaxis.set_tick_params(labelsize=fontSizeAxes, labelrotation = 30)
        rij += 1
    for i in range(0, 8):
        if observedTss[-1] == "f":
            axs[i].set_ylabel("flux (m/day)", size=fontSizeAxes)
        else:
            axs[i].set_ylabel("storage (m)", size=fontSizeAxes)
    #axs[1]._shared_axes['y'].remove(axs[0])
    axs[0].set_ylabel("precipitation\n(m/day)", size=fontSizeAxes, color = 'blue')
    axTemp.set_ylabel("temperature\n(C)", size=fontSizeAxes)
    axs[8].xaxis.set_tick_params(labelsize=fontSizeAxes, labelrotation = 30)
    plt.subplots_adjust(wspace=0, hspace=0)
    if EGU:
        axs[2].remove()
        axs[3].remove()
        axs[4].remove()
        axs[5].remove()
        axs[6].remove()
        axs[7].remove()
        axs[8].remove()
    fig.savefig(figure_directory + "tss_modartcomp_" + observedTss + ".pdf")
    plt.close(fig)

def timeSeriesPlotByScenario(modelledTssEs, observedTssEs, scenario, start, end):
    fig = plt.figure(dpi=dpi_figures)
    gs = fig.add_gridspec(7, 3, hspace=0, wspace=0)
    fig, axs = plt.subplots(7, 1)
    set_share_axes(axs[1:], sharex=True)
    fig.set_size_inches(8.27, 11.69)
    rij = 1
    # print temperature and precipitation in first row
    # get the first scenario and the first run from that scenario
    firstSc = (df[df["sc"] == scenario]).iloc[0]
    axs[0].bar(
            firstSc["valid_date"][start:end],
            firstSc["valid_ts_precipitation"][start:end],
            linewidth = 1.5,
            color = "blue"
            )
    axs[0].yaxis.set_tick_params(labelsize=fontSizeAxes)
    axs[0].set(xticklabels=[])
    axTemp = axs[0].twinx()
    axTemp.plot(
            firstSc["valid_date"][start:end],
            firstSc["valid_ts_temperature"][start:end] - 0.93,
            linewidth = 1.0,
            color = "black"
            )
    axTemp.plot(
            firstSc["valid_date"][start:end],
            (firstSc["valid_ts_temperature"][start:end] - 0.93)/100000,
            linestyle = 'dashed',
            linewidth = 0.5,
            color = "black",
            )
    axTemp.yaxis.set_tick_params(labelsize=fontSizeAxes)
    #axs[1]._shared_axes['y'].remove(axs[0])
    # Plot model output in rows starting in row 2.
    tssNumber = 0
    for tss in modelledTssEs:
        for i in range(0,4):
            a = (df[df["sc"] == scenario].sort_values(by="lossTrainingValue")).iloc[i]
            # Plot observed timeseries either from artificial data or from observations.
            observed_tss = observedTssEs[tssNumber]
            if not(observed_scenario) or (observed_tss == 'valid_ts_OBS'):
                axs[rij].plot(
                    a["valid_date"][start:end],
                    a[observed_tss][start:end],
                    linewidth = 0.5,
                    #color=green
                    color='black'
                )
            # Plot modelled timeseries.
            if i == 0:
                line_width = 2.0
                line_style = 'solid'
            else:
                line_width = 0.5
                line_style = 'dashed'
            axs[rij].plot(
                a["valid_date"][start:end],
                a[tss][start:end],
                #linewidth=1.0,
                linewidth = line_width,
                linestyle = line_style,
                #color="black"
                color = a["color"]
            )
        axs[rij].yaxis.set_tick_params(labelsize=fontSizeAxes)
        axs[rij].locator_params(axis='y', nbins=2)
        axs[rij].xaxis.set_tick_params(labelsize=fontSizeAxes, labelrotation = 30)
        if rij < 6:
            axs[rij].set(xticklabels=[])
        if tss[-1] == "f":
            axs[rij].set_ylabel("flux (m/day)", size=fontSizeAxes)
        else:
            axs[rij].set_ylabel("storage (m)", size=fontSizeAxes)
        rij += 1
        tssNumber += 1
    axs[0].set_ylabel("precipitation\n(m/day)", size=fontSizeAxes, color = 'blue')
    axTemp.set_ylabel("temperature (C)", size=fontSizeAxes)
    axs[6].xaxis.set_tick_params(labelsize=fontSizeAxes, labelrotation = 30)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(figure_directory + "tss_modartcomp_" + scenario + ".pdf")
    plt.close(fig)


startTimeTss = 2 * 365
endTimeTss = 4 * 365

i = 0
while i < tssVariables:
    modelledTss = modelled_tss_list[i]
    observedTss = observed_tss_list[i]
    timeseries_plot_by_variable(scenarios_to_plot, modelledTss, observedTss, startTimeTss, endTimeTss)
    i = i + 1

i = 0
for scenario in scenarios_to_plot:
    timeSeriesPlotByScenario(modelled_tss_list, observed_tss_list, scenario, startTimeTss, endTimeTss)
    i = i + 1


# scatterplots

endTimeTss = len(observed_tss_list[0])
# endTimeTss = 6 * 365 # len(df['valid_ts_OBS']) - 1


def rSquaredFormatted(x, y):
    rM = numpy.corrcoef(x, y)
    r = rM[0][1]
    rSq = r * r
    rSqFor = "{:.3f}".format(rSq)
    return rSqFor


def scatterPlot(scenarios, modelledTss, observedTss, start, end):
    fig = plt.figure(dpi=dpi_figures)
    # gs = fig.add_gridspec(8, 3, hspace=0, wspace=0)
    # fig, axs = plt.subplots(8, 1, sharex='col', sharey = True)
    fig, axs = plt.subplots(8, 1)
    fig.set_size_inches(8.27, 11.69)
    if EGU:
        print('')
    else:
        fig.subplots_adjust(hspace=0.1)
    # fig.set_size_inches(8.27/3.0,11.69)
    rij = 0
    for sc in scenarios:
        a = (df[df["sc"] == sc].sort_values(by="lossTrainingValue")).iloc[0]
        x = a[observedTss][start:end]
        y = a[modelledTss][start:end]
        hb = axs[rij].hexbin(
            x, y, gridsize=20, cmap="Greens", bins="log", linewidths=0.0
        )
        fig.colorbar(hb, ax=axs[rij], pad=0.01)
        axs[rij].plot([0, 10], [0, 10], color="black", linewidth=0.5)
        rSqFor = rSquaredFormatted(x, y)
        axs[rij].text(
            0.99, 0.01, rSqFor, ha="right", va="bottom", transform=axs[rij].transAxes
        )
        # axs[rij].scatter(a[observedTss][start:end], a[modelledTss][start:end], s = 0.5)
        axs[rij].set_ylim(0, max(a[observedTss][start:end]))
        axs[rij].set_xlim(0, max(a[observedTss][start:end]))
        if rij < len(scenarios) - 1:
            axs[rij].set(xticklabels=[])
        axs[rij].set_aspect("equal")
        rij += 1
    # plt.subplots_adjust(wspace=0, hspace=0)
    if EGU:
        axs[1].remove()
        axs[2].remove()
        axs[3].remove()
        axs[4].remove()
        axs[5].remove()
        axs[6].remove()
        axs[7].remove()
    if EGU:
        # Save just the portion _inside_ the second axis's boundaries
        extent = full_extent(axs[0]).transformed(fig.dpi_scale_trans.inverted())
        # Alternatively,
        #extent = axs[0].get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(figure_directory + "sca_modartcomp_" + observedTss + ".pdf", bbox_inches=extent)
    else:
        fig.savefig(figure_directory + "sca_modartcomp_" + observedTss + ".pdf")
    plt.close(fig)


startTimeTss = 2 * 365
endTimeTss = 6 * 365  # len(df['valid_ts_OBS']) - 1

i = 0
while i < tssVariables:
    modelledTss = modelled_tss_list[i]
    observedTss = observed_tss_list[i]
    scatterPlot(scenarios_to_plot, modelledTss, observedTss, startTimeTss, endTimeTss)
    i = i + 1


# df['lossStop'] = lossStopList
# df['lossTest'] = lossValiList

# a = df[ (df['sc'] == 'fit_sub') & (df ['ts'] == 1)]

##
# scenarios = ['fit_eva', 'fit_sno', 'fit_sub', 'fit_sne', 'fit_sue', 'fit_sus', 'fit_thr', 'fit_exp']

print("#########################################################")

variables = [
    "sc",
    "ts",
    "rs",
    "lossTrainingValue",
    "lossStoppingValue",
    "lossValidationValue",
    "NSEVal",
]
# print(df[df['sc'] == 'fit_eva'].sort_values(by="lossTrainingValue").loc[:,['sc','ts','rs','lossTrainingValue', 'lossStoppingValue','lossValidationValue', 'NSEVal']])
for scen in scenarios:
    print(df[df["sc"] == scen].sort_values(by="lossTrainingValue").loc[:, variables])

def nsePlot():
    fig = plt.figure(dpi=dpi_figures, figsize = [2.5,2], tight_layout = {'pad': 1})
    #fig.set_size_inches(8.27/4, 11.69/4)

    scenarios_to_plot = [
        "fit_eva",
        "fit_sno",
        "fit_sub",
        "fit_sne",
        "fit_sue",
        "fit_sus",
        "fit_thr"
        ]

    scenariosToPlotExp = [
        'fit_xva',
        'fit_xno',
        'fit_xub',
        'fit_xne',
        'fit_xue',
        'fit_xus',
        'fit_xhr'
        ]

    names = [
           "E",
           "S",
           "G" ,
           "ES",
           "EG",
           "SG",
           "ESG"
        ]


    i = 0
    for scen in scenarios_to_plot:
        nse = (df[df["sc"] == scen].sort_values(by="lossTrainingValue")).iloc[0]["NSEVal"]
        if i == 0:
            plt.plot(names[i], nse, '.', color="black", label = "neural network")
        else:
            plt.plot(names[i], nse, '.', color="black")
        i += 1

    i = 0
    for scen in scenariosToPlotExp:
        nse = (df[df["sc"] == scen].sort_values(by="lossTrainingValue")).iloc[0]["NSEVal"]
        if i == 0:
            plt.plot(names[i], nse, '+', color="black", label = "expert model")
        else:
            plt.plot(names[i], nse, '+', color="black")
        i += 1
    plt.legend()
    plt.ylabel("NSE")
    fig.savefig(figure_directory + "nse.pdf")
    plt.close(fig)


#nsePlot()
