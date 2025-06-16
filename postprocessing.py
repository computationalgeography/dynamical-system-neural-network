import numpy
import pandas
import os
import string
from matplotlib import pyplot as plt
from itertools import product
from matplotlib.transforms import Bbox
from matplotlib.lines import Line2D

#######################
# main configurations #
#######################

observed_scenario = False
one_area = True

create_scatter = False
create_timeseries = False
create_r2_by_variable = True
create_nse = False
print_stats = False

data_dir = '../data/scenarios/runs_from_sonic_velocity/'
number_of_rerun_scenarios = 2  # 2 for all except fitting on observations for one area (where one can use 4)


##################
# other settings #
##################

plt.rcParams["font.size"] = 8
plt.rcParams.update({'figure.max_open_warning': 0})

dpi_figures = 600

EGU = False

actual_snow_flux = True

if EGU:
    font_size_axes = 9
else:
    font_size_axes = 8 

number_of_fits_to_plot = 4

labels_variables = ['evapotranspiration',
                    'snow melt',
                    'snow storage',
                    'outflow subsurface storage (streamflow)',
                    'subsurface storage'
                    ]

labels_variables_tight = ['evapotranspiration',
                    'snow melt',
                    'snow storage',
                    'outflow subsurface\nstorage (streamflow)',
                    'subsurface storage'
                    ]



if observed_scenario:
    if one_area:
        results_folder = 'kals_model_fit_on_observations/results/'
    else:
        results_folder = 'kals_model_two_areas_fit_on_observations/results/'
    scenario_directory = data_dir + results_folder 
else:
    # No error in streamflow (replaced) and precipitation and temperature
    # copied into the results folder
    if one_area:
        results_folder = 'kals_model_fit_on_arti_data_with_error_subf_val_noerror/results/'
    else:
        results_folder = 'kals_model_two_areas_fit_on_artificial_data_with_error/results/'
        #results_folder = 'kals_model_two_areas_fit_on_artificial_data/results/'
    scenario_directory = data_dir + results_folder
    # Note that the streamflow for validation is with error
    #scenario_directory = data_dir + "kals_model_fit_on_arti_data_with_error/results/"

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

if observed_scenario:
    # all scenarios
    if one_area:
        scenarios = ['fit_eva', 'fit_sno', 'fit_sub', 'fit_sne', 'fit_sue', 'fit_sus', 'fit_thr', \
                 'fit_xva', 'fit_xno', 'fit_xub', 'fit_xne', 'fit_xue', 'fit_xus', 'fit_xhr']
    else:
        scenarios = ['fit_eva', 'fit_sno', 'fit_sub', 'fit_sne', 'fit_sue', 'fit_sus', 'fit_thr', \
                 'fit_xva', 'fit_xno', 'fit_xub', 'fit_xne', 'fit_xue', 'fit_xus', 'fit_xhr']

    ## only nn scenarios
    if one_area:
        scenarios_to_plot = [
            "fit_eva",
            "fit_sno",
            "fit_sub",
            "fit_sne",
            "fit_sue",
            "fit_sus",
            "fit_thr",
            "fit_xhr"  
        ]
    else:
        scenarios_to_plot = [
            "fit_eva",
            "fit_sno",
            "fit_sub",
            "fit_sne",
            "fit_sue",
            "fit_sus",
            "fit_thr",
            "fit_xhr"  
        ]

else:
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

names = [
           "E",
           "S",
           "G" ,
           "ES",
           "EG",
           "SG",
           "ESG",
           "Exp"
        ]

#if EGU:
#    #scenarios = ["fit_sno", "fit_eva", "fit_thr"]
#    scenarios = ["fit_thr"]
#    scenarios_to_plot = scenarios

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

training_scenarios = ["1", "2", "3", "4"]

# rerun scenarios
#number_of_rerun_scenarios = 2
aRange = numpy.arange(1, number_of_rerun_scenarios + 1)
rerun_scenarios = []
for s in aRange:
    rerun_scenarios.append(str(s))

folderWithArrays = (
    scenario_directory
    + "/"
    + scenarios[0]
    + "/"
    + training_scenarios[0]
    + "/"
    + rerun_scenarios[0]
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

for sc, ts, rs in product(scenarios, training_scenarios, rerun_scenarios):
    scList.append(sc)
    tsList.append(int(ts))
    rsList.append(int(rs))
df["sc"] = scList
df["ts"] = tsList
df["rs"] = rsList

for array in arrays:
    arrayContents = []
    for sc, ts, rs in product(scenarios, training_scenarios, rerun_scenarios):
        folder = scenario_directory + sc + "/" + ts + "/" + rs + "/arrays/"
        arrayName = folder + array + ".npy"
        arrayContent = numpy.load(arrayName, allow_pickle=True)
        arrayContents.append(arrayContent)
    df[array] = arrayContents

df["lossTrainingValue"] = df["lossTraining"].apply(lambda x: x[-1])
df["lossValidationValue"] = df["lossValidation"].apply(lambda x: x[-1])
df["lossStoppingValue"] = df["lossStopping"].apply(lambda x: x[-1])

#df["minLossValidationValue"] = df["lossValidation"].apply(lambda x: x.min())
#df["lossRatioValidationValue"] = df["minLossValidationValue"]/df["lossValidationValue"]
#
#for scen in scenarios:
#    #print(df[df["sc"] == scen].sort_values(by="lossTrainingValue").loc[:, variables])
#    a = df[df["sc"] == scen]
#    print(scen, a["lossRatioValidationValue"].mean())

# Replace potential snow melt by actual snow melt both for synthetic data
# and for ML modelled data
if actual_snow_flux:
    df['valid_ts_sno_f'] = df.apply(lambda x: \
                           numpy.where(x['valid_ts_sno_s'] < 0.0001, 0.0, \
                           x['valid_ts_sno_f']), \
                           axis=1)
    df['val_art_ts_sno_f'] = df.apply(lambda x: \
                           numpy.where(x['val_art_ts_sno_s'] < 0.0001, 0.0, \
                           x['val_art_ts_sno_f']), \
                           axis=1)


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

# load the reference artificial models for plotting
df_first = df[(df["ts"] == 1) & (df["rs"] == 1)]
response_eva_x = numpy.array(df_first[df_first["sc"] == "fit_sno"]["response_eva_x"])[0]
response_eva_y = numpy.array(df_first[df_first["sc"] == "fit_sno"]["response_eva_y"])[0]
response_sno_x = numpy.array(df_first[df_first["sc"] == "fit_eva"]["response_sno_x"])[0]
response_sno_y = numpy.array(df_first[df_first["sc"] == "fit_eva"]["response_sno_y"])[0]
response_sub_x = numpy.array(df_first[df_first["sc"] == "fit_eva"]["response_sub_x"])[0]
response_sub_y = numpy.array(df_first[df_first["sc"] == "fit_eva"]["response_sub_y"])[0]

line_width_best = 2.5
line_width_other = 0.7

rij = 0
for sc in scenarios_to_plot:
    #a = df[(df["sc"] == sc)]
    a = (df[df["sc"] == sc].sort_values(by="lossTrainingValue"))
    i = 0
    for index, row in a.iterrows():
        if i < number_of_fits_to_plot:
            if i == 1:
                label_to_print = 'modelled (2nd - 4th), all colours'
            else:
                label_to_print = ''
            if i == 0:
                label_to_print = 'modelled (best), all colours'
            # Plot the modelled response curves.
            if i == 0:
                line_width = line_width_best 
                line_style = 'solid'
            else:
                line_width = line_width_other 
                line_style = 'dashed'
            axs[rij, 0].plot(
                row["response_eva_x"], row["response_eva_y"], color=row["color"],
                linewidth = line_width,
                linestyle = line_style,
                label = label_to_print 
            )
            axs[rij, 1].plot(
                row["response_sno_x"], row["response_sno_y"], color=row["color"],
                linewidth = line_width,
                linestyle = line_style
            )
            axs[rij, 2].plot(
                row["response_sub_x"], row["response_sub_y"], color=row["color"],
                linewidth = line_width,
                linestyle = line_style
            )
            if True: # observed_scenario:
                # Clear subplots
                if row["sc"] == 'fit_eva':
                    axs[rij,1].cla()
                    axs[rij,2].cla()
                if row["sc"] == 'fit_sno':
                    axs[rij,0].cla()
                    axs[rij,2].cla()
                if row["sc"] == 'fit_sub':
                    axs[rij,0].cla()
                    axs[rij,1].cla()
                if row["sc"] == 'fit_sne':
                    axs[rij,2].cla()
                if row["sc"] == 'fit_sue':
                    axs[rij,1].cla()
                if row["sc"] == 'fit_sus':
                    axs[rij,0].cla()
            i += 1


    i = 0
    for index, row in a.iterrows():
        if i == 0:
            # Plot the reference response curves.
            n = names[rij]
            o = observed_scenario == True
            if not ((n == "E") or (n == "ES") or (n == "EG") or (n == "ESG") or (n == "Exp")) & o:
                axs[rij, 0].plot(
                    response_eva_x, response_eva_y,
                    linewidth = 1,
                    linestyle = 'solid',
                    color='black',
                    zorder = -10,
                    label = 'synthetic'
                )
            if not ((n == "S") or (n == "ES") or (n == "SG") or (n == "ESG") or (n == "Exp")) & o:
                    axs[rij, 1].plot(
                    response_sno_x, response_sno_y,
                    linewidth = 1,
                    linestyle = 'solid',
                    color='black',
                    zorder = -10
                )
            if not ((n == "G") or (n == "EG") or (n == "SG") or (n == "ESG") or (n == "Exp")) & o:
                axs[rij, 2].plot(
                    response_sub_x, response_sub_y,
                    linewidth = 1,
                    linestyle = 'solid',
                    color='black',
                    zorder = -10
                )
            if not EGU:
                # Plot the panel labels
                axs[rij,0].text(.05, .93, n, ha='left', va='top', transform=axs[rij,0].transAxes, size = font_size_axes * 1.5)


        axs[rij, 0].set_yticks([0.0, 0.010])
        axs[rij, 0].set_yticklabels([0.0, 0.010], size=font_size_axes)
        i += 1
    rij += 1
custom_lines = [Line2D([0], [0], color=green, lw=line_width_best, ls='solid'),
                Line2D([0], [0], color=green, lw=line_width_other, ls='dashed'),
                Line2D([0], [0], color='black', lw=1)]
if observed_scenario:
    legend_text = ['model, best (all colours)', 'model, 2nd-4th best (all colours)', 'conceptual']
else:
    legend_text = ['model, best (all colours)', 'model, 2nd-4th best (all colours)', 'synthetic']
axs[7,0].legend(custom_lines, legend_text, loc = 'upper center', bbox_to_anchor = (1.5, -0.5), ncol = 3)

axs[0, 0].set_ylim(0, 0.020)

axs[0, 0].set_xlim(-10, 15)
axs[0, 1].set_xlim(-2, 8)
axs[0, 2].set_xlim(0, 0.4)

axs[7, 0].set_xticks([-10, -5, 0, 5, 10])
axs[7, 0].set_xticklabels([-10, -5, 0, 5, 10], size=font_size_axes)

labels = [-2, 0, 2, 4, 6]
axs[7, 1].set_xticks(labels)
axs[7, 1].set_xticklabels(labels, size=font_size_axes)

axs[7, 0].set_xlabel("temperature ($\degree$C)", fontsize=font_size_axes)
axs[7,0].xaxis.set_tick_params(labelsize=font_size_axes)
axs[7,1].xaxis.set_tick_params(labelsize=font_size_axes)
axs[7,2].xaxis.set_tick_params(labelsize=font_size_axes)
axs[7, 1].set_xlabel("temperature ($\degree$C)", fontsize=font_size_axes)
axs[7, 2].set_xlabel("subsurface storage (m)", fontsize=font_size_axes)
for i in range(0, 8):
    axs[i, 0].set_ylabel("flux (m/day)", fontsize=font_size_axes)
axs[0, 0].set_title("evapotranspiration\nincluding sublimation", fontsize=font_size_axes)
axs[0, 1].set_title("snow melt", fontsize=font_size_axes)
axs[0, 2].set_title("outflow subsurf. storage", fontsize=font_size_axes)
if not EGU:
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
    "valid_ts_sub_s"
]

if observed_scenario:
    observed_tss_list = [
        "val_art_ts_eva_f",
        "val_art_ts_sno_f",
        "val_art_ts_sno_s",
        "valid_ts_OBS",
        "val_art_ts_sub_s"
    ]
else:
    observed_tss_list = [
        "val_art_ts_eva_f",
        "val_art_ts_sno_f",
        "val_art_ts_sno_s",
        "val_art_ts_sub_f",
        "val_art_ts_sub_s"
    ]

tss_variables = len(observed_tss_list)

# timeseries

def timeseries_plot_by_variable(scenarios, modelled_tss, observed_tss, start, end):
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
    axs[0].yaxis.set_tick_params(labelsize=font_size_axes)
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
    axTemp.yaxis.set_tick_params(labelsize=font_size_axes)
    #axs[1]._shared_axes['y'].remove(axs[0])
    # print rest in remaining rows
    for sc in scenarios:
        a = (df[df["sc"] == sc].sort_values(by="lossTrainingValue")).iloc[0]
        axs[rij].plot(
            a["valid_date"][start:end],
            a[observed_tss][start:end],
            linewidth=1.0,
            color=green
        )
        axs[rij].plot(
            a["valid_date"][start:end],
            a[modelled_tss][start:end],
            linewidth=1.0,
            color="black"
        )
        axs[rij].yaxis.set_tick_params(labelsize=font_size_axes)
        axs[rij].locator_params(axis='y', nbins=2)
        axs[rij].xaxis.set_tick_params(labelsize=font_size_axes, labelrotation = 30)
        rij += 1
    for i in range(0, 8):
        if observed_tss[-1] == "f":
            axs[i].set_ylabel("flux (m/day)", size=font_size_axes)
        else:
            axs[i].set_ylabel("storage (m)", size=font_size_axes)
    #axs[1]._shared_axes['y'].remove(axs[0])
    axs[0].set_ylabel("precipitation\n(m/day)", size=font_size_axes, color = 'blue')
    axTemp.set_ylabel("temperature\n($\degree$C)", size=font_size_axes)
    axs[8].xaxis.set_tick_params(labelsize=font_size_axes, labelrotation = 30)
    plt.subplots_adjust(wspace=0, hspace=0)
    if EGU:
        axs[2].remove()
        axs[3].remove()
        axs[4].remove()
        axs[5].remove()
        axs[6].remove()
        axs[7].remove()
        axs[8].remove()
    fig.savefig(figure_directory + "tss_modartcomp_" + observed_tss + ".pdf")
    plt.close(fig)

def timeseries_plot_by_scenario(modelled_tss_es, observed_tss_es, scenario, start, end):
    rows_in_figure = 6
    fig = plt.figure(dpi=dpi_figures)
    gs = fig.add_gridspec(rows_in_figure, 3, hspace=0, wspace=0)
    fig, axs = plt.subplots(rows_in_figure, 1)
    set_share_axes(axs[1:], sharex=True)
    fig.set_size_inches(8.27, 11.69)
    rij = 1
    # print temperature and precipitation in first row
    # get the first scenario and the first run from that scenario
    firstSc = (df[df["sc"] == scenario]).iloc[0]
    axs[0].bar(
            firstSc["valid_date"][start:end],
            firstSc["valid_ts_precipitation"][start:end]/1000,
            linewidth = 1.5,
            color = "blue"
            )
    axs[0].yaxis.set_tick_params(labelsize=font_size_axes)
    axs[0].set(xticklabels=[])
    axTemp = axs[0].twinx()
    axTemp.plot(
            firstSc["valid_date"][start:end],
            firstSc["valid_ts_temperature"][start:end] - 0.93,
            linewidth = 1.0,
            color = red 
            )
    axTemp.plot(
            firstSc["valid_date"][start:end],
            (firstSc["valid_ts_temperature"][start:end] - 0.93)/100000,
            linestyle = 'dashed',
            linewidth = 0.5,
            color = red,
            )
    axTemp.yaxis.set_tick_params(labelsize=font_size_axes)
    #axs[1]._shared_axes['y'].remove(axs[0])
    # Plot model output in rows starting in row 2.
    tssNumber = 0
    for tss in modelled_tss_es:
        for i in range(0,number_of_fits_to_plot):
            a = (df[df["sc"] == scenario].sort_values(by="lossTrainingValue")).iloc[i]
            # Plot observed timeseries either from artificial data or from observations.
            observed_tss = observed_tss_es[tssNumber]
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
                line_width_best = 1.5
                line_width = line_width_best 
                line_style = 'solid'
                z_order = 10
            else:
                line_width_other = 0.5
                line_width = line_width_other 
                line_style = 'dashed'
                z_order = -10
            axs[rij].plot(
                a["valid_date"][start:end],
                a[tss][start:end],
                #linewidth=1.0,
                linewidth = line_width,
                linestyle = line_style,
                #color="black"
                color = a["color"],
                zorder = z_order
            )
            axs[rij].text(.02, .93, labels_variables[rij-1], ha='left', va='top', \
                          transform=axs[rij].transAxes, size = font_size_axes,
                          zorder = 10, backgroundcolor = 'white')
        axs[rij].yaxis.set_tick_params(labelsize=font_size_axes)
        axs[rij].locator_params(axis='y', nbins=3)
        axs[rij].xaxis.set_tick_params(labelsize=font_size_axes, ) #labelrotation = 30)
        if rij < rows_in_figure - 1:
            axs[rij].set(xticklabels=[])
        if tss[-1] == "f":
            axs[rij].set_ylabel("flux (m/day)", size=font_size_axes)
        else:
            axs[rij].set_ylabel("storage (m)", size=font_size_axes)
        rij += 1
        tssNumber += 1
    axs[0].set_ylabel("precipitation\n(m/day)", size=font_size_axes, color = 'blue')
    axTemp.set_ylabel("temperature ($\degree$C)", size=font_size_axes, color = red)
    axs[rows_in_figure - 1].xaxis.set_tick_params(labelsize=font_size_axes, ) # labelrotation = 30)
    custom_lines = [Line2D([0], [0], color=green, lw=line_width_best, ls='solid'),
                    Line2D([0], [0], color=green, lw=line_width_other, ls='dashed'),
                    Line2D([0], [0], color='black', lw=1)]
    if observed_scenario:
        legend_text = ['model, best (all colours)', 'model, 2nd-4th best (all colours)', 'observed']
    else:
        legend_text = ['model, best (all colours)', 'model, 2nd-4th best (all colours)', 'synthetic']
    axs[rows_in_figure - 1].legend(custom_lines, legend_text, loc = 'upper center', \
                                   bbox_to_anchor = (0.5, -0.25), ncol = 3)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(figure_directory + "tss_modartcomp_" + scenario + ".pdf")
    plt.close(fig)


startTimeTss = 2 * 365
endTimeTss = 4 * 365


if create_timeseries:
    # Plot for each variable all scenarios
    #i = 0
    #while i < tss_variables:
    #    modelled_tss = modelled_tss_list[i]
    #    observed_tss = observed_tss_list[i]
    #    timeseries_plot_by_variable(scenarios_to_plot, modelled_tss, observed_tss, startTimeTss, endTimeTss)
    #    i = i + 1
    
    # Plot for each scenario all variables
    i = 0
    for scenario in scenarios_to_plot:
        timeseries_plot_by_scenario(modelled_tss_list, observed_tss_list, scenario, startTimeTss, endTimeTss)
        i = i + 1


# scatterplots

def rSquared(x, y):
    rM = numpy.corrcoef(x, y)
    r = rM[0][1]
    rSq = r * r
    return rSq

def rSquaredFormatted(x, y):
    rSq = rSquared(x, y)
    r_sq_for = "{:.3f}".format(rSq)
    return r_sq_for

def scatter_plot_by_variable(scenarios, modelled_tss, observed_tss, start, end):
    fig = plt.figure(dpi=dpi_figures)
    # gs = fig.add_gridspec(8, 3, hspace=0, wspace=0)
    # fig, axs = plt.subplots(8, 1, sharex='col', sharey = True)
    #fig, axs = plt.subplots(8, 1)
    #fig.set_size_inches(8.27, 11.69)
    fig, axen = plt.subplots(3, 3)
    axs = [axen[0,0], axen[0,1], axen[0,2],
           axen[1,0], axen[1,1], axen[1,2],
           axen[2,0], axen[2,1], axen[2,2]]
    fig.set_size_inches(8.27, 5.2 * 1.5)
    if not EGU:
        fig.subplots_adjust(hspace=0.1)
    # fig.set_size_inches(8.27/3.0,11.69)
    rij = 0
    for sc in scenarios:
        a = (df[df["sc"] == sc].sort_values(by="lossTrainingValue")).iloc[0]
        x = a[observed_tss][start:end]
        y = a[modelled_tss][start:end]
        hb = axs[rij].hexbin(
            x, y, gridsize=15, cmap="Greens", bins="log", linewidths=0.0
        )
        ##fig.colorbar(hb, ax=axs[rij], pad=0.01)
        cbar = fig.colorbar(hb, ax=axs[rij], fraction=0.046, pad=0.04)
        axs[rij].plot([0, 10], [0, 10], color="black", linewidth=0.5)
        r_sq_for = rSquaredFormatted(x, y)
        axs[rij].text(
            0.99, 0.01, r_sq_for, ha="right", va="bottom", transform=axs[rij].transAxes
        )
        # axs[rij].scatter(a[observed_tss][start:end], a[modelled_tss][start:end], s = 0.5)
        axs[rij].set_ylim(0, max(a[observed_tss][start:end]))
        axs[rij].set_xlim(0, max(a[observed_tss][start:end]))
        #if rij < len(scenarios) - 1:
        #    axs[rij].set(xticklabels=[])
        axs[rij].set_aspect("equal")
        axs[rij].xaxis.set_tick_params(labelsize=font_size_axes * 0.9)
        axs[rij].yaxis.set_tick_params(labelsize=font_size_axes * 0.9)
        axs[rij].text(.05, .95, names[rij], ha='left', va='top', \
                      transform=axs[rij].transAxes, size = font_size_axes)
        axs[rij].locator_params(axis='y', nbins=2)
        axs[rij].locator_params(axis='x', nbins=2)
        rij += 1
    axs[8].remove()
    plt.subplots_adjust(wspace=0.5, hspace=0)
    fig.savefig(figure_directory + "sca_modartcomp_" + observed_tss + ".pdf")
    plt.close(fig)

def scatter_plot_by_scenario(modelled_tss_es, observed_tss_es, scenario, name, start, end):
    plt.close('all')
    fig = plt.figure(dpi=dpi_figures)
    # gs = fig.add_gridspec(8, 3, hspace=0, wspace=0)
    # fig, axs = plt.subplots(8, 1, sharex='col', sharey = True)
    #fig, axs = plt.subplots(5, 1)
    fig, axen = plt.subplots(2, 3)
    axs = [axen[0,0], axen[0,1], axen[0,2], axen[1,0], axen[1,1], axen[1,2]]
    fig.set_size_inches(8.27, 5.0)
    rij = 0
    tssNumber = 0
    for tss in modelled_tss_es:
        a = (df[df["sc"] == scenario].sort_values(by="lossTrainingValue")).iloc[0]
        observed_tss = observed_tss_es[tssNumber]
        x = a[observed_tss][start:end]
        y = a[tss][start:end]
        hb = axs[rij].hexbin(
            x, y, gridsize=15, cmap="Greens", bins="log", linewidths=0.0
        )
        axs[rij].set_ylim(0, max(x))
        axs[rij].set_xlim(0, max(x))
        cbar = fig.colorbar(hb, ax=axs[rij], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=font_size_axes)
        axs[rij].plot([0, 10], [0, 10], color="black", linewidth=1.0, linestyle = 'dashed')
        r_sq_for = rSquaredFormatted(x, y)
        axs[rij].text(
            0.99, 0.01, '$r^2$ = ' + r_sq_for, ha="right", va="bottom", transform=axs[rij].transAxes, size = font_size_axes
        )
        #if rij < len(scenarios) - 1:
        #    axs[rij].set(xticklabels=[])
        axs[rij].set_aspect("equal")
        axs[rij].xaxis.set_tick_params(labelsize=font_size_axes * 0.9)
        axs[rij].yaxis.set_tick_params(labelsize=font_size_axes * 0.9)
        if not EGU:
            if tss[-1] == "f":
                axs[rij].set_ylabel("flux (m/day)", size=font_size_axes)
                axs[rij].set_xlabel("flux (m/day)", size=font_size_axes)
            else:
                axs[rij].set_ylabel("storage (m)", size=font_size_axes)
                axs[rij].set_xlabel("storage (m)", size=font_size_axes)
        axs[rij].text(.05, .95, labels_variables_tight[rij], ha='left', va='top', \
                      transform=axs[rij].transAxes, size = font_size_axes)
        axs[rij].locator_params(axis='y', nbins=2)
        axs[rij].locator_params(axis='x', nbins=2)
        rij += 1
        tssNumber += 1
    #axs[0].set_title(name)
    axs[5].remove()
    plt.subplots_adjust(wspace=0.6, hspace=0)
    fig.savefig(figure_directory + "sca_modartcomp_" + scenario + ".pdf")
    plt.close(fig)


startTimeTss = 1 * 365
endTimeTss = len(df['val_art_ts_eva_f'].iloc[0])

if create_scatter:
    # Plot for each variable all scenarios
    i = 0
    while i < tss_variables:
        modelled_tss = modelled_tss_list[i]
        observed_tss = observed_tss_list[i]
        scatter_plot_by_variable(scenarios_to_plot, modelled_tss, observed_tss, startTimeTss, endTimeTss)
        i = i + 1
    
    # Plot for each scenario all variables
    i = 0
    for scenario in scenarios_to_plot:
        name = names[i]
        scatter_plot_by_scenario(modelled_tss_list, observed_tss_list, scenario, name, startTimeTss, endTimeTss)
        i = i + 1

def r2_by_variable(scenarios, tss_variables, start, end):
    fig, axen = plt.subplots(2, 3)
    axs = [axen[0,0], axen[0,1], axen[0,2],
           axen[1,0], axen[1,1], axen[1,2],
           ]
    fig.set_size_inches(8.27, 4.69)
    i = 0
    while i < tss_variables:
        modelled_tss = modelled_tss_list[i]
        observed_tss = observed_tss_list[i]
        xVal = []
        yVal = []
        rij = 0
        for sc in scenarios[:-1]:
            a = (df[df["sc"] == sc].sort_values(by="lossTrainingValue")).iloc[0]
            x = a[observed_tss][start:end]
            y = a[modelled_tss][start:end]
            r_sq_for = rSquared(x, y)
            xVal.append(names[rij])
            yVal.append(r_sq_for)
            rij += 1
        axs[i].plot(xVal,yVal, '.', markersize=12)
        axs[i].text(.05, .95, labels_variables_tight[i], ha='left', va='top', \
                      transform=axs[i].transAxes, size = font_size_axes)
        i = i + 1
    axs[5].remove()
    plt.tight_layout()
    fig.savefig(figure_directory + "r2_by_variable.pdf")
    plt.close(fig)

if create_r2_by_variable:
    r2_by_variable(scenarios_to_plot, tss_variables, startTimeTss, endTimeTss)


# df['lossStop'] = lossStopList
# df['lossTest'] = lossValiList

# a = df[ (df['sc'] == 'fit_sub') & (df ['ts'] == 1)]

##
# scenarios = ['fit_eva', 'fit_sno', 'fit_sub', 'fit_sne', 'fit_sue', 'fit_sus', 'fit_thr', 'fit_exp']

if print_stats:
    # overfitting
    df["minLossValidationValue"] = df["lossValidation"].apply(lambda x: x.min())
    df["lossRatioValidationValue"] = df["minLossValidationValue"]/df["lossValidationValue"]
    print('overall mean overfit metric is ', df["lossRatioValidationValue"].mean())
    print('and by scenario: ')
    for scen in scenarios:
        a = df[df["sc"] == scen]
        print(scen, a["lossRatioValidationValue"].mean())
    df["minNSEVal"] = 1.0 - (df["minLossValidationValue"] / valid_ss_q_mean)

    # performance metrics
    variables = [
        "sc",
        "ts",
        "rs",
        "lossTrainingValue",
        "lossStoppingValue",
        "lossValidationValue",
        "lossRatioValidationValue",
        "minNSEVal",
        "NSEVal",
    ]
    # print(df[df['sc'] == 'fit_eva'].sort_values(by="lossTrainingValue").loc[:,['sc','ts','rs','lossTrainingValue', 'lossStoppingValue','lossValidationValue', 'NSEVal']])
    for scen in scenarios:
        print(df[df["sc"] == scen].sort_values(by="lossTrainingValue").loc[:, variables])
    
def nsePlot(black):
    fig = plt.figure(dpi=dpi_figures, figsize = [2.5,2], tight_layout = {'pad': 1})

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

    if black:
        nr_fits = 1
    else:
        nr_fits = number_of_fits_to_plot
    i = 0
    fit = 0
    while fit < nr_fits:
        i = 0
        for scen in scenarios_to_plot:
            if (fit == 0) & (i == 0):
                label = "neural network"
            else:
                label = ""
            nse = (df[df["sc"] == scen].sort_values(by="lossTrainingValue")).iloc[fit]["NSEVal"]
            if black:
                color = "black"
            else:
                color = (df[df["sc"] == scen].sort_values(by="lossTrainingValue")).iloc[fit]["color"]
            if fit == 0:
                plt.plot(names[i], nse, '.', color=color, label = label, markersize = 12)
            else:
                plt.plot(names[i], nse, '.', color=color, markersize = 2)
            i += 1
        fit += 1

 
    i = 0
    fit = 0
    while fit < nr_fits:
        i = 0
        for scen in scenariosToPlotExp:
            if (fit == 0) & (i == 0):
                label = "conceptual model"
            else:
                label = ""
            nse = (df[df["sc"] == scen].sort_values(by="lossTrainingValue")).iloc[fit]["NSEVal"]
            if black:
                color = "black"
            else:
                color = (df[df["sc"] == scen].sort_values(by="lossTrainingValue")).iloc[fit]["color"]
            if fit == 0:
                plt.plot(names[i], nse, '_', color=color, label = label, markersize = 10)
            else:
                plt.plot(names[i], nse, '_', color=color, markersize = 3)
            i += 1
        fit += 1
    plt.legend(fontsize = font_size_axes * 0.8)
    plt.ylabel("NSE")
    fig.savefig(figure_directory + "nse.pdf")
    plt.close(fig)

if create_nse:
    if observed_scenario:
        nsePlot(False)



### unused code
    #if EGU:
    #    # Save just the portion _inside_ the second axis's boundaries
    #    extent = full_extent(axs[0]).transformed(fig.dpi_scale_trans.inverted())
    #    # Alternatively,
    #    #extent = axs[0].get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
    #    fig.savefig(figure_directory + "sca_modartcomp_" + scenario + ".pdf", bbox_inches=extent)

