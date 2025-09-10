import numpy
import pandas
import os
import string
import sys
from matplotlib import pyplot as plt
from itertools import product
from matplotlib.transforms import Bbox
from matplotlib.lines import Line2D

#######################
# main configurations #
#######################

run = sys.argv[1]

if run == "art_one":
    observed_scenario = False
    one_area = True

if run == "art_two":
    observed_scenario = False
    one_area = False

if run == "obs_one":
    observed_scenario = True
    one_area = True

if run == "obs_two":
    observed_scenario = True
    one_area = False



create_scatter = True
create_timeseries = True
create_r2_by_variable = True
create_r2_by_scenario = False
create_nse = True
print_stats = False
create_histogram = False
create_act_melt_vs_temp = True

#figure_directory = "../figures/"

# True: use training set or False: combination of training and stopping
# False gives overall somewhat better NSE for streamflow
modelSelectionWithTraining = False 


GFS = False

data_dir = '../data/scenarios/LAND/final_runs/' 
number_of_rerun_scenarios = 4  # CHANGE TO 4 FOR FINAL RUNS


##################
# other settings #
##################

number_of_fits_to_plot = 4

plt.rcParams["font.size"] = 8
plt.rcParams.update({'figure.max_open_warning': 0})

dpi_figures = 600

EGU = False

actual_snow_flux = True

if EGU:
    font_size_axes = 9
else:
    font_size_axes = 8 

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
        #results_folder = '2507_oneArea_observations_7reruns/results/'
        #results_folder = 'land_onearea_observations_cali_xhr/'
        results_folder = 'land_obs_one/'
    else:
        #results_folder = '2507_twoArea_observations_7reruns/'
        #results_folder = 'land_twoareas_observations_cali_xhr/'
        results_folder = 'land_obs_two/'
    scenario_directory = data_dir + results_folder 
else:
    # No error in streamflow (replaced) and precipitation and temperature
    # copied into the results folder
    if one_area:
        #results_folder = 'kals_model_fit_on_arti_data_with_error_subf_val_noerror/results/'
        results_folder = 'land_art_one/'
    else:
        #results_folder = 'kals_model_two_areas_fit_on_artificial_data_with_error/results/'
        #results_folder = 'kals_model_two_areas_fit_on_artificial_data/results/'
        results_folder = 'land_art_two/'
    scenario_directory = data_dir + results_folder
    # Note that the streamflow for validation is with error
    #scenario_directory = data_dir + "kals_model_fit_on_arti_data_with_error/results/"

figure_directory = "../figures/" + results_folder


# only nn scenarios
scenarios = [
    "fit_eva",
    "fit_sno",
    "fit_sub",
    "fit_sne",
    "fit_sue",
    "fit_sus",
    "fit_thr",
    "fit_xhr",
]

if observed_scenario:
    # all scenarios
    if one_area:
        if GFS:
            #scenarios = ['fit_xhr']
            scenarios = ['fit_eva', 'fit_sno', 'fit_sub', 'fit_sne', 'fit_sue', 'fit_sus', 'fit_thr', \
                     'fit_xva', 'fit_xno', 'fit_xub', 'fit_xne', 'fit_xue', 'fit_xus', 'fit_xhr']
        else:
            scenarios = ['fit_eva', 'fit_sno', 'fit_sub', 'fit_sne', 'fit_sue', 'fit_sus', 'fit_thr', \
                     'fit_xhr']
    else:
        if GFS:
            #scenarios = ['fit_xhr']
            scenarios = ['fit_eva', 'fit_sno', 'fit_sub', 'fit_sne', 'fit_sue', 'fit_sus', 'fit_thr', \
                     'fit_xva', 'fit_xno', 'fit_xub', 'fit_xne', 'fit_xue', 'fit_xus', 'fit_xhr']
        else:
            scenarios = ['fit_eva', 'fit_sno', 'fit_sub', 'fit_sne', 'fit_sue', 'fit_sus', 'fit_thr', \
                     'fit_xhr']

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
        "fit_xhr",
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
        "fit_xhr"     # or xhr for observational data
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
    #+ rerun_scenarios[5]
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

## add additional_validation_data
## cosero model, sub_s groundwater
#arrayContents = []
#for sc, ts, rs in product(scenarios, training_scenarios, rerun_scenarios):
#    arrayContent = numpy.load(data_dir + "additional_validation_data/" + "val_cosero_sub_s_gw.npy" , allow_pickle=True)
#    arrayContents.append(arrayContent)
#df["val_cosero_sub_s_gw"] = arrayContents
## cosero model, sub_s soil
#arrayContents = []
#for sc, ts, rs in product(scenarios, training_scenarios, rerun_scenarios):
#    arrayContent = numpy.load(data_dir + "additional_validation_data/" + "val_cosero_sub_s_soil.npy" , allow_pickle=True)
#    arrayContents.append(arrayContent)
#df["val_cosero_sub_s_soil"] = arrayContents
## cosero model, eva_f soil
#arrayContents = []
#for sc, ts, rs in product(scenarios, training_scenarios, rerun_scenarios):
#    arrayContent = numpy.load(data_dir + "additional_validation_data/" + "val_cosero_eva_f.npy" , allow_pickle=True)
#    arrayContents.append(arrayContent)
#df["val_cosero_eva_f"] = arrayContents

# add additional_validation_data
# cosero model, sub_s groundwater
coseroVariables = ["val_cosero_bw0", \
                   "val_cosero_bw1", \
                   "val_cosero_bw2", \
                   "val_cosero_bw3", \
                   "val_cosero_bw4", \
                   "val_cosero_eva_f", \
                   "val_cosero_glacmelt", \
                   "val_cosero_melt", \
                   "val_cosero_smelt", \
                   "val_cosero_sub_s_gw", \
                   "val_cosero_sub_s_soil"]

for coseroVariable in coseroVariables:
    arrayContents = []
    for sc, ts, rs in product(scenarios, training_scenarios, rerun_scenarios):
        arrayContent = numpy.load(data_dir + "additional_validation_data/" + coseroVariable + ".npy", allow_pickle=True)
        arrayContents.append(arrayContent)
    df[coseroVariable] = arrayContents


# loss values
df["lossTrainingValue"] = df["lossTraining"].apply(lambda x: x[-1])
df["lossValidationValue"] = df["lossValidation"].apply(lambda x: x[-1])
df["lossStoppingValue"] = df["lossStopping"].apply(lambda x: x[-1])

if modelSelectionWithTraining:
    df["lossModelSelection"] = df["lossTrainingValue"]
else:
    df["lossModelSelection"] = df["lossTrainingValue"] * 0.75 + df["lossStoppingValue"] * 0.25



#df["minLossValidationValue"] = df["lossValidation"].apply(lambda x: x.min())
#df["lossRatioValidationValue"] = df["minLossValidationValue"]/df["lossValidationValue"]
#
#for scen in scenarios:
#    #print(df[df["sc"] == scen].sort_values(by="lossTrainingValue").loc[:, variables])
#    a = df[df["sc"] == scen]
#    print(scen, a["lossRatioValidationValue"].mean())

# Replace potential snow melt by actual snow melt both for synthetic data
# and for ML modelled data
# assume zero for no snow (too simple)
#if actual_snow_flux:
#    df['valid_ts_sno_f'] = df.apply(lambda x: \
#                           numpy.where(x['valid_ts_sno_s'] < 0.0001, 0.0, \
#                   x['valid_ts_sno_f']), \
#                           axis=1)
#    df['val_art_ts_sno_f'] = df.apply(lambda x: \
#                          numpy.where(x['val_art_ts_sno_s'] < 0.0001, 0.0, \
#                   x['val_art_ts_sno_f']), \
#                   axis=1)
# actually calculate it like in the model
if actual_snow_flux:
    df['valid_ts_sno_f'] = df.apply(lambda x: \
                           numpy.where(x['valid_ts_sno_s'] < x['valid_ts_sno_f'], x['valid_ts_sno_s'], \
                           x['valid_ts_sno_f']), \
                           axis=1)
    df['val_art_ts_sno_f'] = df.apply(lambda x: \
                           numpy.where(x['val_art_ts_sno_s'] < x['val_art_ts_sno_f'], x['val_art_ts_sno_s'], \
                           x['val_art_ts_sno_f']), \
                           axis=1)

# sum cosero sub_s for soil and groundwater
#df["val_cosero_sub_s"] = df["val_cosero_sub_s_gw"] + df["val_cosero_sub_s_soil"]
#min_val_cosero_sub_s = df["val_cosero_sub_s"].apply(lambda x: x.min())
#df["val_cosero_sub_s"] = df["val_cosero_sub_s"] - min_val_cosero_sub_s

min_val_cosero_sub_s_soil = df["val_cosero_sub_s_soil"].apply(lambda x: x.min())
df["val_cosero_sub_s_soil"] = df["val_cosero_sub_s_soil"] - min_val_cosero_sub_s_soil

min_val_cosero_sub_s_gw = df["val_cosero_sub_s_gw"].apply(lambda x: x.min())
df["val_cosero_sub_s_gw"] = df["val_cosero_sub_s_gw"] - min_val_cosero_sub_s_gw

df["val_cosero_sub_s"] = df["val_cosero_sub_s_gw"] + df["val_cosero_sub_s_soil"]
#df["val_cosero_sub_s"] = df["val_cosero_sub_s_soil"]

# sum cosero sub_s for all storages, additional cosero data
df["val_cosero_sub_s_additional"] = df["val_cosero_bw0"] + df["val_cosero_bw1"] + df["val_cosero_bw2"] + df["val_cosero_bw3"]
min_val_cosero_sub_s_additional = df["val_cosero_sub_s_additional"].apply(lambda x: x.min())
df["val_cosero_sub_s_additional"] = df["val_cosero_sub_s_additional"] - min_val_cosero_sub_s_additional

#df["val_cosero_sno_f_additional"] = df["val_cosero_smelt"] + df["val_cosero_glacmelt"]
df["val_cosero_sno_f_additional"] = df["val_cosero_smelt"]



# for calculation of nash sutcliffe
# remove first year (366) which was also not used for training
# For streamflow
# assumes observed q for validation is the same across scenarios of course
# valid_ts_OBS is observed streamflow (not artificial, not in any case)
if observed_scenario:
    valid_mean_q = (df["valid_ts_OBS"].apply(lambda x: x[366:].mean()))[0]
    valid_q = df["valid_ts_OBS"][0][366:]
    valid_ss_q_mean = ((valid_q - valid_mean_q) ** 2.0).mean()
else:
    valid_mean_q = (df["val_art_ts_sub_f"].apply(lambda x: x[366:].mean()))[0]
    valid_q = df["val_art_ts_sub_f"][0][366:]
    valid_ss_q_mean = ((valid_q - valid_mean_q) ** 2.0).mean()
df["NSEVal"] = 1.0 - (df["lossValidationValue"] / valid_ss_q_mean)
# For snow cover
if observed_scenario:
    valid_mean_sno_s = (df["val_lan_ts_sno_s"].apply(lambda x: x[366:].mean()))[0]
    valid_sno_s = df["val_lan_ts_sno_s"][0][366:]
    valid_ss_sno_s_mean = ((valid_sno_s - valid_mean_sno_s) ** 2.0).mean()
    df["mss_sno_s"] = df.apply(lambda x: ((valid_sno_s - x["valid_ts_sno_s"][366:]) ** 2.0).mean(), axis = 1)
else:
    xxx = 0
if observed_scenario:
    df["NSEValSnoS"] = 1.0 - (df["mss_sno_s"] / valid_ss_sno_s_mean)
# For evapotranspiration 
# eva from LAND (val_lan..) or from cosero (val_cosero..)
#validation_eva_file_name = "val_lan_ts_eva_f"
validation_eva_file_name = "val_cosero_eva_f"
if observed_scenario:
    valid_mean_eva_f = (df[validation_eva_file_name].apply(lambda x: x[366:].mean()))[0]
    valid_eva_f = df[validation_eva_file_name][0][366:]
    valid_ss_eva_f_mean = ((valid_eva_f - valid_mean_eva_f) ** 2.0).mean()
    df["mss_eva_f"] = df.apply(lambda x: ((valid_eva_f - x["valid_ts_eva_f"][366:]) ** 2.0).mean(), axis = 1)
else:
    xxx = 0
if observed_scenario:
    df["NSEValEvaF"] = 1.0 - (df["mss_eva_f"] / valid_ss_eva_f_mean)
# For sub_s 
if observed_scenario:
    valid_mean_sub_s = (df["val_cosero_sub_s"].apply(lambda x: x[366:].mean()))[0]
    valid_sub_s = df["val_cosero_sub_s"][0][366:]
    valid_ss_sub_s_mean = ((valid_sub_s - valid_mean_sub_s) ** 2.0).mean()
    df["mss_sub_s"] = df.apply(lambda x: ((valid_sub_s - x["valid_ts_sub_s"][366:]) ** 2.0).mean(), axis = 1)
else:
    xxx = 0
if observed_scenario:
    df["NSEValSubS"] = 1.0 - (df["mss_sub_s"] / valid_ss_sub_s_mean)




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
        "lossModelSelection",
        "lossStoppingValue",
        "lossValidationValue",
        "lossRatioValidationValue",
        "minNSEVal",
        "NSEVal",
        #"eva_parameter",
        #"sub_parameter",
        #"sno_parameter",

    ]
    # print(df[df['sc'] == 'fit_eva'].sort_values(by="lossModelSelection").loc[:,['sc','ts','rs','lossModelSelection', 'lossStoppingValue','lossValidationValue', 'NSEVal']])
    for scen in scenarios:
        print(df[df["sc"] == scen].sort_values(by="lossModelSelection").loc[:, variables])


###################
# response curves #
###################


fig = plt.figure(dpi=dpi_figures)

if observed_scenario:
    response_scenarios_to_plot = scenarios_to_plot
    response_nr_rows = 8
else:
    response_scenarios_to_plot = scenarios_to_plot[:-1]
    response_nr_rows = 7

gs = fig.add_gridspec(response_nr_rows, 3, hspace=0, wspace=0)
fig, axs = plt.subplots(response_nr_rows, 3, sharex="col", sharey=True)
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

line_width_best = 3.0
line_width_other = 1.0

rij = 0

for sc in response_scenarios_to_plot:
    #a = df[(df["sc"] == sc)]
    a = (df[df["sc"] == sc].sort_values(by="lossModelSelection"))
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
    legend_text = ['model, best (all colours)', 'model, 2nd-4th best (all colours)', 'expert']
else:
    legend_text = ['model, best (all colours)', 'model, 2nd-4th best (all colours)', 'synthetic']
#axs[7,0].legend(custom_lines, legend_text, loc = 'upper center', bbox_to_anchor = (1.5, -0.5), ncol = 3)
axs[response_nr_rows - 1,0].legend(custom_lines, legend_text, loc = 'upper center', bbox_to_anchor = (1.5, -0.5), ncol = 3)

if observed_scenario:
    axs[0, 0].set_ylim(0, 0.0205)
else:
    axs[0, 0].set_ylim(0, 0.013)

# x axis 
#axs[0, 0].set_xlim(-10, 15)
axs[0, 0].set_xlim(-10, 12)
#axs[0, 1].set_xlim(-2, 8)
axs[0, 1].set_xlim(-2, 6)
if observed_scenario:
    if one_area:
        axs[0, 2].set_xlim(0, 0.4)
    else:
        axs[0, 2].set_xlim(0, 0.12)
else:
    #axs[0, 2].set_xlim(0.0, 0.15)
    axs[0, 2].set_xlim(0, 0.18)

#axs[7, 0].set_xticks([-10, -5, 0, 5, 10])
axs[response_nr_rows - 1, 0].set_xticks([-10, -5, 0, 5, 10])
#axs[7, 0].set_xticklabels([-10, -5, 0, 5, 10], size=font_size_axes)
axs[response_nr_rows - 1, 0].set_xticklabels([-10, -5, 0, 5, 10], size=font_size_axes)

labels = [-2, 0, 2, 4]
axs[response_nr_rows - 1, 1].set_xticks(labels)
axs[response_nr_rows - 1, 1].set_xticklabels(labels, size=font_size_axes)

#axs[7, 0].set_xlabel("temperature ($\degree$C)", fontsize=font_size_axes)
#axs[7,0].xaxis.set_tick_params(labelsize=font_size_axes)
#axs[7,1].xaxis.set_tick_params(labelsize=font_size_axes)
#axs[7,2].xaxis.set_tick_params(labelsize=font_size_axes)
#axs[7, 1].set_xlabel("temperature ($\degree$C)", fontsize=font_size_axes)
#axs[7, 2].set_xlabel("subsurface storage (m)", fontsize=font_size_axes)

axs[response_nr_rows - 1, 0].set_xlabel("temperature ($\degree$C)", fontsize=font_size_axes)
axs[response_nr_rows - 1,0].xaxis.set_tick_params(labelsize=font_size_axes)
axs[response_nr_rows - 1,1].xaxis.set_tick_params(labelsize=font_size_axes)
axs[response_nr_rows - 1,2].xaxis.set_tick_params(labelsize=font_size_axes)
axs[response_nr_rows - 1, 1].set_xlabel("temperature ($\degree$C)", fontsize=font_size_axes)
axs[response_nr_rows - 1, 2].set_xlabel("subsurface storage (m)", fontsize=font_size_axes)

for i in range(0, response_nr_rows):
    axs[i, 0].set_ylabel("flux (m/day)", fontsize=font_size_axes)
axs[0, 0].set_title("evapotranspiration\nincluding sublimation", fontsize=font_size_axes)
axs[0, 1].set_title("snow melt", fontsize=font_size_axes)
axs[0, 2].set_title("outflow subsurf. storage\n(streamflow)", fontsize=font_size_axes)
if not EGU:
    plt.subplots_adjust(wspace=0, hspace=0)
fig.savefig(figure_directory + "response.pdf")
plt.close(fig)

##################################################
# modelled vs artificial (or measured benchmark) #
##################################################

modelled_tss_list = [
    "valid_ts_eva_f",
    "valid_ts_sno_f",
    "valid_ts_sno_s",
    "valid_ts_sub_f",
    "valid_ts_sub_s"
]

if observed_scenario:
    observed_tss_list = [
        #"val_lan_ts_eva_f",
        "val_cosero_eva_f",
        "val_cosero_sno_f_additional",
        "val_lan_ts_sno_s",
        "valid_ts_OBS",
        "val_cosero_sub_s_additional"
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
        a = (df[df["sc"] == sc].sort_values(by="lossModelSelection")).iloc[0]
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

def timeseries_plot_by_scenario(modelled_tss_es, observed_tss_es, scenario, start, end, best_fit_only):
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
    if best_fit_only:
        number_of_fits_to_plot = 1
    for tss in modelled_tss_es:
        for i in range(0,number_of_fits_to_plot):
            a = (df[df["sc"] == scenario].sort_values(by="lossModelSelection")).iloc[i]
            # Plot observed timeseries either from artificial data or from observations.
            observed_tss = observed_tss_es[tssNumber]
            if not(observed_scenario) or (observed_tss == 'valid_ts_OBS') \
                                         or (observed_tss == 'val_lan_ts_sno_s') \
                                         or (observed_tss == 'val_cosero_eva_f'): 
                axs[rij].plot(
                    a["valid_date"][start:end],
                    a[observed_tss][start:end],
                    linewidth = 0.5,
                    #color=green
                    color='black'
                )
            #if not(observed_scenario) and (observed_tss == 'val_cosero_sub_s'):
            if observed_scenario and (observed_tss == 'val_cosero_sub_s_additional'):
                axs[rij].plot(
                    a["valid_date"][start:end],
                    a["val_cosero_sub_s_additional"][start:end],
                    linewidth = 0.5,
                    #color=green
                    color='black'
                )
            if observed_scenario and (observed_tss == 'val_cosero_sno_f_additional'):
                axs[rij].plot(
                    a["valid_date"][start:end],
                    a["val_cosero_sno_f_additional"][start:end],
                    linewidth = 0.5,
                    #color=green
                    color='black'
                )
#                axs[rij].plot(
#                    a["valid_date"][start:end],
#                    a["val_cosero_sub_s_soil"][start:end],
#                    linewidth = 0.5,
#                    #color=green
#                    color='black'
#                )
#                axs[rij].plot(
#                    a["valid_date"][start:end],
#                    a["val_cosero_sub_s_gw"][start:end],
#                    linewidth = 0.5,
#                    #color=green
#                    color='black',
#                    linestyle = (0, (1,5))
#                )
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
            if observed_scenario:
                if rij == 2:
                    axs[rij].set_ylim(0,0.022)
                if rij == 3:
                    axs[rij].set_ylim(0,0.8)
                if rij == 4:
                    axs[rij].set_ylim(0,0.025)
                if rij == 5:
                    axs[rij].set_ylim(0,0.45)

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
    if best_fit_only:
        print('.')
    else:
        custom_lines = [Line2D([0], [0], color=green, lw=line_width_best, ls='solid'),
                        Line2D([0], [0], color=green, lw=line_width_other, ls='dashed'),
                        Line2D([0], [0], color='black', lw=1)]
    if observed_scenario:
        legend_text = ['model, best (all colours)', 'model, 2nd-4th best (all colours)', 'observed']
    else:
        legend_text = ['model, best (all colours)', 'model, 2nd-4th best (all colours)', 'synthetic']
    if best_fit_only:
        print('.')
    else:
        axs[rows_in_figure - 1].legend(custom_lines, legend_text, loc = 'upper center', \
                                       bbox_to_anchor = (0.5, -0.25), ncol = 3)
    plt.subplots_adjust(wspace=0, hspace=0)
    if best_fit_only:
        fig.savefig(figure_directory + "tss_modartcomp_best_fit_only_" + scenario + ".pdf")
    else:
        fig.savefig(figure_directory + "tss_modartcomp_" + scenario + ".pdf")
    plt.close(fig)


startTimeTss = 4 * 365
#endTimeTss = 4 * 365
#endTimeTss = 6 * 365
endTimeTss = 10 * 365


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
        timeseries_plot_by_scenario(modelled_tss_list, observed_tss_list, scenario, startTimeTss, endTimeTss, True)
        #timeseries_plot_by_scenario(modelled_tss_list, observed_tss_list, scenario, startTimeTss, endTimeTss, False)
        i = i + 1


# scatterplots

def ns(x, y):
    # nash sutcliffe
    # x is observed
    # y is modelled
    return 1 - (((x - y) ** 2.0).mean() / ((x - x.mean()) ** 2.0).mean())

def nsFormatted(x, y):
    nsValue = ns(x, y)
    ns_for = "{:.3f}".format(nsValue)
    return ns_for

def rmse_calc(x, y):
    # normalized RMSE or coefficient of variation
    return numpy.sqrt(((x - y) ** 2.0).mean())/y.mean()

def rmseFormatted(x, y):
    rmse = rmse_calc(x, y)
    rmse_for = "{:.4f}".format(rmse)
    return rmse_for

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
        a = (df[df["sc"] == sc].sort_values(by="lossModelSelection")).iloc[0]
        x = a[observed_tss][start:end]
        y = a[modelled_tss][start:end]
        hb = axs[rij].hexbin(
            x, y, gridsize=15, cmap="Greens", bins="log", linewidths=0.0
        )
        ##fig.colorbar(hb, ax=axs[rij], pad=0.01)
        cbar = fig.colorbar(hb, ax=axs[rij], fraction=0.046, pad=0.04)
        axs[rij].plot([0, 10], [0, 10], color="black", linewidth=0.5)
        #r_sq_for = rSquaredFormatted(x, y)
        #rmse_for = rmseFormatted(x, y)
        ns_for = nsFormatted(x, y)
        axs[rij].text(
            #0.99, 0.01, r_sq_for, ha="right", va="bottom", transform=axs[rij].transAxes
            #0.99, 0.01, rmse_for, ha="right", va="bottom", transform=axs[rij].transAxes
            0.99, 0.01, ns_for, ha="right", va="bottom", transform=axs[rij].transAxes
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
    # Plot for each scenario all variables
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
        a = (df[df["sc"] == scenario].sort_values(by="lossModelSelection")).iloc[0]
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
        #rmse_for = rmseFormatted(x, y)
        ns_for = nsFormatted(x, y)
        axs[rij].text(
            0.99, 0.01, '$r^2$ = ' + r_sq_for, ha="right", va="bottom", transform=axs[rij].transAxes, size = font_size_axes
            #0.99, 0.01, '$r^2$ = ' + rmse_for, ha="right", va="bottom", transform=axs[rij].transAxes, size = font_size_axes
            #0.99, 0.01, 'NSE = ' + ns_for, ha="right", va="bottom", transform=axs[rij].transAxes, size = font_size_axes
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

def histogram_by_scenario(hist_tss, scenario, name, start, end):
    plt.close('all')
    fig = plt.figure(dpi=dpi_figures)
    fig, axen = plt.subplots(2, 3)
    axs = [axen[0,0], axen[0,1], axen[0,2], axen[1,0], axen[1,1], axen[1,2]]
    fig.set_size_inches(8.27, 5.0)
    a = (df[df["sc"] == scenario].sort_values(by="lossModelSelection")).iloc[0]
    #snow = (a["valid_ts_sno_s"][start:end] > 0.0001)
    snow = (a["val_art_ts_sno_s"][start:end] > 0.0001)
    temperature = (a["valid_ts_temperature"][start:end])
    temp_with_snow = temperature[snow]
    daysWithTempAboveThreshold = numpy.sum(temp_with_snow > 4.0)
    daysWithTempBelowThreshold = numpy.sum(temp_with_snow <= 4.0)
    proportionDays = daysWithTempAboveThreshold / (daysWithTempAboveThreshold + daysWithTempBelowThreshold)
    print(proportionDays)
    label = ['temp', 'temp with snow', 'sub_s', 'sno_s']
    rij = 0
    tssNumber = 0
    for tss in hist_tss:
        y = a[tss][start:end]
        if tssNumber == 1:
            hb = axs[rij].hist(temp_with_snow, bins = 20)
        else:
            hb = axs[rij].hist(y, bins = 20)
        axs[rij].set_xlabel(label[tssNumber], size=font_size_axes)
        rij += 1
        tssNumber += 1
    axs[5].remove()
    fig.savefig(figure_directory + "hist_modartcomp_" + scenario + ".pdf")
    plt.close(fig)

hist_tss = [
    "valid_ts_temperature",
    "valid_ts_temperature",
    "valid_ts_sub_s",
    "valid_ts_sno_s"
]

if create_histogram:
    # Plot for each scenario all variables
    i = 0
    for scenario in scenarios_to_plot:
        name = names[i]
        histogram_by_scenario(hist_tss, scenario, name, startTimeTss, endTimeTss)
        i = i + 1

def actual_melt_vs_temperature(start, end):
    plt.close('all')
    fig = plt.figure(dpi=dpi_figures)
    fig, axen = plt.subplots(2, 3)
    axs = [axen[0,0], axen[0,1], axen[0,2], axen[1,0], axen[1,1], axen[1,2]]
    fig.set_size_inches(8.27, 5.0)
    a = (df[df["sc"] == "fit_xhr"].sort_values(by="lossModelSelection")).iloc[0]
    snow = (a["val_art_ts_sno_s"][start:end] > 0.0001)
    # actual snow melt from synthetic observational data set
    actual_snow_melt = (a["val_art_ts_sno_f"][start:end])[snow]
    temperature = (a["valid_ts_temperature"][start:end])[snow]
    #axs[0].plot(temperature,actual_snow_melt, '.')
    hb = axs[0].hexbin(
            temperature, actual_snow_melt, gridsize=50, cmap="Greens", bins="log", linewidths=0.0
        )
    axs[0].set_xlim(-2,15)
    fig.savefig(figure_directory + "act_melt_vs_temp.pdf")

if create_act_melt_vs_temp:
    if not observed_scenario:
        actual_melt_vs_temperature(startTimeTss, endTimeTss)



def r2_by_variable(scenarios, tss_variables, start, end):
    if one_area:
        colour = green
        size = 12
    else:
        colour = blue
        size = 8
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
        # nn scenarios
        for sc in scenarios[:-1]:
            a = (df[df["sc"] == sc].sort_values(by="lossModelSelection")).iloc[0]
            x = a[observed_tss][start:end]
            y = a[modelled_tss][start:end]
            #if (modelled_tss == 'valid_ts_sub_s') and (observed_scenario):
            #    r_sq_for = rSquared(x, y)
            #else:
            r_sq_for = ns(x, y)
            xVal.append(names[rij])
            yVal.append(r_sq_for)
            rij += 1
        axs[i].plot(xVal,yVal, '.', markersize=size, color = colour)
        xValExp = []
        yValExp = []
        rij = 0
        # add expert model scenario as reference, for observed scenario only
        if observed_scenario:
            # expert scenario
            for sc in scenarios[:-1]:
                a = (df[df["sc"] == "fit_xhr"].sort_values(by="lossModelSelection")).iloc[0]
                x = a[observed_tss][start:end]
                y = a[modelled_tss][start:end]
                if (modelled_tss == 'valid_ts_sub_s') and (observed_scenario):
                    r_sq_for = rSquared(x, y)
                else:
                    r_sq_for = ns(x, y)
                xValExp.append(names[rij])
                yValExp.append(r_sq_for)
                rij += 1
            axs[i].plot(xValExp,yValExp, '_', markersize=10, color = colour)
                #plt.plot(names[i], nse, '_', color=color, label = label, markersize = 10)
        axs[i].text(.05, .95, labels_variables_tight[i], ha='left', va='top', \
                      transform=axs[i].transAxes, size = font_size_axes)
        i = i + 1
    if observed_scenario:
        axs[0].set_ylim(-0.8,-0.25) # eva_f
        axs[1].set_ylim(0.1,0.6) # sno_f 
        axs[2].set_ylim(0.6,1.03)   # sno_s
        axs[3].set_ylim(0.78,0.83)  # sub_f
        axs[4].set_ylim(-1.6,0.8)  # sub_s
    else:
        #axs[0].set_ylim(0.94,1.01)
        #axs[1].set_ylim(0.87,1.02)
        #axs[2].set_ylim(0.9965,1.001)
        #axs[3].set_ylim(0.995,1.001)
        #axs[4].set_ylim(0.974,1.005)
        axs[0].set_ylim(0.94,1.0004)
        axs[1].set_ylim(0.87,1.0004)
        axs[2].set_ylim(0.9980,1.0001)
        axs[3].set_ylim(0.9965,1.0001)
        axs[4].set_ylim(0.974,1.0002)
    plt.tight_layout()
    axs[5].remove()
    #if observed_scenario:
    #    # do not plot snow melt
    #    pos1 = axs[1].get_position()
    #    axs[1].set_position(axs[2].get_position())
    #    axs[2].set_position(pos1)
    #    axs[1].remove()
    fig.savefig(figure_directory + "r2_by_variable.pdf", transparent = True)
    plt.close(fig)

if create_r2_by_variable:
    r2_by_variable(scenarios_to_plot, tss_variables, startTimeTss, endTimeTss)

def r2_by_scenario(scenarios, tss_variables, start, end):
    fig, axen = plt.subplots(3, 3)
    axs = [axen[0,0], axen[0,1], axen[0,2],
           axen[1,0], axen[1,1], axen[1,2],
           axen[2,0], axen[2,1], axen[2,2]
           ]
    fig.set_size_inches(8.27, 1.5 * 4.69)

    x_labels = ['E_f', 'S_f', 'S_s', 'G_f', 'G_s']
    rij = 0
    for sc in scenarios[:-1]:
        a = (df[df["sc"] == sc].sort_values(by="lossModelSelection")).iloc[0]
        #xVal = []
        yVal = []
        i = 0
        while i < tss_variables:
            modelled_tss = modelled_tss_list[i]
            observed_tss = observed_tss_list[i]
            x = a[observed_tss][start:end]
            y = a[modelled_tss][start:end]
            #r_sq_for = rSquared(x, y)
            r_sq_for = ns(x, y)
            #xVal.append(names[i])
            yVal.append(r_sq_for)
            i = i + 1
        if one_area:
            axs[rij].plot(x_labels,yVal, '.', markersize=12)
        else:
            axs[rij].plot(x_labels,yVal, '+', markersize=12, color="black")
        #axs[rij].set_ylim(0.88, 1.005)
        #axs[rij].set_ylim(0.875, 1.02)
        if observed_scenario:
            axs[rij].set_ylim(-2.0, 1.02)
        else:
            axs[rij].set_ylim(0.84, 1.02)
        axs[rij].text(.05, .95, names[rij], ha='left', va='top', \
                          transform=axs[rij].transAxes, size = font_size_axes)
        rij += 1
    axs[7].remove()
    axs[8].remove()
    #plt.tight_layout()
    fig.savefig(figure_directory + "r2_by_scenario.pdf", transparent=True)
    plt.close(fig)

if create_r2_by_scenario:
    r2_by_scenario(scenarios_to_plot, tss_variables, startTimeTss, endTimeTss)


# df['lossStop'] = lossStopList
# df['lossTest'] = lossValiList

# a = df[ (df['sc'] == 'fit_sub') & (df ['ts'] == 1)]

##
# scenarios = ['fit_eva', 'fit_sno', 'fit_sub', 'fit_sne', 'fit_sue', 'fit_sus', 'fit_thr', 'fit_exp']

# this has been moved up a bit can be deleted later on
#if print_stats:
#    # overfitting
#    df["minLossValidationValue"] = df["lossValidation"].apply(lambda x: x.min())
#    df["lossRatioValidationValue"] = df["minLossValidationValue"]/df["lossValidationValue"]
#    print('overall mean overfit metric is ', df["lossRatioValidationValue"].mean())
#    print('and by scenario: ')
#    for scen in scenarios:
#        a = df[df["sc"] == scen]
#        print(scen, a["lossRatioValidationValue"].mean())
#    df["minNSEVal"] = 1.0 - (df["minLossValidationValue"] / valid_ss_q_mean)
#
#    # performance metrics
#    variables = [
#        "sc",
#        "ts",
#        "rs",
#        "lossModelSelection",
#        "lossStoppingValue",
#        "lossValidationValue",
#        "lossRatioValidationValue",
#        "minNSEVal",
#        "NSEVal",
#    ]
#    # print(df[df['sc'] == 'fit_eva'].sort_values(by="lossModelSelection").loc[:,['sc','ts','rs','lossModelSelection', 'lossStoppingValue','lossValidationValue', 'NSEVal']])
#    for scen in scenarios:
#        print(df[df["sc"] == scen].sort_values(by="lossModelSelection").loc[:, variables])
    
def nsePlot(black, variable):
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

    if GFS:
        scenariosToPlotExp = [
            'fit_xva',
            'fit_xno',
            'fit_xub',
            'fit_xne',
            'fit_xue',
            'fit_xus',
            'fit_xhr'
            ]
    else:
        scenariosToPlotExp = [
            'fit_xhr',
            'fit_xhr',
            'fit_xhr',
            'fit_xhr',
            'fit_xhr',
            'fit_xhr',
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
            if variable == 'sub_f':
                nse = (df[df["sc"] == scen].sort_values(by="lossModelSelection")).iloc[fit]["NSEVal"]
            if variable == 'sno_s':
                nse = (df[df["sc"] == scen].sort_values(by="lossModelSelection")).iloc[fit]["NSEValSnoS"]
            if variable == 'eva_f':
                nse = (df[df["sc"] == scen].sort_values(by="lossModelSelection")).iloc[fit]["NSEValEvaF"]
            if variable == 'sub_s':
                nse = (df[df["sc"] == scen].sort_values(by="lossModelSelection")).iloc[fit]["NSEValSubS"]
            if black:
                color = "black"
            else:
                color = (df[df["sc"] == scen].sort_values(by="lossModelSelection")).iloc[fit]["color"]
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
                label = "expert model"
            else:
                label = ""
            if variable == 'sub_f':
                nse = (df[df["sc"] == scen].sort_values(by="lossModelSelection")).iloc[fit]["NSEVal"]
            if variable == 'sno_s':
                nse = (df[df["sc"] == scen].sort_values(by="lossModelSelection")).iloc[fit]["NSEValSnoS"]
            if variable == 'eva_f':
                nse = (df[df["sc"] == scen].sort_values(by="lossModelSelection")).iloc[fit]["NSEValEvaF"]
            if variable == 'sub_s':
                nse = (df[df["sc"] == scen].sort_values(by="lossModelSelection")).iloc[fit]["NSEValSubS"]
            if black:
                color = "black"
            else:
                color = (df[df["sc"] == scen].sort_values(by="lossModelSelection")).iloc[fit]["color"]
            if fit == 0:
                plt.plot(names[i], nse, '_', color=color, label = label, markersize = 10)
            else:
                plt.plot(names[i], nse, '_', color=color, markersize = 3)
            i += 1
        fit += 1
    plt.legend(fontsize = font_size_axes * 0.8)
    plt.ylabel("NSE")
    if variable == 'sub_f':
        plt.ylim(0.78, 0.83)
        fig.savefig(figure_directory + "nse_sub_f.pdf")
    if variable == 'sno_s':
        plt.ylim(0.55, 0.98)
        fig.savefig(figure_directory + "nse_sno_s.pdf")
    if variable == 'eva_f':
        fig.savefig(figure_directory + "nse_eva_f.pdf")
    if variable == 'sub_s':
        fig.savefig(figure_directory + "nse_sub_s.pdf")
    plt.close(fig)

if create_nse:
    if observed_scenario:
       nsePlot(False, 'sub_f')
       nsePlot(False, 'sno_s')
       nsePlot(False, 'eva_f')
       nsePlot(False, 'sub_s')



### unused code
    #if EGU:
    #    # Save just the portion _inside_ the second axis's boundaries
    #    extent = full_extent(axs[0]).transformed(fig.dpi_scale_trans.inverted())
    #    # Alternatively,
    #    #extent = axs[0].get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
    #    fig.savefig(figure_directory + "sca_modartcomp_" + scenario + ".pdf", bbox_inches=extent)

