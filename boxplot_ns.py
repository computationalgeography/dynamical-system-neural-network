import pandas
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy
import xlsxwriter

#metric = "NS"
#metric = "bias"
metric = "CC"
#metric = "pbias"

folder_one = "../figures/land_obs_one/"
folder_two = "../figures/land_obs_two/"

df_nen_one = pandas.read_csv(folder_one + "nen_models_ns.csv")
df_nen_two = pandas.read_csv(folder_two + "nen_models_ns.csv")
df_exp_one = pandas.read_csv(folder_one + "exp_models_ns.csv")
df_exp_two = pandas.read_csv(folder_two + "exp_models_ns.csv")

#df_nen_one.rename(columns={"E": "E_nen_one", \
        #                           "S": "S_nen_one", \
        #                   "G": "G_nen_one", \
        #                   "ES": "ES_nen_one", \
        #                   "EG": "EG_nen_one", \
        #                   "SG": "SG_nen_one", \
        #                   "ESG": "ESG_nen_one",}) 

df_nen_one.columns = ["id_nen_one", "variable_nen_one","E_nen_one", "S_nen_one", "G_nen_one", "ES_nen_one", "EG_nen_one", "SG_nen_one", "ESG_nen_one"]
df_nen_two.columns = ["id_nen_two", "variable_nen_two","E_nen_two", "S_nen_two", "G_nen_two", "ES_nen_two", "EG_nen_two", "SG_nen_two", "ESG_nen_two"]
df_exp_one.columns = ["id_exp_one", "variable_exp_one","E_exp_one", "S_exp_one", "G_exp_one", "ES_exp_one", "EG_exp_one", "SG_exp_one", "ESG_exp_one"]
df_exp_two.columns = ["id_exp_two", "variable_exp_two","E_exp_two", "S_exp_two", "G_exp_two", "ES_exp_two", "EG_exp_two", "SG_exp_two", "ESG_exp_two"]

df = pandas.concat([df_nen_one, df_nen_two, df_exp_one, df_exp_two], axis=1)

names = ["E", "S", "G", "ES", "EG", "SG", "ESG"]
nn_one_scenarios = ([f'{i}_nen_one' for i in names])
pb_one_scenarios = ([f'{i}_exp_one' for i in names])
nn_two_scenarios = ([f'{i}_nen_two' for i in names])
pb_two_scenarios = ([f'{i}_exp_two' for i in names])

def create_statistics(scenario):
    scenarios = ([(f'{i}_' + scenario) for i in names])
    a_series = df.groupby(["variable_nen_one"])[scenarios].apply(numpy.median)
    a_df = a_series.to_frame()
    a_df.columns = [scenario]
    if metric == "NS":
        rounded = a_df.round(2)
    if metric == "CC":
        rounded = a_df.round(2)
    if metric == "pbias":
        rounded = a_df.round(0)
    if metric == "bias":
        rounded = a_df
    return rounded


nen_one = create_statistics("nen_one")
exp_one = create_statistics("exp_one")
nen_two = create_statistics("nen_two")
exp_two = create_statistics("exp_two")
print(nen_one)
total = pandas.concat([nen_one, exp_one, nen_two, exp_two],axis = 1 )
total.to_excel("../figures/merged/boxplot_statistics_" + metric + ".xlsx",  engine="xlsxwriter")


green = "#4daf4a"
#blue = "#377eb8"
blue = "#b0d6f5"

dpi_figures = 600
#fig = plt.figure(dpi=dpi_figures,figsize=(8.27, 11.69))
fig = plt.figure(dpi=dpi_figures,figsize=(11.69, 8.27))
fig, axs = plt.subplots(5)

variables = ["eva_f", "sno_f", "sno_s", "sub_f", "sub_s"]
names = ["evapo-\ntrans-\npiration", \
         "snow\nmelt", \
         "snow\nstorage", \
         "outflow\nsub-\nsurface\nstorage\n(stream-\nflow)",
         "subsurface\nstorage", \
         ]

i=0
for variable in variables:
    df_mod = df[df["variable_nen_one"] == variable]
    axs[i],props = df_mod.boxplot(column=[
                                   "E_nen_one", \
                                   "E_nen_two", \
                                   "S_nen_one", \
                                   "S_nen_two", \
                                   "G_nen_one", \
                                   "G_nen_two", \
                                   "ES_nen_one", \
                                   "ES_nen_two", \
                                   "EG_nen_one", \
                                   "EG_nen_two", \
                                   "SG_nen_one", \
                                   "SG_nen_two", \
                                   "ESG_nen_one", \
                                   "ESG_nen_two", \
                                   "E_exp_one", \
                                   "E_exp_two", \
                                   ],
                                   sym='.',
                                   patch_artist=True,
                                   grid = True,
                                   return_type = "both",
                                   medianprops=dict(color="black", lw=2),
                                   rot=0,
                                   ax = axs[i],
                                   showfliers = False)
    if metric == "pbias":
        a = 2
    else:
        a = 2
        #axs[i].yaxis.set_major_locator(MultipleLocator(0.05))

    seq = [green, blue]
    colors = 8 * seq
    for patch,color in zip(props['boxes'],colors):
        patch.set_facecolor(color)

    [axs[i].axvline(x, color = 'black', linestyle='-', linewidth='0.5') for x in \
                               [2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5]]
    axs[i].get_xaxis().set_ticks([])
    axs[i].text(-0.11,0.5,names[i],transform=axs[i].transAxes, va='center', ha='right')

    i += 1

a = ([*range(1,17)])
plt.xticks(a, 8 * ['lumped', 'semi-distr'], rotation = 90, size=8)
x = 0.05
step = 0.125
y = 1.2
fontje = 14
axs[0].text(x, y, 'E', transform=axs[0].transAxes, fontsize=fontje, va='top')
axs[0].text(x + step, y, 'S', transform=axs[0].transAxes, fontsize=fontje, va='top')
axs[0].text(x + 2*step, y, 'G', transform=axs[0].transAxes, fontsize=fontje, va='top')
axs[0].text(x + 3*step, y, 'ES', transform=axs[0].transAxes, fontsize=fontje, va='top')
axs[0].text(x + 4*step, y, 'EG', transform=axs[0].transAxes, fontsize=fontje, va='top')
axs[0].text(x + 5*step, y, 'SG', transform=axs[0].transAxes, fontsize=fontje, va='top')
axs[0].text(x + 6*step, y, 'ESG', transform=axs[0].transAxes, fontsize=fontje, va='top')
axs[0].text(x + 7*step, y, 'pb', transform=axs[0].transAxes, fontsize=fontje, va='top')
fig.set_size_inches(8.27, 11.69)
plt.subplots_adjust(left = 0.25, hspace=0.05)
plt.savefig('../figures/merged/boxplot_ns.pdf')


##### plot either lumped or distributed

def box_lumped_or_distributed(lumped): 
    #fig = plt.figure(dpi=dpi_figures,figsize=(11.69, 8.27))
    fig = plt.figure(dpi=dpi_figures)
    fig, axs = plt.subplots(5)

    colors = dict(boxes=blue) 
    i=0
    if lumped:
        ext = "one"
    else:
        ext = "two"
    for variable in variables:
        df_mod = df[df["variable_nen_one"] == variable]
        axs[i],props = df_mod.boxplot(column=[
                                       "E_nen_" + ext, \
                                       "S_nen_" + ext, \
                                       "G_nen_" + ext, \
                                       "ES_nen_" + ext, \
                                       "EG_nen_" + ext, \
                                       "SG_nen_" + ext, \
                                       "ESG_nen_" + ext, \
                                       "E_exp_" + ext, \
                                       ],
                                       sym='.',
                                       patch_artist=True,
                                       grid = False,
                                       return_type = "both",
                                       medianprops=dict(color="black", lw=2.0),
                                       rot=0,
                                       ax = axs[i],
                                       color = colors,
                                       fontsize = 8,
                                       showmeans = False,
                                       meanprops=dict(color="red", lw=1.5, marker = '.'),
                                       showfliers = False)

        if (metric == "bias") or (metric == "pbias") or (metric == "NS"):
            a = 2
        else:
            axs[i].yaxis.set_major_locator(MultipleLocator(0.1))
        axs[i].get_xaxis().set_ticks([])
        if metric == "cc":
            axs[i].text(-0.38,0.5,names[i],transform=axs[i].transAxes, va='center', ha='right', size = 9)
        axs[i].grid(True, axis="y")

        # https://towardsdatascience.com/how-to-fetch-the-exact-values-from-a-boxplot-python-8b8a648fc813/
        medians = [item.get_ydata()[0] for item in props['medians']]
        #axs[i].scatter([*range(1,9)], medians, color="blue", zorder=3)
        k = 0
        for j in medians:
            axs[i].text([*range(1,9)][k], medians[k], "{:.2f}\n".format(medians[k]), size = 3, verticalalignment='bottom', horizontalalignment='center')
            k += 1
        i += 1

    plt.xticks([*range(1,9)], ["E", "S", "G", "ES", "EG", "SG", "ESG", "pb"])
    axs[4].set_xticklabels(axs[4].get_xticklabels(), rotation=45)
    #fig.set_size_inches(8.27, 11.69)
    fig.set_size_inches((8.27-1.0)/3.0, 11.69)
    plt.subplots_adjust(left = 0.45, hspace=0.05)
    if metric == "NS":
        if lumped:
            axs[1].set_ylim(top = 0.6)
            axs[1].yaxis.set_major_locator(MultipleLocator(0.1))
        else:
            axs[1].set_ylim(bottom = -3.2, top = 0.6)
            axs[1].yaxis.set_major_locator(MultipleLocator(0.4))
        axs[0].yaxis.set_major_locator(MultipleLocator(0.2))
        axs[2].yaxis.set_major_locator(MultipleLocator(0.1))
        axs[3].yaxis.set_major_locator(MultipleLocator(0.05))
        axs[4].yaxis.set_major_locator(MultipleLocator(0.3))
    axs[0].set_title(metric)
    if lumped:
        plt.savefig('../figures/merged/boxplot_ns_lumped.pdf')
    else:
        plt.savefig('../figures/merged/boxplot_ns_semi_distributed.pdf')

box_lumped_or_distributed(True)
box_lumped_or_distributed(False)
