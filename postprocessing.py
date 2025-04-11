import numpy
import pandas
import os
import string
from matplotlib import pyplot as plt
from itertools import product
#test

dpi_figures = 600

scenarioDirectory = '../data/scenarios/runs_from_sonic_velocity/kals_model_fit_on_observations/results/'
#scenarioDirectory = '../data/scenarios/runs_from_sonic_velocity/kals_model_fit_on_arti_data/results/'
#scenarioDirectory = '../data/scenarios/runs_from_sonic_velocity/kals_model_fit_on_arti_data_with_error/results/'   # note that the streamflow for validation is with error
figureDirectory = '../figures/'

scenarios = ['fit_eva', 'fit_sno', 'fit_sub', 'fit_sne', 'fit_sue', 'fit_sus', 'fit_thr', 'fit_exp']
#scenarios = ['fit_sno', 'fit_sub', 'fit_sne']

trainingScenarios = ['1', '2', '3', '4']

# rerun scenarios
numberOfScenarios = 2
aRange = numpy.arange(1,numberOfScenarios + 1)
reRunScenarios = []
for s in aRange:
    reRunScenarios.append(str(s))

folderWithArrays = scenarioDirectory + '/' + scenarios[0] + '/' + trainingScenarios[0] + '/' + reRunScenarios[0] + '/arrays'
arrayFiles = os.listdir(folderWithArrays)


arrays = []
for arrayFile in arrayFiles:
    arrays.append(arrayFile.split('.')[0])

df = pandas.DataFrame()
pandas.set_option('display.precision',7)


# load the scenarios, the training scenarios and the rerun scenario numbers
scList = []
tsList = []
rsList = []

for sc, ts, rs in product(scenarios, trainingScenarios, reRunScenarios):
    scList.append(sc)
    tsList.append(int(ts))
    rsList.append(int(rs))
df['sc'] = scList
df['ts'] = tsList
df['rs'] = rsList


for array in arrays:
    arrayContents = []
    for sc, ts, rs in product(scenarios, trainingScenarios, reRunScenarios):
        folder = scenarioDirectory + sc + '/' + ts + '/' + rs + '/arrays/' 
        arrayName = folder + array + '.npy'
        arrayContent = numpy.load(arrayName, allow_pickle = True)
        arrayContents.append(arrayContent)
    df[array] = arrayContents

df['lossTrainingValue'] = (df['lossTraining'].apply(lambda x: x[-1]))
df['lossValidationValue'] = (df['lossValidation'].apply(lambda x: x[-1]))
df['lossStoppingValue'] = (df['lossStopping'].apply(lambda x: x[-1]))

# for calculation of nash sutcliffe
# remove first year (366) which was also not used for training
# assumes observed q for validation is the same across scenarios of course
valid_mean_q = (df['valid_ts_OBS'].apply(lambda x: x[366:].mean()))[0]
valid_q = df['valid_ts_OBS'][0][366:]
valid_ss_q_mean = ((valid_q-valid_mean_q)**2.0).mean()
df['NSEVal'] = 1.0 - (df['lossValidationValue'] / valid_ss_q_mean)

# colors
green = '#4daf4a'
blue = '#377eb8'
red = '#e41a1c'
purple = '#984ea3'
df['color'] = numpy.where(df['ts'] == 1, green, '1')
df['color'] = numpy.where(df['ts'] == 2, blue, df.color)
df['color'] = numpy.where(df['ts'] == 3, red, df.color)
df['color'] = numpy.where(df['ts'] == 4, purple, df.color)
print(df)

#a = df[ (df['sc'] == 'fit_sub') & (df['ts'] == 1) & df['rs'] == 1] 


###################
# response curves #
###################

fig = plt.figure(dpi = dpi_figures)
gs = fig.add_gridspec(8, 3, hspace=0, wspace=0)
fig, axs = plt.subplots(8, 3, sharex='col', sharey = True)
fig.set_size_inches(8.27,11.69)
rij = 0
for sc in scenarios:
    a = df[ (df['sc'] == sc)] 
    for index, row in a.iterrows():
        #if row['lossValidationValue'] < 8e-6:
        #if row['lossValidationValue'] < 1e-7:
        if row['lossTrainingValue'] < 7.3e-6:
            axs[rij,0].plot(row['response_eva_x'], row['response_eva_y'], color = row['color'] )
            axs[rij,1].plot(row['response_sno_x'], row['response_sno_y'], color = row['color'] )
            axs[rij,2].plot(row['response_sub_x'], row['response_sub_y'], color = row['color'] )
        axs[rij,0].set_yticks([0.0, 0.005, 0.010, 0.015])
        axs[rij,0].set_yticklabels([0.0, 0.005, 0.010, 0.015])
    rij += 1
axs[0,0].set_ylim(0,0.02)
axs[0,2].set_xlim(0,0.5)
axs[7,0].set_xlabel('temperature (C)')
axs[7,1].set_xlabel('temperature (C)')
axs[7,2].set_xlabel('store (m)')
for i in range(0,8):
    axs[i,0].set_ylabel('flux (m/day)')
axs[0,0].set_title('evapotranspiration',fontsize = '10')
axs[0,1].set_title('snow melt',fontsize = '10')
axs[0,2].set_title('outflow subsurface storage', fontsize='10')
plt.subplots_adjust(wspace=0, hspace=0)
fig.savefig(figureDirectory + "response.pdf")
plt.close(fig)

##########################
# modelled vs artificial #
##########################

modelledTssList = ['valid_ts_eva_f', 'valid_ts_sno_f', 'valid_ts_sno_s', 'valid_ts_sub_f', 'valid_ts_sub_s', 'valid_ts_sub_f']
observedTssList = ['val_art_ts_eva_f', 'val_art_ts_sno_f', 'val_art_ts_sno_s', 'val_art_ts_sub_f', 'val_art_ts_sub_s', 'valid_ts_OBS']
tssVariables = len(observedTssList)

# timeseries

def timeseriesPlot(modelledTss, observedTss, start, end):
    fig = plt.figure(dpi = dpi_figures)
    gs = fig.add_gridspec(8, 3, hspace=0, wspace=0)
    fig, axs = plt.subplots(8, 1, sharex='col', sharey = True)
    fig.set_size_inches(8.27,11.69)
    rij = 0
    for sc in scenarios:
        a = (df[df['sc'] == sc].sort_values(by='lossTrainingValue')).iloc[0]
        axs[rij].plot(a['valid_date'][start:end], a[observedTss][start:end], linewidth=1.5, color=green)
        axs[rij].plot(a['valid_date'][start:end], a[modelledTss][start:end], linewidth=1.0, color='black')
        rij += 1
    for i in range(0,8):
        if observedTss[-1] == 'f':
            axs[i].set_ylabel('flux (m/day)')
        else:
            axs[i].set_ylabel('storage (m)')
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(figureDirectory + "tss_modartcomp_" + observedTss + ".pdf")
    plt.close(fig)

startTimeTss = 2 * 365
endTimeTss = 5 * 365

i = 0
while i < tssVariables:
    modelledTss = modelledTssList[i]
    observedTss = observedTssList[i]
    timeseriesPlot(modelledTss, observedTss, startTimeTss, endTimeTss)
    i = i + 1


# scatterplots

def scatterPlot(modelledTss, observedTss, start, end):
    fig = plt.figure(dpi = dpi_figures)
    #gs = fig.add_gridspec(8, 3, hspace=0, wspace=0)
    #fig, axs = plt.subplots(8, 1, sharex='col', sharey = True)
    fig, axs = plt.subplots(8, 1)
    fig.set_size_inches(8.27,11.69)
    #fig.set_size_inches(8.27/3.0,11.69)
    rij = 0
    for sc in scenarios:
        a = (df[df['sc'] == sc].sort_values(by='lossTrainingValue')).iloc[0]
        #axs[rij].hexbin(a[observedTss][start:end], a[modelledTss][start:end])
        #axs[rij].plot(a[observedTss][start:end], a[modelledTss][start:end], markersize = 0.5, marker = '.', linestyle = 'none')
        axs[rij].scatter(a[observedTss][start:end], a[modelledTss][start:end], s = 0.5)
        axs[rij].set_ylim(0,max(a[observedTss][start:end]))
        axs[rij].set_xlim(0,max(a[observedTss][start:end]))
        axs[rij].set_aspect('equal')
        rij += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(figureDirectory + "sca_modartcomp_" + observedTss + ".pdf")
    plt.close(fig)

startTimeTss = 2 * 365
endTimeTss = 6 * 365 # len(df['valid_ts_OBS']) - 1

i = 0
while i < tssVariables:
    modelledTss = modelledTssList[i]
    observedTss = observedTssList[i]
    scatterPlot(modelledTss, observedTss, startTimeTss, endTimeTss)
    i = i + 1


#def scatterplot_rich(sfd, filename, variableOne, variableTwo, xlabel, ylabel, symbol, ylim):
#    fig = plt.figure(dpi=600, figsize=(4,3))
#    #fig = plt.figure(dpi=600)
#    plt.xlabel(xlabel, fontsize = myFontSize)
#    plt.ylabel(ylabel, fontsize = myFontSize)
#    plt.xticks(fontsize = myFontSize)
#    plt.yticks(fontsize = myFontSize)
#    plt.plot(variableOne, variableTwo, symbol)
#    plt.ylim(ylim)
#    maxX = max(variableOne)
#    plt.xlim((min(variableOne), 1.05 * maxX))
#    fig.savefig(sfd + '/' + filename + ".pdf")
#    plt.close(fig)
#def responsePlot(dataFrame):
#    for 

#df['lossStop'] = lossStopList
#df['lossTest'] = lossValiList

#a = df[ (df['sc'] == 'fit_sub') & (df ['ts'] == 1)] 

##
scenarios = ['fit_eva', 'fit_sno', 'fit_sub', 'fit_sne', 'fit_sue', 'fit_sus', 'fit_thr', 'fit_exp']

print('#########################################################')

variables = ['sc','ts','rs','lossTrainingValue', 'lossStoppingValue','lossValidationValue', 'NSEVal']
#print(df[df['sc'] == 'fit_eva'].sort_values(by="lossTrainingValue").loc[:,['sc','ts','rs','lossTrainingValue', 'lossStoppingValue','lossValidationValue', 'NSEVal']])
print(df[df['sc'] == 'fit_eva'].sort_values(by="lossTrainingValue").loc[:,variables])
print(df[df['sc'] == 'fit_sno'].sort_values(by="lossTrainingValue").loc[:,variables])
print(df[df['sc'] == 'fit_sub'].sort_values(by="lossTrainingValue").loc[:,variables])
print(df[df['sc'] == 'fit_sne'].sort_values(by="lossTrainingValue").loc[:,variables])
print(df[df['sc'] == 'fit_sue'].sort_values(by="lossTrainingValue").loc[:,variables])
print(df[df['sc'] == 'fit_sus'].sort_values(by="lossTrainingValue").loc[:,variables])
print(df[df['sc'] == 'fit_thr'].sort_values(by="lossTrainingValue").loc[:,variables])
print(df[df['sc'] == 'fit_exp'].sort_values(by="lossTrainingValue").loc[:,variables])
