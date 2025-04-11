import numpy as numpy
numpy.set_printoptions(precision=2)
import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
from matplotlib import pyplot as plt
import math
import csv
import datetime
import os
import sys
import string
#import matplotlib.pyplot as plt

runInBatch = False

# inputs for running in batch mode
if runInBatch:
    first = sys.argv[1]   # ranges from 1 up to and including the number of fitting scenarios used
    second = sys.argv[2]  # is 1 or 3 (training scenarios, 1 represents 1 and 2, 3 represents 3 and 4)
    third = sys.argv[3]   # is 1 or 1,2 or 1,2,3 etc, splitted at , (rerun scenario)

deepLayer = True   # note that for true, it may need to be in the parameter list of the optimizer

m = torch.nn.ReLU()

seedAll = 4
torch.manual_seed(seedAll)
numpy.random.seed(seedAll)

if deepLayer:
    nodes = 100
else:
    nodes = 1000

oneArea = True
printParameters = True
L1Regularization = False     # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-l1-l2-and-elastic-net-regularization-with-pytorch.md

inputDataDirectory = "../data/inputData/"
outputDataDirectory = "../data/outputData/"

conversionFluxes = 549.3

width = 0.25

# meteo start date is 1 jan 1979, end date is 31 juli 2014
# so do not select a time stem outside this range
# streamflow available always in this period

myFontSize = 7

def seedGenerator(string):
    tot = 0
    i = 1
    for letter in string:
        tot += ord(letter) * i
        i = i + 10
    return(tot + seedAll)

def timeSeriesPlot(dateTimeSeries, variableTimeSeriesList, ylabel):
    fig = plt.figure(dpi=600)
    plt.xlabel("time")
    plt.ylabel(ylabel)
    for item in variableTimeSeriesList:
        plt.plot(dateTimeSeries, item, linewidth=width)
    fig.savefig(ylabel + ".pdf")
    plt.close(fig)

def timeSeriesPlot_rich(sfd, fileName, dateTimeSeries, variableTimeSeriesList, xlabel, ylabel):
    fig = plt.figure(dpi=600, figsize=(6,3))
    plt.xlabel(xlabel, fontsize = myFontSize)
    plt.ylabel(ylabel, fontsize = myFontSize)
    plt.xticks(fontsize = myFontSize)
    plt.yticks(fontsize = myFontSize)
    plt.grid(True, linewidth = width, linestyle = 'dotted')
    plt.locator_params(axis='y', nbins=30)
    for item in variableTimeSeriesList:
        plt.plot(dateTimeSeries, item, linewidth = width)
    fig.savefig(sfd + '/' + fileName + ".pdf")
    plt.close(fig)

def scatterplot(variableOne, variableTwo, ylabel, symbol):
    fig = plt.figure(dpi=600)
    plt.xlabel("obs")
    plt.ylabel(ylabel)
    plt.plot(variableOne, variableTwo, symbol)
    fig.savefig(ylabel + ".pdf")
    plt.close(fig)

def scatterplot_rich(sfd, filename, variableOne, variableTwo, xlabel, ylabel, symbol, ylim):
    fig = plt.figure(dpi=600, figsize=(4,3))
    #fig = plt.figure(dpi=600)
    plt.xlabel(xlabel, fontsize = myFontSize)
    plt.ylabel(ylabel, fontsize = myFontSize)
    plt.xticks(fontsize = myFontSize)
    plt.yticks(fontsize = myFontSize)
    plt.plot(variableOne, variableTwo, symbol)
    plt.ylim(ylim)
    maxX = max(variableOne)
    plt.xlim((min(variableOne), 1.05 * maxX))
    fig.savefig(sfd + '/' + filename + ".pdf")
    plt.close(fig)


def createTrainingIndices():
    a = [1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]
    out = []
    for p in range(0,366):    # add one year spin up
        out.append(1)
    for i in a:
        for j in range(0,365):
            out.append(i)
    out = numpy.array(out)
    outOne = (out == 1)
    outTwo = (out == 2)
    outThree = (out == 3)
    outFour = (out == 4)
    return outOne, outTwo, outThree, outFour





def rand_min_max(mean,proportion):
    max = mean + proportion * mean
    min = mean - proportion * mean
    return torch.rand(1)*(max-min) + min

class weightConstraint(object):
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            w=module.weight.data
            w=w.clamp(0.0,2.0)
            #w=w.clamp(0.0,0.0)
            module.weight.data=w


class EarlyStopper:
    def __init__(self, patience = 1, min_delta_proportion = 1e-10):
        self.patience = patience
        self.min_delta_proportion = min_delta_proportion
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta_proportion * self.min_validation_loss):
            self.counter += 1
            if self.counter >= self.patience:
                return self.counter, True
        return self.counter, False


###############
# data import #
###############


def createMeteoData(inputDataDirectory, outputDataDirectory, startDate, endDate):
    precipitationFile = open(outputDataDirectory + "precipitation.txt", "w")
    temperatureFile = open(outputDataDirectory + "temperature.txt", "w")
    date = datetime.date(1979, 1, 1)
    timestep = datetime.timedelta(days = 1)
    
    temperatureTimeSeries = []
    precipitationTimeSeries = []
    
    with open(inputDataDirectory + 'weatherdata-470125_corrected.csv') as csv_file:
        csv_reader = csv.reader(csv_file, dialect='excel')
        line_count = 0
        line_out_count = 1
        for row in csv_reader:
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                dateFromFileStr = str.split(row[0],"/")
                monthFromFileStr = int(dateFromFileStr[0])
                dayFromFileStr = int(dateFromFileStr[1])
                yearFromFileStr = int(dateFromFileStr[2])
                dateFromFile = datetime.date(yearFromFileStr, monthFromFileStr, dayFromFileStr)
                #print(date, 'from file ', dateFromFile)
                temperature = (float(row[4]) + float(row[5])) / 2.0
                precip = row[6]
                #if (date >= start) and (date <= end):
                if (date >= startDate) and (date <= endDate):
                    precipitationFile.write(str(line_out_count) + " " + precip + "\n")
                    precipitationTimeSeries.append(float(precip))
                    temperatureFile.write(str(line_out_count) + " " + str(temperature) + "\n")
                    temperatureTimeSeries.append(temperature)
                    line_out_count += 1
                line_count += 1
                date = date + timestep

    precipitationFile.close()
    temperatureFile.close()
    return temperatureTimeSeries, precipitationTimeSeries

def createStreamFlowData(inputDataDirectory, outputDataDirectory, start, end):
    streamFlowFile = open(outputDataDirectory + "streamflow.txt", "w")
    date = datetime.date(1951, 1, 1)
    timestep = datetime.timedelta(days = 1)
    
    dischargeFile = open(inputDataDirectory + "6246180_Q_Day.Cmd_noHeader.txt","r") 
    dischargeFileContent = dischargeFile.readlines()
    
    streamFlowTimeSeries = []
    dateTimeSeries = []
    
    line_out_count = 1
    for i in range(0,len(dischargeFileContent)):
        splitted = str.split(dischargeFileContent[i])
        discharge = splitted[1]
        if (date >= start) and (date <= end):
            streamFlowFile.write(str(line_out_count) + " " + discharge + "\n")
            streamFlowTimeSeries.append(float(discharge))
            dateTimeSeries.append(date)
            line_out_count += 1
        date = date + timestep

    dischargeFile.close()
    return streamFlowTimeSeries, dateTimeSeries

## data for training
#temperatureTimeSeries, precipitationTimeSeries = createMeteoData(inputDataDirectory, outputDataDirectory, start, end)
#streamFlowTimeSeries, dateTimeSeries = createStreamFlowData(inputDataDirectory, outputDataDirectory, start, end)
#
## data for validation
#startVal = datetime.date(1997, 10 , 1)
#endVal = datetime.date(2013, 10, 1)
#temperatureTimeSeriesVal, precipitationTimeSeriesVal = createMeteoData(inputDataDirectory, outputDataDirectory, startVal, endVal)
#streamFlowTimeSeriesVal, dateTimeSeriesVal = createStreamFlowData(inputDataDirectory, outputDataDirectory, startVal, endVal)
#
#
#aRange = range(1,len(streamFlowTimeSeries)+1,1)
#f = plt.figure()
#f, axs = plt.subplots(2, 2)
#axs[0,0].scatter(temperatureTimeSeries, streamFlowTimeSeries, s = 1 )
#axs[0,0].set_ylabel('streamflow')
#axs[0,0].set_xlabel('temperature')
#
#width=0.25
#axs[0,1].plot(dateTimeSeries, numpy.asarray(temperatureTimeSeries), linewidth=width)
#axs[0,1].set_ylabel('temperature')
#axs[1,0].plot(dateTimeSeries, numpy.asarray(streamFlowTimeSeries), linewidth=width)
#axs[1,0].set_ylabel('streamflow')
#axs[1,1].plot(dateTimeSeries, numpy.asarray(precipitationTimeSeries), linewidth=width)
#axs[1,1].set_ylabel('precipitation')
#
#axs[1,0].xaxis.set_major_locator(plt.MaxNLocator(3))
#axs[0,1].xaxis.set_major_locator(plt.MaxNLocator(3))
#axs[1,1].xaxis.set_major_locator(plt.MaxNLocator(3))
#
#plt.tight_layout()
#f.savefig(outputDataDirectory + "inputTimeseries.pdf")

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # calibrated parameters used for modes obsCreation
        #self.seepageProportion = 0.029 # calibrated, see txt for info 
        #self.meltRateParameter = 0.0026 # calibrated, see txt for info https://tc.copernicus.org/articles/17/211/2023/#abstract, 0.006 not strange

        self.eva_f_d = nn.Linear(1,nodes)
        self.eva_f_c = nn.Linear(nodes,1)
        self.sno_f_d = nn.Linear(1,nodes)
        self.sno_f_c = nn.Linear(nodes,1)
        self.sub_f_d = nn.Linear(1,nodes)
        self.sub_f_c = nn.Linear(nodes,1)
        if deepLayer:
            self.eva_f_dd = nn.Linear(nodes,nodes)
            self.sno_f_dd = nn.Linear(nodes,nodes)
            self.sub_f_dd = nn.Linear(nodes,nodes)

        if oneArea:
            eva_par = 0.8715501 
            sub_par = 0.0297130 
            sno_par = 0.0021341 
            tem_par = -0.8557543000000001 
            evp_par = 3.52035805  # note input to sigmoid (evp [0,1])
        else:
            eva_par = 0.7017500
            sub_par = 0.0625746
            sno_par = 0.0007901
            tem_par = -0.9351652 
            evp_par = 3.4553902 

        self.eva_parameter_obs_creation = eva_par
        self.sub_parameter_obs_creation = sub_par
        self.sno_parameter_obs_creation = sno_par
        self.tem_parameter_obs_creation = tem_par
        self.evp_parameter_obs_creation = evp_par 



        proportion = 0.5
        self.eva_parameter = nn.Parameter(rand_min_max(eva_par,proportion))
        self.sub_parameter = nn.Parameter(rand_min_max(sub_par,proportion))
        self.sno_parameter = nn.Parameter(rand_min_max(sno_par,proportion))
        self.tem_parameter = nn.Parameter(rand_min_max(tem_par,proportion))
        self.evp_parameter = nn.Parameter(rand_min_max(evp_par,proportion))

        if deepLayer:
            self.params = nn.ModuleDict({
                'eva': nn.ModuleList([self.eva_f_d, self.eva_f_dd, self.eva_f_c]),
                'sno': nn.ModuleList([self.sno_f_d, self.sno_f_dd, self.sno_f_c]),
                'sub': nn.ModuleList([self.sub_f_d, self.sub_f_dd, self.sub_f_c]),
                'exp': nn.ParameterList([self.eva_parameter, self.sub_parameter, self.sno_parameter, self.tem_parameter, self.evp_parameter])
                })
        else:
            self.params = nn.ModuleDict({
                'eva': nn.ModuleList([self.eva_f_d, self.eva_f_c]),
                'sno': nn.ModuleList([self.sno_f_d, self.sno_f_c]),
                'sub': nn.ModuleList([self.sub_f_d, self.sub_f_c]),
                'exp': nn.ParameterList([self.eva_parameter, self.sub_parameter, self.sno_parameter, self.tem_parameter, self.evp_parameter])
                })


    def eva_f_calculate(self,
                        temp,
                        modeEva,
                        deepLayer,
                        linearArt
                        ):
        if modeEva == "obsCreation":
            if linearArt:
                # potential evapotranspiration
                # https://en.wikipedia.org/wiki/Blaney–Criddle_equation (see link for params)
                EPot = torch.max(((0.35 * (0.457 * torch.tensor([temp]) + 8.128))/1000.0) * self.eva_parameter_obs_creation, torch.tensor(0.0))
            else:
                if temp > -15.0:
                    #EPot = torch.max(((0.35 * (0.457 * temp + 8.128))/1000.0) * self.eva_parameter_obs_creation, torch.tensor(0.0))
                    EPot = ((torch.sin((torch.tensor([temp + 15.0]) * 0.25 - 0.5*torch.pi))+1)/1000.0) + 0.0001 * torch.tensor([temp + 15.0])
                else:
                    EPot = torch.tensor([0.0])
        if modeEva == "fit":
            out_eva_f = m(self.eva_f_d((torch.tensor([temp]))))
            if deepLayer:
                out_eva_f = m(self.eva_f_dd(out_eva_f))
            EPot = torch.sigmoid(self.eva_f_c(out_eva_f))/75.0   # /750.0 or /1000.0
        if modeEva == "fitExpert":
            # potential evapotranspiration
            # https://en.wikipedia.org/wiki/Blaney–Criddle_equation (see link for params)
            EPot = torch.max(((0.35 * (0.457 * temp + 8.128))/1000.0) * self.eva_parameter, torch.tensor(0.0))
        return EPot

    def sno_f_calculate(self,
                        temp,
                        modeSno,
                        deepLayer,
                        linearArt
                        ):
        if modeSno == "obsCreation":
            if linearArt:
                melting = temp > 0.0
                if melting:
                    potentialMelt = temp * self.sno_parameter_obs_creation
                else:
                    potentialMelt = torch.tensor(0.0)
            else:
                melting = temp > 0.0
                if melting:
                    potentialMelt = ((torch.sin((torch.tensor([temp]) * 0.6 - 0.5*torch.pi))+1)/200) + 0.0004 * torch.tensor([temp])
                else:
                    potentialMelt = torch.tensor(0.0)
        if modeSno == "fit":
            out_sno_f = m(self.sno_f_d((torch.tensor([temp]))))
            if deepLayer:
                out_sno_f = m(self.sno_f_dd(out_sno_f))
            potentialMelt = torch.sigmoid(self.sno_f_c(out_sno_f))/50.0
        if modeSno == "fitExpert":
            melting = temp > 0.0
            if melting:
                potentialMelt = temp * self.sno_parameter
            else:
                potentialMelt = torch.tensor(0.0)
        return potentialMelt


    def sub_f_calculate(self,
                        sub_s,
                        modeSub,
                        deepLayer,
                        linearArt
                        ):
        if modeSub == "obsCreation":
            if linearArt:
                seepagePot = self.sub_parameter_obs_creation * sub_s 
            else:
                seepagePot = ((torch.sin((torch.tensor([sub_s])*20-0.5*torch.pi))+1)/500) + 0.03 * sub_s
        if modeSub == "fit":
            out_sub_f = m(self.sub_f_d((torch.tensor([sub_s])*2.0)))  
            if deepLayer:
                out_sub_f = m(self.sub_f_dd(out_sub_f))
            #seepagePot = (torch.sigmoid(self.sub_f_c(out_sub_f)) - 0.0) / 10.0 
            proportion = (torch.sigmoid(self.sub_f_c(out_sub_f))) 
            seepagePot = proportion * sub_s
        if modeSub == "fitExpert":
            seepage = self.sub_parameter * sub_s 
            seepagePot = seepage
        return seepagePot

    def com_f_response(self,
                       modeSub,
                       deepLayer,
                       component,
                       linearArt
                       ):
        if component == 'sub':
            com_s_values = numpy.arange(0.0,1.0,0.01)
        if component == 'sno':
            com_s_values = numpy.arange(-10.0,15.0,0.01)
        if component == 'eva':
            com_s_values = numpy.arange(-10.0,15.0,0.01)
        com_f_values = numpy.empty(0)
        for com_s_value in com_s_values:
            if component == 'sub':
                com_f_value = self.sub_f_calculate(com_s_value.item(), modeSub, deepLayer, linearArt)
            if component == 'sno':
                com_f_value = self.sno_f_calculate(com_s_value.item(), modeSub, deepLayer, linearArt)
            if component == 'eva':
                com_f_value = self.eva_f_calculate(com_s_value.item(), modeSub, deepLayer, linearArt)
            if isinstance(com_f_value, torch.Tensor):
                if modeSub == 'fitExpert':
                    com_f_value_detached = com_f_value.detach().numpy()
                else:
                    if component == 'sub':
                        com_f_value_detached = com_f_value.detach().numpy()[0]
                    if component == 'sno':
                        com_f_value_detached = com_f_value.detach().numpy()
                    if component == 'eva':
                        com_f_value_detached = com_f_value.detach().numpy()
            else:
                com_f_value_detached = com_f_value
            com_f_values = numpy.append(com_f_values, com_f_value_detached)
            com_f_values = numpy.maximum(com_f_values, 0.0)
        return com_s_values, com_f_values


    def forward(self,
                linearArt,                 # linear obs creation model
                firstEpochs,               # start of training or not
                sno_s_initial,             # initial snow storage
                sub_s_initial,             # initial subsurface storage
                temperature,               # temperature time series
                precipitation,             # precipitation time series
                modeEva,                   # mode evapotranspiration
                modeSno,                   # mode snow 
                modeSub,                   # mode subsurface
                modeTem,                   # temperature offset
                modeEvP                    # potential proportion of evap to sublimation  
                ):

        nr_timesteps = len(temperature)

        # parameters and inputs for expert model

        #temperatureOffsetExpert = -1.19 # calibrated see txt for info 

        #EPar = 0.85  # calibrated see txt for info
        # offset should actually be (((2115+2797)/2)-1180)*0.005 = 6.38 degrees.. (i.e. lower and higher half,
        # average, multipled by lapse rate
        # difference between lower and higher is (2115-2797) * 0.005 = 3.41 degrees
        # https://journals.ametsoc.org/view/journals/clim/16/7/1520-0442_2003_016_1032_sasvoa_2.0.co_2.xml

        # proportion of evapotranspiration potentially assigned to sno storage
        #evaPropToSnoExpert = 0.995

        if modeTem == "fitExpert":
            temperatureOffset = self.tem_parameter
        if modeTem == "obsCreation":
            temperatureOffset = self.tem_parameter_obs_creation 

        if modeEvP == "fitExpert":
            evaPropToSno = torch.sigmoid(self.evp_parameter)
        if modeEvP == "obsCreation":
            evaPropToSno = torch.sigmoid(torch.tensor(self.evp_parameter_obs_creation))

        if oneArea:
            areaTemperatureOffsets = [0.0]
        else:
            areaTemperatureOffsets = [1.705,-1.705]

        if oneArea:
            numberOfAreas = 1
        else:
            numberOfAreas = 2
        for area in range(0,numberOfAreas):
            # emtpy output timeseries for writing
            # storages
            sno_s_ts = torch.zeros(nr_timesteps)    
            sub_s_ts = torch.zeros(nr_timesteps)   
            # fluxes
            sno_f_ts = torch.zeros(nr_timesteps)
            sub_f_ts = torch.zeros(nr_timesteps)
            # evapotranspiration
            eva_f_ts = torch.zeros(nr_timesteps)
    
            # initial storages
            sno_s = sno_s_initial 
            sub_s = sub_s_initial
    
            for i in range(0,nr_timesteps):
                # get drivers for timestep
                temp = temperature[i] + temperatureOffset + areaTemperatureOffsets[area]

                # potential evapotranspiration
    
                EPot = torch.max(self.eva_f_calculate(temp, modeEva, deepLayer, linearArt), torch.tensor(0.0))

                eva_f = EPot
                eva_f_ts[i] = eva_f

    
                # snow or rain

                precip = precipitation[i] / 1000.0
                snowing = temp < 0.0
                if snowing:
                    snowfall = precip
                    rainfall = torch.tensor(0.0)
                else:
                    snowfall = torch.tensor(0.0)
                    rainfall = precip


                # snow storage

                # snowfall
                sno_s = sno_s + snowfall
                sno_s_ts[i] = sno_s    

                # snow melt
                potentialMelt = self.sno_f_calculate(temp, modeSno, deepLayer, linearArt)

                if firstEpochs:                       # hard cap to prevent zero snow cover
                    if temp < - 5.0:
                        potentialMelt = torch.tensor(0.0)

                sno_f = potentialMelt
                sno_f_ts[i] = sno_f    
    
                potentialMelt = torch.max(potentialMelt,torch.tensor(0))
                actualMelt = torch.min(sno_s, potentialMelt)
                sno_s = sno_s - actualMelt

                # sublimation
                potentialSublimation = evaPropToSno * EPot
                actualSublimation = torch.min(sno_s, potentialSublimation)
                sno_s = sno_s - actualSublimation
                potentialEvapForSub = EPot - actualSublimation

                # sub surface storage

                totalInput = rainfall + actualMelt
                #hortonRunoffProportion = 0.04
                hortonRunoffProportion = 0.0
                hortonRunoff = hortonRunoffProportion * totalInput

                sub_s = sub_s + (1.0 - hortonRunoffProportion) * totalInput
    
                EAct = torch.min(sub_s, potentialEvapForSub)

                sub_s = sub_s - EAct

                sub_s_ts[i] = sub_s    

                seepagePot = self.sub_f_calculate(sub_s, modeSub, deepLayer, linearArt)
                     
                seepageNotNegative = torch.max(seepagePot, torch.tensor(0.0))
                seepageAct = torch.min(seepageNotNegative, sub_s)
    
                sub_f = seepageAct + hortonRunoff

                #sub_f = torch.max(sub_f, torch.tensor(0.0008))

                sub_f_ts[i] = sub_f
    
                sub_s = sub_s - seepageAct

            if area == 0:
                # storages
                sno_s_ts_areas = sno_s_ts
                sub_s_ts_areas = sub_s_ts
                # fluxes
                sno_f_ts_areas = sno_f_ts
                sub_f_ts_areas = sub_f_ts
                # evapotranspiration
                eva_f_ts_areas = eva_f_ts
            else:
                # storages
                sno_s_ts_areas = torch.vstack([sno_s_ts_areas, sno_s_ts])
                sub_s_ts_areas = torch.vstack([sub_s_ts_areas, sub_s_ts])
                # fluxes
                sno_f_ts_areas = torch.vstack([sno_f_ts_areas, sno_f_ts])
                sub_f_ts_areas = torch.vstack([sub_f_ts_areas, sub_f_ts])
                # evapotranspiration
                eva_f_ts_areas = torch.vstack([eva_f_ts_areas, eva_f_ts])

            # if one area add the same ones again ('fake' two areas, each the same)
            if oneArea:
                # storages
                sno_s_ts_areas = torch.vstack([sno_s_ts_areas, sno_s_ts])
                sub_s_ts_areas = torch.vstack([sub_s_ts_areas, sub_s_ts])
                # fluxes
                sno_f_ts_areas = torch.vstack([sno_f_ts_areas, sno_f_ts])
                sub_f_ts_areas = torch.vstack([sub_f_ts_areas, sub_f_ts])
                # evapotranspiration
                eva_f_ts_areas = torch.vstack([eva_f_ts_areas, eva_f_ts])

        return sno_s_ts_areas, sub_s_ts_areas, sno_f_ts_areas, sub_f_ts_areas, eva_f_ts_areas

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()



#############################
# create artificial observations
#############################

def createArtificialObservations(temperatureTimeSeries, precipitationTimeSeries, addErrorToArtificialStreamFlow, linearArt):
    hydromodel = Net()
    sno_s_initial = 0.0
    sub_s_initial = 0.0
    sno_s_ts_areas, sub_s_ts_areas, sno_f_ts_areas, sub_f_ts_areas, eva_f_ts_areas = hydromodel(
                   linearArt,
                   False,
                   torch.tensor(sno_s_initial),
                   torch.tensor(sub_s_initial),
                   torch.tensor(temperatureTimeSeries),
                   torch.tensor(precipitationTimeSeries),
                   modeEva = 'obsCreation',
                   modeSno = 'obsCreation',
                   modeSub = 'obsCreation',
                   modeTem = 'obsCreation',
                   modeEvP = 'obsCreation',
                   )
    sno_s_ts = sno_s_ts_areas.mean(dim = 0)
    sub_s_ts = sub_s_ts_areas.mean(dim = 0)
    sno_f_ts = sno_f_ts_areas.mean(dim = 0)
    sub_f_ts = sub_f_ts_areas.mean(dim = 0)
    eva_f_ts = eva_f_ts_areas.mean(dim = 0)
    if addErrorToArtificialStreamFlow:
        #error = torch.tensor([numpy.random.normal(0,0.01,len(streamFlowTimeSeries))])
        #sma = torch.nn.AvgPool1d(kernel_size=59, stride = 1, padding = 29)
        #error = sma(error)
        #sub_f_ts = torch.max(sub_f_ts + error, torch.tensor(0.0))[0]
        sd = 0.2
        errorMultiplier = numpy.minimum(numpy.maximum(0.0, numpy.random.normal(1,sd, len(streamFlowTimeSeries))), 2)
        #sub_f_ts = torch.max(sub_f_ts * errorMultiplier, torch.tensor(0.0))[0]
        sub_f_ts = torch.max(sub_f_ts * errorMultiplier, torch.tensor(0.0))
    return sno_s_ts, sub_s_ts, sno_f_ts, sub_f_ts, eva_f_ts


def training_loop(n_epochs,
                  stopping,
                  optimizer,
                  model,
                  loss_fn,
                  sno_s_initial,
                  sub_s_initial,
                  temperatureTimeSeries,
                  precipitationTimeSeries,
                  streamFlowTimeSeries,
                  dateTimeSeries,
                  temperatureTimeSeriesSto,
                  precipitationTimeSeriesSto,
                  streamFlowTimeSeriesSto,
                  dateTimeSeriesSto,
                  temperatureTimeSeriesVal,
                  precipitationTimeSeriesVal,
                  streamFlowTimeSeriesVal,
                  dateTimeSeriesVal,
                  sno_s_ts, sub_s_ts, sno_f_ts, sub_f_ts, eva_f_ts,
                  sno_s_tsSto, sub_s_tsSto, sno_f_tsSto, sub_f_tsSto, eva_f_tsSto,
                  sno_s_tsVal, sub_s_tsVal, sno_f_tsVal, sub_f_tsVal, eva_f_tsVal,
                  trainingData,
                  fitOnObservations,
                  modeEvaTrain,
                  modeSnoTrain,
                  modeSubTrain,
                  modeTemTrain,
                  modeEvPTrain,
                  outputDirectory,
                  scenarioDirectory,
                  linearArt
                  ):


    sfd = outputDirectory + scenarioDirectory + '/'
    sfdAr = sfd + 'arrays' + '/'
    if not os.path.exists(sfd):
        os.makedirs(sfd)
    if not os.path.exists(sfdAr):
        os.makedirs(sfdAr)

    # write artificial data to disk, validation
    numpy.save(sfdAr + 'val_art_ts_sno_s.npy', sno_s_tsVal)
    numpy.save(sfdAr + 'val_art_ts_sub_s.npy', sub_s_tsVal)
    numpy.save(sfdAr + 'val_art_ts_sno_f.npy', sno_f_tsVal)
    numpy.save(sfdAr + 'val_art_ts_sub_f.npy', sub_f_tsVal)
    numpy.save(sfdAr + 'val_art_ts_eva_f.npy', eva_f_tsVal)
    # write artificial data to disk, training
    numpy.save(sfdAr + 'train_art_ts_sno_s.npy', sno_s_ts)
    numpy.save(sfdAr + 'train_art_ts_sub_s.npy', sub_s_ts)
    numpy.save(sfdAr + 'train_art_ts_sno_f.npy', sno_f_ts)
    numpy.save(sfdAr + 'train_art_ts_sub_f.npy', sub_f_ts)
    numpy.save(sfdAr + 'train_art_ts_eva_f.npy', eva_f_ts)

    lossTrainingSeries = []
    lossValidationSeries = []
    lossStoppingSeries = []
    epochSeries = []
    stopperCounterSeries = []

    early_stopper = EarlyStopper(patience = 200, min_delta_proportion = 1e-2)

    time = datetime.datetime.now()

    firstEpochs = True
    for epoch in range(1, n_epochs + 1):
        if epoch > 100:
            firstEpochs = False
        firstEpochs = True
        newTime = datetime.datetime.now()
        duration = newTime - time
        time = datetime.datetime.now()

        tr_sno_s_ts_areas, tr_sub_s_ts_areas, tr_sno_f_ts_areas, tr_sub_f_ts_areas, tr_eva_f_ts_areas = hydromodel(
               linearArt,
               firstEpochs,
               torch.tensor(sno_s_initial),
               torch.tensor(sub_s_initial),
               torch.tensor(temperatureTimeSeries),
               torch.tensor(precipitationTimeSeries),
               modeEva = modeEvaTrain,
               modeSno = modeSnoTrain,
               modeSub = modeSubTrain,
               modeTem = modeTemTrain,
               modeEvP = modeEvPTrain,
               )

        tr_sno_s_ts = tr_sno_s_ts_areas.mean(dim = 0)
        tr_sub_s_ts = tr_sub_s_ts_areas.mean(dim = 0)
        tr_sno_f_ts = tr_sno_f_ts_areas.mean(dim = 0)
        tr_sub_f_ts = tr_sub_f_ts_areas.mean(dim = 0)
        tr_eva_f_ts = tr_eva_f_ts_areas.mean(dim = 0)

        training = numpy.logical_not(stopping)

        if fitOnObservations:
            if epoch < -5:
                loss_train = loss_fn(tr_sub_f_ts, sub_f_ts, training)
            else:
                loss_train = loss_fn(tr_sub_f_ts, torch.tensor(streamFlowTimeSeries)/conversionFluxes, training)
        else:
            if trainingData == 'trainingSub':
                loss_train = loss_fn(tr_sub_f_ts, sub_f_ts, training)
            if trainingData == 'trainingEva':
                loss_train = loss_fn(tr_eva_f_ts, eva_f_ts, training)
            if trainingData == 'trainingSno':
                loss_train = loss_fn(tr_sno_f_ts, sno_f_ts, training)
            if trainingData == 'trainingSubAndSnow':
                modifiedSnow = sno_s_ts/2.0 
                loss_train_sno = loss_fn(tr_sno_s_ts, modifiedSnow)
                loss_train_sub = loss_fn(tr_sub_s_ts, sub_s_ts)
                loss_train = (loss_train_sno/4.0 + loss_train_sub)/2.0  # /4.0 to standardize snow

        lossTrainingSeries.append(loss_train.item())
        epochSeries.append(epoch)

        # STOPPING, ie VALIDATION
        if epoch == 1 or epoch % 1 == 0:
#            tr_sno_s_ts_areasSto, tr_sub_s_ts_areasSto, tr_sno_f_ts_areasSto, tr_sub_f_ts_areasSto, tr_eva_f_ts_areasSto = hydromodel(
#                   linearArt,
#                   True,
#                   torch.tensor(sno_s_initial),
#                   torch.tensor(sub_s_initial),
#                   torch.tensor(temperatureTimeSeriesSto),
#                   torch.tensor(precipitationTimeSeriesSto),
#                   modeEva = modeEvaTrain,
#                   modeSno = modeSnoTrain,
#                   modeSub = modeSubTrain,
#                   modeTem = modeTemTrain,
#                   modeEvP = modeEvPTrain
#                   )
#            tr_sno_s_tsSto = tr_sno_s_ts_areasSto.mean(dim = 0)
#            tr_sub_s_tsSto = tr_sub_s_ts_areasSto.mean(dim = 0)
#            tr_sno_f_tsSto = tr_sno_f_ts_areasSto.mean(dim = 0)
#            tr_sub_f_tsSto = tr_sub_f_ts_areasSto.mean(dim = 0)
#            tr_eva_f_tsSto = tr_eva_f_ts_areasSto.mean(dim = 0)
            if fitOnObservations:
                    loss_trainSto = loss_fn(tr_sub_f_ts, torch.tensor(streamFlowTimeSeries)/conversionFluxes, stopping)
            else:
                    loss_trainSto = loss_fn(tr_sub_f_ts, sub_f_ts, stopping)

        lossStoppingSeries.append(loss_trainSto.item())
        counter, stop = early_stopper.early_stop(loss_trainSto)
        stopperCounterSeries.append(counter)

        # VALIDATION, ie TESTING
        if epoch == 1 or epoch % 50 == 0 or stop:
            tr_sno_s_ts_areasVal, tr_sub_s_ts_areasVal, tr_sno_f_ts_areasVal, tr_sub_f_ts_areasVal, tr_eva_f_ts_areasVal = hydromodel(
                   linearArt,
                   True,
                   torch.tensor(sno_s_initial),
                   torch.tensor(sub_s_initial),
                   torch.tensor(temperatureTimeSeriesVal),
                   torch.tensor(precipitationTimeSeriesVal),
                   modeEva = modeEvaTrain,
                   modeSno = modeSnoTrain,
                   modeSub = modeSubTrain,
                   modeTem = modeTemTrain,
                   modeEvP = modeEvPTrain
                   )
            tr_sno_s_tsVal = tr_sno_s_ts_areasVal.mean(dim = 0)
            tr_sub_s_tsVal = tr_sub_s_ts_areasVal.mean(dim = 0)
            tr_sno_f_tsVal = tr_sno_f_ts_areasVal.mean(dim = 0)
            tr_sub_f_tsVal = tr_sub_f_ts_areasVal.mean(dim = 0)
            tr_eva_f_tsVal = tr_eva_f_ts_areasVal.mean(dim = 0)
            validation = numpy.logical_or(stopping, training)
            if fitOnObservations:
                    loss_trainVal = loss_fn(tr_sub_f_tsVal, torch.tensor(streamFlowTimeSeriesVal)/conversionFluxes, validation)
            else:
                    loss_trainVal = loss_fn(tr_sub_f_tsVal, sub_f_tsVal, validation)

        lossValidationSeries.append(loss_trainVal.item())

        if L1Regularization:
            # Compute L1 loss component
            l1_weight = 0.000001
            l1_parameters = []
            for parameter in model.parameters():
                l1_parameters.append(parameter.view(-1))
            l1 = l1_weight * model.compute_l1_loss(torch.cat(l1_parameters))
     
            # loss without reg
            print(loss_train, l1.item())

            # Add L1 loss component
            loss_train += l1

        if epoch == 1 or epoch % 20 == 0 or stop:
            print("\n###############################")
            #print('running scenario: ', scenarioDirectory, ' running one area: ', oneArea)
            print('running scenario: ', scenarioDirectory, ' Running one area: ', oneArea, '. Duration of one epoch:', duration)
            print(f"Epoch {epoch}, Training loss {loss_train.item():.10},", end = '')
            print(f" Stopping (i.e. validation) loss {loss_trainSto.item():.10},", end = '')
            print(f" Validation (i.e. testing) loss {loss_trainVal.item():.10},")
            if fitOnObservations:
                print("Fitting on observations")
            else:
                print("Fitting on artificial data")
            #print('learning rates (eva, sub, sno)', scheduler.get_last_lr())
            print('learning rate sub', optimizer.param_groups[0]["lr"])
            if L1Regularization:
                print('l1', l1.item())
            print(epoch, 'snow flux', tr_sno_f_ts.mean().item(), end = '')
            print(epoch, 'snow stor', tr_sno_s_ts.mean().item(), end = '')
            print(epoch, 'sub flux', tr_sub_f_ts.mean().item(), end = '')
            print(epoch, 'sub stor', tr_sub_s_ts.mean().item(), end = '')
            print(epoch, 'eva flux', tr_eva_f_ts.mean().item())
            if modeSubTrain == 'fitExpert':
                print('seepage parameter:', hydromodel.sub_parameter.detach().numpy()[0], end = ' ')
                numpy.save(sfdAr + 'sub_parameter.npy', hydromodel.sub_parameter.detach().numpy())
            if modeSnoTrain == 'fitExpert':
                print('snow melt parameter:', hydromodel.sno_parameter.detach().numpy()[0], end = ' ')
                numpy.save(sfdAr + 'sno_parameter.npy', hydromodel.sno_parameter.detach().numpy())
            if modeEvaTrain == 'fitExpert':
                print('eva parameter:', hydromodel.eva_parameter.detach().numpy()[0], end = ' ')
                numpy.save(sfdAr + 'eva_parameter.npy', hydromodel.eva_parameter.detach().numpy())
            if modeTemTrain == 'fitExpert':
                print('tem parameter:', hydromodel.tem_parameter.detach().numpy()[0], end = ' ')
                numpy.save(sfdAr + 'tem_parameter.npy', hydromodel.tem_parameter.detach().numpy())
            if modeEvPTrain == 'fitExpert':
                print('evp parameter (par, sigmoid):', hydromodel.evp_parameter.detach().numpy()[0], torch.sigmoid(hydromodel.evp_parameter).detach().numpy()[0])
                numpy.save(sfdAr + 'evp_parameter.npy', hydromodel.evp_parameter.detach().numpy())
                numpy.save(sfdAr + 'evp_sig_parameter.npy', torch.sigmoid(hydromodel.evp_parameter).detach().numpy()[0])

            a, b = hydromodel.com_f_response(modeSubTrain, deepLayer, 'sub', linearArt)
            scatterplot_rich(sfd, 'response_sub', a, b, 'storage (m)', 'potential flux (m/day)', '-', (0,0.02))
            numpy.save(sfdAr + 'response_sub_x.npy',numpy.array(a))
            numpy.save(sfdAr + 'response_sub_y.npy',numpy.array(b))

            a, b = hydromodel.com_f_response(modeSnoTrain, deepLayer, 'sno', linearArt)
            scatterplot_rich(sfd, 'response_sno', a, b, 'temperature (C)', 'potential flux (m/day)', '-', (0,0.02))
            numpy.save(sfdAr + 'response_sno_x.npy',numpy.array(a))
            numpy.save(sfdAr + 'response_sno_y.npy',numpy.array(b))

            a, b = hydromodel.com_f_response(modeEvaTrain, deepLayer, 'eva', linearArt)
            scatterplot_rich(sfd, 'response_eva', a, b, 'temperature (C)', 'potential flux (m/day)', '-', (0,0.02))
            numpy.save(sfdAr + 'response_eva_x.npy',numpy.array(a))
            numpy.save(sfdAr + 'response_eva_y.npy',numpy.array(b))

            # blauw, oranje, groen
            timeSeriesPlot_rich(sfd, 'loss', epochSeries, [torch.min(torch.tensor(lossTrainingSeries), torch.tensor(1e-5)),
                                                           torch.min(torch.tensor(lossStoppingSeries), torch.tensor(1e-5)), 
                                                           torch.min(torch.tensor(lossValidationSeries), torch.tensor(1e-5))] , 'epoch', 'loss')
            timeSeriesPlot_rich(sfd, 'stopCounter', epochSeries, [stopperCounterSeries], 'epoch', 'counter')
            numpy.save(sfdAr + 'epochs.npy', numpy.array(epochSeries))
            numpy.save(sfdAr + 'lossTraining.npy', numpy.array(lossTrainingSeries))
            numpy.save(sfdAr + 'lossStopping.npy', numpy.array(lossStoppingSeries))
            numpy.save(sfdAr + 'lossValidation.npy', numpy.array(lossValidationSeries))
            # mean over area only
            # training
            timeSeriesPlot_rich(sfd, 'train_ts_eva_f', dateTimeSeries, [tr_eva_f_ts.detach().numpy(),eva_f_ts.detach().numpy()], 'time', 'flux (m/day)')
            timeSeriesPlot_rich(sfd, 'train_ts_sub_f', dateTimeSeries, [tr_sub_f_ts.detach().numpy(),sub_f_ts.detach().numpy()], 'time', 'flux (m/day)')
            timeSeriesPlot_rich(sfd, 'train_ts_sub_s', dateTimeSeries, [tr_sub_s_ts.detach().numpy(),sub_s_ts.detach().numpy()], 'time', 'storage (m/day)')
            timeSeriesPlot_rich(sfd, 'train_ts_sno_f', dateTimeSeries, [tr_sno_f_ts.detach().numpy(),sno_f_ts.detach().numpy()], 'time', 'flux (m/day)')
            timeSeriesPlot_rich(sfd, 'train_ts_sno_s', dateTimeSeries, [tr_sno_s_ts.detach().numpy(),sno_s_ts.detach().numpy()], 'time', 'storage (m/day)')
            timeSeriesPlot_rich(sfd, 'train_ts_OBS', dateTimeSeries, [tr_sub_f_ts.detach().numpy(),torch.tensor(streamFlowTimeSeries)/conversionFluxes], 'time', 'flux (m/day')
            numpy.save(sfdAr + 'train_ts_eva_f.npy', numpy.array(tr_eva_f_ts.detach().numpy()))
            numpy.save(sfdAr + 'train_ts_sub_f.npy', numpy.array(tr_sub_f_ts.detach().numpy()))
            numpy.save(sfdAr + 'train_ts_sub_s.npy', numpy.array(tr_sub_s_ts.detach().numpy()))
            numpy.save(sfdAr + 'train_ts_sno_f.npy', numpy.array(tr_sno_f_ts.detach().numpy()))
            numpy.save(sfdAr + 'train_ts_sno_s.npy', numpy.array(tr_sno_s_ts.detach().numpy()))
            numpy.save(sfdAr + 'train_ts_OBS', (torch.tensor(streamFlowTimeSeries)/conversionFluxes).numpy())
            numpy.save(sfdAr + 'train_date.npy', numpy.array(dateTimeSeries))
            # validation
            timeSeriesPlot_rich(sfd, 'valid_ts_eva_f', dateTimeSeriesVal, [tr_eva_f_tsVal.detach().numpy()], 'time', 'flux (m/day)')
            timeSeriesPlot_rich(sfd, 'valid_ts_sub_f', dateTimeSeriesVal, [tr_sub_f_tsVal.detach().numpy()], 'time', 'flux (m/day)')
            timeSeriesPlot_rich(sfd, 'valid_ts_sub_s', dateTimeSeriesVal, [tr_sub_s_tsVal.detach().numpy()], 'time', 'storage (m/day)')
            timeSeriesPlot_rich(sfd, 'valid_ts_sno_f', dateTimeSeriesVal, [tr_sno_f_tsVal.detach().numpy()], 'time', 'flux (m/day)')
            timeSeriesPlot_rich(sfd, 'valid_ts_sno_s', dateTimeSeriesVal, [tr_sno_s_tsVal.detach().numpy()], 'time', 'storage (m/day)')
            timeSeriesPlot_rich(sfd, 'valid_ts_OBS', dateTimeSeriesVal, [tr_sub_f_tsVal.detach().numpy(),torch.tensor(streamFlowTimeSeriesVal)/conversionFluxes], 'time', 'flux (m/day')
            numpy.save(sfdAr + 'valid_ts_eva_f.npy', numpy.array(tr_eva_f_tsVal.detach().numpy()))
            numpy.save(sfdAr + 'valid_ts_sub_f.npy', numpy.array(tr_sub_f_tsVal.detach().numpy()))
            numpy.save(sfdAr + 'valid_ts_sub_s.npy', numpy.array(tr_sub_s_tsVal.detach().numpy()))
            numpy.save(sfdAr + 'valid_ts_sno_f.npy', numpy.array(tr_sno_f_tsVal.detach().numpy()))
            numpy.save(sfdAr + 'valid_ts_sno_s.npy', numpy.array(tr_sno_s_tsVal.detach().numpy()))
            numpy.save(sfdAr + 'valid_ts_OBS', (torch.tensor(streamFlowTimeSeriesVal)/conversionFluxes).numpy())
            numpy.save(sfdAr + 'valid_date.npy', numpy.array(dateTimeSeriesVal))

            # scatterplots, indep vs dep
            # mean over area only
            # evapotranspiration, predicted and observations
            scatterplot_rich(sfd, 'loss_pre_eva_f_sc', temperatureTimeSeries, tr_eva_f_ts.detach().numpy(), 'temperature (C)', 'flux (m/day)', '.', (0,0.01))
            scatterplot_rich(sfd, 'loss_eva_f_sc', temperatureTimeSeries, eva_f_ts, 'temperature (C)', 'flux (m/day)', '.', (0,0.01))
            # snow melt, predicted and observations
            scatterplot_rich(sfd, 'loss_pre_sno_f_sc', temperatureTimeSeries, tr_sno_f_ts.detach().numpy(), 'temperature (C)', 'flux (m/day)', '.', (0,0.02))
            scatterplot_rich(sfd, 'loss_sno_f_sc', temperatureTimeSeries, sno_f_ts, 'temperature (C))', 'flux (m/day)', '.', (0,0.02))
            # subsurface, predicted and observations
            scatterplot_rich(sfd, 'loss_pre_sub_f_sc', tr_sub_s_ts.detach().numpy(),tr_sub_f_ts.detach().numpy(), 'storage (m)', 'flux (m/day)', '.', (0,0.05) )
            scatterplot_rich(sfd, 'loss_sub_f_sc', sub_s_ts, sub_f_ts, 'storage (m)', 'flux (m/day)', '.', (0,0.05) )

            if not oneArea:
                # by area, training
                a = tr_sno_s_ts_areas.detach().numpy()
                timeSeriesPlot_rich(sfd, "train_ts_sno_s_areas", dateTimeSeries, [a[0],a[1]], 'time', 'storage (m)')
                numpy.save(sfdAr + 'train_ts_sno_s_areas.npy', a)
                a = tr_sub_s_ts_areas.detach().numpy()
                timeSeriesPlot_rich(sfd, "train_ts_sub_s_areas", dateTimeSeries, [a[0],a[1]], 'time', 'storage (m)')
                numpy.save(sfdAr + 'train_ts_sub_s_areas.npy', a)
                a = tr_sno_f_ts_areas.detach().numpy()
                timeSeriesPlot_rich(sfd, "train_ts_sno_f_areas", dateTimeSeries, [a[0],a[1]], 'time', 'flux (m/day)')
                numpy.save(sfdAr + 'train_ts_sno_f_areas.npy', a)
                a = tr_sub_f_ts_areas.detach().numpy()
                timeSeriesPlot_rich(sfd, "train_ts_sub_f_areas", dateTimeSeries, [a[0],a[1]], 'time', 'flux (m/day)')
                numpy.save(sfdAr + 'train_ts_sub_f_areas.npy', a)
                a = tr_eva_f_ts_areas.detach().numpy()
                timeSeriesPlot_rich(sfd, "train_ts_eva_f_areas", dateTimeSeries, [a[0],a[1]], 'time', 'flux (m/day)')
                numpy.save(sfdAr + 'train_ts_eva_f_areas.npy', a)
                # by area, validation
                a = tr_sno_s_ts_areasVal.detach().numpy()
                timeSeriesPlot_rich(sfd, "valid_ts_sno_s_areas", dateTimeSeries, [a[0],a[1]], 'time', 'storage (m)')
                numpy.save(sfdAr + 'valid_ts_sno_s_areas.npy', a)
                a = tr_sub_s_ts_areasVal.detach().numpy()
                timeSeriesPlot_rich(sfd, "valid_ts_sub_s_areas", dateTimeSeries, [a[0],a[1]], 'time', 'storage (m)')
                numpy.save(sfdAr + 'valid_ts_sub_s_areas.npy', a)
                a = tr_sno_f_ts_areasVal.detach().numpy()
                timeSeriesPlot_rich(sfd, "valid_ts_sno_f_areas", dateTimeSeries, [a[0],a[1]], 'time', 'flux (m/day)')
                numpy.save(sfdAr + 'valid_ts_sno_f_areas.npy', a)
                a = tr_sub_f_ts_areasVal.detach().numpy()
                timeSeriesPlot_rich(sfd, "valid_ts_sub_f_areas", dateTimeSeries, [a[0],a[1]], 'time', 'flux (m/day)')
                numpy.save(sfdAr + 'valid_ts_sub_f_areas.npy', a)
                a = tr_eva_f_ts_areasVal.detach().numpy()
                timeSeriesPlot_rich(sfd, "valid_ts_eva_f_areas", dateTimeSeries, [a[0],a[1]], 'time', 'flux (m/day)')
                numpy.save(sfdAr + 'valid_ts_eva_f_areas.npy', a)

        if stop:
            print('break')
            break



        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        #scheduler.step()
        scheduler.step(loss_trainSto)

        #counter, stop = early_stopper.early_stop(loss_trainSto)
        #stopperSeries.append(loss_trainVal.item())


def loss_fn(t_p, t_c, period):
    a = 366   #must be used as spin up is embedded in training/stopping and validation data sets
    #periodFloat = torch.tensor(numpy.where(period, 1.0, 0.0))[a:]
    period = period[a:]
    squared_diffs = ((t_p[a:] - t_c[a:])**2.0)[period]
    return squared_diffs.mean()


## with momentum, works very well voor sub (sneller en precieser met zelfde loss)
#optimizer = optim.RMSprop(hydromodel.parameters(), lr = 0.00001, momentum = 0.9) 
#
## plateau https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, eps = 1e-8, patience = 50)



# fit model on observations or on artificial data
fitOnObservations = True 
# if fitting on artificial data, use linear model or non-linear one
linearArtForNotFitOnObservations = False
# add error to artificial data
addErrorToArtificialStreamFlow = False

# training data to calculate loss over (typical sub, i.e. outflow)
trainingData = 'trainingSub'

# folder to store output data, note the / at the end
outputDirectory = '../data/results/'


# if fitting on real data, linear models for components not represented as NN are assumed
# if fitting on artificial data, the option defined above is used
if fitOnObservations:
    linearArt = True
else:
    linearArt = linearArtForNotFitOnObservations 

# data for training or stopping ie validation
# training period 1
startOne = datetime.date(1979, 10, 1)
endOne = datetime.date(1996, 9, 26)
temperatureTimeSeries, precipitationTimeSeries = createMeteoData(inputDataDirectory, outputDataDirectory, startOne, endOne)
streamFlowTimeSeries, dateTimeSeries = createStreamFlowData(inputDataDirectory, outputDataDirectory, startOne, endOne)
sno_s_ts, sub_s_ts, sno_f_ts, sub_f_ts, eva_f_ts = createArtificialObservations( \
                                                           temperatureTimeSeries, precipitationTimeSeries, addErrorToArtificialStreamFlow, linearArt)

# training period 2 NOT USED ANYMORE
startTwo = datetime.date(1990, 10, 1)
#startTwo = datetime.date(1996, 10, 1)
endTwo = datetime.date(2001, 9, 1)
temperatureTimeSeriesTwo, precipitationTimeSeriesTwo = createMeteoData(inputDataDirectory, outputDataDirectory, startTwo, endTwo)
streamFlowTimeSeriesTwo, dateTimeSeriesTwo = createStreamFlowData(inputDataDirectory, outputDataDirectory, startTwo, endTwo)
sno_s_tsTwo, sub_s_tsTwo, sno_f_tsTwo, sub_f_tsTwo, eva_f_tsTwo = createArtificialObservations( \
                                                           temperatureTimeSeriesTwo, precipitationTimeSeriesTwo, addErrorToArtificialStreamFlow, linearArt)

# data for validation, ie testing
startVal = datetime.date(1995, 10 , 1)
endVal = datetime.date(2012, 9, 26)
temperatureTimeSeriesVal, precipitationTimeSeriesVal = createMeteoData(inputDataDirectory, outputDataDirectory, startVal, endVal)
streamFlowTimeSeriesVal, dateTimeSeriesVal = createStreamFlowData(inputDataDirectory, outputDataDirectory, startVal, endVal)
# note that no error is added to artificial streamflow in any case as it is used for validation only
sno_s_tsVal, sub_s_tsVal, sno_f_tsVal, sub_f_tsVal, eva_f_tsVal = createArtificialObservations( \
                                                           temperatureTimeSeriesVal, precipitationTimeSeriesVal, False, linearArt)


sno_s_initial = 0.0
sub_s_initial = 0.0

#####################
# fitting scenarios #
#####################

nrEpochs = 5000

# expert models (n = 7)

xub =       {'name': 'fit_xub',
             'modeEvaTrain': 'obsCreation',
             'modeSnoTrain': 'obsCreation',
             'modeSubTrain': 'fitExpert',
             'modeTemTrain': 'obsCreation',
             'modeEvPTrain': 'obsCreation',
             'nrEpochs': nrEpochs}
xno =       {'name': 'fit_xno',
             'modeEvaTrain': 'obsCreation',
             'modeSnoTrain': 'fitExpert',
             'modeSubTrain': 'obsCreation',
             'modeTemTrain': 'obsCreation',
             'modeEvPTrain': 'obsCreation',
             'nrEpochs': nrEpochs}
xva =       {'name': 'fit_xva',
             'modeEvaTrain': 'fitExpert',
             'modeSnoTrain': 'obsCreation',
             'modeSubTrain': 'obsCreation',
             'modeTemTrain': 'obsCreation',
             'modeEvPTrain': 'obsCreation',
             'nrEpochs': nrEpochs}
xus =       {'name': 'fit_xus',
             'modeEvaTrain': 'obsCreation',
             'modeSnoTrain': 'fitExpert',
             'modeSubTrain': 'fitExpert',
             'modeTemTrain': 'obsCreation',
             'modeEvPTrain': 'obsCreation',
             'nrEpochs': nrEpochs}
xue =       {'name': 'fit_xue',
             'modeEvaTrain': 'fitExpert',
             'modeSnoTrain': 'obsCreation',
             'modeSubTrain': 'fitExpert',
             'modeTemTrain': 'obsCreation',
             'modeEvPTrain': 'obsCreation',
             'nrEpochs': nrEpochs}
xne =       {'name': 'fit_xne',
             'modeEvaTrain': 'fitExpert',
             'modeSnoTrain': 'fitExpert',
             'modeSubTrain': 'obsCreation',
             'modeTemTrain': 'obsCreation',
             'modeEvPTrain': 'obsCreation',
             'nrEpochs': nrEpochs}
xhr =       {'name': 'fit_xhr',
             'modeEvaTrain': 'fitExpert',
             'modeSnoTrain': 'fitExpert',
             'modeSubTrain': 'fitExpert',
             'modeTemTrain': 'obsCreation',      # fixed as not considered here
             'modeEvPTrain': 'obsCreation',      # idem
             'nrEpochs': nrEpochs}


# machine learning models (n = 7)

sub =       {'name': 'fit_sub',
             'modeEvaTrain': 'obsCreation',
             'modeSnoTrain': 'obsCreation',
             'modeSubTrain': 'fit',
             'modeTemTrain': 'obsCreation',
             'modeEvPTrain': 'obsCreation',
             'nrEpochs': nrEpochs}
sno =       {'name': 'fit_sno',
             'modeEvaTrain': 'obsCreation',
             'modeSnoTrain': 'fit',
             'modeSubTrain': 'obsCreation',
             'modeTemTrain': 'obsCreation',
             'modeEvPTrain': 'obsCreation',
             'nrEpochs': nrEpochs}
eva =       {'name': 'fit_eva',
             'modeEvaTrain': 'fit',
             'modeSnoTrain': 'obsCreation',
             'modeSubTrain': 'obsCreation',
             'modeTemTrain': 'obsCreation',
             'modeEvPTrain': 'obsCreation',
             'nrEpochs': nrEpochs}
sus =       {'name': 'fit_sus',
             'modeEvaTrain': 'obsCreation',
             'modeSnoTrain': 'fit',
             'modeSubTrain': 'fit',
             'modeTemTrain': 'obsCreation',
             'modeEvPTrain': 'obsCreation',
             'nrEpochs': nrEpochs}
sue =       {'name': 'fit_sue',
             'modeEvaTrain': 'fit',
             'modeSnoTrain': 'obsCreation',
             'modeSubTrain': 'fit',
             'modeTemTrain': 'obsCreation',
             'modeEvPTrain': 'obsCreation',
             'nrEpochs': nrEpochs}
sne =       {'name': 'fit_sne',
             'modeEvaTrain': 'fit',
             'modeSnoTrain': 'fit',
             'modeSubTrain': 'obsCreation',
             'modeTemTrain': 'obsCreation',
             'modeEvPTrain': 'obsCreation',
             'nrEpochs': nrEpochs}
thr =       {'name': 'fit_thr',
             'modeEvaTrain': 'fit',
             'modeSnoTrain': 'fit',
             'modeSubTrain': 'fit',
             'modeTemTrain': 'obsCreation',
             'modeEvPTrain': 'obsCreation',
             'nrEpochs': nrEpochs}

# fittingScenarios define what components are fitted and what aren't
#fittingScenarios = [xva, xno, xub, xne, xue, xus, xhr]            # expert models, code is x plus second and third letter of corresponding ML scenario
#fittingScenarios = [eva, sno, sub, sne, sue, sus, thr]            # ML models
fittingScenarios = [xva, xno, xub, xne, xue, xus, xhr, eva]        # expert models and eva added for testing

outOne, outTwo, outThree, outFour = createTrainingIndices()

# training scenarios
one =       {'name': '1',
             'stopping': outOne,
             'temperature': temperatureTimeSeries,
             'precipitation': precipitationTimeSeries,
             'streamFlow': streamFlowTimeSeries,
             'date': dateTimeSeries,
             'temperatureSto': temperatureTimeSeriesTwo,
             'precipitationSto': precipitationTimeSeriesTwo,
             'streamFlowSto': streamFlowTimeSeriesTwo,
             'dateSto': dateTimeSeriesTwo,
             'sno_s_ts': sno_s_ts,
             'sub_s_ts': sub_s_ts,
             'sno_f_ts': sno_f_ts,
             'sub_f_ts': sub_f_ts,
             'eva_f_ts': eva_f_ts,
             'sno_s_tsSto': sno_s_tsTwo,
             'sub_s_tsSto': sub_s_tsTwo,
             'sno_f_tsSto': sno_f_tsTwo,
             'sub_f_tsSto': sub_f_tsTwo,
             'eva_f_tsSto': eva_f_tsTwo
             }

two = one.copy()
three = one.copy()
four = one.copy()

two['name'] = '2'
three['name'] = '3'
four['name'] = '4'

two['stopping'] = outTwo 
three['stopping'] = outThree 
four['stopping'] = outFour 

trainingScenarios = [one, two, three, four]
#trainingScenarios = [one, two]
#trainingScenarios = [three, four]

# rerun scenarios
numberOfScenarios = 4
aRange = numpy.arange(1,numberOfScenarios + 1)
reRunScenarios = []
for s in aRange:
    reRunScenarios.append(str(s))

if runInBatch:
    fittingScenarios = [fittingScenarios[int(first)-1]]
    trainingScenarios = trainingScenarios[int(second)-1:int(second)+1]
    reRunScenarios = str.split(third,',')

for fs in fittingScenarios:
    for ts in trainingScenarios:
        for rs in reRunScenarios:
            scenarioDirectory = fs["name"] + "/" + ts["name"] + '/' + rs
            print(scenarioDirectory)
            seed = seedGenerator(scenarioDirectory)
            print('seed is', seed)
            torch.manual_seed(seed)
            hydromodel = Net()
            optimizer = optim.RMSprop([
                {'params': hydromodel.params.sub.parameters(), 'lr': 0.00001, 'momentum': 0.9},
                {'params': hydromodel.params.sno.parameters(), 'lr': 0.00001, 'momentum': 0.9},
                {'params': hydromodel.params.eva.parameters(), 'lr': 0.00001, 'momentum': 0.9},
                {'params': hydromodel.params.exp.parameters(), 'lr': 0.00001, 'momentum': 0.9}
                            ])
            # plateau https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
            #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.95, eps = 1e-8, patience = 50)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.95, eps = 1e-8, patience = 200)

            training_loop(fs['nrEpochs'],
                         ts['stopping'],
                         optimizer,
                         hydromodel,
                         loss_fn,
                         sno_s_initial,
                         sub_s_initial,
                         ts['temperature'],
                         ts['precipitation'],
                         ts['streamFlow'],
                         ts['date'],
                         ts['temperatureSto'],
                         ts['precipitationSto'],
                         ts['streamFlowSto'],
                         ts['dateSto'],
                         temperatureTimeSeriesVal,
                         precipitationTimeSeriesVal,
                         streamFlowTimeSeriesVal,
                         dateTimeSeriesVal,
                         ts['sno_s_ts'], ts['sub_s_ts'], ts['sno_f_ts'], ts['sub_f_ts'], ts['eva_f_ts'],
                         ts['sno_s_tsSto'], ts['sub_s_tsSto'], ts['sno_f_tsSto'], ts['sub_f_tsSto'], ts['eva_f_tsSto'],
                         sno_s_tsVal, sub_s_tsVal, sno_f_tsVal, sub_f_tsVal, eva_f_tsVal,
                         trainingData,
                         fitOnObservations,
                         fs['modeEvaTrain'],
                         fs['modeSnoTrain'],
                         fs['modeSubTrain'],
                         fs['modeTemTrain'],
                         fs['modeEvPTrain'],
                         outputDirectory,
                         scenarioDirectory,
                         linearArt
                         )

exit()

#######

scenarioDirectory = 'fitSub'

hydromodel = Net()
# with momentum, works very well voor sub (sneller en precieser met zelfde loss)
optimizer = optim.RMSprop(hydromodel.parameters(), lr = 0.00001, momentum = 0.9) 
# plateau https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, eps = 1e-8, patience = 50)

# what to fit
modeEvaTrain = 'obsCreation'
#modeEvaTrain = 'fit'
#modeEvaTrain = 'fitExpert'

modeSnoTrain = 'obsCreation'
#modeSnoTrain = 'fit'
#modeSnoTrain = 'fitExpert'

#modeSubTrain = 'obsCreation'
modeSubTrain = 'fit'
#modeSubTrain = 'fitExpert'

modeTemTrain = "obsCreation"
#modeTemTrain = 'fitExpert'

modeEvPTrain = "obsCreation"
#modeEvPTrain = "fitExpert"

training_loop(my_n_epochs, optimizer, hydromodel, loss_fn, sno_s_initial, sub_s_initial, temperatureTimeSeries, precipitationTimeSeries, \
              temperatureTimeSeriesVal, precipitationTimeSeriesVal, trainingData, fitOnObservations, modeEvaTrain, modeSnoTrain, modeSubTrain, \
              modeTemTrain, modeEvPTrain, outputDirectory, scenarioDirectory)

#######
my_n_epochs = 5000

scenarioDirectory = 'fitSno'

hydromodel = Net()
# with momentum, works very well voor sub (sneller en precieser met zelfde loss)
optimizer = optim.RMSprop(hydromodel.parameters(), lr = 0.00001, momentum = 0.9) 
# plateau https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, eps = 1e-8, patience = 50)

# what to fit
modeEvaTrain = 'obsCreation'
#modeEvaTrain = 'fit'
#modeEvaTrain = 'fitExpert'

#modeSnoTrain = 'obsCreation'
modeSnoTrain = 'fit'
#modeSnoTrain = 'fitExpert'

modeSubTrain = 'obsCreation'
#modeSubTrain = 'fit'
#modeSubTrain = 'fitExpert'

modeTemTrain = "obsCreation"
#modeTemTrain = 'fitExpert'

modeEvPTrain = "obsCreation"
#modeEvPTrain = "fitExpert"

training_loop(my_n_epochs, optimizer, hydromodel, loss_fn, sno_s_initial, sub_s_initial, temperatureTimeSeries, precipitationTimeSeries, \
              temperatureTimeSeriesVal, precipitationTimeSeriesVal, trainingData, fitOnObservations, modeEvaTrain, modeSnoTrain, modeSubTrain, \
              modeTemTrain, modeEvPTrain, outputDirectory, scenarioDirectory)

#######
my_n_epochs = 5000

scenarioDirectory = 'fitSubSno'

hydromodel = Net()
# with momentum, works very well voor sub (sneller en precieser met zelfde loss)
# optimizer = optim.RMSprop(hydromodel.parameters(), lr = 0.00001, momentum = 0.9) 
optimizer = optim.RMSprop([
        {'params': hydromodel.params.eva.parameters(), 'lr': 1e-2},
        {'params': hydromodel.params.sub.parameters(), 'lr': 1e-3},
        {'params': hydromodel.params.sno.parameters(), 'lr': 1e-4}
    ])
# plateau https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, eps = 1e-8, patience = 50)

# what to fit
modeEvaTrain = 'obsCreation'
#modeEvaTrain = 'fit'
#modeEvaTrain = 'fitExpert'

#modeSnoTrain = 'obsCreation'
modeSnoTrain = 'fit'
#modeSnoTrain = 'fitExpert'

#modeSubTrain = 'obsCreation'
modeSubTrain = 'fit'
#modeSubTrain = 'fitExpert'

modeTemTrain = "obsCreation"
#modeTemTrain = 'fitExpert'

modeEvPTrain = "obsCreation"
#modeEvPTrain = "fitExpert"

training_loop(my_n_epochs, optimizer, hydromodel, loss_fn, sno_s_initial, sub_s_initial, temperatureTimeSeries, precipitationTimeSeries, \
              temperatureTimeSeriesVal, precipitationTimeSeriesVal, trainingData, fitOnObservations, modeEvaTrain, modeSnoTrain, modeSubTrain, \
              modeTemTrain, modeEvPTrain, outputDirectory, scenarioDirectory)



exit()

#PLOTS FOR ARTIFICIAL OBSERVATIONS, WORKS FOR TWO AREAS
## mean
#timeSeriesPlot(dateTimeSeries, [sno_s_ts], "sno_s_ts")
#timeSeriesPlot(dateTimeSeries, [sub_s_ts], "sub_s_ts")
#timeSeriesPlot(dateTimeSeries, [sno_f_ts], "sno_f_ts")
#timeSeriesPlot(dateTimeSeries, [sub_f_ts], "sub_f_ts")
#timeSeriesPlot(dateTimeSeries, [eva_f_ts], "eva_f_ts")
#
## by area
#timeSeriesPlot(dateTimeSeries, sno_s_ts_areas, "sno_s_ts_areas")
#timeSeriesPlot(dateTimeSeries, sub_s_ts_areas, "sub_s_ts_areas")
#timeSeriesPlot(dateTimeSeries, sno_f_ts_areas, "sno_f_ts_areas")
#timeSeriesPlot(dateTimeSeries, sub_f_ts_areas, "sub_f_ts_areas")
#timeSeriesPlot(dateTimeSeries, eva_f_ts_areas, "eva_f_ts_areas")
#scatterplot(temperatureTimeSeries, sno_f_ts_areas[0], "sno_f_areas_sc_00", '.')
#scatterplot(temperatureTimeSeries, sno_f_ts_areas[1], "sno_f_areas_sc_01", '.')
#scatterplot(sub_s_ts_areas[0],sub_f_ts_areas[0], "sub_f_areas_sc_00", '.')
#scatterplot(sub_s_ts_areas[1],sub_f_ts_areas[1], "sub_f_areas_sc_01", '.')


##########################################
#            ## for separate areas
#            timeSeriesPlot(dateTimeSeries, [tr_eva_f_ts_areas.swapaxes(0,1).detach().numpy()], 'loss_eva_f_areas_check')
#            timeSeriesPlot(dateTimeSeries, [tr_sub_f_ts_areas.swapaxes(0,1).detach().numpy()], 'loss_sub_f_areas_check')
#            timeSeriesPlot(dateTimeSeries, [tr_sub_s_ts_areas.swapaxes(0,1).detach().numpy()], 'loss_sub_s_areas_check')
#            timeSeriesPlot(dateTimeSeries, [tr_sno_f_ts_areas.swapaxes(0,1).detach().numpy()], 'loss_sno_f_areas_check')
#            timeSeriesPlot(dateTimeSeries, [tr_sno_s_ts_areas.swapaxes(0,1).detach().numpy()], 'loss_sno_s_areas_check')
#            # for separate areas
#            # snow
#            scatterplot(temperatureTimeSeries, tr_sno_f_ts_areas.swapaxes(0,1).detach().numpy(), "loss_pre_sno_f_areas_sc",'.')
#            # sub
#            scatterplot(tr_sub_s_ts_areas.swapaxes(0,1).detach().numpy(),tr_sub_f_ts_areas.swapaxes(0,1).detach().numpy(), "loss_pre_sub_f_areas_sc",'.')
#    for name, param in model.named_parameters():
#        if param.requires_grad:
#            print(name, param.data)
#

##########################################

#        if epoch == 1190:
#            for param_group in optimizer.param_groups:
#                param_group['lr'] = 0.0003

#constraintsWeight = weightConstraint()
##hydromodel._modules['eva_f_d'].apply(constraintsWeight)
##hydromodel._modules['eva_f_c'].apply(constraintsWeight)
##hydromodel._modules['sno_f_d'].apply(constraintsWeight)
##hydromodel._modules['sno_f_c'].apply(constraintsWeight)
#hydromodel._modules['sub_f_d'].apply(constraintsWeight)
#hydromodel._modules['sub_f_c'].apply(constraintsWeight)
#if deepLayer:
##    hydromodel._modules['eva_f_dd'].apply(constraintsWeight)
##    hydromodel._modules['sno_f_dd'].apply(constraintsWeight)
#    hydromodel._modules['sub_f_dd'].apply(constraintsWeight)

#for name, param in hydromodel.named_parameters():
#    #print(name, param.shape)
#    print(name, param)


#streamFlowTimeSeries = (torch.tensor(streamFlowTimeSeries) + (sub_f_ts_areas[0] + sub_f_ts_areas[1])/2.0)/2.0
#print("averaging streamflow!!!!!!!!!")




#            for name, param in hydromodel.named_parameters():
#                if param.requires_grad and printParameters:
#                    if 'weight' in name:
#                        #print('data', name, param.data[0][0:2])
#                        if param.grad != None:
#                           #print('grad', name, param.grad[0][0:2].detach().numpy(), end = '')
#                           print('grad', name, param.grad[0].detach().numpy(), end = '')
#                           print('dtype', name, param.grad[0].dtype)
#                    if 'bias' in name:
#                        #print('data', name, param.data[0:2])
#                        if param.grad != None:
#                            #print('grad', name, param.grad[0:2].detach().numpy(), end = '')
#                            print('grad', name, param.grad.detach().numpy(), end = '')

#def loss_fn(t_p, t_c):
#    #squared_diffs = (t_p - t_c)**2
#    #squared_diffs = (t_p[365:] - t_c[365:])**2
#    #squared_diffs = ((t_p - t_c)**2)/(t_c+0.000000000001)
## loss met windowaverage
##    squared_diffs = (t_p[10:] - t_c[10:])**2
##    sma = torch.nn.AvgPool1d(kernel_size=3, stride = 1)
##    t_c_new = t_c.view(1,-1)
##    t_c_win = sma(t_c_new)
##    t_p_new = t_p.view(1,-1)
##    t_p_win = sma(t_p_new)
##    loss = ((t_p_win - t_c_win)**2.0)[0].mean()
##    return loss
## loss absolute error
##    lossje = nn.L1Loss()
##    return lossje(t_p, t_c)
## loss huber error
##    print('tp', t_p)
##    print('tc', t_c)
##    lossje = nn.HuberLoss(delta = 0.002)
##    return lossje(t_p, t_c)
## draw random subset of samples
##    print('%.3e' % ((t_p - t_c)**2.0).mean().detach().numpy(), ' # ', end = '')
##    p = (t_p / 1000.0)+0.5  # equal prob
##    #index = p.multinomial(num_samples=20, replacement=False)
##    index = p.multinomial(num_samples=60, replacement=False)
##    t_p = t_p[index]
##    t_c = t_c[index]
## loss standaard
#    #a = int(2 * 365)

## exponential
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.0001)  

#optimizer = optim.SGD(hydromodel.parameters(), lr=1e-1) # <1>
#optimizer = optim.Adam(mymodel.parameters(), lr=1e-3) # <1>
#optimizer = optim.Adam(mymodel.parameters(), lr=1e-2) # <1>  # works good for s1 free and s2 linear
#optimizer = optim.Adam(hydromodel.parameters(), lr=1e-3) # 
#optimizer = optim.Adam(hydromodel.parameters(), lr=1e-3) # 
#optimizer = optim.Adam(hydromodel.parameters(), lr=1e-3) # 


#if fitOnObservations:
#    #optimizer = optim.SGD(hydromodel.parameters(), lr=100)
#    #optimizer = optim.SGD(hydromodel.parameters(), lr=10.0)
##    optimizer = optim.SGD([
##        {'params': hydromodel.params.eva.parameters(), 'lr': 1e-3 },
##        {'params': hydromodel.params.sub.parameters(), 'lr': 10.0 },
##        {'params': hydromodel.params.sno.parameters(), 'lr': 5000}
##        ])
#    optimizer = optim.Adam([
#        {'params': hydromodel.params.eva.parameters(), 'lr': 1e-3},
#        {'params': hydromodel.params.sub.parameters(), 'lr': 1e-2},
#        {'params': hydromodel.params.sno.parameters(), 'lr': 1e-2}
#    ])
#else:
#    #optimizer = optim.SGD(hydromodel.parameters(), lr=100)
#    #optimizer = optim.SGD(hydromodel.parameters(), lr=10.0)
##    optimizer = optim.SGD([
##        {'params': hydromodel.params.eva.parameters(), 'lr': 1e-3 },
##        {'params': hydromodel.params.sub.parameters(), 'lr': 10.0 },
##        {'params': hydromodel.params.sno.parameters(), 'lr': 5000}
##        ])
#    optimizer = optim.Adam([
#        {'params': hydromodel.params.eva.parameters(), 'lr': 1e-3},
#        {'params': hydromodel.params.sub.parameters(), 'lr': 1e-2},
#        {'params': hydromodel.params.sno.parameters(), 'lr': 1e-2}
#    ])



pre_sno_s_ts, pre_sub_s_ts, pre_sno_f_ts, pre_sub_f_ts, pre_eva_f_ts = hydromodel(
               False,
               torch.tensor(sno_s_initial),
               torch.tensor(sub_s_initial),
               torch.tensor(temperatureTimeSeries),
               torch.tensor(precipitationTimeSeries),
               modeEva = modeEvaTrain,
               modeSno = modeSnoTrain,
               modeSub = modeSubTrain
               )

# timeseries, observed and predicted in one plot
# fluxes
timeSeriesPlot(dateTimeSeries, [eva_f_ts, pre_eva_f_ts.detach().numpy()], "val_eva_f_ts")
timeSeriesPlot(dateTimeSeries, [sno_f_ts, pre_sno_f_ts.detach().numpy()], "val_sno_f_ts")
timeSeriesPlot(dateTimeSeries, [sub_f_ts, pre_sub_f_ts.detach().numpy()], "val_sub_f_ts")
# storages
timeSeriesPlot(dateTimeSeries, [sub_s_ts, pre_sub_s_ts.detach().numpy()], "val_sub_s_ts")
timeSeriesPlot(dateTimeSeries, [sno_s_ts, pre_sno_s_ts.detach().numpy()], "val_sub_s_ts")

# scatterplots, indep vs dep
# evapotranspiration, predicted and observations
scatterplot(temperatureTimeSeries, pre_eva_f_ts.detach().numpy(), "pre_eva_f_sc",'.')
scatterplot(temperatureTimeSeries, eva_f_ts, "eva_f_sc",'.')
# snow melt, predicted and observations
scatterplot(temperatureTimeSeries, pre_sno_f_ts.detach().numpy(), "pre_sno_f_sc",'.')
scatterplot(temperatureTimeSeries, sno_f_ts, "sno_f_sc",'.')
# subsurface, predicted and observations
scatterplot(pre_sub_s_ts.detach().numpy(),pre_sub_f_ts.detach().numpy(), "pre_sub_f_sc",'.')
scatterplot(sub_s_ts,sub_f_ts, "sub_f_sc",'.')

