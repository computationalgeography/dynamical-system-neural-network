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


run_in_batch = False

# inputs for running in batch mode
if run_in_batch:
    # ranges from 1 up to and including the number of fitting scenarios used
    first = sys.argv[1]
    # is 1 or 3 (training scenarios, 1 represents 1 and 2, 3 represents 3 and 4)
    second = sys.argv[2]
    # is 1 or 1,2 or 1,2,3 etc, splitted at , (rerun scenario)
    third = sys.argv[3]

# note that for true, it may need to be in the parameter list of the optimizer
deep_layer = (True)

m = torch.nn.ReLU()

seedAll = 4
torch.manual_seed(seedAll)
numpy.random.seed(seedAll)

if deep_layer:
    nodes = 100
else:
    nodes = 1000

one_area = True
print_parameters = True
# https://github.com/christianversloot/machine-learning-articles/
# blob/main/how-to-use-l1-l2-and-elastic-net-regularization-with-pytorch.md
l1_regularization = False

input_data_directory = "../data/inputData/"
output_data_directory = "../data/outputData/"

conversion_fluxes = 549.3

# line width in plots
width = 0.25

# font size in plots
myFontSize = 7

# meteo start date is 1 jan 1979, end date is 31 juli 2014
# so do not select a time stem outside this range
# streamflow available always in this period



def seed_generator(string):
    tot = 0
    i = 1
    for letter in string:
        tot += ord(letter) * i
        i = i + 10
    return tot + seedAll


def timeSeriesPlot(date_time_series, variableTimeSeriesList, ylabel):
    fig = plt.figure(dpi=600)
    plt.xlabel("time")
    plt.ylabel(ylabel)
    for item in variableTimeSeriesList:
        plt.plot(date_time_series, item, linewidth=width)
    fig.savefig(ylabel + ".pdf")
    plt.close(fig)


def timeSeriesPlot_rich(
    sfd, fileName, date_time_series, variableTimeSeriesList, xlabel, ylabel
):
    fig = plt.figure(dpi=600, figsize=(6, 3))
    plt.xlabel(xlabel, fontsize=myFontSize)
    plt.ylabel(ylabel, fontsize=myFontSize)
    plt.xticks(fontsize=myFontSize)
    plt.yticks(fontsize=myFontSize)
    plt.grid(True, linewidth=width, linestyle="dotted")
    plt.locator_params(axis="y", nbins=30)
    for item in variableTimeSeriesList:
        plt.plot(date_time_series, item, linewidth=width)
    fig.savefig(sfd + "/" + fileName + ".pdf")
    plt.close(fig)


def scatterplot(variableOne, variableTwo, ylabel, symbol):
    fig = plt.figure(dpi=600)
    plt.xlabel("obs")
    plt.ylabel(ylabel)
    plt.plot(variableOne, variableTwo, symbol)
    fig.savefig(ylabel + ".pdf")
    plt.close(fig)


def scatterplot_rich(
    sfd, filename, variableOne, variableTwo, xlabel, ylabel, symbol, ylim
):
    fig = plt.figure(dpi=600, figsize=(4, 3))
    # fig = plt.figure(dpi=600)
    plt.xlabel(xlabel, fontsize=myFontSize)
    plt.ylabel(ylabel, fontsize=myFontSize)
    plt.xticks(fontsize=myFontSize)
    plt.yticks(fontsize=myFontSize)
    plt.plot(variableOne, variableTwo, symbol)
    plt.ylim(ylim)
    maxX = max(variableOne)
    plt.xlim((min(variableOne), 1.05 * maxX))
    fig.savefig(sfd + "/" + filename + ".pdf")
    plt.close(fig)


def createTrainingIndices():
    a = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    out = []
    for p in range(0, 366):  # add one year spin up
        out.append(1)
    for i in a:
        for j in range(0, 365):
            out.append(i)
    out = numpy.array(out)
    outOne = out == 1
    outTwo = out == 2
    outThree = out == 3
    outFour = out == 4
    return outOne, outTwo, outThree, outFour


def rand_min_max(mean, proportion):
    max = mean + proportion * mean
    min = mean - proportion * mean
    return torch.rand(1) * (max - min) + min


class weightConstraint(object):
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, "weight"):
            w = module.weight.data
            w = w.clamp(0.0, 2.0)
            # w=w.clamp(0.0,0.0)
            module.weight.data = w


class EarlyStopper:
    def __init__(self, patience=1, min_delta_proportion=1e-10):
        self.patience = patience
        self.min_delta_proportion = min_delta_proportion
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (
            self.min_validation_loss
            + self.min_delta_proportion * self.min_validation_loss
        ):
            self.counter += 1
            if self.counter >= self.patience:
                return self.counter, True
        return self.counter, False


###############
# data import #
###############


def createMeteoData(input_data_directory, output_data_directory, startDate, endDate):
    precipitation_file = open(output_data_directory + "precipitation.txt", "w")
    temperatureFile = open(output_data_directory + "temperature.txt", "w")
    date = datetime.date(1979, 1, 1)
    timestep = datetime.timedelta(days=1)

    temperature_time_series = []
    precipitationTimeSeries = []

    with open(input_data_directory + "weatherdata-470125_corrected.csv") as csv_file:
        csv_reader = csv.reader(csv_file, dialect="excel")
        line_count = 0
        line_out_count = 1
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                dateFromFileStr = str.split(row[0], "/")
                monthFromFileStr = int(dateFromFileStr[0])
                dayFromFileStr = int(dateFromFileStr[1])
                yearFromFileStr = int(dateFromFileStr[2])
                dateFromFile = datetime.date(
                    yearFromFileStr, monthFromFileStr, dayFromFileStr
                )
                # print(date, 'from file ', dateFromFile)
                temperature = (float(row[4]) + float(row[5])) / 2.0
                precip = row[6]
                # if (date >= start) and (date <= end):
                if (date >= startDate) and (date <= endDate):
                    precipitation_file.write(str(line_out_count) + " " + precip + "\n")
                    precipitationTimeSeries.append(float(precip))
                    temperatureFile.write(
                        str(line_out_count) + " " + str(temperature) + "\n"
                    )
                    temperature_time_series.append(temperature)
                    line_out_count += 1
                line_count += 1
                date = date + timestep

    precipitation_file.close()
    temperatureFile.close()
    return temperature_time_series, precipitationTimeSeries


def create_streamflow_data(input_data_directory, output_data_directory, start, end):
    streamFlowFile = open(output_data_directory + "streamflow.txt", "w")
    date = datetime.date(1951, 1, 1)
    timestep = datetime.timedelta(days=1)

    dischargeFile = open(input_data_directory + "6246180_Q_Day.Cmd_noHeader.txt", "r")
    dischargeFileContent = dischargeFile.readlines()

    streamFlowTimeSeries = []
    date_time_series = []

    line_out_count = 1
    for i in range(0, len(dischargeFileContent)):
        splitted = str.split(dischargeFileContent[i])
        discharge = splitted[1]
        if (date >= start) and (date <= end):
            streamFlowFile.write(str(line_out_count) + " " + discharge + "\n")
            streamFlowTimeSeries.append(float(discharge))
            date_time_series.append(date)
            line_out_count += 1
        date = date + timestep

    dischargeFile.close()
    return streamFlowTimeSeries, date_time_series


## data for training
# temperature_time_series, precipitationTimeSeries = createMeteoData(input_data_directory, output_data_directory, start, end)
# streamFlowTimeSeries, date_time_series = create_streamflow_data(input_data_directory, output_data_directory, start, end)
#
## data for validation
# startVal = datetime.date(1997, 10 , 1)
# endVal = datetime.date(2013, 10, 1)
# temperature_time_series_val, precipitation_time_series_val = createMeteoData(input_data_directory, output_data_directory, startVal, endVal)
# streamflow_time_series_val, date_time_series_val = create_streamflow_data(input_data_directory, output_data_directory, startVal, endVal)
#
#
# aRange = range(1,len(streamFlowTimeSeries)+1,1)
# f = plt.figure()
# f, axs = plt.subplots(2, 2)
# axs[0,0].scatter(temperature_time_series, streamFlowTimeSeries, s = 1 )
# axs[0,0].set_ylabel('streamflow')
# axs[0,0].set_xlabel('temperature')
#
# width=0.25
# axs[0,1].plot(date_time_series, numpy.asarray(temperature_time_series), linewidth=width)
# axs[0,1].set_ylabel('temperature')
# axs[1,0].plot(date_time_series, numpy.asarray(streamFlowTimeSeries), linewidth=width)
# axs[1,0].set_ylabel('streamflow')
# axs[1,1].plot(date_time_series, numpy.asarray(precipitationTimeSeries), linewidth=width)
# axs[1,1].set_ylabel('precipitation')
#
# axs[1,0].xaxis.set_major_locator(plt.MaxNLocator(3))
# axs[0,1].xaxis.set_major_locator(plt.MaxNLocator(3))
# axs[1,1].xaxis.set_major_locator(plt.MaxNLocator(3))
#
# plt.tight_layout()
# f.savefig(output_data_directory + "inputTimeseries.pdf")


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # calibrated parameters used for modes obsCreation
        # self.seepageProportion = 0.029 # calibrated, see txt for info
        # self.meltRateParameter = 0.0026 # calibrated, see txt for info https://tc.copernicus.org/articles/17/211/2023/#abstract, 0.006 not strange

        self.eva_f_d = nn.Linear(1, nodes)
        self.eva_f_c = nn.Linear(nodes, 1)
        self.sno_f_d = nn.Linear(1, nodes)
        self.sno_f_c = nn.Linear(nodes, 1)
        self.sub_f_d = nn.Linear(1, nodes)
        self.sub_f_c = nn.Linear(nodes, 1)
        if deep_layer:
            self.eva_f_dd = nn.Linear(nodes, nodes)
            self.sno_f_dd = nn.Linear(nodes, nodes)
            self.sub_f_dd = nn.Linear(nodes, nodes)

        if one_area:
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
        self.eva_parameter = nn.Parameter(rand_min_max(eva_par, proportion))
        self.sub_parameter = nn.Parameter(rand_min_max(sub_par, proportion))
        self.sno_parameter = nn.Parameter(rand_min_max(sno_par, proportion))
        self.tem_parameter = nn.Parameter(rand_min_max(tem_par, proportion))
        self.evp_parameter = nn.Parameter(rand_min_max(evp_par, proportion))

        if deep_layer:
            self.params = nn.ModuleDict(
                {
                    "eva": nn.ModuleList([self.eva_f_d, self.eva_f_dd, self.eva_f_c]),
                    "sno": nn.ModuleList([self.sno_f_d, self.sno_f_dd, self.sno_f_c]),
                    "sub": nn.ModuleList([self.sub_f_d, self.sub_f_dd, self.sub_f_c]),
                    "exp": nn.ParameterList(
                        [
                            self.eva_parameter,
                            self.sub_parameter,
                            self.sno_parameter,
                            self.tem_parameter,
                            self.evp_parameter,
                        ]
                    ),
                }
            )
        else:
            self.params = nn.ModuleDict(
                {
                    "eva": nn.ModuleList([self.eva_f_d, self.eva_f_c]),
                    "sno": nn.ModuleList([self.sno_f_d, self.sno_f_c]),
                    "sub": nn.ModuleList([self.sub_f_d, self.sub_f_c]),
                    "exp": nn.ParameterList(
                        [
                            self.eva_parameter,
                            self.sub_parameter,
                            self.sno_parameter,
                            self.tem_parameter,
                            self.evp_parameter,
                        ]
                    ),
                }
            )

    def eva_f_calculate(self, temp, mode_eva, deep_layer, linearArt):
        if mode_eva == "obsCreation":
            if linearArt:
                # potential evapotranspiration
                # https://en.wikipedia.org/wiki/Blaney–Criddle_equation (see link for params)
                EPot = torch.max(
                    ((0.35 * (0.457 * torch.tensor([temp]) + 8.128)) / 1000.0)
                    * self.eva_parameter_obs_creation,
                    torch.tensor(0.0),
                )
            else:
                if temp > -15.0:
                    # EPot = torch.max(((0.35 * (0.457 * temp + 8.128))/1000.0) * self.eva_parameter_obs_creation, torch.tensor(0.0))
                    EPot = (
                        (
                            torch.sin(
                                (torch.tensor([temp + 15.0]) * 0.25 - 0.5 * torch.pi)
                            )
                            + 1
                        )
                        / 1000.0
                    ) + 0.0001 * torch.tensor([temp + 15.0])
                else:
                    EPot = torch.tensor([0.0])
        if mode_eva == "fit":
            out_eva_f = m(self.eva_f_d((torch.tensor([temp]))))
            if deep_layer:
                out_eva_f = m(self.eva_f_dd(out_eva_f))
            EPot = torch.sigmoid(self.eva_f_c(out_eva_f)) / 75.0  # /750.0 or /1000.0
        if mode_eva == "fitExpert":
            # potential evapotranspiration
            # https://en.wikipedia.org/wiki/Blaney–Criddle_equation (see link for params)
            EPot = torch.max(
                ((0.35 * (0.457 * temp + 8.128)) / 1000.0) * self.eva_parameter,
                torch.tensor(0.0),
            )
        return EPot

    def sno_f_calculate(self, temp, modeSno, deep_layer, linearArt):
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
                    potentialMelt = (
                        (torch.sin((torch.tensor([temp]) * 0.6 - 0.5 * torch.pi)) + 1)
                        / 200
                    ) + 0.0004 * torch.tensor([temp])
                else:
                    potentialMelt = torch.tensor(0.0)
        if modeSno == "fit":
            out_sno_f = m(self.sno_f_d((t.tensor([temp]))))
            if deep_layer:
                out_sno_f = m(self.sno_f_dd(out_sno_f))
            potentialMelt = torch.sigmoid(self.sno_f_c(out_sno_f)) / 50.0
        if modeSno == "fitExpert":
            melting = temp > 0.0
            if melting:
                potentialMelt = temp * self.sno_parameter
            else:
                potentialMelt = torch.tensor(0.0)
        return potentialMelt

    def sub_f_calculate(self, sub_s, modeSub, deep_layer, linearArt):
        if modeSub == "obsCreation":
            if linearArt:
                seepagePot = self.sub_parameter_obs_creation * sub_s
            else:
                seepagePot = (
                    (torch.sin((torch.tensor([sub_s]) * 20 - 0.5 * torch.pi)) + 1) / 500
                ) + 0.03 * sub_s
        if modeSub == "fit":
            out_sub_f = m(self.sub_f_d((torch.tensor([sub_s]) * 2.0)))
            if deep_layer:
                out_sub_f = m(self.sub_f_dd(out_sub_f))
            proportion = torch.sigmoid(self.sub_f_c(out_sub_f))
            seepagePot = proportion * sub_s
        if modeSub == "fitExpert":
            seepage = self.sub_parameter * sub_s
            seepagePot = seepage
        return seepagePot

    def com_f_response(self, modeSub, deep_layer, component, linearArt):
        if component == "sub":
            com_s_values = numpy.arange(0.0, 1.0, 0.01)
        if component == "sno":
            com_s_values = numpy.arange(-10.0, 15.0, 0.01)
        if component == "eva":
            com_s_values = numpy.arange(-10.0, 15.0, 0.01)
        com_f_values = numpy.empty(0)
        for com_s_value in com_s_values:
            if component == "sub":
                com_f_value = self.sub_f_calculate(
                    com_s_value.item(), modeSub, deep_layer, linearArt
                )
            if component == "sno":
                com_f_value = self.sno_f_calculate(
                    com_s_value.item(), modeSub, deep_layer, linearArt
                )
            if component == "eva":
                com_f_value = self.eva_f_calculate(
                    com_s_value.item(), modeSub, deep_layer, linearArt
                )
            if isinstance(com_f_value, torch.Tensor):
                if modeSub == "fitExpert":
                    com_f_value_detached = com_f_value.detach().numpy()
                else:
                    if component == "sub":
                        com_f_value_detached = com_f_value.detach().numpy()[0]
                    if component == "sno":
                        com_f_value_detached = com_f_value.detach().numpy()
                    if component == "eva":
                        com_f_value_detached = com_f_value.detach().numpy()
            else:
                com_f_value_detached = com_f_value
            com_f_values = numpy.append(com_f_values, com_f_value_detached)
            com_f_values = numpy.maximum(com_f_values, 0.0)
        return com_s_values, com_f_values

    def forward(
        self,
        linearArt,  # linear obs creation model
        firstEpochs,  # start of training or not
        sno_s_initial,  # initial snow storage
        sub_s_initial,  # initial subsurface storage
        temperature,  # temperature time series
        precipitation,  # precipitation time series
        mode_eva,  # mode evapotranspiration
        modeSno,  # mode snow
        modeSub,  # mode subsurface
        modeTem,  # temperature offset
        modeEvP,  # potential proportion of evap to sublimation
    ):

        nr_timesteps = len(temperature)

        # parameters and inputs for expert model

        # temperatureOffsetExpert = -1.19 # calibrated see txt for info

        # EPar = 0.85  # calibrated see txt for info
        # offset should actually be (((2115+2797)/2)-1180)*0.005 = 6.38 degrees.. (i.e. lower and higher half,
        # average, multipled by lapse rate
        # difference between lower and higher is (2115-2797) * 0.005 = 3.41 degrees
        # https://journals.ametsoc.org/view/journals/clim/16/7/1520-0442_2003_016_1032_sasvoa_2.0.co_2.xml

        # proportion of evapotranspiration potentially assigned to sno storage
        # evaPropToSnoExpert = 0.995

        if modeTem == "fitExpert":
            temperatureOffset = self.tem_parameter
        if modeTem == "obsCreation":
            temperatureOffset = self.tem_parameter_obs_creation

        if modeEvP == "fitExpert":
            evaPropToSno = torch.sigmoid(self.evp_parameter)
        if modeEvP == "obsCreation":
            evaPropToSno = torch.sigmoid(torch.tensor(self.evp_parameter_obs_creation))

        if one_area:
            areaTemperatureOffsets = [0.0]
        else:
            areaTemperatureOffsets = [1.705, -1.705]

        if one_area:
            numberOfAreas = 1
        else:
            numberOfAreas = 2
        for area in range(0, numberOfAreas):
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

            for i in range(0, nr_timesteps):
                # get drivers for timestep
                temp = temperature[i] + temperatureOffset + areaTemperatureOffsets[area]

                # potential evapotranspiration

                EPot = torch.max(
                    self.eva_f_calculate(temp, mode_eva, deep_layer, linearArt),
                    torch.tensor(0.0),
                )

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
                potentialMelt = self.sno_f_calculate(
                    temp, modeSno, deep_layer, linearArt
                )

                if firstEpochs:  # hard cap to prevent zero snow cover
                    if temp < -5.0:
                        potentialMelt = torch.tensor(0.0)

                sno_f = potentialMelt
                sno_f_ts[i] = sno_f

                potentialMelt = torch.max(potentialMelt, torch.tensor(0))
                actualMelt = torch.min(sno_s, potentialMelt)
                sno_s = sno_s - actualMelt

                # sublimation
                potentialSublimation = evaPropToSno * EPot
                actualSublimation = torch.min(sno_s, potentialSublimation)
                sno_s = sno_s - actualSublimation
                potentialEvapForSub = EPot - actualSublimation

                # sub surface storage

                totalInput = rainfall + actualMelt
                # hortonRunoffProportion = 0.04
                hortonRunoffProportion = 0.0
                hortonRunoff = hortonRunoffProportion * totalInput

                sub_s = sub_s + (1.0 - hortonRunoffProportion) * totalInput

                EAct = torch.min(sub_s, potentialEvapForSub)

                sub_s = sub_s - EAct

                sub_s_ts[i] = sub_s

                seepagePot = self.sub_f_calculate(sub_s, modeSub, deep_layer, linearArt)

                seepageNotNegative = torch.max(seepagePot, torch.tensor(0.0))
                seepageAct = torch.min(seepageNotNegative, sub_s)

                sub_f = seepageAct + hortonRunoff

                # sub_f = torch.max(sub_f, torch.tensor(0.0008))

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
            if one_area:
                # storages
                sno_s_ts_areas = torch.vstack([sno_s_ts_areas, sno_s_ts])
                sub_s_ts_areas = torch.vstack([sub_s_ts_areas, sub_s_ts])
                # fluxes
                sno_f_ts_areas = torch.vstack([sno_f_ts_areas, sno_f_ts])
                sub_f_ts_areas = torch.vstack([sub_f_ts_areas, sub_f_ts])
                # evapotranspiration
                eva_f_ts_areas = torch.vstack([eva_f_ts_areas, eva_f_ts])

        return (
            sno_s_ts_areas,
            sub_s_ts_areas,
            sno_f_ts_areas,
            sub_f_ts_areas,
            eva_f_ts_areas,
        )

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()


#############################
# create artificial observations
#############################


def createArtificialObservations(
    temperature_time_series,
    precipitationTimeSeries,
    addErrorToArtificialStreamFlow,
    linearArt,
):
    hydromodel = Net()
    sno_s_initial = 0.0
    sub_s_initial = 0.0
    sno_s_ts_areas, sub_s_ts_areas, sno_f_ts_areas, sub_f_ts_areas, eva_f_ts_areas = (
        hydromodel(
            linearArt,
            False,
            torch.tensor(sno_s_initial),
            torch.tensor(sub_s_initial),
            torch.tensor(temperature_time_series),
            torch.tensor(precipitationTimeSeries),
            mode_eva="obsCreation",
            modeSno="obsCreation",
            modeSub="obsCreation",
            modeTem="obsCreation",
            modeEvP="obsCreation",
        )
    )
    sno_s_ts = sno_s_ts_areas.mean(dim=0)
    sub_s_ts = sub_s_ts_areas.mean(dim=0)
    sno_f_ts = sno_f_ts_areas.mean(dim=0)
    sub_f_ts = sub_f_ts_areas.mean(dim=0)
    eva_f_ts = eva_f_ts_areas.mean(dim=0)
    if addErrorToArtificialStreamFlow:
        # error = torch.tensor([numpy.random.normal(0,0.01,len(streamFlowTimeSeries))])
        # sma = torch.nn.AvgPool1d(kernel_size=59, stride = 1, padding = 29)
        # error = sma(error)
        # sub_f_ts = torch.max(sub_f_ts + error, torch.tensor(0.0))[0]
        sd = 0.2
        errorMultiplier = numpy.minimum(
            numpy.maximum(0.0, numpy.random.normal(1, sd, len(streamFlowTimeSeries))), 2
        )
        # sub_f_ts = torch.max(sub_f_ts * errorMultiplier, torch.tensor(0.0))[0]
        sub_f_ts = torch.max(sub_f_ts * errorMultiplier, torch.tensor(0.0))
    return sno_s_ts, sub_s_ts, sno_f_ts, sub_f_ts, eva_f_ts


def training_loop(
    n_epochs,
    stopping,
    optimizer,
    model,
    loss_fn,
    sno_s_initial,
    sub_s_initial,
    temperature_time_series,
    precipitationTimeSeries,
    streamFlowTimeSeries,
    date_time_series,
    temperatureTimeSeriesSto,
    precipitationTimeSeriesSto,
    streamFlowTimeSeriesSto,
    dateTimeSeriesSto,
    temperature_time_series_val,
    precipitation_time_series_val,
    streamflow_time_series_val,
    date_time_series_val,
    sno_s_ts,
    sub_s_ts,
    sno_f_ts,
    sub_f_ts,
    eva_f_ts,
    sno_s_tsSto,
    sub_s_tsSto,
    sno_f_tsSto,
    sub_f_tsSto,
    eva_f_tsSto,
    sno_s_tsVal,
    sub_s_tsVal,
    sno_f_tsVal,
    sub_f_tsVal,
    eva_f_tsVal,
    trainingData,
    fitOnObservations,
    mode_eva_train,
    modeSnoTrain,
    modeSubTrain,
    modeTemTrain,
    modeEvPTrain,
    output_directory,
    scenario_directory,
    linearArt,
):

    sfd = output_directory + scenario_directory + "/"
    sfdAr = sfd + "arrays" + "/"
    if not os.path.exists(sfd):
        os.makedirs(sfd)
    if not os.path.exists(sfdAr):
        os.makedirs(sfdAr)

    # write artificial data to disk, validation
    numpy.save(sfdAr + "val_art_ts_sno_s.npy", sno_s_tsVal)
    numpy.save(sfdAr + "val_art_ts_sub_s.npy", sub_s_tsVal)
    numpy.save(sfdAr + "val_art_ts_sno_f.npy", sno_f_tsVal)
    numpy.save(sfdAr + "val_art_ts_sub_f.npy", sub_f_tsVal)
    numpy.save(sfdAr + "val_art_ts_eva_f.npy", eva_f_tsVal)
    # write artificial data to disk, training
    numpy.save(sfdAr + "train_art_ts_sno_s.npy", sno_s_ts)
    numpy.save(sfdAr + "train_art_ts_sub_s.npy", sub_s_ts)
    numpy.save(sfdAr + "train_art_ts_sno_f.npy", sno_f_ts)
    numpy.save(sfdAr + "train_art_ts_sub_f.npy", sub_f_ts)
    numpy.save(sfdAr + "train_art_ts_eva_f.npy", eva_f_ts)

    lossTrainingSeries = []
    lossValidationSeries = []
    lossStoppingSeries = []
    epochSeries = []
    stopperCounterSeries = []

    early_stopper = EarlyStopper(patience=200, min_delta_proportion=1e-2)

    time = datetime.datetime.now()

    firstEpochs = True
    for epoch in range(1, n_epochs + 1):
        if epoch > 100:
            firstEpochs = False
        firstEpochs = True
        newTime = datetime.datetime.now()
        duration = newTime - time
        time = datetime.datetime.now()

        (
            tr_sno_s_ts_areas,
            tr_sub_s_ts_areas,
            tr_sno_f_ts_areas,
            tr_sub_f_ts_areas,
            tr_eva_f_ts_areas,
        ) = hydromodel(
            linearArt,
            firstEpochs,
            torch.tensor(sno_s_initial),
            torch.tensor(sub_s_initial),
            torch.tensor(temperature_time_series),
            torch.tensor(precipitationTimeSeries),
            mode_eva=mode_eva_train,
            modeSno=modeSnoTrain,
            modeSub=modeSubTrain,
            modeTem=modeTemTrain,
            modeEvP=modeEvPTrain,
        )

        tr_sno_s_ts = tr_sno_s_ts_areas.mean(dim=0)
        tr_sub_s_ts = tr_sub_s_ts_areas.mean(dim=0)
        tr_sno_f_ts = tr_sno_f_ts_areas.mean(dim=0)
        tr_sub_f_ts = tr_sub_f_ts_areas.mean(dim=0)
        tr_eva_f_ts = tr_eva_f_ts_areas.mean(dim=0)

        training = numpy.logical_not(stopping)

        if fitOnObservations:
            if epoch < -5:
                loss_train = loss_fn(tr_sub_f_ts, sub_f_ts, training)
            else:
                loss_train = loss_fn(
                    tr_sub_f_ts,
                    torch.tensor(streamFlowTimeSeries) / conversion_fluxes,
                    training,
                )
        else:
            if trainingData == "trainingSub":
                loss_train = loss_fn(tr_sub_f_ts, sub_f_ts, training)
            if trainingData == "trainingEva":
                loss_train = loss_fn(tr_eva_f_ts, eva_f_ts, training)
            if trainingData == "trainingSno":
                loss_train = loss_fn(tr_sno_f_ts, sno_f_ts, training)
            if trainingData == "trainingSubAndSnow":
                modifiedSnow = sno_s_ts / 2.0
                loss_train_sno = loss_fn(tr_sno_s_ts, modifiedSnow)
                loss_train_sub = loss_fn(tr_sub_s_ts, sub_s_ts)
                loss_train = (
                    loss_train_sno / 4.0 + loss_train_sub
                ) / 2.0  # /4.0 to standardize snow

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
            #                   mode_eva = mode_eva_train,
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
                loss_trainSto = loss_fn(
                    tr_sub_f_ts,
                    torch.tensor(streamFlowTimeSeries) / conversion_fluxes,
                    stopping,
                )
            else:
                loss_trainSto = loss_fn(tr_sub_f_ts, sub_f_ts, stopping)

        lossStoppingSeries.append(loss_trainSto.item())
        counter, stop = early_stopper.early_stop(loss_trainSto)
        stopperCounterSeries.append(counter)

        # VALIDATION, ie TESTING
        if epoch == 1 or epoch % 50 == 0 or stop:
            (
                tr_sno_s_ts_areasVal,
                tr_sub_s_ts_areasVal,
                tr_sno_f_ts_areasVal,
                tr_sub_f_ts_areasVal,
                tr_eva_f_ts_areasVal,
            ) = hydromodel(
                linearArt,
                True,
                torch.tensor(sno_s_initial),
                torch.tensor(sub_s_initial),
                torch.tensor(temperature_time_series_val),
                torch.tensor(precipitation_time_series_val),
                mode_eva=mode_eva_train,
                modeSno=modeSnoTrain,
                modeSub=modeSubTrain,
                modeTem=modeTemTrain,
                modeEvP=modeEvPTrain,
            )
            tr_sno_s_tsVal = tr_sno_s_ts_areasVal.mean(dim=0)
            tr_sub_s_tsVal = tr_sub_s_ts_areasVal.mean(dim=0)
            tr_sno_f_tsVal = tr_sno_f_ts_areasVal.mean(dim=0)
            tr_sub_f_tsVal = tr_sub_f_ts_areasVal.mean(dim=0)
            tr_eva_f_tsVal = tr_eva_f_ts_areasVal.mean(dim=0)
            validation = numpy.logical_or(stopping, training)
            if fitOnObservations:
                loss_trainVal = loss_fn(
                    tr_sub_f_tsVal,
                    torch.tensor(streamflow_time_series_val) / conversion_fluxes,
                    validation,
                )
            else:
                loss_trainVal = loss_fn(tr_sub_f_tsVal, sub_f_tsVal, validation)

        lossValidationSeries.append(loss_trainVal.item())

        if l1_regularization:
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
            # print('running scenario: ', scenario_directory, ' running one area: ', one_area)
            print(
                "running scenario: ",
                scenario_directory,
                " Running one area: ",
                one_area,
                ". Duration of one epoch:",
                duration,
            )
            print(f"Epoch {epoch}, Training loss {loss_train.item():.10},", end="")
            print(
                f" Stopping (i.e. validation) loss {loss_trainSto.item():.10},", end=""
            )
            print(f" Validation (i.e. testing) loss {loss_trainVal.item():.10},")
            if fitOnObservations:
                print("Fitting on observations")
            else:
                print("Fitting on artificial data")
            # print('learning rates (eva, sub, sno)', scheduler.get_last_lr())
            print("learning rate sub", optimizer.param_groups[0]["lr"])
            if l1_regularization:
                print("l1", l1.item())
            print(epoch, "snow flux", tr_sno_f_ts.mean().item(), end="")
            print(epoch, "snow stor", tr_sno_s_ts.mean().item(), end="")
            print(epoch, "sub flux", tr_sub_f_ts.mean().item(), end="")
            print(epoch, "sub stor", tr_sub_s_ts.mean().item(), end="")
            print(epoch, "eva flux", tr_eva_f_ts.mean().item())
            if modeSubTrain == "fitExpert":
                print(
                    "seepage parameter:",
                    hydromodel.sub_parameter.detach().numpy()[0],
                    end=" ",
                )
                numpy.save(
                    sfdAr + "sub_parameter.npy",
                    hydromodel.sub_parameter.detach().numpy(),
                )
            if modeSnoTrain == "fitExpert":
                print(
                    "snow melt parameter:",
                    hydromodel.sno_parameter.detach().numpy()[0],
                    end=" ",
                )
                numpy.save(
                    sfdAr + "sno_parameter.npy",
                    hydromodel.sno_parameter.detach().numpy(),
                )
            if mode_eva_train == "fitExpert":
                print(
                    "eva parameter:",
                    hydromodel.eva_parameter.detach().numpy()[0],
                    end=" ",
                )
                numpy.save(
                    sfdAr + "eva_parameter.npy",
                    hydromodel.eva_parameter.detach().numpy(),
                )
            if modeTemTrain == "fitExpert":
                print(
                    "tem parameter:",
                    hydromodel.tem_parameter.detach().numpy()[0],
                    end=" ",
                )
                numpy.save(
                    sfdAr + "tem_parameter.npy",
                    hydromodel.tem_parameter.detach().numpy(),
                )
            if modeEvPTrain == "fitExpert":
                print(
                    "evp parameter (par, sigmoid):",
                    hydromodel.evp_parameter.detach().numpy()[0],
                    torch.sigmoid(hydromodel.evp_parameter).detach().numpy()[0],
                )
                numpy.save(
                    sfdAr + "evp_parameter.npy",
                    hydromodel.evp_parameter.detach().numpy(),
                )
                numpy.save(
                    sfdAr + "evp_sig_parameter.npy",
                    torch.sigmoid(hydromodel.evp_parameter).detach().numpy()[0],
                )

            a, b = hydromodel.com_f_response(modeSubTrain, deep_layer, "sub", linearArt)
            scatterplot_rich(
                sfd,
                "response_sub",
                a,
                b,
                "storage (m)",
                "potential flux (m/day)",
                "-",
                (0, 0.02),
            )
            numpy.save(sfdAr + "response_sub_x.npy", numpy.array(a))
            numpy.save(sfdAr + "response_sub_y.npy", numpy.array(b))

            a, b = hydromodel.com_f_response(modeSnoTrain, deep_layer, "sno", linearArt)
            scatterplot_rich(
                sfd,
                "response_sno",
                a,
                b,
                "temperature (C)",
                "potential flux (m/day)",
                "-",
                (0, 0.02),
            )
            numpy.save(sfdAr + "response_sno_x.npy", numpy.array(a))
            numpy.save(sfdAr + "response_sno_y.npy", numpy.array(b))

            a, b = hydromodel.com_f_response(mode_eva_train, deep_layer, "eva", linearArt)
            scatterplot_rich(
                sfd,
                "response_eva",
                a,
                b,
                "temperature (C)",
                "potential flux (m/day)",
                "-",
                (0, 0.02),
            )
            numpy.save(sfdAr + "response_eva_x.npy", numpy.array(a))
            numpy.save(sfdAr + "response_eva_y.npy", numpy.array(b))

            # blauw, oranje, groen
            timeSeriesPlot_rich(
                sfd,
                "loss",
                epochSeries,
                [
                    torch.min(torch.tensor(lossTrainingSeries), torch.tensor(1e-5)),
                    torch.min(torch.tensor(lossStoppingSeries), torch.tensor(1e-5)),
                    torch.min(torch.tensor(lossValidationSeries), torch.tensor(1e-5)),
                ],
                "epoch",
                "loss",
            )
            timeSeriesPlot_rich(
                sfd,
                "stopCounter",
                epochSeries,
                [stopperCounterSeries],
                "epoch",
                "counter",
            )
            numpy.save(sfdAr + "epochs.npy", numpy.array(epochSeries))
            numpy.save(sfdAr + "lossTraining.npy", numpy.array(lossTrainingSeries))
            numpy.save(sfdAr + "lossStopping.npy", numpy.array(lossStoppingSeries))
            numpy.save(sfdAr + "lossValidation.npy", numpy.array(lossValidationSeries))
            # mean over area only
            # training
            timeSeriesPlot_rich(
                sfd,
                "train_ts_eva_f",
                date_time_series,
                [tr_eva_f_ts.detach().numpy(), eva_f_ts.detach().numpy()],
                "time",
                "flux (m/day)",
            )
            timeSeriesPlot_rich(
                sfd,
                "train_ts_sub_f",
                date_time_series,
                [tr_sub_f_ts.detach().numpy(), sub_f_ts.detach().numpy()],
                "time",
                "flux (m/day)",
            )
            timeSeriesPlot_rich(
                sfd,
                "train_ts_sub_s",
                date_time_series,
                [tr_sub_s_ts.detach().numpy(), sub_s_ts.detach().numpy()],
                "time",
                "storage (m/day)",
            )
            timeSeriesPlot_rich(
                sfd,
                "train_ts_sno_f",
                date_time_series,
                [tr_sno_f_ts.detach().numpy(), sno_f_ts.detach().numpy()],
                "time",
                "flux (m/day)",
            )
            timeSeriesPlot_rich(
                sfd,
                "train_ts_sno_s",
                date_time_series,
                [tr_sno_s_ts.detach().numpy(), sno_s_ts.detach().numpy()],
                "time",
                "storage (m/day)",
            )
            timeSeriesPlot_rich(
                sfd,
                "train_ts_OBS",
                date_time_series,
                [
                    tr_sub_f_ts.detach().numpy(),
                    torch.tensor(streamFlowTimeSeries) / conversion_fluxes,
                ],
                "time",
                "flux (m/day",
            )
            numpy.save(
                sfdAr + "train_ts_eva_f.npy", numpy.array(tr_eva_f_ts.detach().numpy())
            )
            numpy.save(
                sfdAr + "train_ts_sub_f.npy", numpy.array(tr_sub_f_ts.detach().numpy())
            )
            numpy.save(
                sfdAr + "train_ts_sub_s.npy", numpy.array(tr_sub_s_ts.detach().numpy())
            )
            numpy.save(
                sfdAr + "train_ts_sno_f.npy", numpy.array(tr_sno_f_ts.detach().numpy())
            )
            numpy.save(
                sfdAr + "train_ts_sno_s.npy", numpy.array(tr_sno_s_ts.detach().numpy())
            )
            numpy.save(
                sfdAr + "train_ts_OBS",
                (torch.tensor(streamFlowTimeSeries) / conversion_fluxes).numpy(),
            )
            numpy.save(sfdAr + "train_date.npy", numpy.array(date_time_series))
            numpy.save(sfdAr + "train_ts_temperature.npy", numpy.array(temperature_time_series))
            numpy.save(sfdAr + "train_ts_precipitation.npy", numpy.array(precipitationTimeSeries))
            # validation
            timeSeriesPlot_rich(
                sfd,
                "valid_ts_eva_f",
                date_time_series_val,
                [tr_eva_f_tsVal.detach().numpy()],
                "time",
                "flux (m/day)",
            )
            timeSeriesPlot_rich(
                sfd,
                "valid_ts_sub_f",
                date_time_series_val,
                [tr_sub_f_tsVal.detach().numpy()],
                "time",
                "flux (m/day)",
            )
            timeSeriesPlot_rich(
                sfd,
                "valid_ts_sub_s",
                date_time_series_val,
                [tr_sub_s_tsVal.detach().numpy()],
                "time",
                "storage (m/day)",
            )
            timeSeriesPlot_rich(
                sfd,
                "valid_ts_sno_f",
                date_time_series_val,
                [tr_sno_f_tsVal.detach().numpy()],
                "time",
                "flux (m/day)",
            )
            timeSeriesPlot_rich(
                sfd,
                "valid_ts_sno_s",
                date_time_series_val,
                [tr_sno_s_tsVal.detach().numpy()],
                "time",
                "storage (m/day)",
            )
            timeSeriesPlot_rich(
                sfd,
                "valid_ts_OBS",
                date_time_series_val,
                [
                    tr_sub_f_tsVal.detach().numpy(),
                    torch.tensor(streamflow_time_series_val) / conversion_fluxes,
                ],
                "time",
                "flux (m/day",
            )
            numpy.save(
                sfdAr + "valid_ts_eva_f.npy",
                numpy.array(tr_eva_f_tsVal.detach().numpy()),
            )
            numpy.save(
                sfdAr + "valid_ts_sub_f.npy",
                numpy.array(tr_sub_f_tsVal.detach().numpy()),
            )
            numpy.save(
                sfdAr + "valid_ts_sub_s.npy",
                numpy.array(tr_sub_s_tsVal.detach().numpy()),
            )
            numpy.save(
                sfdAr + "valid_ts_sno_f.npy",
                numpy.array(tr_sno_f_tsVal.detach().numpy()),
            )
            numpy.save(
                sfdAr + "valid_ts_sno_s.npy",
                numpy.array(tr_sno_s_tsVal.detach().numpy()),
            )
            numpy.save(
                sfdAr + "valid_ts_OBS",
                (torch.tensor(streamflow_time_series_val) / conversion_fluxes).numpy(),
            )
            numpy.save(sfdAr + "valid_date.npy", numpy.array(date_time_series_val))
            numpy.save(sfdAr + "valid_ts_temperature.npy", numpy.array(temperature_time_series_val))
            numpy.save(sfdAr + "valid_ts_precipitation.npy", numpy.array(precipitation_time_series_val))

            # scatterplots, indep vs dep
            # mean over area only
            # evapotranspiration, predicted and observations
            scatterplot_rich(
                sfd,
                "loss_pre_eva_f_sc",
                temperature_time_series,
                tr_eva_f_ts.detach().numpy(),
                "temperature (C)",
                "flux (m/day)",
                ".",
                (0, 0.01),
            )
            scatterplot_rich(
                sfd,
                "loss_eva_f_sc",
                temperature_time_series,
                eva_f_ts,
                "temperature (C)",
                "flux (m/day)",
                ".",
                (0, 0.01),
            )
            # snow melt, predicted and observations
            scatterplot_rich(
                sfd,
                "loss_pre_sno_f_sc",
                temperature_time_series,
                tr_sno_f_ts.detach().numpy(),
                "temperature (C)",
                "flux (m/day)",
                ".",
                (0, 0.02),
            )
            scatterplot_rich(
                sfd,
                "loss_sno_f_sc",
                temperature_time_series,
                sno_f_ts,
                "temperature (C))",
                "flux (m/day)",
                ".",
                (0, 0.02),
            )
            # subsurface, predicted and observations
            scatterplot_rich(
                sfd,
                "loss_pre_sub_f_sc",
                tr_sub_s_ts.detach().numpy(),
                tr_sub_f_ts.detach().numpy(),
                "storage (m)",
                "flux (m/day)",
                ".",
                (0, 0.05),
            )
            scatterplot_rich(
                sfd,
                "loss_sub_f_sc",
                sub_s_ts,
                sub_f_ts,
                "storage (m)",
                "flux (m/day)",
                ".",
                (0, 0.05),
            )

            if not one_area:
                # by area, training
                a = tr_sno_s_ts_areas.detach().numpy()
                timeSeriesPlot_rich(
                    sfd,
                    "train_ts_sno_s_areas",
                    date_time_series,
                    [a[0], a[1]],
                    "time",
                    "storage (m)",
                )
                numpy.save(sfdAr + "train_ts_sno_s_areas.npy", a)
                a = tr_sub_s_ts_areas.detach().numpy()
                timeSeriesPlot_rich(
                    sfd,
                    "train_ts_sub_s_areas",
                    date_time_series,
                    [a[0], a[1]],
                    "time",
                    "storage (m)",
                )
                numpy.save(sfdAr + "train_ts_sub_s_areas.npy", a)
                a = tr_sno_f_ts_areas.detach().numpy()
                timeSeriesPlot_rich(
                    sfd,
                    "train_ts_sno_f_areas",
                    date_time_series,
                    [a[0], a[1]],
                    "time",
                    "flux (m/day)",
                )
                numpy.save(sfdAr + "train_ts_sno_f_areas.npy", a)
                a = tr_sub_f_ts_areas.detach().numpy()
                timeSeriesPlot_rich(
                    sfd,
                    "train_ts_sub_f_areas",
                    date_time_series,
                    [a[0], a[1]],
                    "time",
                    "flux (m/day)",
                )
                numpy.save(sfdAr + "train_ts_sub_f_areas.npy", a)
                a = tr_eva_f_ts_areas.detach().numpy()
                timeSeriesPlot_rich(
                    sfd,
                    "train_ts_eva_f_areas",
                    date_time_series,
                    [a[0], a[1]],
                    "time",
                    "flux (m/day)",
                )
                numpy.save(sfdAr + "train_ts_eva_f_areas.npy", a)
                # by area, validation
                a = tr_sno_s_ts_areasVal.detach().numpy()
                timeSeriesPlot_rich(
                    sfd,
                    "valid_ts_sno_s_areas",
                    date_time_series,
                    [a[0], a[1]],
                    "time",
                    "storage (m)",
                )
                numpy.save(sfdAr + "valid_ts_sno_s_areas.npy", a)
                a = tr_sub_s_ts_areasVal.detach().numpy()
                timeSeriesPlot_rich(
                    sfd,
                    "valid_ts_sub_s_areas",
                    date_time_series,
                    [a[0], a[1]],
                    "time",
                    "storage (m)",
                )
                numpy.save(sfdAr + "valid_ts_sub_s_areas.npy", a)
                a = tr_sno_f_ts_areasVal.detach().numpy()
                timeSeriesPlot_rich(
                    sfd,
                    "valid_ts_sno_f_areas",
                    date_time_series,
                    [a[0], a[1]],
                    "time",
                    "flux (m/day)",
                )
                numpy.save(sfdAr + "valid_ts_sno_f_areas.npy", a)
                a = tr_sub_f_ts_areasVal.detach().numpy()
                timeSeriesPlot_rich(
                    sfd,
                    "valid_ts_sub_f_areas",
                    date_time_series,
                    [a[0], a[1]],
                    "time",
                    "flux (m/day)",
                )
                numpy.save(sfdAr + "valid_ts_sub_f_areas.npy", a)
                a = tr_eva_f_ts_areasVal.detach().numpy()
                timeSeriesPlot_rich(
                    sfd,
                    "valid_ts_eva_f_areas",
                    date_time_series,
                    [a[0], a[1]],
                    "time",
                    "flux (m/day)",
                )
                numpy.save(sfdAr + "valid_ts_eva_f_areas.npy", a)

        if stop:
            print("break")
            break

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        # scheduler.step()
        scheduler.step(loss_trainSto)

        # counter, stop = early_stopper.early_stop(loss_trainSto)
        # stopperSeries.append(loss_trainVal.item())


def loss_fn(t_p, t_c, period):
    a = 366  # must be used as spin up is embedded in training/stopping and validation data sets
    # periodFloat = torch.tensor(numpy.where(period, 1.0, 0.0))[a:]
    period = period[a:]
    squared_diffs = ((t_p[a:] - t_c[a:]) ** 2.0)[period]
    return squared_diffs.mean()


## with momentum, works very well voor sub (sneller en precieser met zelfde loss)
# optimizer = optim.RMSprop(hydromodel.parameters(), lr = 0.00001, momentum = 0.9)
#
## plateau https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, eps = 1e-8, patience = 50)


# fit model on observations or on artificial data
fitOnObservations = True
# if fitting on artificial data, use linear model or non-linear one
linearArtForNotFitOnObservations = False
# add error to artificial data
addErrorToArtificialStreamFlow = False

# training data to calculate loss over (typical sub, i.e. outflow)
trainingData = "trainingSub"

# folder to store output data, note the / at the end
output_directory = "../data/results/"


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
temperature_time_series, precipitationTimeSeries = createMeteoData(
    input_data_directory, output_data_directory, startOne, endOne
)
streamFlowTimeSeries, date_time_series = create_streamflow_data(
    input_data_directory, output_data_directory, startOne, endOne
)
sno_s_ts, sub_s_ts, sno_f_ts, sub_f_ts, eva_f_ts = createArtificialObservations(
    temperature_time_series,
    precipitationTimeSeries,
    addErrorToArtificialStreamFlow,
    linearArt,
)

# training period 2 NOT USED ANYMORE
startTwo = datetime.date(1990, 10, 1)
# startTwo = datetime.date(1996, 10, 1)
endTwo = datetime.date(2001, 9, 1)
temperatureTimeSeriesTwo, precipitationTimeSeriesTwo = createMeteoData(
    input_data_directory, output_data_directory, startTwo, endTwo
)
streamFlowTimeSeriesTwo, dateTimeSeriesTwo = create_streamflow_data(
    input_data_directory, output_data_directory, startTwo, endTwo
)
sno_s_tsTwo, sub_s_tsTwo, sno_f_tsTwo, sub_f_tsTwo, eva_f_tsTwo = (
    createArtificialObservations(
        temperatureTimeSeriesTwo,
        precipitationTimeSeriesTwo,
        addErrorToArtificialStreamFlow,
        linearArt,
    )
)

# data for validation, ie testing
startVal = datetime.date(1995, 10, 1)
endVal = datetime.date(2012, 9, 26)
temperature_time_series_val, precipitation_time_series_val = createMeteoData(
    input_data_directory, output_data_directory, startVal, endVal
)
streamflow_time_series_val, date_time_series_val = create_streamflow_data(
    input_data_directory, output_data_directory, startVal, endVal
)
# note that no error is added to artificial streamflow in any case as it is used for validation only
sno_s_tsVal, sub_s_tsVal, sno_f_tsVal, sub_f_tsVal, eva_f_tsVal = (
    createArtificialObservations(
        temperature_time_series_val, precipitation_time_series_val, False, linearArt
    )
)


sno_s_initial = 0.0
sub_s_initial = 0.0

#####################
# fitting scenarios #
#####################

nr_epochs = 5000

# expert models (n = 7)

xub = {
    "name": "fit_xub",
    "modeEvaTrain": "obsCreation",
    "modeSnoTrain": "obsCreation",
    "modeSubTrain": "fitExpert",
    "modeTemTrain": "obsCreation",
    "modeEvPTrain": "obsCreation",
    "nrEpochs": nr_epochs,
}
xno = {
    "name": "fit_xno",
    "modeEvaTrain": "obsCreation",
    "modeSnoTrain": "fitExpert",
    "modeSubTrain": "obsCreation",
    "modeTemTrain": "obsCreation",
    "modeEvPTrain": "obsCreation",
    "nrEpochs": nr_epochs,
}
xva = {
    "name": "fit_xva",
    "modeEvaTrain": "fitExpert",
    "modeSnoTrain": "obsCreation",
    "modeSubTrain": "obsCreation",
    "modeTemTrain": "obsCreation",
    "modeEvPTrain": "obsCreation",
    "nrEpochs": nr_epochs,
}
xus = {
    "name": "fit_xus",
    "modeEvaTrain": "obsCreation",
    "modeSnoTrain": "fitExpert",
    "modeSubTrain": "fitExpert",
    "modeTemTrain": "obsCreation",
    "modeEvPTrain": "obsCreation",
    "nrEpochs": nr_epochs,
}
xue = {
    "name": "fit_xue",
    "modeEvaTrain": "fitExpert",
    "modeSnoTrain": "obsCreation",
    "modeSubTrain": "fitExpert",
    "modeTemTrain": "obsCreation",
    "modeEvPTrain": "obsCreation",
    "nrEpochs": nr_epochs,
}
xne = {
    "name": "fit_xne",
    "modeEvaTrain": "fitExpert",
    "modeSnoTrain": "fitExpert",
    "modeSubTrain": "obsCreation",
    "modeTemTrain": "obsCreation",
    "modeEvPTrain": "obsCreation",
    "nrEpochs": nr_epochs,
}
xhr = {
    "name": "fit_xhr",
    "modeEvaTrain": "fitExpert",
    "modeSnoTrain": "fitExpert",
    "modeSubTrain": "fitExpert",
    "modeTemTrain": "obsCreation",  # fixed as not considered here
    "modeEvPTrain": "obsCreation",  # idem
    "nrEpochs": nr_epochs,
}


# machine learning models (n = 7)

sub = {
    "name": "fit_sub",
    "modeEvaTrain": "obsCreation",
    "modeSnoTrain": "obsCreation",
    "modeSubTrain": "fit",
    "modeTemTrain": "obsCreation",
    "modeEvPTrain": "obsCreation",
    "nrEpochs": nr_epochs,
}
sno = {
    "name": "fit_sno",
    "modeEvaTrain": "obsCreation",
    "modeSnoTrain": "fit",
    "modeSubTrain": "obsCreation",
    "modeTemTrain": "obsCreation",
    "modeEvPTrain": "obsCreation",
    "nrEpochs": nr_epochs,
}
eva = {
    "name": "fit_eva",
    "modeEvaTrain": "fit",
    "modeSnoTrain": "obsCreation",
    "modeSubTrain": "obsCreation",
    "modeTemTrain": "obsCreation",
    "modeEvPTrain": "obsCreation",
    "nrEpochs": nr_epochs,
}
sus = {
    "name": "fit_sus",
    "modeEvaTrain": "obsCreation",
    "modeSnoTrain": "fit",
    "modeSubTrain": "fit",
    "modeTemTrain": "obsCreation",
    "modeEvPTrain": "obsCreation",
    "nrEpochs": nr_epochs,
}
sue = {
    "name": "fit_sue",
    "modeEvaTrain": "fit",
    "modeSnoTrain": "obsCreation",
    "modeSubTrain": "fit",
    "modeTemTrain": "obsCreation",
    "modeEvPTrain": "obsCreation",
    "nrEpochs": nr_epochs,
}
sne = {
    "name": "fit_sne",
    "modeEvaTrain": "fit",
    "modeSnoTrain": "fit",
    "modeSubTrain": "obsCreation",
    "modeTemTrain": "obsCreation",
    "modeEvPTrain": "obsCreation",
    "nrEpochs": nr_epochs,
}
thr = {
    "name": "fit_thr",
    "modeEvaTrain": "fit",
    "modeSnoTrain": "fit",
    "modeSubTrain": "fit",
    "modeTemTrain": "obsCreation",
    "modeEvPTrain": "obsCreation",
    "nrEpochs": nr_epochs,
}

# fitting_scenarios define what components are fitted and what aren't
# fitting_scenarios = [xva, xno, xub, xne, xue, xus, xhr]            # expert models, code is x plus second and third letter of corresponding ML scenario
# fitting_scenarios = [eva, sno, sub, sne, sue, sus, thr]            # ML models
fitting_scenarios = [
    xva,
    xno,
    xub,
    xne,
    xue,
    xus,
    xhr,
    eva,
]  # expert models and eva added for testing

outOne, outTwo, outThree, outFour = createTrainingIndices()

# training scenarios
one = {
    "name": "1",
    "stopping": outOne,
    "temperature": temperature_time_series,
    "precipitation": precipitationTimeSeries,
    "streamFlow": streamFlowTimeSeries,
    "date": date_time_series,
    "temperatureSto": temperatureTimeSeriesTwo,
    "precipitationSto": precipitationTimeSeriesTwo,
    "streamFlowSto": streamFlowTimeSeriesTwo,
    "dateSto": dateTimeSeriesTwo,
    "sno_s_ts": sno_s_ts,
    "sub_s_ts": sub_s_ts,
    "sno_f_ts": sno_f_ts,
    "sub_f_ts": sub_f_ts,
    "eva_f_ts": eva_f_ts,
    "sno_s_tsSto": sno_s_tsTwo,
    "sub_s_tsSto": sub_s_tsTwo,
    "sno_f_tsSto": sno_f_tsTwo,
    "sub_f_tsSto": sub_f_tsTwo,
    "eva_f_tsSto": eva_f_tsTwo,
}

two = one.copy()
three = one.copy()
four = one.copy()

two["name"] = "2"
three["name"] = "3"
four["name"] = "4"

two["stopping"] = outTwo
three["stopping"] = outThree
four["stopping"] = outFour

training_scenarios = [one, two, three, four]
# training_scenarios = [one, two]
# training_scenarios = [three, four]

# rerun scenarios
numberOfScenarios = 4
aRange = numpy.arange(1, numberOfScenarios + 1)
re_run_scenarios = []
for s in aRange:
    re_run_scenarios.append(str(s))

if run_in_batch:
    fitting_scenarios = [fitting_scenarios[int(first) - 1]]
    training_scenarios = training_scenarios[int(second) - 1 : int(second) + 1]
    re_run_scenarios = str.split(third, ",")

for fs in fitting_scenarios:
    for ts in training_scenarios:
        for rs in re_run_scenarios:
            scenario_directory = fs["name"] + "/" + ts["name"] + "/" + rs
            print(scenario_directory)
            seed = seed_generator(scenario_directory)
            print("seed is", seed)
            torch.manual_seed(seed)
            hydromodel = Net()
            optimizer = optim.RMSprop(
                [
                    {
                        "params": hydromodel.params.sub.parameters(),
                        "lr": 0.00001,
                        "momentum": 0.9,
                    },
                    {
                        "params": hydromodel.params.sno.parameters(),
                        "lr": 0.00001,
                        "momentum": 0.9,
                    },
                    {
                        "params": hydromodel.params.eva.parameters(),
                        "lr": 0.00001,
                        "momentum": 0.9,
                    },
                    {
                        "params": hydromodel.params.exp.parameters(),
                        "lr": 0.00001,
                        "momentum": 0.9,
                    },
                ]
            )
            # plateau https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.95, eps = 1e-8, patience = 50)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", factor=0.95, eps=1e-8, patience=200
            )

            training_loop(
                fs["nrEpochs"],
                ts["stopping"],
                optimizer,
                hydromodel,
                loss_fn,
                sno_s_initial,
                sub_s_initial,
                ts["temperature"],
                ts["precipitation"],
                ts["streamFlow"],
                ts["date"],
                ts["temperatureSto"],
                ts["precipitationSto"],
                ts["streamFlowSto"],
                ts["dateSto"],
                temperature_time_series_val,
                precipitation_time_series_val,
                streamflow_time_series_val,
                date_time_series_val,
                ts["sno_s_ts"],
                ts["sub_s_ts"],
                ts["sno_f_ts"],
                ts["sub_f_ts"],
                ts["eva_f_ts"],
                ts["sno_s_tsSto"],
                ts["sub_s_tsSto"],
                ts["sno_f_tsSto"],
                ts["sub_f_tsSto"],
                ts["eva_f_tsSto"],
                sno_s_tsVal,
                sub_s_tsVal,
                sno_f_tsVal,
                sub_f_tsVal,
                eva_f_tsVal,
                trainingData,
                fitOnObservations,
                fs["modeEvaTrain"],
                fs["modeSnoTrain"],
                fs["modeSubTrain"],
                fs["modeTemTrain"],
                fs["modeEvPTrain"],
                output_directory,
                scenario_directory,
                linearArt,
            )

