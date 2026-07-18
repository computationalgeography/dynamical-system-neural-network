import numpy as numpy
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

# Dynamical Neural Network Model
# by Derek Karssenberg, Utrecht University

numpy.set_printoptions(precision=2)


########################
# main configurations #
#######################

# Run in batch by providing inputs on command line
run_in_batch = True

# running in batch (typical usage) is done by for instance
# set run_in_batch to True
# for running kals catchment only use
# python kals_model.py sub 1 1 observations one test kals
# for running one of all catchments use for catchment twelve
# python kals_model.py sub 1 1 observations one 12 all_catch
# 1st value is scenario (here sub)
# 2nd value is training fold (1-4) (here 1)
# 3rd value is rerun scenario (1, 2, 3,...) (here 1)
# 4th value is observations indicating fitting on observational data
# or something else (here observations)
# 5th value is one or two, representing lumped or semi-distributed (here one)
# 6th value is name of output folder (here test), for all_catch this should be catchment ID
# 7th value is all_catch for running over all catch or else run Kals only
# these same related settings below are ignored when running in batch

# max number of epochs to run (will run for this number of epochs
# if validation (stopping) does not make it stop
nr_epochs = 5000

# run one area or else two
one_area = True

# multiply learning rate for quick tests (use 1.0 for real runs)
learning_rate_multiplier_value = 1.0  # 5.0 is possible for expert model fit

# fit model on observations or on artificial data
fitOnObservations = True

# add error to artificial data
addErrorToArtificialStreamFlow = False

# input directory with input data
#input_data_directory = " ../data/inputData/"
input_data_directory = "../data/inputData/"

# output directories
# folder where results folder (scen_directory) is written
#out_folder = " ../data/results/"
out_folder = "../data/results/"

# name of results folder
scen_directory = "test"


########################
# other configurations #
########################

# what weather data to use, if True GFS, else ECMWF Land
GFS = False

# training data to calculate loss over (typical sub, i.e. outflow)
training_data = "trainingSub"

# if fitting on artificial data, use linear model or non-linear one
linearArtForNotFitOnObservations = False

print_parameters = True

if run_in_batch:
    batch_scenario = sys.argv[1]
    training_scenario = sys.argv[2]
    re_run_scenario = sys.argv[3]
    fitOnObservationsOne = sys.argv[4]
    oneArea = sys.argv[5]
    scen_directory = sys.argv[6]
    if fitOnObservationsOne == "observations":
        fitOnObservations = True
    else:
        fitOnObservations = False
    if oneArea == "one":
        one_area = True
    else:
        one_area = False
    ID = sys.argv[6]    # ID of catchment from LamaH-CE, when running all catchm.
    if sys.argv[7] == "all_catch":
        all_catch = True
        print("Running one catchment from all catchm., catchment ID is: ", ID)
        scen_directory = ID
    else:
        all_catch = False




output_directory = out_folder + scen_directory + "/"

# print status for epochs to screen
print("##### RUN INFO ######")
print("running scenario: ", batch_scenario)
print("running training fold: ", training_scenario)
print("running rerun scenario: ", re_run_scenario)
print("fitting on observations: ", fitOnObservations)
print("running one area: ", one_area)
print("results written to: ", output_directory)
print("#### END RUN INFO ######")


# Create the run directory
try:
    os.mkdir(output_directory)
    print(f"Directory '{output_directory}' created successfully.")
except FileExistsError:
    print(f"Directory '{output_directory}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{output_directory}'.")
except Exception as e:
    print(f"An error occurred: {e}")


####################
# some definitions #
####################

# define if an additional NN layer is used (typically True in runs thus far)
# note that for true, it may need to be in the parameter list of the optimizer
deep_layer = True

m = torch.nn.ReLU()

# set the seed for the random initialization
seedAll = 5
torch.manual_seed(seedAll)
numpy.random.seed(seedAll)

# set the number of nodes in the NN layers
if deep_layer:
    nodes = 100
else:
    nodes = 1000

l1_regularization = False

# conversion from m3/s to m/day, catchment area 47.5 km3 (own calculation from DEM)
# GRDC gives 47.0, Lama-H gives 47.242 km3 which are very similar
conversion_fluxes = 549.3

# line width in plots
width = 0.25

# font size in plots
myFontSize = 7

# meteo start date is 1 jan 1979, end date is 31 juli 2014
# so do not select a time stem outside this range
# streamflow available always in this period


def seed_generator(string):
    """
    generator of seed to support different (and known)
    seed when rerunning the model
    """
    tot = 0
    i = 1
    for letter in string:
        tot += ord(letter) * i
        i = i + 10
    return tot + seedAll


def timeSeriesPlot_rich(
    sfd, fileName, date_time_series, variableTimeSeriesList, xlabel, ylabel
):
    """
    function to plot timeseries while running
    the optimization
    """
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


def scatterplot_rich(
    sfd, filename, variableOne, variableTwo, xlabel, ylabel, symbol, ylim
):
    """
    function to plot scatterplots while running
    the optimization
    """
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
    if GFS:
        spin_up_duration = 366
    else:
        spin_up_duration = 365
    for p in range(0, spin_up_duration):  # add one year spin up
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


# class weightConstraint(object):
#    def __init__(self):
#        pass
#
#    def __call__(self, module):
#        if hasattr(module, "weight"):
#            w = module.weight.data
#            w = w.clamp(0.0, 2.0)
#            # w=w.clamp(0.0,0.0)
#            module.weight.data = w


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


####################
# Data preparation #
####################


def createMeteoData(input_data_directory, output_directory, startDate, endDate, ID):
    """
    Creates meteodata set from input data
    Note that start date for ECMWF Land from LamaH-CH is assumed to be 1981 for
    all catchments. This is taken care of in catch_sel.py
    """
    precipitation_file = open(output_directory + "precipitation.txt", "w")
    temperatureFile = open(output_directory + "temperature.txt", "w")
    if GFS:
        date = datetime.date(1979, 1, 1)
    else:
        date = datetime.date(1981, 1, 1)

    timestep = datetime.timedelta(days=1)

    temperature_time_series = []  # temperature
    precipitation_time_series = []  # precipitation
    land_sno_time_series = []  # snow
    land_eva_time_series = []  # evapotranspiration

    if GFS:
        weatherDataCSV = "weatherdata-470125_corrected.csv"
    else:
        if all_catch:
            weatherDataCSV = "ID_" + ID + ".csv"
        else:
            weatherDataCSV = "ID_535.csv"
        print("input meteodata file is: ", weatherDataCSV)

    with open(input_data_directory + "A_basins_total_upstrm/" + weatherDataCSV) as csv_file:
        if GFS:
            csv_reader = csv.reader(csv_file, dialect="excel")
        else:
            csv_reader = csv.reader(csv_file, dialect="excel", delimiter=";")
        line_count = 0
        line_out_count = 1
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                if GFS:
                    temperature = (float(row[4]) + float(row[5])) / 2.0
                else:
                    temperature = float(row[5])
                if GFS:
                    precip = row[6]
                else:
                    precip = row[22]
                if GFS:
                    land_sno = -9.0
                    land_eva = -9.0
                else:
                    land_sno = float(row[15]) / 1000.0
                    land_eva = float(row[21]) / 1000.0
                if (date >= startDate) and (date <= endDate):
                    precipitation_file.write(str(line_out_count) + " " + precip + "\n")
                    precipitation_time_series.append(float(precip))
                    temperatureFile.write(
                        str(line_out_count) + " " + str(temperature) + "\n"
                    )
                    temperature_time_series.append(temperature)
                    land_sno_time_series.append(land_sno)
                    land_eva_time_series.append(land_eva)
                    line_out_count += 1
                line_count += 1
                date = date + timestep
        if line_count < 14245:
            print("number of days in meteo data is too limited")
            exit()

    precipitation_file.close()
    temperatureFile.close()
    return (
        temperature_time_series,
        precipitation_time_series,
        land_sno_time_series,
        land_eva_time_series,
    )


def create_cosero_data(input_data_directory, output_directory, startDate, endDate, ID):
    """
    Creates data set from cosero runs in Lama
    """
    date = datetime.date(1981, 1, 1)

    timestep = datetime.timedelta(days=1)

    cosero_sub_s_soil_time_series = []  # subsurface storage, soil
    cosero_sub_s_gw_time_series = []  # subsurface storage, groundwater
    cosero_eva_f_time_series = []  # evapotranspiration flux

    coseroDataCSV = "ID_535_cosero.csv"

    if all_catch:
        coseroDataCSV = "ID_" + ID + ".csv"
    else:
        coseroDataCSV = "ID_535.csv"
    print("input cosero data file is: ", coseroDataCSV)

    with open(input_data_directory + "F_hydrol_model/" + coseroDataCSV) as csv_file:
        csv_reader = csv.reader(csv_file, dialect="excel", delimiter=";")
        line_count = 0
        line_out_count = 1
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                sub_s_soil = float(row[11]) / 1000.0
                sub_s_gw = float(row[13]) / 1000.0
                eva_f = float(row[9]) / 1000.0
                if (date >= startDate) and (date <= endDate):
                    cosero_sub_s_soil_time_series.append(sub_s_soil)
                    cosero_sub_s_gw_time_series.append(sub_s_gw)
                    cosero_eva_f_time_series.append(eva_f)
                    line_out_count += 1
                line_count += 1
                date = date + timestep
    numpy.save(
        output_directory + "cosero_sub_s_soil.npy", cosero_sub_s_soil_time_series
    )
    numpy.save(output_directory + "cosero_sub_s_gw.npy", cosero_sub_s_gw_time_series)
    numpy.save(output_directory + "cosero_eva_f.npy", cosero_eva_f_time_series)
    return (
        cosero_sub_s_soil_time_series,
        cosero_sub_s_gw_time_series,
        cosero_eva_f_time_series,
    )


def create_cosero_data_additional(
    input_data_directory, output_directory, startDate, endDate
):
    """
    read model outputs from cosero runs in LamaH, additional attributes
    bw 0 item 8
    bw 1 item 9
    bw 2 item 10
    bw 3 item 11
    bw 4 item 12
    melt item 21
    smelt item 36
    glacmelt item 40
    """
    date = datetime.date(1981, 1, 1)

    timestep = datetime.timedelta(days=1)

    cosero_bw0_time_series = []
    cosero_bw1_time_series = []
    cosero_bw2_time_series = []
    cosero_bw3_time_series = []
    cosero_bw4_time_series = []
    cosero_melt_time_series = []
    cosero_smelt_time_series = []
    cosero_glacmelt_time_series = []

    coseroDataCSV = "monitor_sb0276.txt"
    print('update this')
    exit()

    with open(input_data_directory + coseroDataCSV) as csv_file:
        csv_reader = csv.reader(
            csv_file, dialect="excel", delimiter=" ", skipinitialspace=True
        )
        line_count = 0
        line_out_count = 1
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                bw0 = float(row[8]) / 1000.0
                bw1 = float(row[9]) / 1000.0
                bw2 = float(row[10]) / 1000.0
                bw3 = float(row[11]) / 1000.0
                bw4 = float(row[12]) / 1000.0
                melt = float(row[21]) / 1000.0
                smelt = float(row[36]) / 1000.0
                glacmelt = float(row[40]) / 1000.0
                if (date >= startDate) and (date <= endDate):
                    cosero_bw0_time_series.append(bw0)
                    cosero_bw1_time_series.append(bw1)
                    cosero_bw2_time_series.append(bw2)
                    cosero_bw3_time_series.append(bw3)
                    cosero_bw4_time_series.append(bw4)
                    cosero_melt_time_series.append(melt)
                    cosero_smelt_time_series.append(smelt)
                    cosero_glacmelt_time_series.append(glacmelt)
                    line_out_count += 1
                line_count += 1
                date = date + timestep
    numpy.save(output_directory + "cosero_bw0.npy", cosero_bw0_time_series)
    numpy.save(output_directory + "cosero_bw1.npy", cosero_bw1_time_series)
    numpy.save(output_directory + "cosero_bw2.npy", cosero_bw2_time_series)
    numpy.save(output_directory + "cosero_bw3.npy", cosero_bw3_time_series)
    numpy.save(output_directory + "cosero_bw4.npy", cosero_bw4_time_series)
    numpy.save(output_directory + "cosero_melt.npy", cosero_melt_time_series)
    numpy.save(output_directory + "cosero_smelt.npy", cosero_smelt_time_series)
    numpy.save(output_directory + "cosero_glacmelt.npy", cosero_glacmelt_time_series)


def create_streamflow_data(input_data_directory, output_directory, start, end):
    """
    Create data set for streamflow data
    """
    streamflow_file = open(output_directory + "streamflow.txt", "w")
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
            streamflow_file.write(str(line_out_count) + " " + discharge + "\n")
            streamFlowTimeSeries.append(float(discharge))
            date_time_series.append(date)
            line_out_count += 1
        date = date + timestep

    dischargeFile.close()
    return streamFlowTimeSeries, date_time_series


#########################################
# Dynamical System Neural Network model #
#########################################


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # neural network layers for each of the three neural network
        # model components
        self.eva_f_d = nn.Linear(1, nodes)  # eva, evapotranspiration
        self.eva_f_c = nn.Linear(nodes, 1)
        self.sno_f_d = nn.Linear(1, nodes)  # sno, snow
        self.sno_f_c = nn.Linear(nodes, 1)
        self.sub_f_d = nn.Linear(1, nodes)  # sub, subsurface
        self.sub_f_c = nn.Linear(nodes, 1)

        # option to include an additional layer in each component
        if deep_layer:
            self.eva_f_dd = nn.Linear(nodes, nodes)
            self.sno_f_dd = nn.Linear(nodes, nodes)
            self.sub_f_dd = nn.Linear(nodes, nodes)

        # calibrated parameters of process-based model components
        if GFS:
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
        else:
            if one_area:
                eva_par = 0.014076318
                sub_par = 0.03213018
                sno_par = 0.0029198169
                tem_par = 0.0  # assumed zero
                evp_par = 3.5  # note input to sigmoid (evp [0,1]), 3.5 equals 0.97
            else:
                eva_par = 0.072542064
                sub_par = 0.113195494
                sno_par = 0.0012788665
                tem_par = 0.0  # assumed zero
                evp_par = 3.5

        self.eva_parameter_obs_creation = eva_par
        self.sub_parameter_obs_creation = sub_par
        self.sno_parameter_obs_creation = sno_par
        self.tem_parameter_obs_creation = tem_par
        self.evp_parameter_obs_creation = evp_par

        # random initialisation of parameters of process-based model components
        # as starting value for calibrating the process-based model
        proportion = 0.5
        self.eva_parameter = nn.Parameter(rand_min_max(eva_par, proportion))
        self.sub_parameter = nn.Parameter(rand_min_max(sub_par, proportion))
        self.sno_parameter = nn.Parameter(rand_min_max(sno_par, proportion))
        self.tem_parameter = nn.Parameter(rand_min_max(tem_par, proportion))
        self.evp_parameter = nn.Parameter(rand_min_max(evp_par, proportion))

        # create dictionary with model parameters for accessing these
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
        """
        Evapotranspiration component model
        Calculates evapotranspiration from temperature (temp argument)
        mode_eva defines what type of component is used:
        obsCreation: process-based component model is used (without training)
            linearArt: for synthetic data set generation model (True)
                       or process-based (expert) model
        fit: machine learning component model is trained
        fitExpert: process-based component model is calibrated (for real-world data
                   set only)
        """
        if mode_eva == "obsCreation":
            if linearArt:
                EPot = torch.max(
                    ((0.35 * (0.457 * torch.tensor([temp]) + 8.128)) / 1000.0)
                    * self.eva_parameter_obs_creation,
                    torch.tensor(0.0),
                )
            else:
                if temp > -15.0:
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
            EPot = torch.sigmoid(self.eva_f_c(out_eva_f)) / 75.0

        if mode_eva == "fitExpert":
            EPot = torch.max(
                ((0.35 * (0.457 * temp + 8.128)) / 1000.0) * self.eva_parameter,
                torch.tensor(0.0),
            )
        return EPot

    def sno_f_calculate(self, temp, mode_sno, deep_layer, linearArt):
        """
        Snow component model
        Calculates potential snowmelt from temperature (temp argument)
        mode_sno defines what type of component is used:
        obsCreation: process-based component model is used (without training)
            linearArt: for synthetic data set generation model (True)
                       or process-based (expert) model
        fit: machine learning component model is trained
        fitExpert: process-based component model is calibrated (for real-world data
                   set only)
        """
        if mode_sno == "obsCreation":
            if linearArt:
                melting = temp > 0.0
                if melting:
                    potential_melt = temp * self.sno_parameter_obs_creation
                else:
                    potential_melt = torch.tensor(0.0)
            else:
                melting = temp > 0.0
                if melting:
                    potential_melt = (
                        (torch.sin((torch.tensor([temp]) * 0.6 - 0.5 * torch.pi)) + 1)
                        / 200
                    ) + 0.0004 * torch.tensor([temp])
                else:
                    potential_melt = torch.tensor(0.0)
        if mode_sno == "fit":
            out_sno_f = m(self.sno_f_d((torch.tensor([temp]))))
            if deep_layer:
                out_sno_f = m(self.sno_f_dd(out_sno_f))
            potential_melt = torch.sigmoid(self.sno_f_c(out_sno_f)) / 50.0
        if mode_sno == "fitExpert":
            melting = temp > 0.0
            if melting:
                potential_melt = temp * self.sno_parameter
            else:
                potential_melt = torch.tensor(0.0)
        return potential_melt

    def sub_f_calculate(self, sub_s, mode_sub, deep_layer, linearArt):
        """
        Subsurface water component model
        Calculates snowmelt from subsurface storage (sub_s argument)
        mode_sno defines what type of component is used:
        obsCreation: process-based component model is used (without training)
            linearArt: for synthetic data set generation model (True)
                       or process-based (expert) model
        fit: machine learning component model is trained
        fitExpert: process-based component model is calibrated (for real-world data
                   set only)
        """
        if mode_sub == "obsCreation":
            if linearArt:
                seepage_pot = self.sub_parameter_obs_creation * sub_s
            else:
                seepage_pot = (
                    (torch.sin((torch.tensor([sub_s]) * 20 - 0.5 * torch.pi)) + 1) / 500
                ) + 0.03 * sub_s
        if mode_sub == "fit":
            out_sub_f = m(self.sub_f_d((torch.tensor([sub_s]) * 2.0)))
            if deep_layer:
                out_sub_f = m(self.sub_f_dd(out_sub_f))
            proportion = torch.sigmoid(self.sub_f_c(out_sub_f))
            seepage_pot = proportion * sub_s
        if mode_sub == "fitExpert":
            seepage = self.sub_parameter * sub_s
            seepage_pot = seepage
        return seepage_pot

    def com_f_response(self, mode_sub, deep_layer, component, linearArt):
        """
        Creates data to plot governing equation ('response function') with
        used type of model component.
        It first creates a range of drivers (com_s_values) and then
        passes these to the model component which returns the range of
        corresponding fluxes (com_f_value).
        """
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
                    com_s_value.item(), mode_sub, deep_layer, linearArt
                )
            if component == "sno":
                com_f_value = self.sno_f_calculate(
                    com_s_value.item(), mode_sub, deep_layer, linearArt
                )
            if component == "eva":
                com_f_value = self.eva_f_calculate(
                    com_s_value.item(), mode_sub, deep_layer, linearArt
                )
            if isinstance(com_f_value, torch.Tensor):
                if mode_sub == "fitExpert":
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
        first_epochs,  # start of training or not
        sno_s_initial,  # initial snow storage
        sub_s_initial,  # initial subsurface storage
        temperature,  # temperature time series
        precipitation,  # precipitation time series
        mode_eva,  # mode evapotranspiration
        mode_sno,  # mode snow
        mode_sub,  # mode subsurface
        modeTem,  # temperature offset
        modeEvP,  # potential proportion of evap to sublimation
    ):
        # number of time steps in model run
        nr_timesteps = len(temperature)

        if modeTem == "fitExpert":
            temperatureOffset = self.tem_parameter
        if modeTem == "obsCreation":
            temperatureOffset = self.tem_parameter_obs_creation

        if modeEvP == "fitExpert":
            evaPropToSno = torch.sigmoid(self.evp_parameter)
        if modeEvP == "obsCreation":
            evaPropToSno = torch.sigmoid(torch.tensor(self.evp_parameter_obs_creation))

        # difference between lower and higher half of the area is (2115 m - 2797 m)
        # multiplied by lapse rate of 0.005 gives 3.41 degrees difference in temperature
        # use this for semi-distributed model only
        if one_area:
            areaTemperatureOffsets = [0.0]
        else:
            areaTemperatureOffsets = [1.705, -1.705]

        if one_area:
            numberOfAreas = 1
        else:
            numberOfAreas = 2

        # loop over the areas (for distributed model)
        for area in range(0, numberOfAreas):
            # create emtpy output timeseries for writing
            # storages
            sno_s_ts = torch.zeros(nr_timesteps)
            sub_s_ts = torch.zeros(nr_timesteps)
            # fluxes
            sno_f_ts = torch.zeros(nr_timesteps)
            sub_f_ts = torch.zeros(nr_timesteps)
            eva_f_ts = torch.zeros(nr_timesteps)

            # set initial storages
            sno_s = sno_s_initial
            sub_s = sub_s_initial

            # loop over the time steps, i.e. the actual dynamical model
            for i in range(0, nr_timesteps):
                # get system drivers for timestep
                temp = temperature[i] + temperatureOffset + areaTemperatureOffsets[area]

                ######################
                # evapotranspiration #
                ######################
                EPot = torch.max(
                    self.eva_f_calculate(temp, mode_eva, deep_layer, linearArt),
                    torch.tensor(0.0),
                )

                eva_f = EPot
                eva_f_ts[i] = eva_f

                ################
                # snow or rain #
                ################

                # convert from mm to m
                precip = precipitation[i] / 1000.0
                snowing = temp < 0.0
                if snowing:
                    snowfall = precip
                    rainfall = torch.tensor(0.0)
                else:
                    snowfall = torch.tensor(0.0)
                    rainfall = precip

                ################
                # snow storage #
                ################

                # add snowfall to snow storage
                sno_s = sno_s + snowfall

                sno_s_ts[i] = sno_s

                # snow melt
                potential_melt = self.sno_f_calculate(
                    temp, mode_sno, deep_layer, linearArt
                )

                # limit snow melt to prevent zero snow cover
                # over first epochs, for more efficient training
                if first_epochs:
                    if temp < -5.0:
                        potential_melt = torch.tensor(0.0)

                sno_f = potential_melt

                sno_f_ts[i] = sno_f

                potential_melt = torch.max(potential_melt, torch.tensor(0))
                actualMelt = torch.min(sno_s, potential_melt)
                sno_s = sno_s - actualMelt

                # sublimation
                potentialSublimation = evaPropToSno * EPot
                actualSublimation = torch.min(sno_s, potentialSublimation)
                sno_s = sno_s - actualSublimation
                potentialEvapForSub = EPot - actualSublimation

                #######################
                # sub surface storage #
                #######################

                # total input to subsurface storage
                totalInput = rainfall + actualMelt
                hortonRunoffProportion = 0.0
                hortonRunoff = hortonRunoffProportion * totalInput

                sub_s = sub_s + (1.0 - hortonRunoffProportion) * totalInput

                EAct = torch.min(sub_s, potentialEvapForSub)

                sub_s = sub_s - EAct

                sub_s_ts[i] = sub_s

                seepage_pot = self.sub_f_calculate(
                    sub_s, mode_sub, deep_layer, linearArt
                )

                seepageNotNegative = torch.max(seepage_pot, torch.tensor(0.0))
                seepageAct = torch.min(seepageNotNegative, sub_s)

                sub_f = seepageAct + hortonRunoff

                sub_f_ts[i] = sub_f

                sub_s = sub_s - seepageAct

            # data reorganisation in case of two areas (vstack)
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

            # if one area add the same ones again (mimic two areas, each the same
            # such that processing is the same for one or two areas
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


##################################
# Create artificial observations #
##################################


def create_artificial_observations(
    temperature_time_series,
    precipitation_time_series,
    addErrorToArtificialStreamFlow,
    linearArt,
):
    """
    Create synthetic data set
    """
    hydromodel = Net()
    sno_s_initial = 0.0
    sub_s_initial = 0.0
    (
        sno_s_ts_areas,
        sub_s_ts_areas,
        sno_f_ts_areas,
        sub_f_ts_areas,
        eva_f_ts_areas,
    ) = hydromodel(
        linearArt,
        False,
        torch.tensor(sno_s_initial),
        torch.tensor(sub_s_initial),
        torch.tensor(temperature_time_series),
        torch.tensor(precipitation_time_series),
        mode_eva="obsCreation",
        mode_sno="obsCreation",
        mode_sub="obsCreation",
        modeTem="obsCreation",
        modeEvP="obsCreation",
    )
    sno_s_ts = sno_s_ts_areas.mean(dim=0)
    sub_s_ts = sub_s_ts_areas.mean(dim=0)
    sno_f_ts = sno_f_ts_areas.mean(dim=0)
    sub_f_ts = sub_f_ts_areas.mean(dim=0)
    eva_f_ts = eva_f_ts_areas.mean(dim=0)
    if addErrorToArtificialStreamFlow:
        sd = 0.2
        errorMultiplier = numpy.minimum(
            numpy.maximum(0.0, numpy.random.normal(1, sd, len(streamFlowTimeSeries))), 2
        )
        sub_f_ts = torch.max(sub_f_ts * errorMultiplier, torch.tensor(0.0))
    return (
        sno_s_ts,
        sub_s_ts,
        sno_f_ts,
        sub_f_ts,
        eva_f_ts,
        sno_s_ts_areas,
        sub_s_ts_areas,
        sno_f_ts_areas,
        sub_f_ts_areas,
        eva_f_ts_areas,
    )


############
# Training #
############


def training_loop(
    n_epochs,
    stopping,
    optimizer,
    model,
    loss_fn,
    sno_s_initial,
    sub_s_initial,
    temperature_time_series,
    precipitation_time_series,
    streamFlowTimeSeries,
    land_sno_time_series,
    land_eva_time_series,
    date_time_series,
    temperature_time_series_val,
    precipitation_time_series_val,
    streamflow_time_series_val,
    land_sno_time_series_val,
    land_eva_time_series_val,
    date_time_series_val,
    sno_s_ts,
    sub_s_ts,
    sno_f_ts,
    sub_f_ts,
    eva_f_ts,
    sno_s_tsVal,
    sub_s_tsVal,
    sno_f_tsVal,
    sub_f_tsVal,
    eva_f_tsVal,
    sno_s_ts_areasVal,
    sub_s_ts_areasVal,
    sno_f_ts_areasVal,
    sub_f_ts_areasVal,
    eva_f_ts_areasVal,
    training_data,
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
    # create output directories for writing data below
    sfd = output_directory + scenario_directory + "/"
    sfdAr = sfd + "arrays" + "/"
    if not os.path.exists(sfd):
        os.makedirs(sfd)
    if not os.path.exists(sfdAr):
        os.makedirs(sfdAr)

    # write artificial data to disk, training
    numpy.save(sfdAr + "train_art_ts_sno_s.npy", sno_s_ts)
    numpy.save(sfdAr + "train_art_ts_sub_s.npy", sub_s_ts)
    numpy.save(sfdAr + "train_art_ts_sno_f.npy", sno_f_ts)
    numpy.save(sfdAr + "train_art_ts_sub_f.npy", sub_f_ts)
    numpy.save(sfdAr + "train_art_ts_eva_f.npy", eva_f_ts)

    # write artificial data to disk, validation, total of catchment and areas
    numpy.save(sfdAr + "val_art_ts_sno_s.npy", sno_s_tsVal)
    numpy.save(sfdAr + "val_art_ts_sub_s.npy", sub_s_tsVal)
    numpy.save(sfdAr + "val_art_ts_sno_f.npy", sno_f_tsVal)
    numpy.save(sfdAr + "val_art_ts_sub_f.npy", sub_f_tsVal)
    numpy.save(sfdAr + "val_art_ts_eva_f.npy", eva_f_tsVal)
    numpy.save(sfdAr + "val_art_ts_sno_s_areas.npy", sno_s_ts_areasVal)
    numpy.save(sfdAr + "val_art_ts_sub_s_areas.npy", sub_s_ts_areasVal)
    numpy.save(sfdAr + "val_art_ts_sno_f_areas.npy", sno_f_ts_areasVal)
    numpy.save(sfdAr + "val_art_ts_sub_f_areas.npy", sub_f_ts_areasVal)
    numpy.save(sfdAr + "val_art_ts_eva_f_areas.npy", eva_f_ts_areasVal)

    # write internal observed variables to disk, validation
    numpy.save(sfdAr + "val_lan_ts_sno_s.npy", land_sno_time_series_val)
    numpy.save(sfdAr + "val_lan_ts_eva_f.npy", land_eva_time_series_val)

    # write internal observed variables to disk, training
    numpy.save(sfdAr + "train_lan_ts_sno_s.npy", land_sno_time_series)
    numpy.save(sfdAr + "train_lan_ts_eva_f.npy", land_eva_time_series)

    # create empty lists for information related to each epoch
    # in particular loss, epoch number, stopper
    loss_training_series = []
    loss_validation_series = []
    loss_stopping_series = []
    epoch_series = []
    stopper_counter_series = []

    early_stopper = EarlyStopper(patience=200, min_delta_proportion=1e-2)

    # set time for tracking run times
    time = datetime.datetime.now()

    # loop over the epochs
    first_epochs = True
    for epoch in range(1, n_epochs + 1):
        # set first_epochs indicating initial part of training
        # is used in dynamical model for snow storage component
        # to make training more efficient
        if epoch > 100:
            first_epochs = False
        first_epochs = True

        # set time for tracking run times
        newTime = datetime.datetime.now()
        duration = newTime - time
        time = datetime.datetime.now()

        # run the dynamical model for training
        (
            tr_sno_s_ts_areas,
            tr_sub_s_ts_areas,
            tr_sno_f_ts_areas,
            tr_sub_f_ts_areas,
            tr_eva_f_ts_areas,
        ) = hydromodel(
            linearArt,
            first_epochs,
            torch.tensor(sno_s_initial),
            torch.tensor(sub_s_initial),
            torch.tensor(temperature_time_series),
            torch.tensor(precipitation_time_series),
            mode_eva=mode_eva_train,
            mode_sno=modeSnoTrain,
            mode_sub=modeSubTrain,
            modeTem=modeTemTrain,
            modeEvP=modeEvPTrain,
        )

        # average model outputs over 2 areas (in case of 1 area
        # two similar timeseries are averaged)
        tr_sno_s_ts = tr_sno_s_ts_areas.mean(dim=0)
        tr_sub_s_ts = tr_sub_s_ts_areas.mean(dim=0)
        tr_sno_f_ts = tr_sno_f_ts_areas.mean(dim=0)
        tr_sub_f_ts = tr_sub_f_ts_areas.mean(dim=0)
        tr_eva_f_ts = tr_eva_f_ts_areas.mean(dim=0)

        training = numpy.logical_not(stopping)

        # calculate training loss over observed streamflow (fitOnObservations)
        # or over synthetic streamflow
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
            # select what is trained on for synthetic run (so far for
            # production always Sub, i.e. streamflow)
            if training_data == "trainingSub":
                loss_train = loss_fn(tr_sub_f_ts, sub_f_ts, training)
            if training_data == "trainingEva":
                loss_train = loss_fn(tr_eva_f_ts, eva_f_ts, training)
            if training_data == "trainingSno":
                loss_train = loss_fn(tr_sno_f_ts, sno_f_ts, training)
            if training_data == "trainingSubAndSnow":
                modified_snow = sno_s_ts / 2.0
                loss_train_sno = loss_fn(tr_sno_s_ts, modified_snow)
                loss_train_sub = loss_fn(tr_sub_s_ts, sub_s_ts)
                loss_train = (
                    loss_train_sno / 4.0 + loss_train_sub
                ) / 2.0  # /4.0 to standardize snow

        loss_training_series.append(loss_train.item())
        epoch_series.append(epoch)

        # STOPPING, ie VALIDATION
        # calculate validation loss in a similar fashion
        # it is used here to decide on early stopping of training
        # and is thus referred to as 'stopping'
        if epoch == 1 or epoch % 1 == 0:
            if fitOnObservations:
                loss_trainSto = loss_fn(
                    tr_sub_f_ts,
                    torch.tensor(streamFlowTimeSeries) / conversion_fluxes,
                    stopping,
                )
            else:
                loss_trainSto = loss_fn(tr_sub_f_ts, sub_f_ts, stopping)

        loss_stopping_series.append(loss_trainSto.item())
        counter, stop = early_stopper.early_stop(loss_trainSto)
        stopper_counter_series.append(counter)

        # run the dynamical model for VALIDATION, ie TESTING, on independent
        # data not used for training or validation
        # this is done only for every 50 epochs to save run time
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
                mode_sno=modeSnoTrain,
                mode_sub=modeSubTrain,
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

        loss_validation_series.append(loss_trainVal.item())

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

        # every 20 epochs, print status, create intermediate figures, and save outputs ###################
        if epoch == 1 or epoch % 20 == 0 or stop:
            print("\n###############################")
            # print('running scenario: ', scenario_directory, ' running one area: ', one_area)
            print(
                "running scenario:",
                scenario_directory,
                " Running one area:",
                one_area,
                " Duration of one epoch:",
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

            a, b = hydromodel.com_f_response(
                mode_eva_train, deep_layer, "eva", linearArt
            )
            if GFS:
                y_max = 0.02
            else:
                y_max = 0.005
            scatterplot_rich(
                sfd,
                "response_eva",
                a,
                b,
                "temperature (C)",
                "potential flux (m/day)",
                "-",
                (0, y_max),
            )
            numpy.save(sfdAr + "response_eva_x.npy", numpy.array(a))
            numpy.save(sfdAr + "response_eva_y.npy", numpy.array(b))

            timeSeriesPlot_rich(
                sfd,
                "loss",
                epoch_series,
                [
                    torch.min(torch.tensor(loss_training_series), torch.tensor(1e-5)),
                    torch.min(torch.tensor(loss_stopping_series), torch.tensor(1e-5)),
                    torch.min(torch.tensor(loss_validation_series), torch.tensor(1e-5)),
                ],
                "epoch",
                "loss",
            )
            timeSeriesPlot_rich(
                sfd,
                "stopCounter",
                epoch_series,
                [stopper_counter_series],
                "epoch",
                "counter",
            )
            numpy.save(sfdAr + "epochs.npy", numpy.array(epoch_series))
            numpy.save(sfdAr + "lossTraining.npy", numpy.array(loss_training_series))
            numpy.save(sfdAr + "lossStopping.npy", numpy.array(loss_stopping_series))
            numpy.save(
                sfdAr + "lossValidation.npy", numpy.array(loss_validation_series)
            )
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
            numpy.save(
                sfdAr + "train_ts_temperature.npy", numpy.array(temperature_time_series)
            )
            numpy.save(
                sfdAr + "train_ts_precipitation.npy",
                numpy.array(precipitation_time_series),
            )
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
                "flux (m/day)",
            )
            timeSeriesPlot_rich(
                sfd,
                "valid_ts_lan_sno",
                date_time_series_val,
                [tr_sno_s_tsVal.detach().numpy(), land_sno_time_series_val],
                "time",
                "storage (m)",
            )
            timeSeriesPlot_rich(
                sfd,
                "valid_ts_lan_eva",
                date_time_series_val,
                [tr_eva_f_tsVal.detach().numpy(), land_eva_time_series_val],
                "time",
                "flux (m/day)",
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
            numpy.save(
                sfdAr + "valid_ts_temperature.npy",
                numpy.array(temperature_time_series_val),
            )
            numpy.save(
                sfdAr + "valid_ts_precipitation.npy",
                numpy.array(precipitation_time_series_val),
            )

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
            print("\n")
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


if fitOnObservations:
    linearArt = True
else:
    linearArt = linearArtForNotFitOnObservations


#####################
# Data set creation #
#####################

# data for training and stopping ie validation

# small modification for non GFS data
if GFS:
    yearIncrease = 0
else:
    yearIncrease = 2
startOne = datetime.date(1979 + yearIncrease, 10, 1)
endOne = datetime.date(1996 + yearIncrease, 9, 26)

(
    temperature_time_series,
    precipitation_time_series,
    land_sno_time_series,
    land_eva_time_series,
) = createMeteoData(input_data_directory, output_directory, startOne, endOne, ID)

streamFlowTimeSeries, date_time_series = create_streamflow_data(
    input_data_directory, output_directory, startOne, endOne
)

(
    sno_s_ts,
    sub_s_ts,
    sno_f_ts,
    sub_f_ts,
    eva_f_ts,
    sno_s_ts_areas,
    sub_s_ts_areas,
    sno_f_ts_areas,
    sub_f_ts_areas,
    eva_f_ts_areas,
) = create_artificial_observations(
    temperature_time_series,
    precipitation_time_series,
    addErrorToArtificialStreamFlow,
    linearArt,
)

# data for validation, ie testing

startVal = datetime.date(1995 + yearIncrease, 10, 1)
endVal = datetime.date(2012 + yearIncrease, 9, 26)

(
    temperature_time_series_val,
    precipitation_time_series_val,
    land_sno_time_series_val,
    land_eva_time_series_val,
) = createMeteoData(input_data_directory, output_directory, startVal, endVal, ID)

streamflow_time_series_val, date_time_series_val = create_streamflow_data(
    input_data_directory, output_directory, startVal, endVal
)

(
    cosero_sub_s_soil_time_series_val,
    cosero_sub_s_gw_time_series_val,
    cosero_eva_f_time_series_val,
) = create_cosero_data(input_data_directory, output_directory, startVal, endVal, ID)

create_cosero_data_additional(input_data_directory, output_directory, startVal, endVal)

# note that no error is added to artificial streamflow in any case as it is used for validation only
(
    sno_s_tsVal,
    sub_s_tsVal,
    sno_f_tsVal,
    sub_f_tsVal,
    eva_f_tsVal,
    sno_s_ts_areasVal,
    sub_s_ts_areasVal,
    sno_f_ts_areasVal,
    sub_f_ts_areasVal,
    eva_f_ts_areasVal,
) = create_artificial_observations(
    temperature_time_series_val, precipitation_time_series_val, False, linearArt
)


# define initial snow and subsurface storages
sno_s_initial = 0.0
sub_s_initial = 0.0


#####################
# Fitting scenarios #
#####################

# so called expert models where the process-based
# model parameters are fitted, each for a component
# ('fitExpert') as specified below
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
# for the different scenarios of model configurations
# each time fitting one or more model components represented
# as neural network ("fit")

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
# fitting_scenarios = [eva, sno, sub, sne, sue, sus, thr]              # ML models

# 8 scenarios: 7 ML scenarios and 1 expert model scenario for reference
if fitOnObservations:
    fitting_scenarios = [eva, sno, sub, sne, sue, sus, thr, xhr]
# 14 scenarios: 7 ML scenarios and 7 expert scenarios
else:
    fitting_scenarios = [
        eva,
        sno,
        sub,
        sne,
        sue,
        sus,
        thr,
        xva,
        xno,
        xub,
        xne,
        xue,
        xus,
        xhr,
    ]

if run_in_batch:
    if batch_scenario == "eva":
        fitting_scenarios = [eva]
    if batch_scenario == "sno":
        fitting_scenarios = [sno]
    if batch_scenario == "sub":
        fitting_scenarios = [sub]
    if batch_scenario == "sne":
        fitting_scenarios = [sne]
    if batch_scenario == "sue":
        fitting_scenarios = [sue]
    if batch_scenario == "sus":
        fitting_scenarios = [sus]
    if batch_scenario == "thr":
        fitting_scenarios = [thr]
    if batch_scenario == "xva":
        fitting_scenarios = [xva]
    if batch_scenario == "xno":
        fitting_scenarios = [xno]
    if batch_scenario == "xub":
        fitting_scenarios = [xub]
    if batch_scenario == "xne":
        fitting_scenarios = [xne]
    if batch_scenario == "xue":
        fitting_scenarios = [xue]
    if batch_scenario == "xus":
        fitting_scenarios = [xus]
    if batch_scenario == "xhr":
        fitting_scenarios = [xhr]

outOne, outTwo, outThree, outFour = createTrainingIndices()

# training folds
one = {
    "name": "1",
    "stopping": outOne,
    "temperature": temperature_time_series,
    "precipitation": precipitation_time_series,
    "streamFlow": streamFlowTimeSeries,
    "land_sno": land_sno_time_series,
    "land_eva": land_eva_time_series,
    "date": date_time_series,
    "sno_s_ts": sno_s_ts,
    "sub_s_ts": sub_s_ts,
    "sno_f_ts": sno_f_ts,
    "sub_f_ts": sub_f_ts,
    "eva_f_ts": eva_f_ts,
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

if run_in_batch:
    if training_scenario == "1":
        training_scenarios = [one]
    if training_scenario == "2":
        training_scenarios = [two]
    if training_scenario == "3":
        training_scenarios = [three]
    if training_scenario == "4":
        training_scenarios = [four]

# rerun scenarios, i.e. the same run each time
# with different random initial parameters
number_of_scenarios = 4
aRange = numpy.arange(1, number_of_scenarios + 1)
re_run_scenarios = []
for s in aRange:
    re_run_scenarios.append(str(s))

if run_in_batch:
    re_run_scenarios = [re_run_scenario]

for fs in fitting_scenarios:
    for ts in training_scenarios:
        for rs in re_run_scenarios:
            scenario_directory = fs["name"] + "/" + ts["name"] + "/" + rs
            print(scenario_directory)
            seed = seed_generator(scenario_directory)
            print("seed is", seed)
            torch.manual_seed(seed)
            hydromodel = Net()
            learning_rate_multiplier = learning_rate_multiplier_value
            optimizer = optim.RMSprop(
                [
                    {
                        "params": hydromodel.params.sub.parameters(),
                        "lr": 0.00001 * learning_rate_multiplier,
                        "momentum": 0.9,
                    },
                    {
                        "params": hydromodel.params.sno.parameters(),
                        "lr": 0.00001 * learning_rate_multiplier,
                        "momentum": 0.9,
                    },
                    {
                        "params": hydromodel.params.eva.parameters(),
                        "lr": 0.00001 * learning_rate_multiplier,
                        "momentum": 0.9,
                    },
                    {
                        "params": hydromodel.params.exp.parameters(),
                        "lr": 0.00001 * learning_rate_multiplier,
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
                ts["land_sno"],
                ts["land_eva"],
                ts["date"],
                temperature_time_series_val,
                precipitation_time_series_val,
                streamflow_time_series_val,
                land_sno_time_series_val,
                land_eva_time_series_val,
                date_time_series_val,
                ts["sno_s_ts"],
                ts["sub_s_ts"],
                ts["sno_f_ts"],
                ts["sub_f_ts"],
                ts["eva_f_ts"],
                sno_s_tsVal,
                sub_s_tsVal,
                sno_f_tsVal,
                sub_f_tsVal,
                eva_f_tsVal,
                sno_s_ts_areasVal,
                sub_s_ts_areasVal,
                sno_f_ts_areasVal,
                sub_f_ts_areasVal,
                eva_f_ts_areasVal,
                training_data,
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

print("\n")
print("end of script")
