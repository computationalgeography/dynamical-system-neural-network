import numpy
import pandas

pandas.set_option('display.max_columns', None)  # or 1000
pandas.set_option('display.max_rows', None)  # or 1000
pandas.set_option('display.max_colwidth', None)  # or 199

# script to select catchments


#####
# gauge attributes
#####

data_folder_gauge = "../data/2_LamaH-CE_daily_klinger2021/D_gauges/1_attributes/"

# gauge attributes
g_a = pandas.read_csv(data_folder_gauge + "Gauge_attributes.csv", sep = ";")

# select austria 
g_a["austria"] = g_a["country"] == "AUT"

# select gauge above 1000 m
g_a["high"] = g_a["elev"] > 1000 

# select catchment area gt 45 km
g_a["large"] = g_a["area_gov"] > 45 

# no or very limited urban impact
g_a["impact_u"] = g_a["degimpact"] == "u" 
g_a["impact_l"] = g_a["degimpact"] == "l" 
g_a["no_impact"] = g_a.impact_u | g_a.impact_l

# proper time span
g_a["start_early_enough"] = g_a["obsbeg_day"] < 1982

# no gaps in hourly time series
g_a["no_gaps"] = g_a["gaps_post"] == 0

# total selection
#g_a["total_selection"] = g_a["austria"] & g_a["high"] & g_a["large"] & g_a["no_impact"] & \
#                         g_a["start_early_enough"] & g_a["no_gaps"]
g_a["total_selection"] = g_a["high"] & g_a["large"] & g_a["no_impact"] & \
                         g_a["start_early_enough"] & g_a["no_gaps"]
g_a_selected = g_a.loc[g_a.total_selection == 1]

####
# basin attributes
#####

data_folder_basin = "../data/2_LamaH-CE_daily_klinger2021/A_basins_total_upstrm/1_attributes/"
b_a = pandas.read_csv(data_folder_basin + "Catchment_attributes.csv", sep = ";")

# no urban influence
b_a["no_urban"] = b_a["urban_fra"] < 0.01

# enough snowfall
b_a["snow"] = b_a["frac_snow"] > 0.2

# no glaciers
b_a["no_glaciers"] = b_a["glac_fra"] < 0.15

b_a["total_selection"] = b_a["no_urban"] & b_a["snow"] & b_a["no_glaciers"]
b_a_selected = b_a.loc[b_a.total_selection == 1]

######
# join
#####

joined = pandas.merge(g_a_selected,b_a_selected, how = "inner", left_on="ID", right_on="ID")
print(joined)

print(joined[["ID", "country", "area_gov", "elev", "urban_fra","frac_snow", "glac_fra", \
              "name", "river", "region", "lon", "lat"]])
