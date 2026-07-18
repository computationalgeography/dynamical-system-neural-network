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

# select gauge above 1400 m
g_a["high"] = g_a["elev"] > 1400 

# select catchment area gt 50 km
g_a["large"] = g_a["area_gov"] > 45 

# no or very limited urban impact
g_a["impact_u"] = g_a["degimpact"] == "u" 
g_a["impact_l"] = g_a["degimpact"] == "l" 
g_a["no_impact"] = g_a.impact_u | g_a.impact_l


# total selection
g_a["total_selection"] = g_a["austria"] & g_a["high"] & g_a["large"] & g_a["no_impact"]
g_a_selected = g_a.loc[g_a.total_selection == 1]


####
# basin attributes
#####

data_folder_basin = "../data/2_LamaH-CE_daily_klinger2021/A_basins_total_upstrm/1_attributes/"
b_a = pandas.read_csv(data_folder_basin + "Catchment_attributes.csv", sep = ";")

# no urban influence
b_a["no_urban"] = b_a["urban_fra"] < 0.001

b_a["total_selection"] = b_a["no_urban"]
b_a_selected = b_a.loc[b_a.total_selection == 1]
print(b_a_selected)

