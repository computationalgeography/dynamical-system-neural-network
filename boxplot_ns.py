import pandas
import matplotlib.pyplot as plt

folder_one = "../figures/land_obs_one/"
folder_two = "../figures/land_obs_two/"

df_nen_one = pandas.read_csv(folder_one + "nen_models_ns.csv")
df_nen_two = pandas.read_csv(folder_two + "nen_models_ns.csv")
df_exp_one = pandas.read_csv(folder_one + "exp_models_ns.csv")
df_exp_two = pandas.read_csv(folder_two + "exp_models_ns.csv")
print(df_nen_one)

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

green = "#4daf4a"
blue = "#377eb8"

df_mod = df[df["variable_nen_one"] == "sub_f"]
print(df_mod)

fig,ax = plt.subplots(figsize=(9,3))

ax,props = df_mod.boxplot(column=[
                                   "E_nen_one", \
                                   "E_exp_one", \
                                   "E_nen_two", \
                                   "E_exp_two", \
                                   "S_nen_one", \
                                   "S_exp_one", \
                                   "S_nen_two", \
                                   "S_exp_two", \
                                   "G_nen_one", \
                                   "G_exp_one", \
                                   "G_nen_two", \
                                   "G_exp_two", \
                                   "ES_nen_one", \
                                   "ES_exp_one", \
                                   "ES_nen_two", \
                                   "ES_exp_two", \
                                   "EG_nen_one", \
                                   "EG_exp_one", \
                                   "EG_nen_two", \
                                   "EG_exp_two", \
                                   "SG_nen_one", \
                                   "SG_exp_one", \
                                   "SG_nen_two", \
                                   "SG_exp_two", \
                                   "ESG_nen_one", \
                                   "ESG_exp_one", \
                                   "ESG_nen_two", \
                                   "ESG_exp_two", \
                                   ],
                                   sym='.',
                                   patch_artist=True,
                                   grid = False,
                                   return_type = "both",
                                   medianprops=dict(color="black", lw=2),
                                   rot=0)

test = [green, green, blue, blue]
colors = test + test + test + test
colors = colors + colors
for patch,color in zip(props['boxes'],colors):
    patch.set_facecolor(color)

[ax.axvline(x, color = 'black', linestyle='-', linewidth='0.5') for x in \
                       [4.5, 8.5, 12.5, 16.5, 20.5, 24.5]]
a = ([*range(1,29)])
plt.xticks(a, 7 * ['n', 'e', 'n', 'e'])
x = 0.05
step = 0.14
fontje = 12
ax.text(x, 1.1, 'E', transform=ax.transAxes, fontsize=fontje, va='top')
ax.text(x + step, 1.1, 'S', transform=ax.transAxes, fontsize=fontje, va='top')
ax.text(x + 2*step, 1.1, 'G', transform=ax.transAxes, fontsize=fontje, va='top')
ax.text(x + 3*step, 1.1, 'ES', transform=ax.transAxes, fontsize=fontje, va='top')
ax.text(x + 4*step, 1.1, 'EG', transform=ax.transAxes, fontsize=fontje, va='top')
ax.text(x + 5*step, 1.1, 'SG', transform=ax.transAxes, fontsize=fontje, va='top')
ax.text(x + 6*step, 1.1, 'ESG', transform=ax.transAxes, fontsize=fontje, va='top')


plt.savefig('test.pdf')
