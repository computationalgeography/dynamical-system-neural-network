# number of runs at the same time is number of id's
#
ids='35 68 247 528 534 535 565 815 818'


for id in $ids
do
    python postprocessing.py obs_one $id
    python postprocessing.py obs_two $id
done
