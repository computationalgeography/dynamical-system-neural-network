# number of runs at the same time is number of id's
#
#ids='35 68 247 528 534 535 565 815 818'
ids='35 68 247 528'


for id in $ids
do
    nohup bash run_all_catch.sh $id &
done
