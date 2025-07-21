training_scenarios='1'
rerun_scenarios='1'
# number of runs at the same time is number of training scenarios
# multiplied by number of rerun scenarios defined above
# all combinations of training and rerun scenario
# are then executed at the same time in the order
# defined in runall_one_by_one.sh


for training_scenario in $training_scenarios
do
    for rerun_scenario in $rerun_scenarios
    do
	    nohup bash runall_one_by_one.sh $training_scenario $rerun_scenario &
    done
done
