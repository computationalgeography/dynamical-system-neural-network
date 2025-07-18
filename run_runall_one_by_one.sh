training_scenarios='1 2'
rerun_scenarios='1 2'
for training_scenario in $training_scenarios
do
    for rerun_scenario in $rerun_scenarios
    do
	    nohup bash run_runall_one_by_one.sh $training_scenario $rerun_scenario &
    done
done
