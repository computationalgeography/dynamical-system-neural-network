# training scenario, 1, 2, 3, or 4 (folds)
t_s='1'
# re-run scenario, 1, 2, 3, or 4 (reruns, can also greater than 4, eg 5, it always runs one)
re_run_s='1'
# catchment id (also directory name)
#directory='68'
directory=$1

batch_scenarios='thr xhr'

##############################
# observational data fitting #
##############################

#python kals_model.py thr 1 1 observations one 68 all_catch land_obs_one

# one area

observations='observations'
areas='one'

for b_s in $batch_scenarios
do
    nohup python -u kals_model.py $b_s $t_s $re_run_s $observations $areas $directory all_catch all_land_obs_one > nohup_bat_sc_$b_s.tr_sc_$t_s.rr_sc_$re_run_s.obs_$observations.area_$areas.folder_$directory.out
done


# two areas

observations='observations'
areas='two'

for b_s in $batch_scenarios
do
    nohup python -u kals_model.py $b_s $t_s $re_run_s $observations $areas $directory all_catch all_land_obs_two > nohup_bat_sc_$b_s.tr_sc_$t_s.rr_sc_$re_run_s.obs_$observations.area_$areas.folder_$directory.out
done

