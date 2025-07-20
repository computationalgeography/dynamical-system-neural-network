# training scenario, 1, 2, 3, or 4 (folds)
t_s=$1
# re-run scenario, 1, 2, 3, or 4 (reruns, can also be more than 4, eg 5)
re_run_s=$2

#batch_scenarios='eva sno sub sne sue sus thr xhr'

# SURF -> two areas
#batch_scenarios='eva'
#batch_scenarios='sno'
#batch_scenarios='sub'
#batch_scenarios='sne'
#batch_scenarios='sue'
#batch_scenarios='sus'
#batch_scenarios='thr'
#batch_scenarios='xhr'

# VELOCITY -> one area
#batch_scenarios='eva sno'
#batch_scenarios='sub sne'
#batch_scenarios='sue sus'
#batch_scenarios='thr'
#batch_scenarios='xhr'

##############################
# observational data fitting #
##############################


# one area

observations='observations'
areas='one'
directory='land_obs_one'

for b_s in $batch_scenarios
do
    nohup python -u kals_model.py $b_s $t_s $re_run_s $observations $areas $directory > nohup_bat_sc_$b_s.tr_sc_$t_s.rr_sc_$re_run_s.obs_$observations.area_$areas.folder_$directory.out 
done


# two areas

observations='observations'
areas='two'
directory='land_obs_two'

for b_s in $batch_scenarios
do
    nohup python -u kals_model.py $b_s $t_s $re_run_s $observations $areas $directory > nohup_bat_sc_$b_s.tr_sc_$t_s.rr_sc_$re_run_s.obs_$observations.area_$areas.folder_$directory.out 
done


##############################
# artificial data fitting #
##############################


# one area

observations='arti'
areas='one'
directory='land_art_one'

for b_s in $batch_scenarios
do
    nohup python -u kals_model.py $b_s $t_s $re_run_s $observations $areas $directory > nohup_bat_sc_$b_s.tr_sc_$t_s.rr_sc_$re_run_s.obs_$observations.area_$areas.folder_$directory.out 
done


# two areas

observations='arti'
areas='two'
directory='land_art_two'

for b_s in $batch_scenarios
do
    nohup python -u kals_model.py $b_s $t_s $re_run_s $observations $areas $directory > nohup_bat_sc_$b_s.tr_sc_$t_s.rr_sc_$re_run_s.obs_$observations.area_$areas.folder_$directory.out 
done


#nohup python -u kals_model.py eva $t_s $re_run_s observations > nohup_eva_tr_sc_$t_s.rr_sc_$re_run_s.out 
#nohup python -u kals_model.py sno $t_s $re_run_s observations > nohup_sno_tr_sc_$t_s.rr_sc_$re_run_s.out 
#nohup python -u kals_model.py sub $t_s $re_run_s observations > nohup_sub_tr_sc_$t_s.rr_sc_$re_run_s.out 
#nohup python -u kals_model.py sne $t_s $re_run_s observations > nohup_sne_tr_sc_$t_s.rr_sc_$re_run_s.out 
#nohup python -u kals_model.py sue $t_s $re_run_s observations > nohup_sue_tr_sc_$t_s.rr_sc_$re_run_s.out 
#nohup python -u kals_model.py sus $t_s $re_run_s observations > nohup_sus_tr_sc_$t_s.rr_sc_$re_run_s.out 
#nohup python -u kals_model.py thr $t_s $re_run_s observations > nohup_thr_tr_sc_$t_s.rr_sc_$re_run_s.out 
#nohup python -u kals_model.py xhr $t_s $re_run_s observations > nohup_xhr_tr_sc_$t_s.rr_sc_$re_run_s.out 

#nohup python -u kals_model.py xva $t_s $re_run_s > nohup_xva_tr_sc_$t_s.rr_sc_$re_run_s.out &
#nohup python -u kals_model.py xno $t_s $re_run_s > nohup_xno_tr_sc_$t_s.rr_sc_$re_run_s.out &
#nohup python -u kals_model.py xub $t_s $re_run_s > nohup_xub_tr_sc_$t_s.rr_sc_$re_run_s.out &
#nohup python -u kals_model.py xne $t_s $re_run_s > nohup_xne_tr_sc_$t_s.rr_sc_$re_run_s.out &
#nohup python -u kals_model.py xue $t_s $re_run_s > nohup_xue_tr_sc_$t_s.rr_sc_$re_run_s.out &
#nohup python -u kals_model.py xus $t_s $re_run_s > nohup_xus_tr_sc_$t_s.rr_sc_$re_run_s.out &
