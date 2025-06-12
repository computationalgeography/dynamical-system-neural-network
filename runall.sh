# training scenario, 1, 2, 3, or 4 (folds)
training_scenario="1"
# re-run scenario, 1, 2, 3, or 4 (reruns, can also be more than 4, eg 5)
re_run_scenario="3"
nohup python -u kals_model.py eva $training_scenario $re_run_scenario > nohup_eva_training_sc_$training_scenario.rerun_sc_$re_run_scenario.out &
nohup python -u kals_model.py sno $training_scenario $re_run_scenario > nohup_sno_training_sc_$training_scenario.rerun_sc_$re_run_scenario.out &
nohup python -u kals_model.py sub $training_scenario $re_run_scenario > nohup_sub_training_sc_$training_scenario.rerun_sc_$re_run_scenario.out &
nohup python -u kals_model.py sne $training_scenario $re_run_scenario > nohup_sne_training_sc_$training_scenario.rerun_sc_$re_run_scenario.out &
nohup python -u kals_model.py sue $training_scenario $re_run_scenario > nohup_sue_training_sc_$training_scenario.rerun_sc_$re_run_scenario.out &
nohup python -u kals_model.py sus $training_scenario $re_run_scenario > nohup_sus_training_sc_$training_scenario.rerun_sc_$re_run_scenario.out &
nohup python -u kals_model.py thr $training_scenario $re_run_scenario > nohup_thr_training_sc_$training_scenario.rerun_sc_$re_run_scenario.out &

#nohup python -u kals_model.py xhr $training_scenario $re_run_scenario > nohup_xhr_training_sc_$training_scenario.rerun_sc_$re_run_scenario.out &
#nohup python -u kals_model.py xva $training_scenario $re_run_scenario > nohup_xva_training_sc_$training_scenario.rerun_sc_$re_run_scenario.out &
#nohup python -u kals_model.py xno $training_scenario $re_run_scenario > nohup_xno_training_sc_$training_scenario.rerun_sc_$re_run_scenario.out &
#nohup python -u kals_model.py xub $training_scenario $re_run_scenario > nohup_xub_training_sc_$training_scenario.rerun_sc_$re_run_scenario.out &
#nohup python -u kals_model.py xne $training_scenario $re_run_scenario > nohup_xne_training_sc_$training_scenario.rerun_sc_$re_run_scenario.out &
#nohup python -u kals_model.py xue $training_scenario $re_run_scenario > nohup_xue_training_sc_$training_scenario.rerun_sc_$re_run_scenario.out &
#nohup python -u kals_model.py xus $training_scenario $re_run_scenario > nohup_xus_training_sc_$training_scenario.rerun_sc_$re_run_scenario.out &
