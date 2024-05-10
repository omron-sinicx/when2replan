#! /bin/bash

# Run all methods for all scenarios

model_dir="../trained_model"

# Dijkstra x DWA at 16 pillar
model_name="DijkstraxDWA/16pillar.zip"
scenario_path="config/scenario/sixteen_pillar/random.yaml"
log_save_dir="run/DijkstraxDWA/16pillar"
seeds=[37,39,62,68]
python3 run.py\
        run.model_dir=$model_dir\
        run.model_name=$model_name\
        run.log_save_dir=$log_save_dir\
        run.navigation_scenarios=[$scenario_path]\
        run.view_animation=False\
        run.seeds=$seeds\
        navigation.common.global_planner="Dijkstra"\
        navigation.common.local_planner="DWA"

# Dijkstra x DWA at 9 pillar
model_name="DijkstraxDWA/9pillar.zip"
scenario_path="config/scenario/nine_pillar/random.yaml"
log_save_dir="run/DijkstraxDWA/9pillar"
seeds=[10,39,51,52,55]
python3 run.py\
        run.model_dir=$model_dir\
        run.model_name=$model_name\
        run.log_save_dir=$log_save_dir\
        run.navigation_scenarios=[$scenario_path]\
        run.view_animation=False\
        run.seeds=$seeds\
        navigation.common.global_planner="Dijkstra"\
        navigation.common.local_planner="DWA
        

# Dijkstra x DWA at 25 pillar
model_name="DijkstraxDWA/25pillar.zip"
scenario_path="config/scenario/twenty_five_pillar/random.yaml"
log_save_dir="run/DijkstraxDWA/25pillar"
seeds=[16,21,58]
python3 run.py\
        run.model_dir=$model_dir\
        run.model_name=$model_name\
        run.log_save_dir=$log_save_dir\
        run.navigation_scenarios=[$scenario_path]\
        run.view_animation=False\
        run.seeds=$seeds\
        navigation.common.global_planner="Dijkstra"\
        navigation.common.local_planner="DWA

# Dijkstra x MPC at 16 pillar
model_name="DijkstraxMPC/16pillar.zip"
scenario_path="config/scenario/sixteen_pillar/random.yaml"
log_save_dir="run/DijkstraxMPC/16pillar"
seeds=[14,41]
python3 run.py\
        run.model_dir=$model_dir\
        run.model_name=$model_name\
        run.log_save_dir=$log_save_dir\
        run.navigation_scenarios=[$scenario_path]\
        run.view_animation=False\
        run.seeds=$seeds\
        navigation.common.global_planner="Dijkstra"\
        navigation.common.local_planner="MPC"

# RRT x DWA at 16 pillar
model_name="RRTxDWA/16pillar.zip"
scenario_path="config/scenario/sixteen_pillar/random.yaml"
log_save_dir="run/RRTxDWA/16pillar"
seeds=[52,77]
python3 run.py\
        run.model_dir=$model_dir\
        run.model_name=$model_name\
        run.log_save_dir=$log_save_dir\
        run.navigation_scenarios=[$scenario_path]\
        run.view_animation=False\
        run.seeds=$seeds\
        navigation.common.global_planner="RRT-star"\
        navigation.common.local_planner="DWA"

# PRM x DWA at 16 pillar
model_name="PRMxDWA/16pillar.zip"
scenario_path="config/scenario/sixteen_pillar/random.yaml"
log_save_dir="run/PRMxDWA/16pillar"
seeds=[304,319]
python3 run.py\
        run.model_dir=$model_dir\
        run.model_name=$model_name\
        run.log_save_dir=$log_save_dir\
        run.navigation_scenarios=[$scenario_path]\
        run.view_animation=False\
        run.seeds=$seeds\
        navigation.common.global_planner="PRM"\
        navigation.common.local_planner="DWA"

# visualize all results
python3 pygame_visualizer.py run_all ../log/run