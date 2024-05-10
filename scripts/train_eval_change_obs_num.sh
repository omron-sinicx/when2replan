#!/bin/bash

# Train and Evaluation for changing number of obstacles

model_dir="../trained_model"
process_num=20 # number of processes
num_eval_episodes=100
num_test_episodes=1

# Obs num = 0
model_name="DijkstraxDWA/16pillar/obs0"
scenario_path="config/scenario/sixteen_pillar/change_obs_num/obs0.yaml"
# Train
python3 train.py\
     train.model_dir=$model_dir\
     train.model_name=$model_name\
     train.navigation_scenarios=[$scenario_path]\
     train.seed=10001\
     train.num_test_episodes=$num_test_episodes\
     train.num_processes=$process_num\
     navigation.common.global_planner="Dijkstra"\
     navigation.common.local_planner="DWA"

# Eval
python3 eval.py\
     eval.model_dir=$model_dir\
     eval.model_name=$model_name\
     eval.navigation_scenarios=[$scenario_path]\
     eval.seed=0\
     eval.num_eval_episodes=$num_eval_episodes\
     eval.num_processes=$process_num\
     navigation.common.global_planner="Dijkstra"\
     navigation.common.local_planner="DWA"\

# Obs num = 5
model_name="DijkstraxDWA/16pillar/obs5"
scenario_path="config/scenario/sixteen_pillar/change_obs_num/obs5.yaml"
# Train
python3 train.py\
     train.model_dir=$model_dir\
     train.model_name=$model_name\
     train.navigation_scenarios=[$scenario_path]\
     train.seed=10001\
     train.num_test_episodes=$num_test_episodes\
     train.num_processes=$process_num\
     navigation.common.global_planner="Dijkstra"\
     navigation.common.local_planner="DWA"

# Eval
python3 eval.py\
     eval.model_dir=$model_dir\
     eval.model_name=$model_name\
     eval.navigation_scenarios=[$scenario_path]\
     eval.seed=0\
     eval.num_eval_episodes=$num_eval_episodes\
     eval.num_processes=$process_num\
     navigation.common.global_planner="Dijkstra"\
     navigation.common.local_planner="DWA"\

# Obs num = 15
model_name="DijkstraxDWA/16pillar/obs15"
scenario_path="config/scenario/sixteen_pillar/change_obs_num/obs15.yaml"
# Train
python3 train.py\
     train.model_dir=$model_dir\
     train.model_name=$model_name\
     train.navigation_scenarios=[$scenario_path]\
     train.seed=10001\
     train.num_test_episodes=$num_test_episodes\
     train.num_processes=$process_num\
     navigation.common.global_planner="Dijkstra"\
     navigation.common.local_planner="DWA"

# Eval
python3 eval.py\
     eval.model_dir=$model_dir\
     eval.model_name=$model_name\
     eval.navigation_scenarios=[$scenario_path]\
     eval.seed=0\
     eval.num_eval_episodes=$num_eval_episodes\
     eval.num_processes=$process_num\
     navigation.common.global_planner="Dijkstra"\
     navigation.common.local_planner="DWA"\