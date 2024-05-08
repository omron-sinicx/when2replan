from __future__ import annotations

from collections import defaultdict
from typing import Callable, Tuple, Union
import time
import gym
import os
import yaml
import datetime
import csv
from omegaconf import OmegaConf, DictConfig
import hydra
import mlflow
import pandas as pd
from matplotlib import pyplot as plt
from navigation_stack_py.gym_env import NavigationStackEnv
from navigation_stack_py.utils import DataLogger
from navigation_stack_py.rl_modules.dqn.dqn import DQN

from map_creator import generate_random_maps
from utils import HyperParameters, evaluate_policy


def init_log_dir(params: HyperParameters) -> defaultdict:
    # log dir
    current_log_dir = os.path.join(
        params.log_dir, time.strftime("%Y%m%d-%H%M%S", time.localtime())
    )
    os.makedirs(current_log_dir, exist_ok=True)

    # make symbolic link to log dir
    abs_log_root_dir = os.path.abspath(params.log_dir)
    symbolic_link = os.path.join(abs_log_root_dir, "latest")

    # delete symbolic link if it exists
    if os.path.exists(symbolic_link):
        os.remove(symbolic_link)
    abs_log_dir = os.path.abspath(current_log_dir)
    os.symlink(abs_log_dir, symbolic_link, target_is_directory=True)

    return current_log_dir


def metric(data_logger: DataLogger) -> dict:
    # calculate metric
    metric_dict = {}

    # collision or not
    collision = data_logger.max("collision")
    metric_dict["collision"] = collision

    if collision:
        seed = data_logger.max("seed")
        print("Collision occured at seed: {}".format(seed))

    # Sum collision time
    collision_time = data_logger.sum("collision_time")
    metric_dict["collision_time"] = collision_time

    # average speed
    average_speed = data_logger.mean("linear_speed")
    metric_dict["average_speed"] = average_speed

    # Sum travel distance
    travel_dist = data_logger.sum("travel_distance")
    metric_dict["travel_distance"] = travel_dist

    # Goal reached or not
    reach_goal = data_logger.max("reach_goal")
    metric_dict["reach_goal"] = reach_goal

    # Goal time, if not reached, goal time is -1
    goal_time = data_logger.max("goal_time")
    metric_dict["goal_time"] = goal_time

    # SGT: Success weighted normalized Goal Time: 0.0 ~ 1.0
    SGT = data_logger.max("SGT")
    metric_dict["SGT"] = SGT

    # SPL: Success weighted normalized Path Length: 0.0 ~ 1.0
    SPL = data_logger.max("SPL")
    metric_dict["SPL"] = SPL

    # No. of updating global planner
    num_global_planner_update = data_logger.sum("update_global_planner")
    metric_dict["update_global_planner"] = num_global_planner_update

    # average stuck time
    ave_stuck_time = data_logger.mean("stuck_time")
    metric_dict["average_stuck_time"] = ave_stuck_time

    # max stuck time
    max_stuck_time = data_logger.max("stuck_time")
    metric_dict["max_stuck_time"] = max_stuck_time

    # oscillation num
    oscillation_num = data_logger.sum("is_oscillation")
    metric_dict["oscillation_num"] = oscillation_num

    return metric_dict


def analyze_metric(log_list: list[dict]) -> dict:
    result_dict = {}

    # sum collision
    result_dict["collision"] = sum([log["collision"] for log in log_list])

    # Average collision time
    result_dict["collision_time"] = sum(
        [log["collision_time"] for log in log_list]
    ) / len(log_list)

    # average speed
    result_dict["average_speed"] = sum(
        [log["average_speed"] for log in log_list]
    ) / len(log_list)

    # Sum travel distance
    result_dict["travel_distance"] = max([log["travel_distance"] for log in log_list])

    # Sum reach goal
    result_dict["reach_goal"] = sum([log["reach_goal"] for log in log_list])

    # Average goal time
    # goalに到達していないものは除外している
    goal_time_list = [log["goal_time"] for log in log_list if log["goal_time"] != -1]
    if len(goal_time_list) > 0:
        result_dict["goal_time"] = sum(goal_time_list) / len(goal_time_list)
    else:
        result_dict["goal_time"] = -1  # not reached

    # Average SGT
    result_dict["SGT"] = sum([log["SGT"] for log in log_list]) / len(log_list)

    # Average SPL
    result_dict["SPL"] = sum([log["SPL"] for log in log_list]) / len(log_list)

    # Sum no. of updating global planner
    result_dict["update_global_planner"] = sum(
        [log["update_global_planner"] for log in log_list]
    )

    #  average stuck time
    result_dict["average_stuck_time"] = sum(
        [log["average_stuck_time"] for log in log_list]
    ) / len(log_list)

    # max stuck time
    result_dict["max_stuck_time"] = max([log["max_stuck_time"] for log in log_list])

    # sum oscillation num
    result_dict["oscillation_num"] = sum([log["oscillation_num"] for log in log_list])

    return result_dict


def print_result(name, result: dict, params: HyperParameters) -> None:
    print("====================")
    print("Result: {}".format(name))
    print(
        "No. of collision: {} / {}".format(
            result["collision"], params.num_eval_episodes
        )
    )
    print("Average collision time[s]: {}".format(result["collision_time"]))
    print("Average speed[m/s]: {}".format(result["average_speed"]))
    print("Sum travel distance[m]: {}".format(result["travel_distance"]))
    print(
        "No. of reach goal: {} / {}".format(
            result["reach_goal"], params.num_eval_episodes
        )
    )
    print("Average goal time[s]: {}".format(result["goal_time"]))
    print("Success weighted Goal Time: {}".format(result["SGT"]))
    print("Success weighted Path Length: {}".format(result["SPL"]))
    print("Average stuck time[s]: {}".format(result["average_stuck_time"]))
    print("Max stuck time[s]: {}".format(result["max_stuck_time"]))
    print("No. of oscillation: {}".format(result["oscillation_num"]))
    print("No. of updating global planner: {}".format(result["update_global_planner"]))
    print("====================")


def save_report(
    result_dict: defaultdict[dict], save_dir: str, params: HyperParameters
) -> None:
    # save meta data as yaml
    meta_data_dict: defaultdict = defaultdict(lambda: {})
    meta_data_dict["scenario"] = params.navigation_scenario_path_list
    meta_data_dict["model"] = params.rl_algorithm
    meta_data_dict["num_test"] = params.num_eval_episodes
    with open(os.path.join(save_dir, "meta_data.yaml"), "w") as f:
        yaml.dump(meta_data_dict, f)

    # summarize result as table
    report_path_csv = os.path.join(save_dir, "summary_report.csv")
    with open(report_path_csv, "w") as f:
        f = csv.writer(f)
        f.writerow(
            [
                "method",
                "No. of collision",
                "Average collision time[s]",
                "Average speed[m/s]",
                "Sum travel distance[m]",
                "No. of reach goal",
                "Average goal time[s]",
                "SGT[-]",
                "SPL[-]",
                "No. of updating global planner",
                "Average stuck time[s]",
                "Max stuck time[s]",
                "No. of oscillation",
            ]
        )

        for result_name, result in result_dict.items():
            f.writerow(
                [
                    result_name,
                    result["collision"],
                    result["collision_time"],
                    result["average_speed"],
                    result["travel_distance"],
                    result["reach_goal"],
                    result["goal_time"],
                    result["SGT"],
                    result["SPL"],
                    result["update_global_planner"],
                    result["average_stuck_time"],
                    result["max_stuck_time"],
                    result["oscillation_num"],
                ]
            )

    return report_path_csv


def eval(params: HyperParameters, current_log_dir: str):
    print("====================")
    print("Scenario: ", params.navigation_scenario_path_list)
    print("Start to evaluate random {} trials".format(params.num_eval_episodes))
    print("with {} processes".format(params.num_processes))
    print("====================")

    # create save dirs for evaluated methods
    log_dir_each_method = defaultdict(lambda: [])
    for eval_name in params.eval_method_list:
        dir = os.path.join(current_log_dir, eval_name)
        os.makedirs(dir, exist_ok=True)
        log_dir_each_method[eval_name] = dir

    log_dict = defaultdict(lambda: defaultdict(lambda: None))
    for method_name in params.eval_method_list:
        print("Evaluating: {}".format(method_name))
        if method_name == "rl_based_replan":
            method = DQN.load(params.saved_model_path)
        else:
            method = method_name

        log_list = evaluate_policy(
            method,
            params=params,
            n_eval_episodes=params.num_eval_episodes,
            num_workers=params.num_processes,
            rank=params.seed,
        )

        log_dict[method_name] = log_list

    if params.save_result:
        for method_name, log_list in log_dict.items():
            for i, log in enumerate(log_list):
                log_path = os.path.join(
                    log_dir_each_method[method_name], "{}.csv".format(i)
                )
                log.save(log_path)

    # convert data
    result_lists: defaultdict[list] = defaultdict(lambda: [])
    for method_name, log_list in log_dict.items():
        for log in log_list:
            result_lists[method_name].append(metric(log))

    # calculate metric for each method
    result_dict = defaultdict(lambda: defaultdict(lambda: None))
    for method_name, result_list in result_lists.items():
        result_dict[method_name] = analyze_metric(result_list)

    # Print result
    for result_name, result in result_dict.items():
        print_result(name=result_name, result=result, params=params)

    # Save report as csv
    report_csv = save_report(result_dict, current_log_dir, params)

    return result_dict, report_csv


def log_graph(result_dict: dict):
    """
    Log results as bar graph to mlflow
    """
    # dict to dataframe
    df = pd.DataFrame.from_dict(result_dict, orient="index")

    # bar graph
    for col in df.columns:
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0.3)
        ax.barh(df.index, df[col])
        plt.title(col)
        # save as pdf
        tmp = col + ".png"
        plt.savefig(tmp)
        plt.clf()
        plt.close()
        mlflow.log_artifact(tmp)
        os.remove(tmp)


@hydra.main(config_name="default.yaml", config_path="../config/eval", version_base=None)
def main(cfg: DictConfig):
    # mlflow setup
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")

    mlflow.set_experiment("eval")

    # params
    eval_config = OmegaConf.to_container(cfg, resolve=True)
    params = HyperParameters(config=eval_config, mode="eval")

    # generate random maps
    if params.is_generate_random_map:
        generate_random_maps(params.map_config, params.random_map_num, params.seed)

    # init log dir
    current_log_dir = init_log_dir(params=params)

    time_stamp = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_name = time_stamp
    for navigation_scenario_path in params.navigation_scenario_path_list:
        name = (
            navigation_scenario_path.split("/")[-2]
            + "_"
            + navigation_scenario_path.split("/")[-1].split(".")[0]
        )
        run_name += "_" + name
    with mlflow.start_run(run_name=run_name):
        # log parameters
        mlflow.log_param("log_dir", current_log_dir)
        mlflow.log_params(eval_config["eval"])
        mlflow.log_params(eval_config["navigation"])
        global_planner = eval_config["navigation"]["common"]["global_planner"]
        local_planner = eval_config["navigation"]["common"]["local_planner"]
        mlflow.log_param("global_planner", global_planner)
        mlflow.log_param("local_planner", local_planner)

        # Eval
        result_dict, report_csv = eval(params=params, current_log_dir=current_log_dir)

        # log result to mlflow
        mlflow.log_artifact(report_csv)
        log_graph(result_dict)


if __name__ == "__main__":
    main()
