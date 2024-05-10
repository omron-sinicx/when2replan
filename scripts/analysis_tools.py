# Plot and animation tools

import math
import os
import fire
from collections import defaultdict

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import markers, patches
import numpy as np
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from stable_baselines3.common.results_plotter import load_results, ts2xy


def load_csv(input_csv):
    return pd.read_csv(input_csv)


def plot(input_csv, x, y):
    """
    Plot a single metric
    x and y are the column names in the csv file
    """

    df = load_csv(input_csv)
    x_elem = df[x]
    y_elem = df[y]
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.plot(x_elem, y_elem)
    ax.set_xlabel(x, fontsize=18)
    ax.set_ylabel(y, fontsize=18)
    ax.grid(c="gainsboro", zorder=9)
    name = x + " - " + y
    plt.title(name, fontsize=18)
    plt.tight_layout()
    plt.show()


def plot_q_value(input_csv, normalize=True, show_plot=False):
    df = load_csv(input_csv)

    if "q_value_0" not in df.columns:
        raise ValueError("q_value column not found in csv file")

    time = df["time_step"]

    _reward = df["reward"]
    # get all q_value columns
    q_values = df.filter(regex="q_value")

    _q_val_not_replan = q_values["q_value_0"]
    _q_val_replan = q_values["q_value_1"]

    if normalize:
        q_val_not_replan = _q_val_not_replan / (_q_val_not_replan + _q_val_replan)
        q_val_replan = _q_val_replan / (_q_val_not_replan + _q_val_replan)
        if _reward.max() != 0:
            reward = _reward / _reward.max()
        else:
            reward = _reward
    else:
        q_val_not_replan = _q_val_not_replan
        q_val_replan = _q_val_replan
        reward = _reward

    # plot all q_value columns
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.plot(time, q_val_not_replan, label="q_value[NOT-REPLAN]")
    ax.plot(time, q_val_replan, label="q_value[REPLAN]")
    if not normalize:
        ax.plot(time, reward, c="r", label="reward")
        ax.set_xlabel("time_step", fontsize=18)
        ax.set_ylabel("q_value", fontsize=18)
    else:
        ax.set_xlabel("time_step", fontsize=18)
        ax.set_ylabel("q_value (normalized)", fontsize=18)
        # set y axis to [-1, 1]
        ax.set_ylim(0, 1)
    ax.grid(c="gainsboro", zorder=9)
    ax.legend()
    plt.tight_layout()

    if show_plot:
        plt.show()
    else:
        # save
        path = os.path.dirname(input_csv)
        # remove .csv
        path = os.path.splitext(path)[0]
        path = os.path.join(path, "_q_value.png")
        plt.savefig(path)


def summarize_results(
    input_folder="../log/eval/latest", output_csv="../log/eval/latest/summary.csv"
):
    """
    Get summary of results
    Args:
        input_folder: folder containing the results
        output_csv: output csv file
    """
    df = pd.DataFrame()
    for sub_dir in os.listdir(input_folder):
        # check if sub_dir is a directory
        if os.path.isdir(os.path.join(input_folder, sub_dir)):
            for file in os.listdir(input_folder + "/" + sub_dir):
                if file.endswith(".csv"):
                    name = sub_dir
                    data = load_csv(input_folder + "/" + sub_dir + "/" + file)
                    # add name as first column
                    data.insert(0, "name", name)
                    df = pd.concat([df, data])
            df.to_csv(output_csv)


def plot_metrics(input_csv="../log/eval/latest/summary.csv"):
    """
    Plot all metrics
    input_csv: csv file containing the result
    """
    df = load_csv(input_csv)
    df = df.groupby(["method"]).mean()
    # eliminate first column
    df = df.iloc[:, 1:]

    # plot all metrics each in a separate plot
    for metric in df.columns:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        # bar plot with
        ax.bar(df.index, df[metric], label=metric)
        ax.set_xlabel("method", fontsize=18)
        ax.set_ylabel(metric, fontsize=18)
        ax.set_title(metric, fontsize=18)
        y_max = df[metric].max()
        y_min = df[metric].min()
        ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
        ax.set_yticks(
            np.arange(y_min, y_max + 0.1 * (y_max - y_min), 0.1 * (y_max - y_min))
        )
    plt.show()


# TODO: WIP
def plot_replan_history(input_dir: str):
    """
    Plot action and update global planner history
    Args:
        input_dir: directory containing the results
    """
    if not os.path.isdir(input_dir):
        raise ValueError("Input directory does not exist")

    df = pd.DataFrame()
    # add time_step, action, and update_global_planner to the dataframe
    df["time_step"] = []
    df["action"] = []
    df["update_global_planner"] = []
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        data = load_csv(file_path)
        time_steps = data["time_step"]
        actions = data["action"]
        replans = data["update_global_planner"]
        # append to dataframe

    # sort by time step
    df.sort_values(by=["time_step"], inplace=True)
    print(df)
    # plot histogram
    # fig = plt.figure(figsize=(12, 8))


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def get_collision_seed(input_folder):
    """
    Get the seed of the collision
    Args:
        input_folder: folder containing the results
    """
    collision_list = []
    if os.path.isdir(input_folder):
        for file in os.listdir(input_folder):
            if file.endswith(".csv"):
                data = load_csv(input_folder + "/" + file)
                if any(data["collision"]) == 1:
                    seed = data["seed"][0]
                    collision_list.append(seed)
    print("Collision seeds: ", collision_list)


def get_interest_results(
    input_folder="../log/eval/latest", num_results=3, metric="SGT"
):
    """
    Get the interest seeds of results
    Args:
        input_folder: folder containing the results
        num_results: number of results to return
        metric: metric to use for sorting
    """
    dir_list = os.listdir(input_folder)
    if len(dir_list) <= 1:
        raise ValueError("Results is not enough")

    # check model in dir_list
    if not "rl_based_replan" in dir_list:
        raise ValueError("rl_based_replan is not in the folder")

    # Get seeds and metrics
    df_dict = defaultdict(pd.DataFrame)
    for sub_dir in dir_list:
        dir_path = os.path.join(input_folder, sub_dir)
        if os.path.isdir(dir_path):
            name = sub_dir
            df = pd.DataFrame()
            for file in os.listdir(dir_path):
                if file.endswith(".csv"):
                    data = load_csv(os.path.join(dir_path, file))
                    # get seed
                    seed = data["seed"][0]
                    # get metric
                    metric_val = data[metric].max()
                    # add seed and metric to df
                    df = pd.concat(
                        [df, pd.DataFrame({"seed": [seed], metric: [metric_val]})],
                        ignore_index=True,
                    )
                    # sort by seed
                    df = df.sort_values(by=["seed"])
            df_dict[name] = df

    # Compare and extract the interest seeds
    df_seed = pd.DataFrame()
    data_num = len(df_dict["rl_based_replan"])
    for i in range(data_num):
        # model metric
        seed = df_dict["rl_based_replan"]["seed"][i]
        model_val = df_dict["rl_based_replan"][metric][i]
        baseline_vals = []
        for key in df_dict.keys():
            if key != "rl_based_replan":
                if df_dict[key]["seed"][i] != seed:
                    raise ValueError("seed is not the same")
                baseline_vals.append(df_dict[key][metric][i])
        # if model metric is best, add seed and difference to df_seed
        if model_val >= max(baseline_vals) and model_val > 0:
            diff = model_val - max(baseline_vals)
            df_seed = pd.concat(
                [df_seed, pd.DataFrame({"seed": [seed], "diff": [diff]})],
                ignore_index=True,
            )
    # sort by diff
    df_seed = df_seed.sort_values(by=["diff"], ascending=False)

    # get the interest seeds and return
    print("Metrics: {}".format(metric))
    print("The interest seeds are:", df_seed["seed"].values[:num_results].tolist())
    print(
        "Difference between model and baseline:",
        df_seed["diff"].values[:num_results].tolist(),
    )


def get_tensorboard_data(input_tf_path: str, name: str):
    """
    tensobord data to pandas
    """
    event_acc = EventAccumulator(input_tf_path)
    event_acc.Reload()
    scalars = event_acc.Scalars(name)

    df = pd.DataFrame(scalars)

    return df


def get_values_from_tensorboard_log(name: str, log_dir: str, x_range: list):
    """
    get values from tensorboard log in dir
    """

    # get all tensoboard files
    dir_name_list = os.listdir(log_dir)

    values_list = []
    step_list = []
    for dir_name in dir_name_list:
        dir_path = os.path.join(log_dir, dir_name)
        file = os.listdir(dir_path)[0]
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            df = get_tensorboard_data(file_path, name)

            data = df["value"].values
            data_np = np.array(data)
            # moving average
            # data_np = moving_average(data_np, 30)
            values_list.append(data_np)

            step = df["step"].values
            step_np = np.array(step)
            # step_np = moving_average(step_np, 30)
            step_list.append(step_np)

    #  get the corresponding values for x_range
    plot_values_list = []
    for i, steps in enumerate(step_list):
        values = []
        for x in x_range:
            idx = np.abs(steps - x).argmin()
            values.append(values_list[i][idx])
        plot_values_list.append(values)

    return plot_values_list


def plot_learning_curves(*log_dirs: str):
    # plot y value
    y_val: str = "rollout/ep_rew_mean_from_learning_start"
    x_range: list = list(range(10000, 100000, 1))

    # matplotlib setting
    # change font
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = 15
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    # file parse
    dir_paths = list(log_dirs)
    # get folder name
    dir_names = []
    for dir_path in dir_paths:
        dir_names.append(os.path.basename(dir_path))

    # get all tensoboard files
    plot_values_for_each_log = []
    for dir_path in dir_paths:
        plot_values_list = get_values_from_tensorboard_log(y_val, dir_path, x_range)
        plot_values_for_each_log.append(plot_values_list)

    # calculate mean and std
    means = []
    stds = []
    for plot_values_list in plot_values_for_each_log:
        mean = np.mean(plot_values_list, axis=0)
        std = np.std(plot_values_list, axis=0)
        means.append(mean)
        stds.append(std)

    # plot
    for i, mean in enumerate(means):
        sns.lineplot(x=x_range, y=mean, label=dir_names[i])
        plt.fill_between(x_range, mean - stds[i], mean + stds[i], alpha=0.2)

    plt.xlabel("Time steps")
    plt.ylabel("Reward")
    # plt.ylim(0.4, 0.56)
    plt.grid()
    plt.legend()

    # save fig as pdf
    plt.savefig("learning_curve.pdf", bbox_inches="tight", pad_inches=0.5)

    plt.show()


def plot_heatmap(
    input_csv: str, x_label: str = "vx_max [m/s]", y_label: str = "v_theta_sample [-]"
):
    # value = 'ΔSGT'
    value = "ΔSR"

    # matplotlib setting
    # change font
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = 15
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    # read data
    df = pd.read_csv(input_csv)
    df = df.dropna()

    # plot heatmap
    df_pivot = df.pivot_table(index=y_label, columns=x_label, values=value)
    sns.heatmap(df_pivot, cmap="Oranges", annot=True, fmt=".2f", cbar=True)

    # set label
    plt.xlabel(x_label, fontsize=15, fontweight="bold")
    plt.ylabel(y_label, fontsize=15, fontweight="bold")
    plt.savefig("heatmap.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.show()


if __name__ == "__main__":
    fire.Fire()
