import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import glob

from utils import TEST_RESULTS_FOLDER

TEST_FILES = [
    "PPO_screen_box_36x36_03-06_08-52_test.csv",
    "PPO_screen_box_36x36_31-05_23-54_test.csv",
]

PRINT_INDIVIDUAL_STATS = True


def calculate_run_stats(test_result_file: str):
    df = pd.read_csv(os.path.join(TEST_RESULTS_FOLDER, test_result_file))

    scores = df["score"].tolist()

    scores = np.array(scores)

    mean_score = np.mean(scores)
    median = np.median(scores)
    variance = np.var(scores)
    std_deviation = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    percentiles = np.percentile(scores, [25, 50, 75])

    data = dict()
    data["mean_score"] = mean_score
    data["median"] = median
    data["std_deviation"] = std_deviation
    data["min_score"] = min_score
    data["max_score"] = max_score
    data["percentiles"] = percentiles

    if PRINT_INDIVIDUAL_STATS:
        print(f"Mean score: {mean_score:.4f}")
        print(f"Median score: {median}")
        print(f"Variance: {variance:.4f}")
        print(f"Standard Deviation: {std_deviation:.4f}")
        print(f"Min/Max score: {min_score} / {max_score}")
        print(f"25/50/75 percentile scores: {percentiles}")
        print("------------------------------------------")

    return data


def process_all_tests(test_files):
    tests_data = dict()
    tests_data["mean_score"] = 0
    tests_data["median"] = 0
    tests_data["variance"] = 0
    tests_data["std_deviation"] = 0
    tests_data["min_score"] = []
    tests_data["max_score"] = []
    tests_data["percentiles"] = [0, 0, 0]

    for test_result_file in test_files:
        test_data = calculate_run_stats(test_result_file)
        for key, value in test_data.items():
            if key == "percentiles":
                tests_data[key][0] += float(value[0])
                tests_data[key][1] += float(value[1])
                tests_data[key][2] += float(value[2])

            elif key == "min_score" or key == "max_score":
                tests_data[key].append(value)

            else:
                tests_data[key] += value

    average_mean_score = tests_data["mean_score"] / len(test_files)
    average_median = tests_data["median"] / len(test_files)
    average_variance = tests_data["mean_score"] / len(test_files)
    average_std_deviation = tests_data["std_deviation"] / len(test_files)
    min_score = min(tests_data["min_score"])
    max_score = max(tests_data["max_score"])
    average_percentiles = [perc / len(test_files) for perc in tests_data["percentiles"]]

    print(f"Average mean score: {average_mean_score:.4f}")
    print(f"Average median score: {average_median}")
    print(f"Average standard Deviation: {average_std_deviation:.4f}")
    print(f"Min/Max score: {min_score} / {max_score}")
    print(f"Average 25/50/75 percentile scores: {average_percentiles}")


def graph_from_csv(folder_path):
    def format_duration(td):
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours} hours {minutes} minutes {seconds} seconds"

    def millions_formatter(x, pos):
        return f'{int(x / 1_000_000)}M'

    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    all_dfs = []

    for file in csv_files:
        algo_name = os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(file)
        df['Algorithm'] = algo_name
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    combined_df['Wall time'] = pd.to_datetime(combined_df['Wall time'], unit='s')

    plt.figure(figsize=(16, 9))

    for algo_name in combined_df['Algorithm'].unique():
        algo_df = combined_df[combined_df['Algorithm'] == algo_name].sort_values('Step')
        plt.plot(algo_df['Step'], algo_df['Value'], label=algo_name)

    plt.title('Porovnanie algoritmov na úlohe Zbierania minerálových úlomkov')
    plt.xlabel('Krok tréningu')
    plt.ylabel('Odmena za epizódu')
    plt.legend(title='Agenti')
    plt.grid(True)
    plt.xlim(0, 10_000_000)

    plt.gca().xaxis.set_major_formatter(FuncFormatter(millions_formatter))

    def format_duration(td):
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        if hours == 0:
            return f"{minutes} minút {seconds} sekúnd."

        return f"{hours} hodín {minutes} minút {seconds} sekúnd."

    duration_text = "Čas na dokončenie tréningu na 10M krokov:\n"
    for algo_name in combined_df['Algorithm'].unique():
        algo_df = combined_df[combined_df['Algorithm'] == algo_name].sort_values('Step')
        target_step = 10_000_000
        closest_row = algo_df.iloc[(algo_df['Step'] - target_step).abs().argmin()]
        duration = closest_row['Wall time'] - algo_df['Wall time'].min()
        duration_text += f"{algo_name}: {format_duration(duration)}\n"

    plt.tight_layout(rect=(0.0, 0.15, 1.0, 1.0))
    plt.figtext(0.5, 0.01, duration_text, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()


if __name__ == '__main__':
    # process_all_tests(TEST_FILES)
    # calculate_run_stats()
    graph_from_csv("csv_data_compare_alg")


