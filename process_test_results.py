import numpy as np
import pandas as pd
import os

from utils import TEST_RESULTS_FOLDER
TEST_FILES = [
    "local_grid_flattened_env_11x11_test_29-05_14-22.csv",
    "local_grid_flattened_env_11x11_test_29-05_14-24.csv",
    "local_grid_flattened_env_11x11_test_29-05_14-25.csv",
    "local_grid_flattened_env_11x11_test_29-05_14-27.csv",
    "local_grid_flattened_env_11x11_test_29-05_14-29.csv",
    "local_grid_flattened_env_11x11_test_29-05_14-31.csv",
    "local_grid_flattened_env_11x11_test_29-05_14-32.csv",
    "local_grid_flattened_env_11x11_test_29-05_14-34.csv",
    "local_grid_flattened_env_11x11_test_29-05_14-35.csv",
]


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

    print(f"Mean score: {mean_score:.4f}")
    print(f"Median score: {median}")
    print(f"Variance: {variance:.4f}")
    print(f"Standard Deviation: {std_deviation:.4f}")
    print(f"Min/Max score: {min_score} / {max_score}")
    print(f"25/50/75 percentile scores: {percentiles}")

    return data


def process_all_tests(test_files):
    sum_data = dict()
    sum_data["mean_score"] = 0
    sum_data["median"] = 0
    sum_data["variance"] = 0
    sum_data["std_deviation"] = 0
    sum_data["min_score"] = 0
    sum_data["max_score"] = 0
    sum_data["percentiles"] = [0, 0, 0]

    for test_result_file in test_files:
        test_data = calculate_run_stats(test_result_file)
        for key, value in test_data.items():
            if key == "percentiles":
                sum_data[key][0] += value[0]
                sum_data[key][1] += value[1]
                sum_data[key][2] += value[2]

            else:
                sum_data[key] += value

        print("---------------------------------------")

    average_mean_score = sum_data["mean_score"] / len(test_files)
    average_median = sum_data["median"] / len(test_files)
    average_variance = sum_data["mean_score"] / len(test_files)
    average_std_deviation = sum_data["std_deviation"] / len(test_files)
    average_min_score = sum_data["min_score"] / len(test_files)
    average_max_score = sum_data["max_score"] / len(test_files)
    average_percentiles = [perc / len(test_files) for perc in sum_data["percentiles"]]

    print(f"Average mean score: {average_mean_score:.4f}")
    print(f"Average median score: {average_median}")
    print(f"Average standard Deviation: {average_std_deviation:.4f}")
    print(f"Average Min/Max score: {average_min_score} / {average_max_score}")
    print(f"Average 25/50/75 percentile scores: {average_percentiles}")


if __name__ == '__main__':
    process_all_tests(TEST_FILES)
    # calculate_run_stats()


