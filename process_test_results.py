import numpy as np
import pandas as pd
import os

from utils import TEST_RESULTS_FOLDER

TEST_FILES = [
    "screen_36x36_test_31-05_15-51.csv",
    "screen_36x36_test_31-05_15-53.csv",
    "screen_36x36_test_31-05_15-54.csv",
    "screen_36x36_test_31-05_15-56.csv",
    "screen_36x36_test_31-05_15-58.csv",
    "screen_36x36_test_31-05_15-59.csv",
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
                sum_data[key][0] += float(value[0])
                sum_data[key][1] += float(value[1])
                sum_data[key][2] += float(value[2])

            else:
                sum_data[key] += value

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


