""" This file performs some analysis of the continuous data. """
import itertools
import math
import warnings
from collections import Counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from src.data.experiment_data_reader import ExperimentDataReader

EMOTIONS = [
    "anger",
    "surprise",
    "disgust",
    "joy",
    "fear",
    "sadness",
    "neutral",
]
MODALITIES = ["faceapi", "plant", "watch", "image"]


def downsample_array(sample: np.ndarray) -> np.ndarray:
    """
    Gets a sample with shape (window_size * 10000,) and then preprocesses it
    before using it in the classifier.

    :param sample: The data sample to preprocess.
    :return: The preprocessed sample.
    """
    downsampling_factor = 10
    pad_size = downsampling_factor - sample.shape[0] % downsampling_factor
    pad_size = 0 if pad_size == downsampling_factor else pad_size
    dimension1 = 1 if len(sample.shape) == 1 else sample.shape[1]
    padded_sample = np.append(
        sample, np.zeros((pad_size, dimension1)) * np.NaN
    )
    downsampled_sample = np.nanmean(
        padded_sample.reshape((-1, downsampling_factor, dimension1)), axis=1
    )
    return np.squeeze(downsampled_sample)


def correlation(
    data1: np.ndarray,
    data2: np.ndarray,
    corr_type: str,
    differentiate: bool = True,
    downsample: bool = True,
) -> tuple[float, float]:
    """
    Compute correlation values between array1 and array2

    :param data1: First array
    :param data2: Second array
    :param corr_type: Correlation function, pearson or spearman
    :param differentiate: Differentiate both arrays, boolean flag
    :param downsample: Whether to downsample or not
    :return: Tuple with (coefficient, p value)
    """
    if downsample:
        data1 = downsample_array(data1)
        data2 = downsample_array(data2)
    if differentiate:
        data1 = data1[1:] - data1[:-1]
        data2 = data2[1:] - data2[:-1]
    if corr_type == "pearson":
        coeff, p_value = scipy.stats.pearsonr(data1, data2)
    elif corr_type == "spearman":
        coeff, p_value = scipy.stats.spearmanr(data1, data2)
    else:
        raise ValueError("Correlation does not exist.")
    return coeff, p_value


def cramers_v_correlation(
    df: pd.DataFrame, modality1: str, modality2: str, downsample: bool = False
) -> float:
    """
    Cramer's V correlation metric between the classification results of the
    two given modalities.

    :param df: The dataframe with all emotion probabilities
    :param modality1: The first modality
    :param modality2: The second modality
    :param downsample: Whether to downsample or not
    :return: Cramer's V value of the classification results
    """
    data1 = np.zeros((613, 7))
    data2 = np.zeros((613, 7))
    for index, emotion in enumerate(EMOTIONS):
        data1[:, index] = df[f"{modality1}_{emotion}"].values
        data2[:, index] = df[f"{modality2}_{emotion}"].values
    if downsample:
        data1 = downsample_array(data1)
        data2 = downsample_array(data2)
    labels1 = np.argmax(data1, axis=1)
    labels2 = np.argmax(data2, axis=1)
    data = pd.crosstab(labels1, labels2)
    chi2 = scipy.stats.chi2_contingency(data)[0]
    n = data.sum().sum()
    r, c = data.shape
    cramers_v = np.sqrt(chi2 / (n * min(r - 1, c - 1)))
    return cramers_v


def theils_u_correlation(
    df: pd.DataFrame, modality1: str, modality2: str, downsample: bool = False
) -> float:
    """
    Computes the Theil's U, a score for directional conditional cross-entropy
    between the classification results of the two given modalities.

    :param df: The dataframe with all emotion probabilities
    :param modality1: The first modality
    :param modality2: The second modality
    :param downsample: Whether to downsample or not
    :return: Theil's U conditional cross entropy score
    """
    data1 = np.zeros((613, 7))
    data2 = np.zeros((613, 7))
    for index, emotion in enumerate(EMOTIONS):
        data1[:, index] = df[f"{modality1}_{emotion}"].values
        data2[:, index] = df[f"{modality2}_{emotion}"].values
    if downsample:
        data1 = downsample_array(data1)
        data2 = downsample_array(data2)
    labels1 = np.argmax(data1, axis=1)
    labels2 = np.argmax(data2, axis=1)

    y_counter = Counter(labels2)
    xy_counter = Counter(list(zip(labels1, labels2)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy)

    x_counter = Counter(labels1)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = scipy.stats.entropy(p_x)
    theils_u = 1 if s_x == 0 else (s_x - entropy) / s_x
    return theils_u


def analyze_single_experiment(
    experiment_index: int, plots: bool = True
) -> dict[str, Any]:
    """
    This function computes all scores for one experiment and returns a
    dictionary with these scores.

    :param experiment_index: The experiment index to look at
    :param plots: Whether to show plots or not, True or False
    :return: Dictionary of scores.
    """
    data_path = f"data/continuous/{experiment_index:03d}_emotions.csv"
    df = pd.read_csv(data_path, index_col=0)

    scores = {}

    means = df.mean(axis=0)  # Means for every column
    stds = df.std(axis=0)  # Std for every column
    for mean, std, col in zip(means, stds, df.columns):
        scores[f"mean_{col}"] = mean
        scores[f"stdd_{col}"] = std

    # Plot probabilities for all emotions
    if plots:
        fig, axes = plt.subplots(4, 2, figsize=(10, 10))
        plt.suptitle(f"Experiment Index: {experiment_index}")
        for em_id, emotion in enumerate(EMOTIONS):
            row = em_id // 2
            col = em_id % 2
            ax = axes[row, col]
            for modality in MODALITIES:
                ax.plot(df[f"{modality}_{emotion}"], label=modality)
            ax.set_xlabel(emotion)
            ax.legend()
        plt.tight_layout()
        plt.show()

    # Correlations: Pearson, Spearman, Cramer's V and Theil's U
    #   Also use increment array to make the series independent
    downsampling = False
    for items in itertools.combinations(MODALITIES, r=2):
        for emotion in EMOTIONS:
            for corr in ["pearson", "spearman"]:
                scores[
                    f"{corr}_{emotion}_{items[0]}_{items[1]}"
                ] = correlation(
                    df[f"{items[0]}_{emotion}"].values,
                    df[f"{items[1]}_{emotion}"].values,
                    corr_type=corr,
                    differentiate=True,
                    downsample=downsampling,
                )
        scores[f"cramerv_{items[0]}_{items[1]}"] = cramers_v_correlation(
            df, items[0], items[1], downsample=downsampling
        )
        scores[f"theilsu_{items[0]}_{items[1]}"] = theils_u_correlation(
            df, items[0], items[1], downsample=downsampling
        )
        scores[f"theilsu_{items[1]}_{items[0]}"] = theils_u_correlation(
            df, items[1], items[0], downsample=downsampling
        )

    return scores


def main():
    """
    Main analysis function that runs analysis for all experiments.
    """
    warnings.filterwarnings("ignore")

    used_indices = ExperimentDataReader.get_complete_data_indices()[:]
    all_scores = {}
    for experiment_index in used_indices:
        exp_scores = analyze_single_experiment(experiment_index, plots=False)
        all_scores[experiment_index] = exp_scores

    scores_df = pd.DataFrame.from_dict(all_scores.values())
    scores_df.index = used_indices
    assert len(scores_df.columns) == 4 * 7 * 2 + sum(range(4)) * (7 * 2 + 3)

    print("--------\nMean Correlations\n--------")
    for column in scores_df.columns:
        data = scores_df[column].values
        if isinstance(data[0], tuple):
            c, p = (np.nanmean(el) for el in list(zip(*list(data))))
            if p < 0.05:
                print(f"{column} - coeff {c}, p value {p}")
        else:
            if np.nanmean(data) > 0.3:
                print(f"{column} - mean {np.nanmean(data)}")


if __name__ == "__main__":
    main()
