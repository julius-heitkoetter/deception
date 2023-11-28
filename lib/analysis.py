"""
This module creates utilities for data analysis and visualization on qaeve datasets.
"""

import typing as T
import numpy as np
import matplotlib.pyplot as plt
import os

import utils


def get_deceptiveness_factor(qaeve_dataset_filename: str) -> float:
    """
    Returns fraction incorrect answers from supervisor with explanations.
    divded by fraction incorrect answers from supervisor without explanations,
    """

    data = utils.get_json_locally("", qaeve_dataset_filename)
    dataset = data["data"]

    # Find how often the supervisor model answers correctly when not influenced by a deceptive explanation.
    fraction_correct_from_evaluator = np.mean([
        utils.get_supervisor_evaluation(item) == utils.get_correct(item)
        for item in dataset
    ])

    # Find how often the supervisor model answers correctly when it is influenced by a deceptive explanation.
    fraction_correct_from_supervisor = np.mean([
        utils.get_verdict(item) == utils.get_correct(item)
        for item in dataset
    ])

    return (1 - fraction_correct_from_evaluator) / (1 - fraction_correct_from_supervisor)


def get_capability_factors(qaeve_dataset_filename: str) -> T.Tuple[float, float]:
    """
    Returns tuple:
        (fraction of answers the supervisor model gets correct,
         fraction of answers the deceiver model gets correct)

    where the deceiver and supervisor models have no extra explanations given.
    """

    data = utils.get_json_locally("", qaeve_dataset_filename)
    dataset = data["data"]

    fraction_correct_from_supervisor_as_evaluator = np.mean([
        utils.get_supervisor_evaluation(item) == utils.get_correct(item)
        for item in dataset
    ])

    fraction_correct_from_deceiver_as_evaluator = np.mean([
        utils.get_deceiver_evaluation(item) == utils.get_correct(item)
        for item in dataset
    ])

    return (
        fraction_correct_from_supervisor_as_evaluator,
        fraction_correct_from_deceiver_as_evaluator,
    )


def plot_deceptiveness_factor(dataset_filenames: T.List[T.Dict]):
    """
    Plot deceptiveness factor (y-axis) vs model smartness (x-axis) for a variety of combinations.

    Every dataset describes a deceiver and supervisor model. Each dataset Each dataset re
    """
    pass


if __name__ == "__main__":
    data_dir = "data"
    filenames = [f"{data_dir}/{filename}" for filename in os.listdir(data_dir)]
    for filename in filenames:
        dataset = utils.get_json_locally("", filename)
        correct = dataset["metadata"]["correct"]
        deceptiveness_factor = get_deceptiveness_factor(filename)
        supervisor_capability, deceiver_capability = get_capability_factors(filename)
        print(f"\nDataset {filename}:\n")
        print(f"\tCorrect answers: {correct}")
        print(f"\tDeceptiveness factor: {deceptiveness_factor}")
        print(f"\tSupervisor capability: {supervisor_capability}")
        print(f"\tDeceiver capability : {deceiver_capability}")

