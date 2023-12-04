"""
This module creates utilities for data analysis and visualization on qaeve datasets.
"""

import typing as T
import numpy as np
from uncertainties import ufloat
import matplotlib.pyplot as plt
import os

import utils

# The ufloat class tracks standard error!
# Example:
#     a = ufloat(1, 0.1); print(a.n, a.std_dev); a**2; print(a.n, a.std_dev)
# will print out:
#     1, 0.1
#     1, 0.2   (std err after squaring is 0.2 instead of 0.1)


def get_statistical_std_err(arr: T.List) -> float:
    """
    The standard error of a binomial distribution is sqrt(p*(1-p) / n).
    """

    # Remove None values and calculate fraction of True.
    arr = [a for a in arr if a is not None]
    p = len([a for a in arr if a is True]) / len(arr)

    return (p * (1-p) / len(arr)) ** 0.5


def get_systematic_std_err(arr: T.List) -> float:
    """
    The systematic error is the percentage that is None, assuming maximally adversarial
    data in the missing slots.
    """

    return len([a for a in arr if a is None]) / len(arr)


def get_ufloat_correct_fraction(dataset: T.List, correct_fn: T.Callable) -> T.Tuple[ufloat, ufloat]:
    """
    Given a dataset of questions to which we can apply utils.get_correct(item),
    evaluate the given correct_fn for how it performs compared to utils.get_correct.

    Returns a ufloat that tracks standard error, including for missing entries
    when correct_fn provides a value of None.
    """
    
    results = [correct_fn(item) for item in dataset]  # each entry is True, False, or None
    syst_std_err = get_statistical_std_err(results)
    stat_std_err = get_systematic_std_err(results)
    correct_fraction = np.mean([
        results[i] == utils.get_correct(item)
        for i, item in enumerate(dataset)
        if results[i] is not None  # throw out invalid values
    ])
    correct_fraction_syst = ufloat(correct_fraction, syst_std_err)
    correct_fraction_stat = ufloat(correct_fraction, stat_std_err)
    return (correct_fraction_syst, correct_fraction_stat)


def get_deceptiveness_factor(qaeve_dataset_filename: str) -> T.Tuple[ufloat]:
    """
    Returns fraction incorrect answers from supervisor with explanations.
    divded by fraction incorrect answers from supervisor without explanations,

    Return type is ufloat ("uncertain float") that carries a standard error.
    """

    data = utils.get_json_locally("", qaeve_dataset_filename)
    dataset = data["data"]

    # Find how often the supervisor model answers correctly when not influenced by a deceptive explanation.
    (
        evaluator_correct_fraction_syst_err,
        evaluator_correct_fraction_stat_err,
    ) = get_ufloat_correct_fraction(dataset, utils.get_supervisor_evaluation)

    # Find how often the supervisor model answers correctly when it is influenced by a deceptive explanation.
    (
        supervisor_correct_fraction_syst_err,
        supervisor_correct_fraction_stat_err,
    ) = get_ufloat_correct_fraction(dataset, utils.get_verdict)

    if supervisor_correct_fraction_syst_err.n == 1:
        return (None, None)

    return (
        (1 - evaluator_correct_fraction_syst_err) / (1 - supervisor_correct_fraction_syst_err),
        (1 - evaluator_correct_fraction_stat_err) / (1 - supervisor_correct_fraction_stat_err),
    )


def get_capability_factors(qaeve_dataset_filename: str) -> T.Tuple[T.Tuple[ufloat, ufloat], T.Tuple[ufloat, ufloat]]:
    """
    Returns tuple:
        (fraction of answers the supervisor model gets correct,
         fraction of answers the deceiver model gets correct)

    where the deceiver and supervisor models have no extra explanations given.

    Return type is ufloat ("uncertain float") that automatically carries standard error.
    """

    data = utils.get_json_locally("", qaeve_dataset_filename)
    dataset = data["data"]

    (
        supervisor_as_evaluator_correct_fraction_syst_err,
        supervisor_as_evaluator_correct_fraction_stat_err,
    ) = get_ufloat_correct_fraction(
        dataset, utils.get_supervisor_evaluation
    )

    (
        deceiver_as_evaluator_correct_fraction_syst_err,
        deceiver_as_evaluator_correct_fraction_stat_err,
    ) = get_ufloat_correct_fraction(
        dataset, utils.get_deceiver_evaluation
    )

    return (
        (supervisor_as_evaluator_correct_fraction_syst_err, 
        supervisor_as_evaluator_correct_fraction_stat_err),
        (deceiver_as_evaluator_correct_fraction_syst_err,
        deceiver_as_evaluator_correct_fraction_stat_err),
    )


def plot_deceptiveness_factor(dataset_filenames: T.List[T.Dict], deceiver_fixed: bool = False, supervisor_fixed: bool = False, plot_stat_err: bool = True):
    """
    Plot deceptiveness factor (y-axis) vs model smartness (x-axis) for a variety of combinations.

    Parameters:
        dataset_filenames: list of JSON files (qaeved)
        deceiver_fixed: True if there is only one deceiver model across all dataset_filenames
        supervisor_fixed: True if there is only one supervisor model across all dataset_filenames
        plot_stat_err: True if error bars correspond to statistial error, else systematic error
    """
    if deceiver_fixed == supervisor_fixed:
        raise ValueError("Exactly one of deceiver_fixed or supervisor_fixed should be True.")

    # check that deceiver is fixed (or supervisor) across the filenames
    deceivers, supervisors = set(), set()
    for filename in dataset_filenames:
        dataset = utils.get_json_locally("", filename)
        deceivers.add(dataset["metadata"]["deceiver_llm"])
        supervisors.add(dataset["metadata"]["supervisor_llm"])

    if deceiver_fixed and len(deceivers) > 1:
        raise ValueError("Found more than one deceiver even though deceiver_fixed is True.")
    elif supervisor_fixed and len(supervisors) > 1:
        raise ValueError("Found more than one supervisor even though supervisor_fixed is True.")

    fixed_model = list(deceivers)[0] if deceiver_fixed else list(supervisors)[0]

    # Get ufloat ("uncertain float") for each variable we wish to plot.
    # Note: get_capability_factors outputs (supervisor_capability, deceiver_capability). If the deceiver is fixed, we want supervisor capabilities; if the supervisor is fixed, we want deceiver capabilities.
    deceptiveness = [get_deceptiveness_factor(filename) for filename in filenames]
    capability = [get_capability_factors(filename) for filename in filenames]
    capability = [c[supervisor_fixed] for c in capability]

    # select out the systematic error version or the statistical error version
    deceptiveness = [d[plot_stat_err] for d in deceptiveness]
    capability = [c[plot_stat_err] for c in capability]

    # remove None values from undefined deceptiveness factors (when supervisor gets 100% correct)
    capability = [c for i, c in enumerate(capability) if deceptiveness[i] is not None]
    deceptiveness = [d for i, d in enumerate(deceptiveness) if deceptiveness[i] is not None]

    deceptiveness_base = [d.n for d in deceptiveness]
    deceptiveness_std_err = [d.std_dev for d in deceptiveness]
    capability_base = [c.n for c in capability]
    capability_std_err = [c.std_dev for c in capability]

    # Define the plot title
    title = (
        "Deceptiveness vs. Capability\n"
        f"(Fixed {'Deceiver' if deceiver_fixed else 'Supervisor'}, {fixed_model})"
    )

    # Construct the plot of deceptiveness vs. capability
    fig, ax = plt.subplots()
    ax.errorbar(
        capability_base, deceptiveness_base,
        xerr=capability_std_err, yerr=deceptiveness_std_err,
        fmt='o', ecolor='lightblue', elinewidth=3, capsize=0, color='darkblue'
    )

    # Set the titles and axis labels
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel(f"Capability of {'Deceiver' if supervisor_fixed else 'Supervisor'}", fontsize=12)
    ax.set_ylabel('Deceptiveness', fontsize=12)

    # Beautifying the plot
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_facecolor('whitesmoke')

    # LAKER: When saving to a file, record whether the error is SYSTEMATIC or STATISTICAL
    plt.show()

if __name__ == "__main__":
    data_dir = "data"
    filenames = [f"{data_dir}/{filename}" for filename in os.listdir(data_dir) if not os.path.isdir(f"{data_dir}/{filename}")]

    fixed_supervisor_dir = "data/fixed_supervisor_llama_7b"
    fixed_supervisor_filenames = [f"{data_dir}/{filename}" for filename in os.listdir(fixed_supervisor_dir)]

    # print some information about each dataset
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

    # create plots of deceptiveness by capability 
    plot_deceptiveness_factor(fixed_supervisor_filenames, supervisor_fixed=True)
