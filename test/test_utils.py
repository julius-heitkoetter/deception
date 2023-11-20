"""
Test the utility functions in utils.py.
"""

import lib.utils as utils

def test_filename_from_atoms():
    # Test standard case
    expected_filename = "mmlu_econometrics_qae_12345.json"
    filename = utils.filename_from_atoms("mmlu", "econometrics", "qae", "12345")
    assert filename == expected_filename

    # Test the timestamp has the right length (e.g., "2023-11-20-07-27-05-100667")
    timestamped_filename = utils.filename_from_atoms("mmlu", "econometrics", "qae")
    timestamp = timestamped_filename.split(".json")[0].split("_")[-1]
    assert len(timestamp) == 26


def test_atoms_from_filename():
    # Test error on non-JSON
    try:
        utils.atoms_from_filename("something.txt")
        assert False, "Failed to throw error on non-JSON file extension."
    except ValueError:
        pass

    # Test error on invalid number of atoms
    try:
        utils.atoms_from_filename("a_b_c.json")
        assert False, "Failed to throw error on invalid number of atoms."
    except ValueError:
        pass

    try:
        utils.atoms_from_filename("a_b_c_d_e.json")
        assert False, "Failed to throw error on invalid number of atoms."
    except ValueError:
        pass

    # Test standard case
    dataset, category, stage, timestamp = utils.atoms_from_filename("a_b_c_d.json")
    assert dataset == "a"
    assert category == "b"
    assert stage == "c"
    assert timestamp == "d"


def test_next_filename_in_chain():
    # Test error if stage not in chain
    try:
        utils.next_filename_in_chain("mmlu_econometrics_quack_12345.json")
        assert False, "Failed to throw error on incorrect stage 'quack'"
    except ValueError:
        pass

    # Test error if stage is last stage
    try:
        utils.next_filename_in_chain("mmlu_econometrics_qaeve_12345.json")
        assert False, "Failed to throw error on final stage 'quave'"
    except ValueError:
        pass

    # Try standard case
    filename = "mmlu_econometrics_qae_2023-11-20-07-27-05-100667.json"
    expected_next = "mmlu_econometrics_qaev_2023-11-20-07-27-05-100667.json"
    next_filename = utils.next_filename_in_chain(filename)
    assert next_filename == expected_next


if __name__ == "__main__":
    test_filename_from_atoms()
    test_atoms_from_filename()
    test_next_filename_in_chain()

