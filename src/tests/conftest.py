import pytest
import os


@pytest.fixture(scope="session")
def dirname():
    return os.path.dirname(__file__)


@pytest.fixture(scope="session")
def phosphoproteomics_input_path(dirname):
    return os.path.join(dirname, "data/inputs/phosphoproteomics_sequence_input.csv")


@pytest.fixture(scope="session")
def phosphoproteomics_ser_thr_output_path(dirname):
    return os.path.join(dirname, "data/outputs/phosphoproteomics_ser_thr_output.csv")


@pytest.fixture(scope="session")
def phosphoproteomics_tyrosine_output_path(dirname):
    return os.path.join(dirname, "data/outputs/phosphoproteomics_tyrosine_output.csv")


@pytest.fixture(scope="session")
def diff_phos_ser_thr_input_path(dirname):
    return os.path.join(dirname, "data/inputs/diff_phos_ser_thr_input.csv")


@pytest.fixture(scope="session")
def diff_phos_ser_thr_output_path(dirname):
    return os.path.join(dirname, "data/outputs/diff_phos_ser_thr_output.csv")


@pytest.fixture(scope="session")
def diff_phos_tyrosine_input_path(dirname):
    return os.path.join(dirname, "data/inputs/diff_phos_tyrosine_input.csv")


@pytest.fixture(scope="session")
def diff_phos_tyrosine_output_path(dirname):
    return os.path.join(dirname, "data/outputs/diff_phos_tyrosine_output.csv")


@pytest.fixture(scope="session")
def enrichment_foreground_ser_thr_input_path(dirname):
    return os.path.join(dirname, "data/inputs/enrichment_foreground_ser_thr_input.csv")


@pytest.fixture(scope="session")
def enrichment_background_ser_thr_input_path(dirname):
    return os.path.join(dirname, "data/inputs/enrichment_background_ser_thr_input.csv")


@pytest.fixture(scope="session")
def enrichment_ser_thr_output_path(dirname):
    return os.path.join(dirname, "data/outputs/enrichment_ser_thr_output.csv")


@pytest.fixture(scope="session")
def enrichment_foreground_tyrosine_input_path(dirname):
    return os.path.join(dirname, "data/inputs/enrichment_foreground_tyrosine_input.csv")


@pytest.fixture(scope="session")
def enrichment_background_tyrosine_input_path(dirname):
    return os.path.join(dirname, "data/inputs/enrichment_background_tyrosine_input.csv")


@pytest.fixture(scope="session")
def enrichment_tyrosine_output_path(dirname):
    return os.path.join(dirname, "data/outputs/enrichment_tyrosine_output.csv")
