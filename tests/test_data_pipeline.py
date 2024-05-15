import pytest
from src import utilities


def test_non_existent_file():
    with pytest.raises(FileNotFoundError):
        _, _ = utilities.data_pipeline('NonExistentFile.csv', 'target', 0.8)

def test_non_existent_column():
    with pytest.raises(ValueError):
        _, _ = utilities.data_pipeline('data/SG.csv', 'random_column', 0.8)
