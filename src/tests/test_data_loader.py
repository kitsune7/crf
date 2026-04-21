"""Tests for the CLI interface."""

from crf.data_loader import load_data_split, split_data, load_data


def test_load_datasplit():
    """Test the data loader."""
    data = load_data_split()
    assert len(data["train"]) > 0
    assert len(data["test"]) > 0
    assert len(data["validation"]) > 0

def test_split_data():
    """Test the data splitter."""
    data = split_data("['a' 'b' 'c']")
    assert data == ["a", "b", "c"]

def test_load_data():
    """Test the data loader."""
    data = load_data("data/lid_spaeng_train.csv")
    assert len(data) > 0
    assert "words" in data[0]
    assert "lid" in data[0]
    assert len(data[0]["words"]) > 0
    assert len(data[0]["lid"]) > 0