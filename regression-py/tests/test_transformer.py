import numpy as np
import pytest
from production.shared_logic import CombinedAttributesAdder
from sklearn.base import BaseEstimator, TransformerMixin


@pytest.fixture
def sample_data():
    return np.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])


def test_fit(sample_data):
    adder = CombinedAttributesAdder()
    adder.fit(sample_data)
    assert adder is not None


def test_transform_with_bedrooms_per_room(sample_data):
    adder = CombinedAttributesAdder(add_bedrooms_per_room=True)
    transformed_data = adder.transform(sample_data)
    assert transformed_data.shape[1] == sample_data.shape[1] + 3
    assert (
        transformed_data[0, -1] == sample_data[0, 4] / sample_data[0, 3]
    )  # bedrooms_per_room
    assert (
        transformed_data[0, -2] == sample_data[0, 5] / sample_data[0, 6]
    )  # population_per_household
    assert (
        transformed_data[0, -3] == sample_data[0, 3] / sample_data[0, 6]
    )  # rooms_per_household


def test_transform_without_bedrooms_per_room(sample_data):
    adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    transformed_data = adder.transform(sample_data)
    assert transformed_data.shape[1] == sample_data.shape[1] + 2
