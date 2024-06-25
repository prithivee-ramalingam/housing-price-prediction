import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from ta_lib.data_processing.api import Outlier


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    A custom transformer for adding combined attributes to a dataset.

    Parameters
    ----------
    add_bedrooms_per_room : bool, optional (default=True)
        Whether to add the 'bedrooms_per_room' attribute to the dataset.
    """

    def __init__(self, add_bedrooms_per_room=True):
        """
        Initializes the CombinedAttributesAdder with the option to add the 'bedrooms_per_room' attribute.

        Parameters
        ----------
        add_bedrooms_per_room : bool, optional (default=True)
            Whether to add the 'bedrooms_per_room' attribute to the dataset.
        """
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        """
        Fits the transformer to the data. This transformer does not learn from the data,
        so this method just returns itself.

        Parameters
        ----------
        X : array-like
            The input data to fit.
        y : ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self

    def transform(self, X):
        """
        Transforms the input data by adding combined attributes.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to transform.

        Returns
        -------
        array, shape (n_samples, n_features + 2) or (n_samples, n_features + 3)
            The transformed data with additional attributes.
        """
        total_rooms_ix, total_bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        rooms_per_household = X[:, total_rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, total_bedrooms_ix] / X[:, total_rooms_ix]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

    def inverse_transform(self, X):
        """
        Reverts the transformation by removing the combined attributes and restoring the original data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features + 2) or (n_samples, n_features + 3)
            The transformed data to revert.

        Returns
        -------
        array, shape (n_samples, n_features)
            The original data with combined attributes removed.
        """
        total_rooms_ix, total_bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        num_extra_attribs = 2 + (1 if self.add_bedrooms_per_room else 0)

        rooms_per_household_ix = X.shape[1] - num_extra_attribs
        population_per_household_ix = rooms_per_household_ix + 1

        total_rooms = X[:, rooms_per_household_ix] * X[:, households_ix]
        population = X[:, population_per_household_ix] * X[:, households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room_ix = population_per_household_ix + 1
            total_bedrooms = X[:, bedrooms_per_room_ix] * X[:, total_rooms_ix]
            original_data = np.c_[
                X[:, :-num_extra_attribs], total_rooms, total_bedrooms, population
            ]
        else:
            original_data = np.c_[X[:, :-num_extra_attribs], total_rooms, population]

        return original_data


num_columns = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder(add_bedrooms_per_room=False)),
    ]
)
cat_pipeline = Pipeline([("onehot", OneHotEncoder())])

full_feature_pipeline = ColumnTransformer(
    [("num", num_pipeline, num_columns), ("cat", cat_pipeline, ["ocean_proximity"])]
)
