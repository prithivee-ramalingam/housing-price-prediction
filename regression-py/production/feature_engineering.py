"""Processors for the feature engineering step of the worklow.

The step loads cleaned training data, processes the data for outliers,
missing values and any other cleaning steps based on business rules/intuition.

The trained pipeline and any artifacts are then saved to be used in
training/scoring pipelines.
"""

import logging
import os.path as op

import numpy as np
from category_encoders import TargetEncoder
from shared_logic import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from ta_lib.core.api import (
    DEFAULT_ARTIFACTS_PATH,
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataset,
    register_processor,
    save_pipeline,
)
from ta_lib.data_processing.api import Outlier

logger = logging.getLogger(__name__)
artifacts_folder = DEFAULT_ARTIFACTS_PATH


@register_processor("feature-engineering-job", "perform-feature-engineering")
def data_pipeline(context, params):

    # Read data as specified in data_catalog/local.yaml

    train_X = load_dataset(context, "processed/train/features")
    train_y = load_dataset(context, "processed/train/target")

    test_X = load_dataset(context, "processed/test/features")
    test_y = load_dataset(context, "processed/test/target")

    train_X_prepared = full_feature_pipeline.fit_transform(train_X)
    feature_names = get_feature_names_from_column_transformer(full_feature_pipeline)
    rooms_per_household_index = len(num_columns)
    population_per_household_index = rooms_per_household_index + 1

    # Insert new attribute names at appropriate positions
    feature_names.insert(rooms_per_household_index, "rooms_per_household")
    feature_names.insert(population_per_household_index, "population_per_household")
    # Create a DataFrame with the transformed data
    train_X = get_dataframe(train_X_prepared, feature_names)
    curated_columns = list(train_X.columns)
    # saving the list of relevant columns
    save_pipeline(
        curated_columns, op.abspath(op.join(artifacts_folder, "curated_columns.joblib"))
    )
    # save the feature pipeline
    save_pipeline(
        full_feature_pipeline, op.abspath(op.join(artifacts_folder, "features.joblib"))
    )
