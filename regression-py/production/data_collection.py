"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""

import os
import tarfile

import numpy as np
import pandas as pd
from scripts import binned_selling_price
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit
from ta_lib.core.api import (
    create_context,
    custom_train_test_split,
    list_datasets,
    load_dataset,
    register_processor,
    save_dataset,
    string_cleaning,
)


@register_processor("data-collection", "downloading")
def fetch_housing_data(context, params):
    os.makedirs(params["housing_path"], exist_ok=True)
    tgz_path = os.path.join(params["housing_path"], "housing.tgz")
    urllib.request.urlretrieve(params["housing_url"], tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=params["housing_path"])
    housing_tgz.close()


@register_processor("data-collection", "creating_train_test")
def create_test_train_datasest(context, params):
    print(list_datasets(context))
    housing_df = load_dataset(context, "raw/housing")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    housing_df_train, housing_df_test = custom_train_test_split(
        housing_df, splitter, by=binned_selling_price
    )
    target_col = "median_house_value"

    train_X, train_y = (
        housing_df_train
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )
    save_dataset(context, train_X, "processed/train/features")
    save_dataset(context, train_y, "processed/train/target")

    test_X, test_y = (
        housing_df_test
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )
    save_dataset(context, test_X, "processed/test/features")
    save_dataset(context, test_y, "processed/test/target")
    print("Done with train test split")
