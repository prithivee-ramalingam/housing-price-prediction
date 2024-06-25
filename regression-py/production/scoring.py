"""Processors for the model scoring/evaluation step of the worklow."""

import os.path as op

from ta_lib.core.api import (
    DEFAULT_ARTIFACTS_PATH,
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    hash_object,
    load_dataset,
    load_pipeline,
    register_processor,
    save_dataset,
)
from ta_lib.regression.api import RegressionComparison, RegressionReport


@register_processor("model-evaluation-job", "model-scoring")
def score_model(context, params):
    input_features_ds = "processed/train/features"
    input_target_ds = "processed/train/target"
    # load training datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    input_features_ds = "processed/test/features"
    input_target_ds = "processed/test/target"
    output_ds = ["score/linear_reg", "score/decision_tree"]
    artifacts_folder = DEFAULT_ARTIFACTS_PATH
    # load test datasets
    test_X = load_dataset(context, input_features_ds)
    test_y = load_dataset(context, input_target_ds)

    # load the feature pipeline and training pipelines
    curated_columns = load_pipeline(op.join(artifacts_folder, "curated_columns.joblib"))
    features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))
    reg_model_pipeline = load_pipeline(
        op.join(artifacts_folder, "regression_pipeline.joblib")
    )
    decision_tree_model_pipeline = load_pipeline(
        op.join(artifacts_folder, "decisiontree_pipeline.joblib")
    )

    train_X_prepared = features_transformer.fit_transform(train_X)
    feature_names = get_feature_names_from_column_transformer(features_transformer)
    rooms_per_household_index = len(curated_columns)
    population_per_household_index = rooms_per_household_index + 1
    bedrooms_per_room_index = population_per_household_index + 1
    # Insert new attribute names at appropriate positions
    feature_names.insert(rooms_per_household_index, "rooms_per_household")
    feature_names.insert(population_per_household_index, "population_per_household")
    print(feature_names)
    # Create a DataFrame with the transformed data
    train_X = get_dataframe(train_X_prepared, feature_names)

    curated_columns = load_pipeline(op.join(artifacts_folder, "curated_columns.joblib"))
    features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))
    print(test_X.columns)
    test_X_prepared = features_transformer.transform(test_X)
    feature_names = get_feature_names_from_column_transformer(features_transformer)
    rooms_per_household_index = len(curated_columns)
    population_per_household_index = rooms_per_household_index + 1

    # Insert new attribute names at appropriate positions
    feature_names.insert(rooms_per_household_index, "rooms_per_household")
    feature_names.insert(population_per_household_index, "population_per_household")
    # Create a DataFrame with the transformed data
    test_X = get_dataframe(test_X_prepared, feature_names)

    # make a prediction
    test_X["yhat"] = reg_model_pipeline.predict(test_X)
    # store the predictions for any further processing.
    save_dataset(context, test_X, output_ds[0])
    test_X = test_X.drop(columns=["yhat"])
    reg_linear_report = RegressionReport(
        model=reg_model_pipeline,
        x_train=train_X,
        y_train=train_y,
        x_test=test_X,
        y_test=test_y,
        refit=True,
    )
    reg_linear_report.get_report(
        include_shap=False, file_path="reports/linear_regression"
    )
    test_X["yhat"] = decision_tree_model_pipeline.predict(test_X)
    # store the predictions for any further processing.
    save_dataset(context, test_X, output_ds[1])
    test_X = test_X.drop(columns=["yhat"])
    decision_tree_report = RegressionReport(
        model=decision_tree_model_pipeline,
        x_train=train_X,
        y_train=train_y,
        x_test=test_X,
        y_test=test_y,
        refit=True,
    )
    decision_tree_report.get_report(
        include_shap=False, file_path="reports/decision_tree"
    )
