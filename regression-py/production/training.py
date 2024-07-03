"""Processors for the model training step of the worklow."""

import logging
import os.path as op

from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from ta_lib.core.api import (
    DEFAULT_ARTIFACTS_PATH,
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataset,
    load_pipeline,
    register_processor,
    save_pipeline,
)
from ta_lib.regression.api import SKLStatsmodelOLS

logger = logging.getLogger(__name__)


@register_processor("model-training-job", "model-training")
def train_model(context, params):
    artifacts_folder = DEFAULT_ARTIFACTS_PATH
    input_features_ds = "processed/train/features"
    input_target_ds = "processed/train/target"
    # load training datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)
    # load pre-trained feature pipelines and other artifacts
    curated_columns = load_pipeline(op.join(artifacts_folder, "curated_columns.joblib"))
    # print(curated_columns)
    features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))
    train_X_prepared = features_transformer.fit_transform(train_X)
    feature_names = get_feature_names_from_column_transformer(features_transformer)
    rooms_per_household_index = len(curated_columns)
    population_per_household_index = rooms_per_household_index + 1

    # Insert new attribute names at appropriate positions
    feature_names.insert(rooms_per_household_index, "rooms_per_household")
    feature_names.insert(population_per_household_index, "population_per_household")
    # Create a DataFrame with the transformed data
    train_X = get_dataframe(train_X_prepared, feature_names)
    # create training pipeline
    reg_ppln_ols = Pipeline([("estimator", SKLStatsmodelOLS())])

    # fit the training pipeline
    reg_ppln_ols.fit(train_X, train_y.values.ravel())

    # save fitted training pipeline
    save_pipeline(
        reg_ppln_ols,
        op.abspath(op.join(artifacts_folder, "regression_pipeline.joblib")),
    )
    dt_training_pipe_init = Pipeline([("DecisionTree", DecisionTreeRegressor())])
    dt_training_pipe_init.fit(train_X, train_y.values.ravel())
    save_pipeline(
        dt_training_pipe_init,
        op.abspath(op.join(artifacts_folder, "decisiontree_pipeline.joblib")),
    )
    param_distribs_config = params["param_distribs"]
    # Convert param_distribs configuration to actual distribution objects
    param_distribs = {
        key: randint(low=value["randint"]["low"], high=value["randint"]["high"])
        for key, value in param_distribs_config.items()
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )

    rnd_search.fit(train_X, train_y.values.ravel())
    random_search_pipeline_final = Pipeline(
        [("RandomSearch", rnd_search.best_estimator_)]
    )
    random_search_pipeline_final.fit(train_X, train_y.values.ravel())
    save_pipeline(
        random_search_pipeline_final,
        op.abspath(
            op.join(artifacts_folder, "randomsearch_randomforest_pipeline.joblib")
        ),
    )
    param_grid = params["param_grid"]
    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(train_X, train_y.values.ravel())
    grid_search_pipeline_final = Pipeline([("GridSearch", grid_search.best_estimator_)])
    grid_search_pipeline_final.fit(train_X, train_y.values.ravel())
    save_pipeline(
        grid_search_pipeline_final,
        op.abspath(
            op.join(artifacts_folder, "gridsearch_randomforest_pipeline.joblib")
        ),
    )
