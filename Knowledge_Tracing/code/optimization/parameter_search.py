
import skopt

from script_step2 import train_evaluate

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

SEARCH_PARAMS = {'learning_rate': 0.4,
                 'max_depth': 15,
                 'num_leaves': 20,
                 'feature_fraction': 0.8,
                 'subsample': 0.2}


def train_evaluate(search_params, X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1234)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    params = {'objective': 'binary',
              'metric': 'binary_error',
              **search_params}
    model = lgb.train(params, train_data,
                      num_boost_round=300,
                      early_stopping_rounds=30,
                      valid_sets=[valid_data],
                      valid_names=['valid'])

    score = model.best_score['valid']['auc']
    return score




neptune.init('jakub-czakon/blog-hpo')
neptune.create_experiment('hpo-on-any-script', upload_source_files=['*.py'])

SPACE = [
    skopt.space.Real(0.01, 0.5, name='learning_rate', prior='log-uniform'),
    skopt.space.Integer(1, 30, name='max_depth'),
    skopt.space.Integer(2, 100, name='num_leaves'),
    skopt.space.Real(0.1, 1.0, name='feature_fraction', prior='uniform'),
    skopt.space.Real(0.1, 1.0, name='subsample', prior='uniform')]


@skopt.utils.use_named_args(SPACE)
def objective(**params):
    return -1.0 * train_evaluate(params)


monitor = sk_utils.NeptuneMonitor()
results = skopt.forest_minimize(objective, SPACE, n_calls=100, n_random_starts=10, callback=[monitor])
sk_utils.log_results(results)

neptune.stop()