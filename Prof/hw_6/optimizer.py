import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm

import optuna


def fill_nans(data: pd.DataFrame, column: str, time_delta: pd.Timedelta) -> pd.DataFrame:
    all_nans = data[data[column].isna()]
    for row in all_nans.index:
        result_value_0 = result_value_1 = 0
        new_index_0 = row - time_delta
        if new_index_0 in data.index:
            result_value_0 = 0 if np.isnan(data.loc[new_index_0, column]) else data.loc[new_index_0, column]

        new_index_1 = row + time_delta
        if new_index_1 in data.index:
            result_value_1 = 0 if np.isnan(data.loc[new_index_1, column]) else data.loc[new_index_1, column]
        result = sum([1 if i > 0 else 0 for i in [result_value_0, result_value_1]])
        result = (result_value_0 + result_value_1) / result
        data.loc[row, column] = result
    return data


def objective(trial: optuna.Trial):
    p = trial.suggest_int('p', 0, 2, step=1)
    d = trial.suggest_int('d', 0, 2, step=1)
    q = trial.suggest_int('q', 0, 2, step=1)
    P = trial.suggest_int('P', 0, 2, step=1)
    D = trial.suggest_int('D', 0, 2, step=1)
    Q = trial.suggest_int('Q', 0, 2, step=1)
    # s = trial.suggest_int('s', 0, 170, step=1)

    model = sm.tsa.statespace.SARIMAX(y,
                                      order=(p, d, q),
                                      seasonal_order=(P, D, Q, 168),
                                      error_action='ignore',).fit()
    aic = model.aic
    return aic


if __name__ == '__main__':
    data = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')
    data['date_time'] = pd.to_datetime(data['date_time'])
    data = data.drop(['rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'weather_description'], axis=1)
    data = data.drop_duplicates()
    border = pd.Timestamp(year=2018, month=8, day=6, hour=0)
    data = data[data['date_time'] > border]

    y = data['traffic_volume']
    y = pd.DataFrame(data['traffic_volume'], index=pd.to_datetime(np.array(data.index)))

    p = 0
    d = 2
    q = 2
    P = 1
    D = 2
    Q = 1

    model = sm.tsa.statespace.SARIMAX(y,
                                      order=(p, d, q),
                                      seasonal_order=(P, D, Q, 168),
                                      error_action='ignore', ).fit()
    joblib.dump(model, 'sarimax_0-2-2-1-2-1-168.joblib')
    # study = optuna.create_study(directions=["minimize"])
    # study.optimize(objective, n_trials=10, n_jobs=1, gc_after_trial=True)

    # print(study.best_trials)
    print("Model saved")