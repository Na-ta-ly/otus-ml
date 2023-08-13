from typing import List

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

import optuna


def objective(trial: optuna.Trial):
    loss = trial.suggest_categorical('loss', ['squared_error', 'absolute_error'])
    n_estimators = trial.suggest_int('n_estimators', 250, 500, step=50)
    max_features = trial.suggest_int('max_features', 5, 80)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    learning_rate = trial.suggest_float('learning_rate', 0.05, 0.3)
    min_samples_split = trial.suggest_float('min_samples_split', 0.01, 0.3)
    max_depth = trial.suggest_int('max_depth', 3, 80)

    model = GradientBoostingRegressor(random_state=42, loss=loss, n_estimators=n_estimators, max_features=max_features,
                                      subsample=subsample, min_samples_split=min_samples_split,
                                      learning_rate=learning_rate, max_depth=max_depth)
    model.fit(X_train, y_train)

    boosting_prediction = recursive_predict(model, X_test, 5)

    mape = mean_absolute_percentage_error(np.array(y_test), np.array(boosting_prediction)) * 100
    mae = mean_absolute_error(np.array(y_test), np.array(boosting_prediction))

    return mape, mae


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


def timeseries_train_test_split(y: pd.Series, X: pd.DataFrame = None, test_size: int = 168) -> (pd.DataFrame,
                                                                                                pd.DataFrame,
                                                                                                pd.Series,
                                                                                                pd.Series):
    test_index = len(y) - test_size
    if X is None:
        y_train = y.iloc[:test_index]
        y_test = y.iloc[test_index:]
        return y_train, y_test
    else:
        X_train = X.iloc[:test_index]
        y_train = y.iloc[:test_index]
        X_test = X.iloc[test_index:]
        y_test = y.iloc[test_index:]
    return X_train, X_test, y_train, y_test


def recursive_predict(model: object, X_test: pd.DataFrame, first_lag_column: int) -> pd.DataFrame:
    horizon = len(X_test)

    temp_data = X_test.copy()
    temp_data.iloc[1:, first_lag_column:] = np.nan

    predicition = []

    for i in range(horizon):
        small_data = pd.DataFrame(temp_data.iloc[i, :]).T
        next_x = model.predict(small_data)[0]
        if i < horizon - 1:
            temp_data.iloc[i + 1, first_lag_column] = next_x
            temp_data.iloc[i + 1, first_lag_column + 1:] = temp_data.iloc[i, first_lag_column:-1].values
        predicition.append(next_x)

    return pd.DataFrame(predicition, index=X_test.index, columns=['traffic_volume'])


def create_lags(df: pd.DataFrame, series_columns: List[str], end_lag: int) -> pd.DataFrame:
    data = df.copy()
    for c in series_columns:
        for i in range(1, end_lag):
            data["lag_{}_{}".format(c, i)] = data[c].shift(i)
    return data


if __name__ == '__main__':
    data = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')
    data['date_time'] = pd.to_datetime(data['date_time'])
    data = data.drop(['rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'weather_description'], axis=1)
    data = data.drop_duplicates()
    border = pd.Timestamp(year=2015, month=6, day=25, hour=5)
    data = data[data['date_time'] > border]

    delta_border = pd.Timedelta(hours=1)
    index_data = pd.DataFrame(pd.date_range(start=data['date_time'].min(), end=data['date_time'].max(),
                                            freq=delta_border), columns=['date_time'])
    index_data = index_data.set_index('date_time')
    data = data.set_index('date_time')
    full_data = data.join(index_data, how='outer')

    full_data = fill_nans(full_data, 'traffic_volume', pd.Timedelta(days=7))

    holiday_dict = {pd.Timestamp(year=2015, month=10, day=12): 'Columbus Day',
                    pd.Timestamp(year=2016, month=10, day=10): 'Columbus Day',
                    pd.Timestamp(year=2017, month=10, day=9): 'Columbus Day',
                    pd.Timestamp(year=2015, month=11, day=11): 'Veterans Day',
                    pd.Timestamp(year=2016, month=11, day=11): 'Veterans Day',
                    pd.Timestamp(year=2017, month=11, day=10): 'Veterans Day',
                    pd.Timestamp(year=2015, month=11, day=26): 'Thanksgiving Day',
                    pd.Timestamp(year=2016, month=11, day=24): 'Thanksgiving Day',
                    pd.Timestamp(year=2017, month=11, day=23): 'Thanksgiving Day',
                    pd.Timestamp(year=2015, month=12, day=25): 'Christmas Day',
                    pd.Timestamp(year=2016, month=12, day=26): 'Christmas Day',
                    pd.Timestamp(year=2017, month=12, day=25): 'Christmas Day',
                    pd.Timestamp(year=2016, month=1, day=1): 'New Years Day',
                    pd.Timestamp(year=2017, month=1, day=2): 'New Years Day',
                    pd.Timestamp(year=2018, month=1, day=1): 'New Years Day',
                    pd.Timestamp(year=2016, month=2, day=15): 'Washingtons Birthday',
                    pd.Timestamp(year=2017, month=2, day=20): 'Washingtons Birthday',
                    pd.Timestamp(year=2018, month=2, day=19): 'Washingtons Birthday',
                    pd.Timestamp(year=2015, month=5, day=25): 'Memorial Day',
                    pd.Timestamp(year=2016, month=5, day=30): 'Memorial Day',
                    pd.Timestamp(year=2017, month=5, day=29): 'Memorial Day',
                    pd.Timestamp(year=2018, month=5, day=28): 'Memorial Day',
                    pd.Timestamp(year=2015, month=7, day=3): 'Independence Day',
                    pd.Timestamp(year=2016, month=7, day=4): 'Independence Day',
                    pd.Timestamp(year=2017, month=7, day=4): 'Independence Day',
                    pd.Timestamp(year=2018, month=7, day=4): 'Independence Day',
                    pd.Timestamp(year=2015, month=8, day=27): 'State Fair',
                    pd.Timestamp(year=2016, month=8, day=25): 'State Fair',
                    pd.Timestamp(year=2017, month=8, day=24): 'State Fair',
                    pd.Timestamp(year=2018, month=8, day=23): 'State Fair',
                    pd.Timestamp(year=2015, month=9, day=7): 'Labor Day',
                    pd.Timestamp(year=2016, month=9, day=5): 'Labor Day',
                    pd.Timestamp(year=2017, month=9, day=4): 'Labor Day',
                    pd.Timestamp(year=2018, month=9, day=3): 'Labor Day',
                    pd.Timestamp(year=2017, month=1, day=16): 'Martin Luther King Jr Day',
                    pd.Timestamp(year=2018, month=1, day=15): 'Martin Luther King Jr Day',
                    pd.Timestamp(year=2016, month=1, day=18): 'Martin Luther King Jr Day'}

    for row in full_data.index:
        day = holiday_dict.get(row, 0)
        if day != 0:
            full_data.loc[row, 'holiday'] = 1
        else:
            full_data.loc[row, 'holiday'] = 0

    bin_number = 0
    value = 0
    for line in full_data.index:
        if value != line.month:
            value = line.month
            bin_number += 1
        full_data.loc[line, 'bin'] = bin_number

    full_data['temp'] = full_data.groupby(['bin'])['temp'].transform(lambda x: x.fillna(x.mean()))
    full_data.drop('bin', axis=1, inplace=True)
    full_data['weekday'] = full_data.index.day_of_week

    full_data['hour'] = full_data.index.hour
    full_data['hour_cos'] = np.cos(2 * np.pi * full_data["hour"] / 23)
    full_data['hour_sin'] = np.sin(2 * np.pi * full_data["hour"] / 23)

    data_with_lags = create_lags(full_data, ['traffic_volume'], 168)
    data_with_lags = data_with_lags.dropna()

    y = data_with_lags['traffic_volume']
    X = data_with_lags.drop(['traffic_volume', 'hour'], axis=1)
    X_train, X_test, y_train, y_test = timeseries_train_test_split(y, X)

    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=250, n_jobs=6, gc_after_trial=True)

    print(study.best_trials)
