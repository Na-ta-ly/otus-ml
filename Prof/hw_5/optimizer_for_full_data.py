import pandas as pd
from tqdm.notebook import tqdm
import nltk
from functools import lru_cache
import pymorphy2

import optuna

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


@lru_cache(500000)
def new_func_01(data):
    word_tokenizer = nltk.WordPunctTokenizer()
    new_columns = []

    for index, item in tqdm(data):
        text_lower = item.lower()
        tokens = word_tokenizer.tokenize(text_lower)
        tokens = tuple([word for word in tokens
                        if (word.isalpha() and word not in stop_words)])
        new_columns.append(tokens)
    return tuple(new_columns)


def tokenize_data(columns):
    data = tuple(columns.items())
    return new_func_01(data)


@lru_cache(500000)
def lemmatize_data(columns):
    new_columns = pd.Series(dtype='object')
    new_item = ''
    for item in tqdm(columns):
        for word in item:
            new_word = morph.parse(word)[0].normal_form
            new_item = ' '.join([new_item, new_word])
        new_columns = pd.concat([new_columns, pd.Series(new_item)], ignore_index=True)
        new_item = ''
    return pd.DataFrame(new_columns, columns=['Reviews'])


def objective(trial: optuna.Trial):
    review_feature = ['Reviews']

    description_feature = ['Description']

    numerical_features = [
        col for col in data.columns if col not in ['Reviews', 'Description', 'Name', 'Score']
    ]

    max_features_1 = trial.suggest_int('max_features_rvw', 50, 1000, step=50)
    max_ngram_1 = trial.suggest_int('max_ngram_rvw', 2, 4)
    max_df_1 = trial.suggest_float('max_df_rvw', 0.70, 1.0, step=0.05)
    min_df_1 = trial.suggest_float('min_df_rvw', 0.01, 0.1, step=0.01)

    reviewsTextProcessor = Pipeline(steps=[
        ("squeeze", FunctionTransformer(lambda x: x.squeeze())),
        ("tfidf", TfidfVectorizer(max_features=max_features_1, norm=None, ngram_range=(1, max_ngram_1),
                                  max_df=max_df_1, min_df=min_df_1)),
        ("toarray", FunctionTransformer(lambda x: x.toarray()))])

    max_features_2 = trial.suggest_int('max_features_dsc', 50, 500, step=50)
    max_ngram_2 = trial.suggest_int('max_ngram_dsc', 2, 3)
    max_df_2 = trial.suggest_float('max_df_dsc', 0.70, 1.0, step=0.05)
    min_df_2 = trial.suggest_float('min_df_dsc', 0.01, 0.1, step=0.01)

    descriptionTextProcessor = Pipeline(steps=[
        ("squeeze", FunctionTransformer(lambda x: x.squeeze())),
        ("tfidf", TfidfVectorizer(max_features=max_features_2, norm=None, ngram_range=(1, max_ngram_2),
                                  max_df=max_df_2, min_df=min_df_2)),
        ("toarray", FunctionTransformer(lambda x: x.toarray()))])

    numerical_transformer = Pipeline(
        steps=[("scaler", StandardScaler())])

    data_transformer = ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, numerical_features),
            ("review", reviewsTextProcessor, review_feature),
            ("description", descriptionTextProcessor, description_feature)
        ])

    preprocessor = Pipeline(steps=[("data_transformer", data_transformer)])

    alpha = trial.suggest_float('alpha', 0.0, 5, step=0.1)
    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0, step=0.1)

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)

    regression_pipline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', model),
        ])

    regression_pipline.fit(train_x, train_y)
    prediction = regression_pipline.predict(test_x)
    mse = mean_squared_error(test_y, prediction)
    r2 = r2_score(test_y, prediction)

    return r2, mse


if __name__ == '__main__':
    data = pd.read_csv('preprocessed_data.csv')
    data = data.drop('Unnamed: 0', axis=1)

    stop_words = nltk.corpus.stopwords.words('russian')
    morph = pymorphy2.MorphAnalyzer()

    tokenized_reviews = tokenize_data(data['Reviews'])
    lemmatized_reviews = lemmatize_data(tokenized_reviews)
    data = data.drop(['Reviews'], axis=1)
    data = pd.concat([data, lemmatized_reviews], axis=1)

    tokenized_description = tokenize_data(data['Description'])
    lemmatized_description = lemmatize_data(tokenized_description)
    lemmatized_description.rename(columns={'Reviews': 'Description'}, inplace=True)
    data = data.drop(['Description'], axis=1)
    data = pd.concat([data, lemmatized_description], axis=1)

    y = data['Score']
    X_full = data.drop('Name', axis=1)
    train_x, test_x, train_y, test_y = train_test_split(X_full, y,
                                                        test_size=0.2, random_state=42)

    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=2000, n_jobs=6, gc_after_trial=True)

    print(study.best_trials)
