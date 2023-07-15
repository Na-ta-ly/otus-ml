import pandas as pd
from tqdm.notebook import tqdm
import nltk
from functools import lru_cache
import pymorphy2

import optuna

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score, mean_squared_error


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
    max_features = trial.suggest_int('max_features', 50, 1000, step=50)
    max_ngram = trial.suggest_int('max_ngram', 2, 4)
    max_df = trial.suggest_float('max_df', 0.70, 1.0, step=0.05)
    min_df = trial.suggest_float('min_df', 0.01, 0.1, step=0.01)

    # for Lasso model
    # alpha = trial.suggest_float('alpha', 0.0005, 1, step=0.001)

    # for Ridge model
    # alpha = trial.suggest_float('alpha', 0.5, 1.5, step=0.1)

    # for ElasticNet model
    # alpha = trial.suggest_float('alpha', 0.9, 5, step=0.1)
    # l1_ratio = trial.suggest_float('l1_ratio', 0, 1, step=0.1)

    vectorizer = TfidfVectorizer(max_features=max_features, norm=None, ngram_range=(1, max_ngram),
                                 max_df=max_df, min_df=min_df).fit(train_raw_x)
    train_x = vectorizer.transform(train_raw_x)
    test_x = vectorizer.transform(test_raw_x)

    model = LinearRegression()
    # model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    # model = Ridge(alpha=alpha, random_state=42, max_iter=10000)
    # model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)

    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    mse = mean_squared_error(test_y, prediction)
    r2 = r2_score(test_y, prediction)

    return r2, mse



if __name__ == '__main__':
    data = pd.read_csv('preprocessed_data.csv')
    data = data.drop([
        col for col in data.columns if col not in ['Reviews', 'Score']
    ], axis=1)

    stop_words = nltk.corpus.stopwords.words('russian')
    morph = pymorphy2.MorphAnalyzer()

    tokenized_reviews = tokenize_data(data['Reviews'])
    lemmatized_reviews = lemmatize_data(tokenized_reviews)
    data = data.drop(['Reviews'], axis=1)
    data = pd.concat([data, lemmatized_reviews], axis=1)

    y = data['Score']
    X_full = data['Reviews']
    train_raw_x, test_raw_x, train_y, test_y = train_test_split(X_full, y,
                                                                test_size=0.2, random_state=42)

    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=1000, n_jobs=8, gc_after_trial=True)

    print(study.best_trials)
