from typing import List, Any
import re

import numpy as np
from gensim.models import KeyedVectors
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm
import spacy


def flatten_list(sample: list) -> List[Any]:
    """
    Flatten the list from 2 levels to 1
    :param sample: 2D list
    :return: 1D list
    """
    new_list = []
    for item in sample:
        if isinstance(item, list):
            for new_item in item:
                new_list.append(new_item)
        else:
            new_list.append(item)
    return new_list


def multiple_split(text: str, list_separators: List[str]) -> List[str]:
    """
    Splits given text according to all given separators
    :param text: string for separation
    :param list_separators: separators as list of strings
    :return: list of separated parts
    """
    resulting_list = [text]
    temp_list = []
    for sep in list_separators:
        for part in resulting_list:
            if sep in part:
                result = part.split(sep)
                for each_result in result:
                    if each_result != '':
                        temp_list.append(each_result)
            else:
                temp_list.append(part)
        resulting_list = flatten_list(temp_list)
        temp_list = []
    return resulting_list


def lemmatize_column(data, column_name):
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    new_columns = pd.Series(dtype='object')
    new_item = ''
    for item in tqdm(data[column_name]):
        item = re.sub(r'(?<=\d)\.(?=\d)', ':', item)
        for sentence in multiple_split(item, ['\n', '!', '?', '.']):
            doc = nlp(sentence.lower())
            new_item = ' '.join([new_item, ' '.join([token.lemma_ for token in doc])])
        new_columns = pd.concat([new_columns, pd.Series(new_item)], ignore_index=True)
        new_item = ''
    return pd.DataFrame(new_columns, columns=['Lemmatized'])


def question_embeddings(initial_data, column_name):
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=100000)
    data = lemmatize_column(initial_data, column_name)
    full_text = []
    for index in tqdm(range(len(data))):
        string_embedding = np.zeros((300,), dtype=float)
        string = data.loc[index, 'Lemmatized']
        string = string.lower().split(' ')
        length = 0
        for word in string:
            word = word.strip()
            if word.isalpha():
                try:
                    word_embedding = model[word]
                    string_embedding = string_embedding + word_embedding
                    length += 1
                except KeyError:
                    print(f'Word {word} not found')
                    continue
        full_text.append(string_embedding / length)
    return pd.DataFrame({column_name: full_text}, index=initial_data.index)


def sum_embeddings(data, column_name):
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=100000)
    full_data = []
    for index in tqdm(range(len(data))):
        full_item = []
        text = data.loc[index, column_name]
        paragraphs = text.lower().split('\n')
        for paragraph in paragraphs:
            paragraph_words = paragraph.split()
            if len(paragraph) > 1 and len(paragraph_words) >= 2 and (paragraph_words[1].isalpha()
                                                                     or paragraph_words[0].isalpha()):
                paragraph_embedding = np.zeros((300,), dtype=float)
                paragraph_length = 0
                sentences = multiple_split(paragraph, ['?', '!', '.'])
                for sentence in sentences:

                    if len(sentence) > 2:
                        sentence_embedding = np.zeros((300,), dtype=float)
                        sentence_length = 0
                        sentence_lemmatized = nlp(sentence)
                        for word in sentence_lemmatized:
                            word = word.lemma_
                            if word.isalpha():
                                try:
                                    word_embedding = model[word]
                                    sentence_embedding = sentence_embedding + word_embedding
                                    sentence_length += 1
                                except KeyError:
                                    print(f'Word {word} not found')
                                continue
                        if not (sentence_embedding[0] == 0.0 and sentence_embedding[5] == 0.0
                                and sentence_embedding[8] == 0.0):
                            paragraph_embedding = paragraph_embedding + sentence_embedding / sentence_length
                            paragraph_length += 1
                if not (paragraph_embedding[0] == 0.0 and paragraph_embedding[5] == 0.0
                        and paragraph_embedding[8] == 0.0):
                    full_item.append(paragraph_embedding / paragraph_length)

        full_data.append(full_item)
    return pd.DataFrame({'Summ_paragraph': full_data}, index=data.index)


def similarity_counter(base_column, sample_column):
    min_similarity = []
    max_similarity = []
    mean_similarity = []
    for k in range(len(base_column)):
        line = []
        for paragraph in sample_column[k]:
            line.append(cosine_similarity([paragraph, base_column[k]])[0][1])
        min_similarity.append(np.min(line))
        max_similarity.append(np.max(line))
        mean_similarity.append(np.mean(line))
    return pd.DataFrame({'Min_sim': min_similarity,
                         'Max_sim': max_similarity,
                         'Mean_sim': mean_similarity}, index=base_column.index)


def phrase_compare(text: List[str], phrase_for_search: List[str]) -> bool:
    """
    Compares text with phrase_for_search and if text starts from phrase_for_search, returns True
    """
    flag = True
    if len(phrase_for_search) > len(text):
        flag = False
    else:
        for word_number in range(len(phrase_for_search)):
            if ((phrase_for_search[word_number] not in ['somebody', 'something', 'somewhere'])
                    and (text[word_number] != phrase_for_search[word_number])):
                flag = False
                break
    return flag


def transform_column(data, column_name=None):
    if column_name is None:
        column_name = data.columns[0]
    lemmatized_column = lemmatize_column(data, column_name)
    return pd.concat([data, lemmatized_column], axis=1)


def get_word_features(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Counts words of a definite level and adds their number to the data
    :param data: DataFrame for analysis
    :param column_name: column for calculus
    :return: DataFrame with added number of words
    """
    result_data = data.copy()
    result_data['A1'] = 0
    result_data['A2'] = 0
    result_data['B1'] = 0
    result_data['B2'] = 0
    result_data['C1'] = 0
    result_data['C2'] = 0
    result_data['AC'] = 0

    data_a1 = pd.read_csv('preprocessing/a1.csv', sep=';')
    data_a2 = pd.read_csv('preprocessing/a2.csv', sep=';')
    data_b1 = pd.read_csv('preprocessing/b1.csv', sep=',')
    data_b2 = pd.read_csv('preprocessing/b2.csv', sep=';')
    data_c1 = pd.read_csv('preprocessing/c1.csv', sep=';')
    data_c2 = pd.read_csv('preprocessing/c2.csv', sep=';')
    data_ac = pd.read_csv('preprocessing/academic.csv', sep=',')

    data_a1 = transform_column(data_a1, 'Word')
    data_a2 = transform_column(data_a2, 'Word')
    data_b1 = transform_column(data_b1, 'Word')
    data_b2 = transform_column(data_b2, 'Word')
    data_c1 = transform_column(data_c1, 'Word')
    data_c2 = transform_column(data_c2, 'Word')
    data_ac = transform_column(data_ac, 'Word')

    lemm_data = transform_column(data, column_name)
    for index in tqdm(range(len(data))):
        string = lemm_data.loc[index, 'Lemmatized']
        string = string.lower().split(' ')
        for word_num in range(len(string)):
            word = string[word_num].strip()
            if word.isalpha():
                for feature_name, dictionary in zip(['C2', 'C1', 'B2', 'B1', 'A2', 'A1', 'AC'],
                                                    [data_c2, data_c1, data_b2, data_b1, data_a2, data_a1, data_ac]):
                    for lemmatized in dictionary['Lemmatized']:
                        lemma_split = lemmatized.split()
                        if phrase_compare(string[word_num:], lemma_split):
                            value = result_data.loc[index, feature_name]
                            result_data.loc[index, feature_name] = value + 1
    return result_data


def tokenize_text(item):
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    new_item = ''
    item = re.sub('[\n!?]', r'\.', item)
    for sentence in item.split('.'):
        if len(sentence) > 1:
            sentence = sentence.strip()
            new_item = ''.join([new_item, ' '.join([token.text for token in nlp(sentence)]), ' .\n'])
    return new_item[:-1]
