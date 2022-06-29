import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def process_binary(X):
    print('#instances: {}, #features: {}'.format(X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        s0 = np.sum(np.isclose(X[:, i], 0))
        s1 = np.sum(np.isclose(X[:, i], 1))
        print(s0, s1)
        if s0 + s1 == X.shape[0]:
            if s0 < s1:
                # swap 0 and 1 if there are more 1's
                X[:, i] = 1 - X[:, i]
                s0, s1 = s1, s0
            print('feature no.{} is binary, {}% are 0\'s'.format(i, s0 / X.shape[0] * 100))

def load_data(dname, path, SEED):
    if dname == 'bank':
        path = os.path.join(path, 'bank-marketing/bank-additional-full.csv')
        df = pd.read_csv(path, delimiter=';')
        for attr in df.columns:
            if df[attr].dtype == 'object':
                encoder= LabelEncoder().fit(df[attr])
                df[attr] = encoder.transform(df[attr])
        X = df.values[:, :-1]
        Y = df.values[:, -1].astype('int')
        process_binary(X)
        fake_label = np.random.randint(0, 2, (X.shape[0], 1))
        X = np.concatenate([X, fake_label], axis=1)
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1, random_state=SEED)
    elif dname == 'credit':
        path = os.path.join(path, 'default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')
        df = pd.read_csv(path, delimiter=',')
        for attr in df.columns:
            if df[attr].dtype == 'object':
                encoder= LabelEncoder().fit(df[attr])
                df[attr] = encoder.transform(df[attr])
        df = df.drop(columns=['ID'])
        X = df.values[:, :-1]
        Y = df.values[:, -1].astype('int')
        process_binary(X)
        fake_label = np.random.randint(0, 2, (X.shape[0], 1))
        X = np.concatenate([X, fake_label], axis=1)
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1, random_state=SEED)
    elif dname == 'news':
        pass
    elif dname == 'mushroom':
        path = os.path.join(path, 'mushroom-classification/mushrooms.csv')
        df = pd.read_csv(path)
        df = df.drop(columns=['veil-type'])
        for attr in df.columns:
            if df[attr].dtype == 'object':
                encoder= LabelEncoder().fit(df[attr])
                df[attr] = encoder.transform(df[attr])
        X = df.values[:, 1:].astype('float')
        Y = df.values[:, 0].astype('int')
        process_binary(X)
        fake_label = np.random.randint(0, 2, (X.shape[0], 1))
        X = np.concatenate([X, fake_label], axis=1)
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1, random_state=SEED)
    elif dname == 'nursery':
        path = os.path.join(path, 'nursery/nursery.csv')
        df = pd.read_csv(path)
        for attr in df.columns:
            if df[attr].dtype == 'object':
                encoder= LabelEncoder().fit(df[attr])
                df[attr] = encoder.transform(df[attr])
        X = df.values[:, :-1].astype('float')
        Y = df.values[:, -1].astype('int')
        process_binary(X)
        fake_label = np.random.randint(0, 2, (X.shape[0], 1))
        X = np.concatenate([X, fake_label], axis=1)
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1, random_state=SEED)
    min_X = np.min(train_X, axis=0)
    train_X -= min_X
    test_X -= min_X
    max_X = np.max(train_X, axis=0)
    train_X /= max_X
    test_X /= max_X
    return train_X, test_X, train_Y, test_Y
