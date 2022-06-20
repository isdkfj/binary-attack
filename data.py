import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(dname, path, SEED):
    if dname == 'bank':
        path = os.path.join(path, 'bank-marketing/bank-additional-full.csv')
        df = pd.read_csv(path, delimiter=';')
        for attr in df.columns:
            if df[attr].dtype == 'object':
                encoder= LabelEncoder().fit(df[attr])
                df[attr] = encoder.transform(df[attr])
        X = df.values[:, :-1]
        X -= np.min(X, axis=0)
        X /= np.max(X, axis=0)
        Y = df.values[:, -1].astype('int')
        # balance
        X_0 = X[Y == 0]
        X_1 = X[Y == 1]
        X_0 = X_0[np.random.choice(X_0.shape[0], X_1.shape[0])]
        X = np.concatenate((X_0, X_1), axis=0)
        Y = np.ones(X.shape[0], dtype=int)
        Y[:X_0.shape[0]] = 0
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
        X -= np.min(X, axis=0)
        X /= np.max(X, axis=0)
        Y = df.values[:, -1].astype('int')
        # balance
        X_0 = X[Y == 0]
        X_1 = X[Y == 1]
        X_0 = X_0[np.random.choice(X_0.shape[0], X_1.shape[0])]
        X = np.concatenate((X_0, X_1), axis=0)
        Y = np.ones(X.shape[0], dtype=int)
        Y[:X_0.shape[0]] = 0
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
        X -= np.min(X, axis=0)
        X /= np.max(X, axis=0)
        Y = df.values[:, 0].astype('int')
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
        X -= np.min(X, axis=0)
        X /= np.max(X, axis=0)
        Y = df.values[:, -1].astype('int')
        fake_label = np.random.randint(0, 2, (X.shape[0], 1))
        X = np.concatenate([X, fake_label], axis=1)
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1, random_state=SEED)
    return train_X, test_X, train_Y, test_Y
