import logging
import os
import glob

import numpy as np
import scipy.io.wavfile
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron


from utils import Timer
from training_data_creation.config import NEGATIVE_SAMPLES_DIR, \
    POSITIVE_SAMPLES_DIR, SAMPLE_RATE, NUM_FEATURES
from feature_extraction import extract_features

logging.basicConfig(level=logging.INFO)


def load():
    """
    Load all training points into a matrix
    """

    Xs = []
    Ys = []

    for path, label in ((NEGATIVE_SAMPLES_DIR, 0), (POSITIVE_SAMPLES_DIR, 1)):

        audio_files = glob.glob(os.path.join(path, '*.wav'))
        logging.info('Detected {} files for label {}'.format(len(audio_files),
                                                             label))
        Xs.append(np.zeros((len(audio_files), NUM_FEATURES)))
        Ys.extend([label] * len(audio_files))

        for i, audio_file in enumerate(audio_files):
            rate, data = scipy.io.wavfile.read(audio_file)
            assert rate == SAMPLE_RATE

            Xs[-1][i, :] = extract_features(data)

            if i % 1000 == 0:
                logging.info('Loaded {} files'.format(i))

    X = np.vstack(Xs)
    y = np.array(Ys)
    return X, y


def preprocess(X, y):
    """
    Clean dataset
    """
    inf_rows = np.nonzero(np.isinf(X))[0]
    return np.delete(X, inf_rows, 0), np.delete(y, inf_rows)


def train(X, y):
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
    ppn.fit(X_train_std, y_train)

    with Timer() as prediction_time:
        y_pred = ppn.predict(X_test_std)

    print('Misclassified with perceptron: {}. Took {} seconds/point to predict'
          .format((y_test != y_pred).sum()/len(y_test),
                  prediction_time.interval/len(X_test_std)))

    return ppn


if __name__ == "__main__":
    X, y = load()
    X, y = preprocess(X, y)
    Model = train(X, y)
