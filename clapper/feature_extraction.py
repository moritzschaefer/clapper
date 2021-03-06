import numpy as np

from training_data_creation.config import FFT_SIZE, NUM_WINDOWS, NUM_FEATURES


def transform_to_freq(data):
    """
    Apply fourier transformation and return frequency magnitude
    """
    data_frequency_domain = np.log10(np.abs(np.fft.rfft(data)) ** 2)

    return data_frequency_domain


def extract_features(data):
    """
    Extract NUM_WINDOWS times the frequency domain of the corresponding window
    and concatenate the output to a feature vector
    """

    if len(data) >= (NUM_WINDOWS + 1) * FFT_SIZE:
        raise ValueError('data contains too many frames.')

    if len(data) < NUM_WINDOWS * FFT_SIZE:
        raise ValueError('data contains too few frames.')

    # TODO check if np.int16 is correct!
    features = np.zeros((NUM_FEATURES, ), dtype=np.float64)
    for i in range(NUM_WINDOWS):
        transformed = transform_to_freq(data[i*FFT_SIZE:(i+1)*FFT_SIZE])

        features[(i*((FFT_SIZE/2)+1)):(i+1)*((FFT_SIZE/2)+1)] = transformed

    return features
