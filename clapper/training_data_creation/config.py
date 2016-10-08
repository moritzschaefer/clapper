import os
import math

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
RAW_CLAP_DIR = os.path.join(DATA_DIR, 'raw_claps')
RAW_DISTORTION_DIR = os.path.join(DATA_DIR, 'raw_distortions')
NEGATIVE_SAMPLES_DIR = os.path.join(DATA_DIR, 'negative_samples')
POSITIVE_SAMPLES_DIR = os.path.join(DATA_DIR, 'positive_samples')
EXTRACTED_CLAP_DIR = os.path.join(DATA_DIR, 'extracted_claps')

SAMPLE_RATE = 44100

SECONDS_BEFORE = 0.02
SECONDS_AFTER = 0.15

FFT_SIZE = 512
SAMPLE_WIDTH = 2  # one sample has 2 bytes (16 bits)

# number of windows used for the feature vector for classifying a clap
NUM_WINDOWS = math.ceil(SECONDS_BEFORE+SECONDS_AFTER)/(FFT_SIZE/SAMPLE_RATE)


# Clap extraction
EXTRACT_WINDOW_SIZE = 0.7
EXTRACT_WINDOW_STEP = 0.4
EXTRACT_THRESHOLD = 99.0

# Distortion extraction
NUM_DISTORTION_SAMPLES = 10000


# Feature extraction
USE_LOG_EXTRACTION = True
