import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
RAW_CLAP_DIR = os.path.join(DATA_DIR, 'raw_claps')
EXTRACTED_CLAP_DIR = os.path.join(DATA_DIR, 'extracted_claps')

SAMPLE_RATE = 44100

WINDOW_SIZE = 0.7
WINDOW_STEP = 0.4

EXTRACT_THRESHOLD = 99.0

SECONDS_BEFORE = 0.03
SECONDS_AFTER = 0.15
