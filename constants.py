import os

INPUT_FOLDER = 'input'
OUTPUT_FOLDER = 'output'
NETWORK_FOLDER = 'network'


def setup_folders():
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(NETWORK_FOLDER, exist_ok=True)