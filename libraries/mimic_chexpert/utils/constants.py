import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = '/shared/rsaas/nschiou2/'

DATA_DIR = os.path.join(SHARED_DIR, 'CXR', 'data')
DEMO_DATA_DIR = os.path.join(SHARED_DIR, 'CXR', 'data_demo')

REAL_MIMIC_TRAIN_PATH = os.path.join(DATA_DIR, 'train', 'mimic')
REAL_CHEXPERT_TRAIN_PATH = os.path.join(DATA_DIR, 'train', 'chexpert')
REAL_MIMIC_TEST_PATH = os.path.join(DATA_DIR, 'test', 'mimic')
REAL_CHEXPERT_TEST_PATH = os.path.join(DATA_DIR, 'test', 'chexpert')

RESULTS_DIR = os.path.join(SHARED_DIR, 'domain_adapt_cxr')

FINE_TUNING_RESULTS_DIR = os.path.join(RESULTS_DIR, 'fine_tuning')
