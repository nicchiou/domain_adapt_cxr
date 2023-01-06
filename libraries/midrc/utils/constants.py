"""Commonly-used constants for MIDRC experiments on SNAP servers."""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DFS = '/dfs/scratch0/nicchiou/'
LFS = '/lfs/local/0/nicchiou/'

DATA_DIR = os.path.join(LFS, 'MIDRC', 'data_subset', 'cr', 'clean_states')
METADATA_DIR = os.path.join(LFS, 'MIDRC', 'meta_info')

RESULTS_DIR = os.path.join(DFS, 'domain_adapt_cxr', 'results', 'midrc')

RESNET_BLOCK_SIZES = {
        'resnet18': (2, 2, 2, 2),
        'resnet34': (3, 4, 6, 3),
        'resnet50': (3, 4, 6, 3),
        'resnet101': (3, 4, 23, 3),
        'resnet152': (3, 8, 36, 3),
    }
