from pathlib import Path
import os

MOD_DUR = os.path.dirname(Path(__file__).absolute())
PROJ_PATH = os.path.join(MOD_DUR, '..', '..')

M1_DATASET = 'ml-1m'
DATA_DIR = os.path.join(PROJ_PATH, 'data', M1_DATASET)

DATA_OUT_DIR = os.path.join(PROJ_PATH, 'data_out')
# CSV_DIR = os.path.join(SCRIPT_DIR, '..', 'csv')
CHART_DIR = os.path.join(PROJ_PATH, 'charts')
MODEL_DIR = os.path.join(PROJ_PATH, 'models')
MODEL_STATE = os.path.join(MODEL_DIR, 'movie_model.pth')
IMAGE_DIR = os.path.join(PROJ_PATH, 'data', 'Multi_Label_dataset', 'images')

# Dataset related constants
AGE_DICT = {1:"Under 18",
            18:"18-24",
            25:"25-34",
            35:"35-44",
            45:"45-49",
            50:"50-55",
            56:"56+"}

AGE_MAP = {1:0,
            18:1,
            25:2,
            35:3,
            45:4,
            50:5,
            56:6}

OCCUPATION_DICT = {0:"other",
                    1:"academic/educator",
                    2:"artist",
                    3:"clerical/admin",
                    4:"college/grad student",
                    5:"customer service",
                    6:"doctor/health care",
                    7:"executive/managerial",
                    8:"farmer",
                    9:"homemaker",
                    10:"K-12 student",
                    11:"lawyer",
                    12:"programmer",
                    13:"retired",
                    14:"sales/marketing",
                    15:"scientist",
                    16:"self-employed",
                    17:"technician/engineer",
                    18:"tradesman/craftsman",
                    19:"unemployed",
                    20:"writer",}

GENRES_LIST = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
# Globals
# MODEL_NAME = 'weld_resnet50_model_v6'
# SCRIPT_PATH = Path(__file__).absolute()
# SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
# CSV_DIR = os.path.join(SCRIPT_DIR, '..', 'csv')
# CHART_DIR = os.path.join(SCRIPT_DIR, '..', 'charts')
# MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
# DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
# DATA_SRC = 'ss304'
# DATA_SRC_REDUCED = 'ss304_reduced'

if __name__ == '__main__':
    print(os.listdir(MOD_DUR))
    print(os.listdir(PROJ_PATH))
    print(DATA_DIR)