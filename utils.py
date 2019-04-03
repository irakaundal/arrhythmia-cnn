import os

BASE_DIR = os.getcwd()
# initialize the paths to our training and testing CSV files
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"

# initialize the number of epochs to train for and batch size
NUM_EPOCHS = 100
BS = 32

# initialize the total number of training and testing image
NUM_TRAIN_IMAGES = 0
NUM_TEST_IMAGES = 0

encoding = {'APC': 1,
            'LBB': 2,
            'NOR': 3,
            'PAB': 4,
            'PVC': 5,
            'RBB': 6,
            'VEB': 7,
            'VFB': 8}