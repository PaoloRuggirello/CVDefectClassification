import pandas as pd

TEST_PATH = 'bootstrap_folds/test_folds/'
TRAIN_PATH = 'bootstrap_folds/train_folds/'


def look_for_working_cells():
    pass


if __name__ == '__main__':
    labels = pd.read_csv('labels.csv', delim_whitespace=True, header=None)
    print('end')
