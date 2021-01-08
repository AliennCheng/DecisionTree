import numpy as np
import pandas as pd

from decision_tree import DecisionTree
from data_loader import DataLoader

def main(args):
    '''Main function for Decision Tree.

    Args:
        - train: path to the training set
        - test: path to the testing set
        - max_depth: maximal depth of the decision tree
        - output_file: (optional) path to the output file
        - verbose: (optional) whether to print the process
    
    Returns:
        - y_pred: prediction for target if y_ts not provided
        - y_prob: probability of the target to be true
    '''

    train = args.train
    test = args.test
    max_depth = args.max_depth
    output_file = args.output_file
    verbose = args.verbose

    X_tr, y_tr, no, X_ts = DataLoader(train, test)

    dt = DecisionTree(max_depth=max_depth, verbose=verbose)
    dt.fit(X_tr, y_tr)
    y_pred = dt.predict(X_ts)
    y_prob = dt.predict(X_ts, prob=True)

    print('Prediction:')
    print(y_pred)
    print('Probability:')
    print(y_prob)

    if output_file != None:
        out = pd.DataFrame({'No': no, 'Target': y_pred, 'Probability': y_prob})
        out.to_csv('Submission.csv', index=False)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        help='Path to the training dataset',
        type=str,
        required=True
    )
    parser.add_argument(
        '--test',
        help='Path to the testing dataset',
        type=str,
        required=True
    )
    parser.add_argument(
        '--max_depth',
        help='Maximal depth of the decision tree',
        default='30',
        type=int
    )
    parser.add_argument(
        '--output_file',
        help='Path to the output file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--verbose',
        help='Whether to print the process',
        default=False,
        type=bool
    )