import numpy as np
from tree_node import TreeNode

class DecisionTree:
    '''A simple decision tree implementation.
    
    Note that this implementation can only deal with binary categorical and
    numerical features. Multiple categorical features are considered numerical.
    Also, this implementation is only available for single feature prediction.
    
    '''
    
    def __init__(self, max_depth=30, verbose=False):
        self.max_depth = max_depth
        self.verbose = verbose
        self.fitted = False
    
    def fit(self, X, y):
        '''Fit the decision tree to the given data X.
        
        Args:
            - X: training data excluding target feature
            - y: the target of training data
            - max_depth: maximal depth of the tree
        '''
        self.X_tr = X
        self.y_tr = y
        self.root = TreeNode(self.X_tr, self.y_tr, np.arange(self.X_tr.shape[0]), 0, self.max_depth)
        self.fitted = True
        if self.verbose:
            print('Fitting the data successfully.')
    
    def predict(self, X_ts, prob=False):
        '''Predict the target values of data X_ts with the decision tree.
        
        Args:
            - X_ts: the testing data
        
        Returns:
            - y: the target values
        '''
        if not self.fitted:
            raise RuntimeError('Decision Tree not built.')
        if len(X_ts.shape) == 1:
            return self.root.predict(X_ts, prob=prob, verbose=self.verbose)
        else:
            n_instances, _ = X_ts.shape
            y = np.empty(n_instances)
            for i in range(n_instances):
                if self.verbose: print('\nInstance #', i, ' :')
                y[i] = self.root.predict(X_ts[i], prob=prob, verbose=self.verbose)
                if self.verbose: print('========================')
            return y
    
    def show(self):
        '''Visualize the decision tree.
        '''
        self.root.show()