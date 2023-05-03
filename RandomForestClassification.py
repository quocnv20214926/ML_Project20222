import pandas as pd 
import numpy as np
import random
import math
import collections
from calculate import calculate_entropy , calculate_left_value, find_best_split
from joblib import Parallel,delayed
class Tree():
#class attribute declaration
    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None
        self.tree_left = None
        self.tree_right = None
        self.tree = None
    def fit(self, X,y , max_depth=10, min_samples_split =6):
        num_features = X.shape[1]
        self.tree = self.build_tree(X,y,num_features,max_depth,min_samples_split)
    def build_tree(self,X,y,num_feature,max_depth, min_samples_split): 
        #create a new node
        tree = Tree()
        #tính toán giá trị entropy của tập dữ liệu hiện tại
        current_entropy = calculate_entropy(y)
        if max_depth==0 or len(set(y))==1:
            tree.leaf_value= calculate_left_value 
            return tree
        best_feature,  best_value, left_X,left_y, right_X, right_y= find_best_split(X,y,num_feature,current_entropy, min_samples_split)
        if best_feature is None:
            tree.leaf_value= calculate_left_value(y)
            return tree
        tree.split_feature= best_feature
        tree.split_value=best_value
        tree.tree_left=self.build_tree(left_X,left_y,num_feature,max_depth-1,min_samples_split)
        tree.tree_right = self.build_tree(right_X,right_y, num_feature, max_depth -1,min_samples_split)
        return tree
    def calc_predict_value(self,dataset):
        if self.leaf_value is not None:
            return self.leaf_value
        elif dataset[self.split_feature]<= self.split_value:
            return self.tree_left.calc_predict_value(dataset)
        else:
            return self.tree_right.calc_predict_value(dataset)
        
    def describe_tree(self):
        if not self.tree_left and not self.tree_right:
            leaf_info ="{leaf_value:" +str(self.leaf_value)+"}"
            return leaf_info
        left_info = self.tree_left.describe_tree()
        right_info = self.tree_right.describe_tree()
        tree_structure = "{split_feature:" + str(self.split_feature) + \
                         ",split_value:" + str(self.split_value) + \
                         ",left_tree:" + left_info + \
                         ",right_tree:" + right_info + "}"
        return tree_structure

class RandomForestClassification(object):
    def __init__(self, n_estimators =10, max_depth =-1, min_samples_split=2,min_samples_leaf=1, 
                 min_split_gain =0, colsample_bytree = None, subsample = 0.8, random_state= None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth if max_depth != -1 else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_split_gain = min_split_gain
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.random_state = random_state
        self.trees = None
        self.feature_importances_ = dict()

    def fit(self, dataset, targets):
       
        assert targets.unique().__len__() == 2, "There must be two class for targets!"
        targets = targets.to_frame(name='RiskLevel')

        if self.random_state:
            random.seed(self.random_state)
        random_state_stages = random.sample(range(self.n_estimators), self.n_estimators)

       
        if self.colsample_bytree == "sqrt":
            self.colsample_bytree = int(len(dataset.columns) ** 0.5)
        elif self.colsample_bytree == "log2":
            self.colsample_bytree = int(math.log(len(dataset.columns)))
        else:
            self.colsample_bytree = len(dataset.columns)

        self.trees = Parallel(n_jobs=-1, verbose=0, backend="threading")(
            delayed(self._parallel_build_trees)(dataset, targets, random_state)
                for random_state in random_state_stages)
        
    def _parallel_build_trees(self, dataset, targets, random_state):

        subcol_index = random.sample(dataset.columns.tolist(), self.colsample_bytree)
        dataset_stage = dataset.sample(n=int(self.subsample * len(dataset)), replace=True, 
                                        random_state=random_state).reset_index(drop=True)
        dataset_stage = dataset_stage.loc[:, subcol_index]
        targets_stage = targets.sample(n=int(self.subsample * len(dataset)), replace=True, 
                                        random_state=random_state).reset_index(drop=True)

        tree = self._build_single_tree(dataset_stage, targets_stage, depth=0)
        print(tree.describe_tree())
        return tree

    def _build_single_tree(self, dataset, targets, depth):
        
        if len(targets['RiskLevel'].unique()) <= 1 or dataset.__len__() <= self.min_samples_split:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['RiskLevel'])
            return tree

        if depth < self.max_depth:
            best_split_feature, best_split_value, best_split_gain = self.choose_best_feature(dataset, targets)
            left_dataset, right_dataset, left_targets, right_targets = \
                self.split_dataset(dataset, targets, best_split_feature, best_split_value)

            tree = Tree()
          
            if left_dataset.__len__() <= self.min_samples_leaf or \
                    right_dataset.__len__() <= self.min_samples_leaf or \
                    best_split_gain <= self.min_split_gain:
                tree.leaf_value = self.calc_leaf_value(targets['RiskLevel'])
                return tree
            else:
                
                self.feature_importances_[best_split_feature] = \
                    self.feature_importances_.get(best_split_feature, 0) + 1

                tree.split_feature = best_split_feature
                tree.split_value = best_split_value
                tree.tree_left = self._build_single_tree(left_dataset, left_targets, depth+1)
                tree.tree_right = self._build_single_tree(right_dataset, right_targets, depth+1)
                return tree
       
        else:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['RiskLevel'])
            return tree

    def choose_best_feature(self, dataset, targets):
        
        best_split_gain = 1
        best_split_feature = None
        best_split_value = None

        for feature in dataset.columns:
            if dataset[feature].unique().__len__() <= 100:
                unique_values = sorted(dataset[feature].unique().tolist())
           
            else:
                unique_values = np.unique([np.percentile(dataset[feature], x)
                                           for x in np.linspace(0, 100, 100)])

            
            for split_value in unique_values:
                left_targets = targets[dataset[feature] <= split_value]
                right_targets = targets[dataset[feature] > split_value]
                split_gain = self.calc_gini(left_targets['RiskLevel'], right_targets['RiskLevel'])

                if split_gain < best_split_gain:
                    best_split_feature = feature
                    best_split_value = split_value
                    best_split_gain = split_gain
        return best_split_feature, best_split_value, best_split_gain

    @staticmethod
    def calc_leaf_value(targets):
        
        RiskLevel_counts = collections.Counter(targets)
        major_RiskLevel = max(zip(RiskLevel_counts.values(), RiskLevel_counts.keys()))
        return major_RiskLevel[1]

    @staticmethod
    def calc_gini(left_targets, right_targets):
        
        split_gain = 0
        for targets in [left_targets, right_targets]:
            gini = 1
            
            RiskLevel_counts = collections.Counter(targets)
            for key in RiskLevel_counts:
                prob = RiskLevel_counts[key] * 1.0 / len(targets)
                gini -= prob ** 2
            split_gain += len(targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini
        return split_gain

    @staticmethod
    def split_dataset(dataset, targets, split_feature, split_value):
        
        left_dataset = dataset[dataset[split_feature] <= split_value]
        left_targets = targets[dataset[split_feature] <= split_value]
        right_dataset = dataset[dataset[split_feature] > split_value]
        right_targets = targets[dataset[split_feature] > split_value]
        return left_dataset, right_dataset, left_targets, right_targets

    def predict(self, dataset):
        
        res = []
        for _, row in dataset.iterrows():
            pred_list = []
            
            for tree in self.trees:
                pred_list.append(tree.calc_predict_value(row))

            pred_RiskLevel_counts = collections.Counter(pred_list)
            pred_RiskLevel = max(zip(pred_RiskLevel_counts.values(), pred_RiskLevel_counts.keys()))
            res.append(pred_RiskLevel[1])
        return np.array(res)


if __name__ == '__main__':
    df = pd.read_csv("dataset.csv")
    df = df[df['RiskLevel'].isin([1, 2])].sample(frac=1, random_state=66).reset_index(drop=True)
    clf = RandomForestClassification(n_estimators=100,
                                 max_depth=10,
                                 min_samples_split=6,
                                 min_samples_leaf=2,
                                 min_split_gain=0.0,
                                 colsample_bytree="sqrt",
                                 subsample=0.8,
                                 random_state=66)
    train_count = int(0.7 * len(df))
    feature_list = ["Age","SystolicBP","DiastolicBP","BS","BodyTemp","HeartRate"]
    clf.fit(df.loc[:train_count, feature_list], df.loc[:train_count, 'RiskLevel'])

    from sklearn import metrics
    print(metrics.accuracy_score(df.loc[:train_count, 'RiskLevel'], clf.predict(df.loc[:train_count, feature_list])))
    print(metrics.accuracy_score(df.loc[train_count:, 'RiskLevel'], clf.predict(df.loc[train_count:, feature_list])))
