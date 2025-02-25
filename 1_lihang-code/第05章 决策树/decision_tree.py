import numpy as np
import pandas as pd


def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
                ['青年', '否', '否', '好', '否'],
                ['青年', '是', '否', '好', '是'],
                ['青年', '是', '是', '一般', '是'],
                ['青年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '好', '否'],
                ['中年', '是', '是', '好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '好', '是'],
                ['老年', '是', '否', '好', '是'],
                ['老年', '是', '否', '非常好', '是'],
                ['老年', '否', '否', '一般', '否'],
                ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    # 返回数据集和每个维度的名称
    return datasets, labels


def entropy(p):
    if (p == 0.0):
        return 0

    return - p * np.log2(p)


def calc_ent(df: pd.DataFrame, feature_name: str):
    res = 0.0
    unique_label = np.unique(df[feature_name])

    for label in unique_label:
        sub_df = df[df[feature_name] == label]
        res += entropy(len(sub_df) / len(df))

    return res


def info_gain(df: pd.DataFrame, feature_name: str, label_name: str):
    H_D_A = 0.0
    H_D = calc_ent(df, label_name)

    unique_feature = np.unique(df[feature_name])

    for feature in unique_feature:
        sub_df = df[df[feature_name] == feature]
        H_D_A += len(sub_df) / len(df) * calc_ent(sub_df, label_name)

    return H_D - H_D_A


def info_gain_rate(df: pd.DataFrame, feature_name: str, label_name: str):
    gain = info_gain(df, feature_name, label_name)
    h = calc_ent(df, feature_name)

    return gain / h


class Node:
    def __init__(self, feature=None, value=None, label=None):
        self.feature = feature
        self.label = label
        self.value = value
        self.child = {}

    def __repr__(self):
        if self.label is None:
            return f"Node({self.feature}, {self.value})"

        return f"Node({self.label})"

    def is_leaf(self):
        """ 叶子结点负责存储最终的label """
        return self.label is not None


class ID3DecisionTree:
    def __init__(self):
        self.root: Node = None

    def best_split(self, df: pd.DataFrame):
        label_name = df.columns.to_list()[-1]
        feature_info_gain = {}

        feature_names = df.columns.to_list()[:-1]
        for feature_name in feature_names:
            gain = info_gain(df, feature_name, label_name)
            feature_info_gain[feature_name] = gain

        return max(feature_info_gain, key=feature_info_gain.get)

    def build_tree(self, df: pd.DataFrame):
        label_name = df.columns.to_list()[-1]
        df_y = df[label_name]
        unique_y = np.unique(df_y)

        # 所有的样本都属于同一个类别, 返回叶子节点
        if (len(unique_y) == 1):
            return Node(label=unique_y[0])

        # 没有数据可以分类, 将df中的多数类作为叶子节点返回
        if (len(df) == 0):
            return Node(label=df_y.mode())

        best_feature = self.best_split(df)
        root = Node(feature=best_feature)

        for value in df[best_feature].unique():
            sub_df = df[df[best_feature] == value].drop(columns=[best_feature])
            root.child[value] = self.build_tree(sub_df)

        return root

    def train(self, df: pd.DataFrame):
        self.root = self.build_tree(df)

    def predict(self, x):
        node = self.root

        while not node.is_leaf():
            feature_value = x.get(node.feature)
            if feature_value in node.child:
                node = node.child[feature_value]
            else:
                return None

        return node.label

    def print_tree(self, node=None, indent=""):
        """打印决策树"""
        if node is None:
            node = self.root

        if node.is_leaf():
            print(indent + f"Leaf: {node.label}")
        else:
            print(indent + f"Feature: {node.feature}")
            for value, child in node.child.items():
                print(indent + f" ├── {value}")
                self.print_tree(child, indent + " │  ")


if __name__ == "__main__":
    datasets, labels = create_data()
    train_data = pd.DataFrame(datasets, columns=labels)
    model = ID3DecisionTree()

    model.train(train_data)

    test_data = train_data.loc[0, :]
    test_x = test_data.iloc[:-1]
    test_y = test_data.iloc[-1]

    y_pred = model.predict(test_x)
    print(y_pred)
    print(test_y)
