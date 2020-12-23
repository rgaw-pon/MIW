from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data
y = iris.target
x_names = iris.feature_names
y_names = iris.target_names

# 70% for training

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)


class Node:

    def __init__(self, x, y, x_names, y_names, tree_depth):
        self.x = x
        self.y = y
        self.x_names = x_names
        self.y_names = y_names
        self.tree_depth = tree_depth
        self.child_l = None
        self.child_r = None
        self.score = None
        self.feature = None
        self.threshold = None

    def get_Gini(self):
        instances = np.bincount(self.y)
        total = np.sum(instances)
        return 1.0 - np.sum(np.power(instances/total, 2))

    def get_Entropy(self):
        instances = np.bincount(self.y)
        total = np.sum(instances)
        p = instances / total
        return 0.0 - np.sum(np.log(p)*p)

    def get_class_for_node(self):
        instances = np.bincount(self.y)
        return np.argmax(instances, axis=0)

    def create_child_nodes(self, feature, threshold):
        x_l = []
        y_l = []
        x_r = []
        y_r = []
        for features, classification in zip(x, y):
            if features[feature] <= threshold:
                x_l.append(features)
                y_l.append(classification)
            else:
                x_r.append(features)
                y_r.append(classification)
        return np.asarray(x_l), np.asarray(y_l, dtype=np.int64), np.asarray(x_r), np.asarray(y_r, dtype=np.int64)

    def get_score(self, y, y_l, y_r, impurity_measure):
        score_left = impurity_measure()*y_l.shape[0]/y.shape[0]
        score_right = impurity_measure()*y_r.shape[0]/y.shape[0]
        return score_left + score_right

    def split_node_node(self, x, y, granulation, impurity_measure):
        x_l_best = None
        y_l_best = None
        x_r_best = None
        y_r_best = None
        score_best = None
        feature_best = None
        threshold_best = None
        if(x is None or y is None):
            return None
        for feature in range(x.shape[1]):
            start = np.min(x[:, feature])
            end = np.max(x[:, feature])
            step = (end - start) / granulation
            #print('start: {} end: {} step: {}'.format(start,end,step))
            if step != 0:
                for threshold in np.arange(start, end, step):
                    x_l, y_l, x_r, y_r = self.create_child_nodes(
                        feature, threshold)
                    score = self.get_score(y, y_l, y_r, impurity_measure)
                    #print('{} - {} => {}'.format(x_names[feature], threshold, score))
                    if score_best is None or score < score_best:
                        x_l_best = x_l
                        y_l_best = y_l
                        x_r_best = x_r
                        y_r_best = y_r
                        score_best = score
                        feature_best = feature
                        threshold_best = threshold
        self.score = score_best
        self.feature = feature_best
        self.threshold = threshold_best
        return x_l_best, y_l_best, x_r_best, y_r_best, score_best, feature_best, threshold_best

    def isLeaf(self):
        return self.child_l is None and self.child_r is None


Nodes = []


def advanceTree(parent):
    if parent.tree_depth <= 3 and parent.get_Gini() != 0:
        # print(parent.tree_depth)
        x_l_best, y_l_best, x_r_best, y_r_best, score_best, feature_best, threshold_best = parent.split_node_node(
            parent.x, parent.y, 10, parent.get_Gini)
        child_l = Node(x_l_best, y_l_best, x_names,
                       y_names, parent.tree_depth+1)
        child_r = Node(x_r_best, y_r_best, x_names,
                       y_names, parent.tree_depth+1)
        parent.child_l = child_l
        parent.child_r = child_r
        Nodes.append(child_l)
        Nodes.append(child_r)
        advanceTree(child_l)
        advanceTree(child_r)


def get_class_for_vector(y):
    instances = np.bincount(y)
    return np.argmax(instances, axis=0)


def convertTreeToGraph(node, G):
    if node.child_l.isLeaf() and node.child_r.isLeaf():
        G.add_edge(str(x_names[node.feature])+' > '+str(node.threshold),
                   y_names[node.child_l.get_class_for_node()])
        G.add_edge(str(x_names[node.feature])+' > '+str(node.threshold),
                   y_names[node.child_r.get_class_for_node()])
    else:
        G.add_edge(str(x_names[node.feature])+' > '+str(node.threshold),
                   str(x_names[node.child_l.feature])+' > '+str(node.child_l.threshold))
        G.add_edge(str(x_names[node.feature])+' > '+str(node.threshold),
                   str(x_names[node.child_r.feature])+' > '+str(node.child_r.threshold))


def findLeaf(node, x):
    if not node.isLeaf():
        if(x[node.feature] > node.threshold):
            return traverseTree(node.child_r, x)
        else:
            return traverseTree(node.child_l, x)
    else:
        return node


rootNode = Node(x_train, y_train, x_names, y_names, 0)
Nodes.append(rootNode)
advanceTree(rootNode)
correctClassification = 0
incorrectClassification = 0
for i in range(len(x_test)):
    leafNode = findLeaf(rootNode, x_test[i])
    if(leafNode.get_class_for_node() == y_test[i]):
        correctClassification += 1
    else:
        incorrectClassification += 1
print(str(100*correctClassification /
          (correctClassification+incorrectClassification))+'%')
G = nx.Graph()


G = nx.Graph()
G.add_node(rootNode)
bb = nx.betweenness_centrality(G)
nx.set_node_attributes(G, bb, 'test')
G.nodes[1]['something']
plt.subplot(121)
nx.draw(G, with_labels=True, font_weight='bold')
