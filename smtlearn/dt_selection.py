# Given a number of points:
# - Train a DT (scale points?)
# - For every point compute distance to the decision boundary
import sklearn.tree as tree

import generator
from main import checker_problem
import pysmt.shortcuts as smt
import numpy as np

from gurobipy import *


def convert(domain, data):
    def _convert(var, val):
        if domain.var_types[var] == smt.BOOL:
            return 1 if val else 0
        elif domain.var_types[var] == smt.REAL:
            return float(val)

    feature_matrix = []
    labels = []
    for instance, label in data:
        feature_matrix.append([_convert(v, instance[v]) for v in domain.variables])
        labels.append(1 if label else 0)
    return feature_matrix, labels


def learn_dt(feature_matrix, labels, **kwargs):
    # noinspection PyArgumentList
    estimator = tree.DecisionTreeClassifier(**kwargs)
    estimator.fit(feature_matrix, labels)
    return estimator


def export_dt(dt):
    import graphviz
    dot_data = tree.export_graphviz(dt, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("DT")




def get_leafs(tree, domain):

    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [domain.variables[i] for i in tree.tree_.feature]
    value = tree.tree_.value
    le = '<='
    g = '>'

    idx = np.argwhere(left == -1)[:, 0]

    IDS=value[idx]

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'l'
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'
        lineage.append((parent, split, threshold[parent], features[parent]))
        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    rules=[]
    for j, child in enumerate(idx):
        rules.append([])
        for node in recurse(left, right, child):
            if len(str(node)) < 3:
                continue
            i = node
            if i[1] == 'l':
                sign = le
            else:
                sign = g
            rules[-1].append([i[3] , sign , i[2]] )
        rules[-1].append(np.argmax(IDS[j]))

    return rules


def distanceforpositive(domain,rules,point):
    m = Model("D")
    m.setParam('OutputFlag', False)
    e=0.0001
    M=10

    x_i = m.addVars(domain.variables, name="x")
    aux_i = m.addVars(domain.variables, lb=-100,name="A")
    aux_j=m.addVars(domain.variables,lb=-100, name="B")
    auxb = []

    for i in rules:
        if i[-1]==1:
            auxb.append(m.addVar(vtype=GRB.BINARY))
            for j in i[:-1]:
                if j[1] == "<=":
                    m.addConstr(x_i[j[0]] <=j[2]+M*(1-auxb[-1]))

                elif j[1] == ">":
                    m.addConstr(x_i[j[0]] >= j[2]- M*(1-auxb[-1])-e)

        for j, i in zip(range(len(domain.variables)), domain.variables):
            m.addConstr(aux_i[i] == point[j] - x_i[i])

        m.addConstrs(aux_j[i] == abs_(aux_i[i]) for i in domain.variables)

        m.addConstr(quicksum(auxb) == 1)

        m.setObjective(sum(aux_j[i] for i in domain.variables), GRB.MINIMIZE)
        m.optimize()

        distance = {}
        for i in domain.variables:
            distance[i] = aux_j[i].x
        return distance

def distancefornegative(domain,rules,point):
    m = Model("D")
    m.setParam('OutputFlag', False)
    e = 0.0001
    M=2

    x_i = m.addVars(domain.variables,lb=-100, name="x")
    aux_i = m.addVars(domain.variables,lb=-100, name="A")
    aux_j=m.addVars(domain.variables, lb=-100,name="B")

    auxb=[]

    for i in rules:
        if i[-1]==0:
            auxb.append(m.addVar(vtype=GRB.BINARY))
            for j in i[:-1]:
                if j[1] == "<=":
                    m.addConstr(x_i[j[0]] <= j[2]+M*(1-auxb[-1]))

                elif j[1] == ">":
                    m.addConstr(x_i[j[0]] >= j[2]- M*(1-auxb[-1])-e)


    for j, i in zip(range(len(domain.variables)),domain.variables):
        m.addConstr(aux_i[i] == point[j] - x_i[i])

    m.addConstrs(aux_j[i] == abs_(aux_i[i] ) for i in domain.variables)

    m.addConstr(quicksum(auxb) ==1)

    m.setObjective(sum(aux_j[i] for i in domain.variables ), GRB.MINIMIZE)
    m.optimize()

    if m.status==GRB.Status.INFEASIBLE:
        print("in")


    distance={}
    for i in domain.variables:
        distance[i]=aux_j[i].x

    return  distance







def get_distances_dt(dt, domain, feature_matrix):
    # Include more features than trained with?

    leave_id = dt.apply(feature_matrix)
    feature = dt.tree_.feature
    threshold = dt.tree_.threshold
    node_indicator = dt.decision_path(feature_matrix)

    distances = []

    for sample_id in range(len(feature_matrix)):
        distance = dict()
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]]
        for node_id in node_index:
            variable = domain.variables[feature[node_id]]
            if leave_id[sample_id] != node_id and domain.var_types[variable] == smt.REAL:
                new_distance = abs(feature_matrix[sample_id][feature[node_id]] - threshold[node_id])
                if variable not in distance or new_distance < distance[variable]:
                    distance[variable] = new_distance
        distances.append(distance)

    return distances

def get_distances_dtNEW(dt, domain, feature_matrix,labels):
    # Include more features than trained with?
    rules=get_leafs(dt, domain)

    distances = []

    for sample_id,label in zip(range(len(feature_matrix)),labels):
        if label == 0:
            distance=distanceforpositive(domain,rules,feature_matrix[sample_id])
            distances.append(distance)
        elif label ==1:
            distance=distancefornegative(domain,rules,feature_matrix[sample_id])
            distances.append(distance)
    return distances


def get_distances(domain, data):
    feature_matrix, labels = convert(domain, data)
    dt = learn_dt(feature_matrix, labels)
    return get_distances_dt(dt, domain, feature_matrix)
    #return get_distances_dtNEW(dt, domain, feature_matrix,labels)

if __name__ == "__main__":
    def main():
        problem = checker_problem()
        data = generator.get_problem_samples(problem, 10, 1)
        print(get_distances(problem.domain, data))

    main()

