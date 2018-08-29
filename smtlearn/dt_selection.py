# Given a number of points:
# - Train a DT (scale points?)
# - For every point compute distance to the decision boundary
import sklearn.tree as tree

import generator
import pysmt.shortcuts as smt


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


def get_distances(domain, data):
    feature_matrix, labels = convert(domain, data)
    dt = learn_dt(feature_matrix, labels)
    return get_distances_dt(dt, domain, feature_matrix)


if __name__ == "__main__":
    def main():
        from main import checker_problem
        problem = checker_problem()
        data = generator.get_problem_samples(problem, 10, 1)
        print(get_distances(problem.domain, data))

    main()

