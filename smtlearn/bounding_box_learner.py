from pysmt.shortcuts import Real, And

from pywmi import Domain


def bb_solve(domain: Domain, data):
    constraints = []
    for v in domain.real_vars:
        min_val = min([row[0][v] for row in data])
        max_val = max([row[0][v] for row in data])
        constraints.append(domain.get_symbol(v) >= Real(min_val))
        constraints.append(domain.get_symbol(v) <= Real(max_val))
    return And(*constraints)
