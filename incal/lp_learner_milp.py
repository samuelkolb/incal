from pysmt.shortcuts import Real, Plus, Symbol, And, GE
from pysmt.typing import REAL

from .incremental_learner import IncrementalLearner
from .learner import NoFormulaFound

try:
    import gurobipy as gurobi
except ImportError:
    gurobi = None


class LpLearnerMilp(IncrementalLearner):
    def __init__(
        self, half_space_count, selection_strategy, allow_negations=True, symmetries=""
    ):
        IncrementalLearner.__init__(self, "cnf_smt", selection_strategy, False)
        self.half_space_count = half_space_count
        self.allow_negations = allow_negations
        self.symmetries = symmetries
        self.active_indices = set()
        # self.smt_solver = smt_solver

    def learn_partial(self, solver, domain, data, labels, new_active_indices):
        if gurobi is None:
            raise RuntimeError("Gurobi is not installed.")

        # noinspection PyPep8Naming
        GRB = gurobi.GRB
        quicksum = gurobi.quicksum

        m = gurobi.Model("milp")
        # m.setParam('TimeLimit', 60*90)
        m.setParam("OutputFlag", False)

        # constants
        n_c = self.half_space_count
        n_v = len(domain.real_vars)  # number of variables #j
        bigm = 10000000
        ep = 1
        wmax = 10000
        cmax = 10000
        # c_0 = 1
        # c_j = c_0 + (1 - c_0)  # add complexity here

        self.active_indices |= set(new_active_indices)
        example_count = len(self.active_indices)
        x_k_j = [data[i] for i in self.active_indices]
        y_k = [labels[i] for i in self.active_indices]
        # labels = [row[1] for row in new_active_indices]

        # Variables

        # Equation offsets
        b_i = [
            m.addVar(
                vtype=GRB.CONTINUOUS, lb=-cmax, ub=cmax, name="b_i({i})".format(i=i)
            )
            for i in range(n_c)
        ]

        # Equation parameters
        a_i_j = [
            [
                m.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=-wmax,
                    ub=wmax,
                    name="a_i_j({i},{j})".format(i=i, j=j),
                )
                for j in range(n_v)
            ]
            for i in range(n_c)
        ]

        # Equation variable occurrence
        occ_i_j = [
            [
                m.addVar(vtype=GRB.BINARY, name="occ_i_j({i},{j})".format(i=i, j=j))
                for j in range(n_v)
            ]
            for i in range(n_c)
        ]

        # Constraint coverage
        cov_i = [
            m.addVar(vtype=GRB.BINARY, name="cov_i({i})".format(i=i))
            for i in range(n_c)
        ]

        # Example x constraint coverage
        excluded_k_i = [
            [
                m.addVar(
                    vtype=GRB.BINARY, name="excluded_k_i({k},{i})".format(k=k, i=i)
                )
                for i in range(n_c)
            ]
            for k in range(example_count)
        ]

        # Minimize equation variable occurrence => TODO why not constraint occurrence?
        m.setObjective(
            quicksum(1 * occ_i_j[i][j] for i in range(n_c) for j in range(n_v)),
            GRB.MINIMIZE,
        )

        # Constraints

        # 1 sum(wij*xkj<=ci) A positives
        for k in range(example_count):
            if y_k[k]:
                for i in range(n_c):
                    # one per positive example and number of constraints
                    m.addConstr(
                        sum(a_i_j[i][j] * x_k_j[k][j] for j in range(n_v)) <= b_i[i],
                        name="c1({k},{i})".format(k=k, i=i),
                    )

        # 2 sum(wij*sjk>m*ski-M+ci+e) A negatives
        for k in range(example_count):
            if not y_k[k]:
                for i in range(n_c):
                    # one per negative example and number of constraints
                    m.addConstr(
                        sum(a_i_j[i][j] * x_k_j[k][j] for j in range(n_v))
                        >= b_i[i] + ep - bigm * (1 - excluded_k_i[k][i]),
                        name="c2({k},{i})".format(k=k, i=i),
                    )

        # 3 sum(ski>=1) A negatives
        for k in range(example_count):  # one per negative example
            if not y_k[k]:
                m.addConstr(
                    sum(excluded_k_i[k][i] for i in range(n_c)) >= 1,
                    name="c3({k})".format(k=k),
                )

        # 5 wij<=cmax*wbij
        for i in range(n_c):
            for j in range(n_v):
                m.addConstr(
                    a_i_j[i][j] <= wmax * occ_i_j[i][j],
                    name="c5({i},{j})".format(i=i, j=j),
                )
        # 6 wij>=-camxwbij
        for i in range(n_c):
            for j in range(n_v):
                m.addConstr(
                    a_i_j[i][j] >= -wmax * occ_i_j[i][j],
                    name="c6({i},{j})".format(i=i, j=j),
                )
        # 7 ci<=cmaxcbi
        for i in range(n_c):
            m.addConstr(b_i[i] <= cmax * cov_i[i], name="c7({i})".format(i=i))
        # 8 ci>=-cmaxcbi
        for i in range(n_c):
            m.addConstr(b_i[i] >= -cmax * cov_i[i], name="c8({i})".format(i=i))
        # 9 sum wbij>=cbi
        for i in range(n_c):
            m.addConstr(
                sum(occ_i_j[i][j] for j in range(n_v)) >= cov_i[i],
                name="c4({i})".format(i=i),
            )

        m.optimize()

        if m.status == GRB.Status.OPTIMAL:
            inequalitys = []
            for i in range(n_c):
                getvariables = [Symbol(domain.variables[j], REAL) for j in range(n_v)]
                rightside = Real(b_i[i].x + 10 ** -5)
                inequalitys.append(
                    GE(
                        rightside,
                        Plus(Real(a_i_j[i][j].x) * getvariables[j] for j in range(n_v)),
                    )
                )
            theory = And(t for t in inequalitys)

            return theory

        else:
            raise NoFormulaFound(data, labels)
