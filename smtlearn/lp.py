from __future__ import print_function

import pysmt.shortcuts as smt
from pysmt.typing import REAL

from incremental_learner import IncrementalLearner



class LPLearner(IncrementalLearner):
    def __init__(self, half_space_count, selection_strategy, allow_negations=True, symmetries=""):
        IncrementalLearner.__init__(self, "cnf_smt", selection_strategy)
        self.half_space_count = half_space_count
        self.allow_negations = allow_negations
        self.symmetries = symmetries

    def learn_partial(self, solver, domain, data, new_active_indices):

        # Constants
        n_r = len(domain.real_vars)
        n_h = self.half_space_count
        n_d = len(data)
        real_features = [[row[v] for v in domain.real_vars] for row, _ in data]

        labels = [row[1] for row in data]

        # Variables
        a_hr = [[smt.Symbol("a_hr[{}][{}]".format(h, r), REAL) for r in range(n_r)] for h in range(n_h)]
        b_h = [smt.Symbol("b_h[{}]".format(h), REAL) for h in range(n_h)]

        # Aux variables
        s_ih = [[smt.Symbol("s_ih[{}][{}]".format(i, h)) for h in range(n_h)] for i in range(n_d)]

        # Constraints
        for i in new_active_indices:
            x_r, label = real_features[i], labels[i]

            for h in range(n_h):
                sum_coefficients = smt.Plus([a_hr[h][r] * smt.Real(x_r[r]) for r in range(n_r)])
                solver.add_assertion(smt.Iff(s_ih[i][h], sum_coefficients  <= b_h[h]))

                #if "n" in self.symmetries:
                    #solver.add_assertion(smt.Or(smt.Equals(b_h[h], smt.Real(1.0))| smt.Equals(b_h[h], smt.Real(0.0))))
                 #   solver.add_assertion(smt.Equals(b_h[h], smt.Real(1.0)) | smt.Equals(b_h[h], smt.Real(0.0)))

                    #solver.add_assertion(smt.Or(smt.Equals(b_h[h], smt.Real(1.0)), smt.Equals(b_h[h], smt.Real(0.0))))
            if "n" in self.symmetries:
                for h in range(n_h):
                    solver.add_assertion(smt.Equals(b_h[h], smt.Real(1.0)) | smt.Equals(b_h[h], smt.Real(0.0))| smt.Equals(b_h[h], smt.Real(-1.0)))

            if label:
                solver.add_assertion(smt.And([s_ih[i][h] for h in range(n_h)]))
            else:
                solver.add_assertion(smt.Or([~s_ih[i][h] for h in range(n_h)]))
        #solver.set(timeout=1)
        solver.solve()
        model = solver.get_model()

        x_vars = [domain.get_symbol(domain.real_vars[r]) for r in range(n_r)]
        half_spaces = [
            smt.Plus([model.get_value(a_hr[h][r]) * x_vars[r] for r in range(n_r)]) <= model.get_value(b_h[h])
            for h in range(n_h)
        ]

        return smt.And(half_spaces)
        #return return_dict