from __future__ import print_function

from pysmt.exceptions import InternalSolverError
from pysmt.environment import Environment
from pysmt.typing import REAL, BOOL
from z3types import Z3Exception

from incremental_learner import SelectionStrategy


class OneClassStrategy(SelectionStrategy):
    def __init__(self, regular_strategy, thresholds, tight_mode=True, class_label=True):
        self.regular_strategy = regular_strategy
        self.thresholds = thresholds
        self.tight_mode = tight_mode
        assert class_label, "Currently only the positive setting is supported"
        self.class_label = class_label
        self.environment = Environment()

    def find_violating(self, domain, data, formula):
        fm = self.environment.formula_manager
        formula = fm.normalize(formula)
        real_symbols = {name: fm.Symbol(name, REAL) for name in domain.real_vars}
        bool_symbols = {name: fm.Symbol(name, BOOL) for name in domain.bool_vars}
        symbols = real_symbols.copy()
        symbols.update(bool_symbols)
        bounds = domain.get_bounds(fm)
        thresholds = {var: fm.Real(t) for var, t in self.thresholds.items()}
        threshold = thresholds[domain.real_vars[0]]

        def boolean_agreement(ex):
            return fm.And([fm.Iff(bool_symbols[bv], fm.Bool(ex[bv])) for bv in domain.bool_vars])

        def distance(ex):
            return sum(fm.Ite(real_symbols[rv] >= fm.Real(ex[rv]),
                              real_symbols[rv] - fm.Real(ex[rv]),
                              fm.Real(ex[rv]) - real_symbols[rv])
                       for rv in domain.real_vars)

        try:
            with self.environment.factory.Solver() as solver:
                solver.add_assertion(formula)
                solver.add_assertion(bounds)
                # equalities = []
                # for row, label in data:
                #     for real_var in domain.real_vars:
                #         sym = real_symbols[real_var]
                #         val = fm.Real(row[real_var])
                #         t = fm.Real(self.thresholds[real_var])
                #         equalities.append(fm.Ite(sym >= val, fm.Equals(sym - val, t), fm.Equals(val - sym, t)))
                # solver.add_assertion(fm.Or(*equalities))
                for row, label in data:
                    if label == self.class_label:  # Positive example
                        constraint = fm.Implies(boolean_agreement(row), distance(row) >= threshold)
                    else:  # Negative example
                        constraint = fm.Implies(boolean_agreement(row), distance(row) >= threshold)
                    solver.add_assertion(constraint)

                # Distance must be equals (for faster convergence)
                solver.add_assertion(fm.Or([
                    boolean_agreement(row) & fm.Equals(distance(row), threshold)
                    for row, label in data if label == self.class_label
                ]))
                solver.solve()
                model = solver.get_model()
                example = {var: model.get_value(symbols[var]).constant_value() for var in domain.variables}
                example = {var: (float(val) if var in domain.real_vars else bool(val)) for var, val in example.items()}
        except Z3Exception:
            return None
        except InternalSolverError:
            return None
        return example

    def select_active(self, domain, data, formula, active_indices):
        regular_selection = self.regular_strategy.select_active(domain, data, formula, active_indices)
        if len(regular_selection) > 0:
            return regular_selection
        else:
            example = self.find_violating(domain, data, formula)
            if example is None:
                return []
            data.append((example, not self.class_label))
            return [len(data) - 1]


"""
There is a point e, such that for every example e': d(e, e') > t
AND(OR(d(e, e', r) > t, forall r), forall e)
"""
