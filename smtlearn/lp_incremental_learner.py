import time

import pysmt.shortcuts as smt
from pysmt.exceptions import InternalSolverError

import observe
from learner import Learner
from gurobipy import *

from z3.z3types import Z3Exception


class IncrementalObserver(observe.SpecializedObserver):
    def observe_initial(self, data, labels, initial_indices):
        raise NotImplementedError()

    def observe_iteration(self, data, labels, formula, new_active_indices, solving_time, selection_time):
        raise NotImplementedError()


class IncrementalLearner(Learner):
    def __init__(self, name, selection_strategy, smt_solver=False):
        """
        Initializes a new incremental learner
        :param str name: The learner name
        :param SelectionStrategy selection_strategy: The selection strategy
        """
        Learner.__init__(self, "incremental_{}".format(name))
        self.selection_strategy = selection_strategy
        self.observer = observe.DispatchObserver()
        self.smt_solver = smt_solver

    def add_observer(self, observer):
        self.observer.add_observer(observer)

    def learn(self, domain, data, initial_indices=None):
        if self.smt_solver:
            with smt.Solver() as solver:
                print("here")
                formula = self.incremental_loop(domain, data, initial_indices, solver)
        else:
            formula = self.incremental_loop(domain, data, initial_indices, None)
            #print(formula)

        return  formula

    def incremental_loop(self, domain, data, initial_indices, solver):

        active_indices = list(range(len(data))) if initial_indices is None else initial_indices
        all_active_indices = active_indices

        self.observer.observe("initial", active_indices)
        #solver1=self.setup(None,domain,data,all_active_indices)
        print(solver)
        formula = None
        while len(active_indices) > 0:
            solving_start = time.time()
            if solver == None:
                formula = self.learn_partial(solver, domain, data, all_active_indices)
            else:
                formula = self.learn_partial(solver, domain, data, active_indices)


            solving_time = time.time() - solving_start

            selection_start = time.time()
            new_active_indices = \
                self.selection_strategy.select_active(domain, data, formula, all_active_indices)
            active_indices = list(new_active_indices)
            all_active_indices += active_indices
            #print(len(all_active_indices))

            selection_time = time.time() - selection_start
            self.observer.observe("iteration", formula, active_indices, solving_time, selection_time)
        return formula

    def learn_partial(self, solver, domain, data, labels, new_active_indices):
        raise NotImplementedError()
