import random
import numpy as np

from autodora.experiment import Experiment, Parameter, Result, derived
from typing import Union

from incal.generator import SyntheticFormula
from incal import Formula
from incal.k_cnf_smt_learner import KCnfSmtLearner
from incal.violations.core import RandomViolationsStrategy


class LearnExperiment(Experiment):
    formula_filename = Parameter(
        str, None, "The filename of the (synthetic) formula file"
    )
    data_filename = Parameter(str, None, "The filename of the examples")
    labels_filename = Parameter(str, None, "The filename of the labels")

    learner = Parameter(str, None, "The type of learner (options: cnf.[mhv1])")
    initial = Parameter(str, "random.20", "The strategy to pick the initial examples")
    selection = Parameter(
        str, "random.10", "The strategy to pick the violating examples"
    )

    initial_k = Parameter(int, 1, "The initial number of constraints")
    initial_h = Parameter(int, 0, "The initial number of inequalities")

    cost_k = Parameter(float, 1, "The complexity cost of adding a constraints")
    cost_h = Parameter(float, 1, "The complexity cost of adding an inequality")

    seed = Parameter(int, None, "The seed to run the experiment with")
    log = Parameter(str, None, "The filename of the log file to log to")

    k = Result(int, None, "The number of constraints in the solution")
    h = Result(int, None, "The number of inequalities in the solution")
    learned_formula = Result(object, None, "The learned formula")

    @derived(cache=False)
    def formula(self) -> Union[Formula, SyntheticFormula]:
        return Formula.from_file(self["formula"])

    @derived(cache=True)
    def gen_k(self):
        raise NotImplementedError()

    @derived(cache=True)
    def gen_h(self):
        raise NotImplementedError()

    @derived(cache=True)
    def gen_l(self):
        raise NotImplementedError()

    def run_internal(self):
        random.seed(self["seed"])
        np.random.seed(self["seed"])

        formula = Formula.from_file(self["formula_filename"])
        data = np.load(self["data_filename"])
        labels = np.load(self["labels_filename"])

        learner_type, symmetries = self["learner"].split(".")
        assert learner_type == "cnf"

        def cnf_factory(k, h, selection_strategy):
            return KCnfSmtLearner(k, h, selection_strategy, symmetries=symmetries)

        initial_type, initial_count = self["initial"].split(".")
        assert initial_type == "random"

        def random_selection(indices):
            return random.sample(indices, int(initial_count))

        selection_type, selection_count = self["selection"].split(".")
        assert selection_type == "random"

        learn(
            formula.domain,
            data,
            labels,
            cnf_factory,
            random_selection,
            RandomViolationsStrategy(int(selection_count)),
            self["initial_k"],
            self["initial_h"],
            self["cost_k"],
            self["cost_h"],
            self["log"],
        )


LearnExperiment.enable_cli()
