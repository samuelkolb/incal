from __future__ import print_function


import random
import time


import matplotlib as mpl  # comment
from pysmt.shortcuts import *

import plotting
from inc_logging import LoggingObserver
from incremental_learner import AllViolationsStrategy, RandomViolationsStrategy, IncrementalObserver, \
    WeightedRandomViolationsStrategy, MaxViolationsStrategy
from parameter_free_learner import learn_bottom_up
from smt_check import SmtChecker
from parse import smt_to_nested, nested_to_smt

from smt_print import pretty_print
import json
import os

# from virtual_data import OneClassStrategy

mpl.use('TkAgg')
from incalpsmt import LPLearner
from incalpmilp import LPLearnermilp
from reducedmilp import smallmilp
from pywmi import evaluate, Domain, Density
import csv
import pywmi
from pywmi.sample import uniform
from lp_problems import cuben, simplexn, pollutionreduction, police,hexagon,simple_2DProblem

from pebble import ProcessPool
from concurrent.futures import TimeoutError
from scipy.spatial import ConvexHull
from milpcs import MILPCS



def evaluate_assignment(problem, assignment):
    # substitution = {problem.domain.get_symbol(v): val for v, val in assignment.items()}
    # return problem.theory.substitute(substitution).simplify().is_true()
    return SmtChecker(assignment).check(problem.theory)


def evaluate_assignment2(problem, assignment):
    # substitution = {problem.domain.get_symbol(v): val for v, val in assignment.items()}
    # return problem.theory.substitute(substitution).simplify().is_true()
    return SmtChecker(assignment).check(problem)

def normalisation_rightside(theory):
    inequalitys = []
    if len(theory.get_atoms()) == 1:
        if theory.arg(1).constant_value() > 0:
            constant = theory.arg(1)
            leftside = theory.arg(0)
            le = Real(constant.constant_value() / constant.constant_value())

            inequalitys.append(GE(le, Plus(
                [leftside.arg(j).arg(0).constant_value() / constant.constant_value() * leftside.arg(j).arg(1) for j in
                 range(len(leftside.args()))])))
        elif theory.arg(1).constant_value() < 0:
            constant = theory.arg(1)
            leftside = theory.arg(0)
            le = Real(constant.constant_value() / constant.constant_value())
            inequalitys.append(LE(le, Plus(
                [leftside.arg(j).arg(0).constant_value() / constant.constant_value() * leftside.arg(j).arg(1) for j in
                 range(len(leftside.args()))])))
        else:
            return theory

    else:
        for i in range(len(theory.get_atoms())):
            if theory.arg(i).arg(1).constant_value() > 0:
                constant = theory.arg(i).arg(1)
                leftside = theory.arg(i).arg(0)

                le = Real(constant.constant_value() / constant.constant_value())

                inequalitys.append(GE(le, Plus(
                    [leftside.arg(j).arg(0).constant_value() / constant.constant_value() * leftside.arg(j).arg(1) for j
                     in range(len(leftside.args()))])))
            elif theory.arg(i).arg(1).constant_value() < 0:
                constant = theory.arg(i).arg(1)
                leftside = theory.arg(i).arg(0)

                le = Real(constant.constant_value() / constant.constant_value())

                inequalitys.append(LE(le, Plus(
                    [leftside.arg(j).arg(0).constant_value() / constant.constant_value() * leftside.arg(j).arg(1) for j
                     in range(len(leftside.args()))])))
            else:
                constant = theory.arg(i).arg(1)
                leftside = theory.arg(i).arg(0)

                le = Real(constant.constant_value())

                inequalitys.append(GE(le, Plus([leftside.arg(j).arg(0).constant_value() * leftside.arg(j).arg(1) for j in
                                                range(len(leftside.args()))])))

    normalisedtheory = And(inequalitys)
    return normalisedtheory

def sample(problem, n, seed=None):
    """
    Sample n instances for the given problem
    :param problem: The problem to sample from
    :param n: The number of samples
    :param seed: An optional seed
    :return: A list containing n samples and their label (True iff the sample satisfies the theory and False otherwise)
    """
    # TODO Other distributions
    samples = []
    for i in range(n):
        instance = dict()
        for v in problem.domain.variables:
            if problem.domain.var_types[v] == REAL:
                lb, ub = problem.domain.var_domains[v]
                instance[v] = random.uniform(lb, ub)
            elif problem.domain.var_types[v] == BOOL:
                instance[v] = True if random.random() < 0.5 else False
            else:
                raise RuntimeError("Unknown variable type {} for variable {}", problem.domain.var_types[v], v)
        samples.append((instance, evaluate_assignment(problem, instance)))

    return samples


def sample_half_half(problem, n, seed=None):
    """
    Sample n instances for the given problem
    :param problem: The problem to sample from
    :param n: The number of samples
    :param seed: An optional seed
    :return: A list containing n samples and their label (True iff the sample satisfies the theory and False otherwise)
    """
    # TODO Other distributions
    samplespositive = []
    samplesnegative = []
    while len(samplespositive) < n / 2 or len(samplesnegative) < n / 2:
        instance = dict()
        for v in problem.domain.variables:
            if problem.domain.var_types[v] == REAL:
                lb, ub = problem.domain.var_domains[v]
                instance[v] = random.uniform(lb, ub)
            else:
                raise RuntimeError("Unknown variable type {} for variable {}", problem.domain.var_types[v], v)
        # print(evaluate_assignment(problem, instance))

        if evaluate_assignment(problem, instance) and len(samplespositive) < n / 2:
            samplespositive.append((instance, evaluate_assignment(problem, instance)))
        elif len(samplesnegative) < n / 2:
            samplesnegative.append((instance, evaluate_assignment(problem, instance)))

    samples = samplesnegative + samplespositive
    random.shuffle(samples)
    return samples


class TrackingObserver(IncrementalObserver):

    def observe_initial(self, initial_indices):
        self.initials = [i for i in initial_indices]

    def observe_iteration(self, theory, new_active_indices, solving_time, selection_time):
        for i in new_active_indices:
            self.initials.append(i)

    def __init__(self, initial_indices):
        self.initials = [i for i in initial_indices]


class ParameterObserver:

    def __init__(self):
        self.seednumber = 0
        self.dimension = 0
        self.samplesize = 0
        self.time = 0
        self.numberofconstrains = 0
        self.numberofconstraintsmodel = 0
        self.overallruns = []
        self.tpr = 0
        self.tnr = 0

    def updating(self):
        self.overallruns.append(
            [self.seednumber, self.dimension, self.samplesize, self.time, self.numberofconstrains,
             self.numberofconstraintsmodel, self.tpr, self.tnr])

def loadintermediatresult(directory, version):
    lastresult= len(os.listdir(directory)) - 1

    with open(directory + version + str(lastresult) + ".csv") as f:
        data = [json.loads(line) for line in f]
    last = data[-1]
    return nested_to_smt(last["theory"])



def get_dt_weights(m, data):
    import dt_selection
    dt_weights = [min(d.values()) for d in dt_selection.get_distances(m.domain, data)] #sum
    return dt_weights


def learn_parameter_free(problem, data, seed,method="smt",synthetic=False):
    #feat_x, feat_y = problem.domain.real_vars[:2]# needed for plotting observer


    w = get_dt_weights(problem, data)
    p = zip(range(len(data)), [w[i] for i in range(len(data))])
    p = sorted(p, key=lambda t: t[1])
    d = [t[0] for t in p[0:20]]
    o = TrackingObserver(d)


    def learn_inc(_data, i, _k, _h):



        if method=="smt":
            learner=LPLearner(_h, MaxViolationsStrategy(1, w))
            #learner = LPLearner(_h, RandomViolationsStrategy(1))
            #learner = LPLearner(_h, MaxViolationsStrategy(1, w) ,symmetries="n")
        else:
            #learner = LPLearnermilp(_h, RandomViolationsStrategy(1))
            learner = LPLearnermilp(_h, MaxViolationsStrategy(1, w))

        if synthetic:
            dir_nameO = "../output/{}/{}/{}/{}/{}/observer{}:{}:{}.csv".format(method,"syn", len(problem.theory.get_atoms()),len(data), seed, len(data),
                                                                    len(problem.domain.variables), _h)
        else:
            dir_nameO = "../output/{}/{}/{}/{}/observer{}:{}:{}.csv".format(method,problem.name, len(data), seed, len(data),
                                                                     len(problem.domain.variables), _h)


        #img_name = "{}_{}_{}_{}_{}_{}_{}".format(learner.name, len(problem.domain.variables), i, _k, _h, len(data),
         #                                        seed)
        #dir_nameP = "../output/{}/{}/{}/".format(problem.name, len(data), seed, len(data))
        #learner.add_observer(plotting.PlottingObserver(problem.domain, data, dir_nameP, img_name, feat_x, feat_y))

        learner.add_observer(LoggingObserver(dir_nameO, verbose=False))

        learner.add_observer(o)

        if len(o.initials) > 20:#20
            initial_indices = random.sample(o.initials, 20)
        else:
            initial_indices = o.initials

        learned_theory = learner.learn(problem.domain, data, initial_indices)

        print("Learned theory:\n{}".format(pretty_print(learned_theory)))
        return learned_theory

    return learn_bottom_up(data, learn_inc, 1000000,1)


def learndouble(problem,data,seed):
    w = get_dt_weights(problem, data)
    p = zip(range(len(data)), [w[i] for i in range(len(data))])
    p = sorted(p, key=lambda t: t[1])
    d = [t[0] for t in p[0:20]] #[-20:]
    print("1",d)
    da=[data[i] for i in d]
    x,y,z=learn_parameter_free(problem,da,seed)
    print(z)

    return learn_parameter_free(problem,data,seed,z)


def dictionary(generated):
    samples=[]
    for s,l in zip(generated.samples, generated.labels):

        variables = {}
        for i ,p in zip(generated.formula.domain.variables ,range(len(generated.formula.domain.variables))):
            variables[i]=s[p].item()
        samples.append((variables,l))
    return samples






def testing(nbseeds, nbdimensions, nbsamplesize, modelinput,incal="smt"):
    parameter = ParameterObserver()
    evalation_sample_count = 1000000

    for i in nbdimensions:

        model = modelinput(i)
        dir_name_for_equaltions = "../output/{}/{}/{}/".format(incal,model.name, "theories_learned")
        if not os.path.exists(dir_name_for_equaltions):
            os.makedirs(dir_name_for_equaltions)

        for j in nbsamplesize:
            parameter.seednumber = 0
            equationsfound = []
            seeds = random.sample(range(10000000), nbseeds)
            dir_name_for_samplesize = "../output/{}/{}/{}".format(incal,model.name, j)
            if not os.path.exists(dir_name_for_samplesize):
                os.makedirs(dir_name_for_samplesize)

            for k in seeds:

                dir_name_for_seed = "../output/{}/{}/{}/{}".format(incal,model.name, j, k)
                if not os.path.exists(dir_name_for_seed):
                    os.makedirs(dir_name_for_seed)
                print(time.asctime(), i, j, k)
                parameter.seednumber = parameter.seednumber + 1
                parameter.dimension = i
                parameter.samplesize = j

                parameter.numberofconstrains = len(model.theory.get_atoms())

                data = sample_half_half(model, j, k)

                start = time.time()
                if incal == "smt":
                    with ProcessPool() as pool:
                        future = pool.schedule(learn_parameter_free, args=[model, data, k,incal], timeout=90*60)

                    try:
                        result = future.result()
                        learned_theory = result[0]
                        numberofconstraints = result[2]
                        parameter.time = time.time() - start
                    except TimeoutError:
                        dir_nameO = "../output/{}/{}/{}/{}/".format(incal,model.name, len(data), k)
                        dir_run = "observer{}:{}:".format(len(data), len(model.domain.variables))
                        parameter.time = False
                        learned_theory = loadintermediatresult(dir_nameO, dir_run)
                        numberofconstraints = len(learned_theory.get_atoms())
                        print("Function took longer than seconds")
                elif incal == "milp":
                    with ProcessPool() as pool:
                        future = pool.schedule(learn_parameter_free, args=[model, data, k,incal], timeout=90*60)

                    try:
                        result = future.result()
                        learned_theory = result[0]
                        numberofconstraints = result[2]
                        parameter.time = time.time() - start
                    except TimeoutError:
                        dir_nameO = "../output/{}/{}/{}/{}/".format(incal,model.name, len(data), k)
                        dir_run = "observer{}:{}:".format(len(data), len(model.domain.variables))
                        parameter.time = False
                        learned_theory = loadintermediatresult(dir_nameO, dir_run)
                        numberofconstraints = len(learned_theory.get_atoms())
                        print("Function took longer than seconds")
                elif incal=="smallmilp":
                    learned_theory, numberofconstraints, timelimit = smallmilp(model.domain, data, 14)
                    if timelimit == True:
                        parameter.time = 90*60

                    else:
                        parameter.time = time.time() - start
                elif incal=="bigmilp":
                    learned_theory, numberofconstraints, timelimit = MILPCS(model.domain, data, 14)
                    if timelimit == True:
                        parameter.time = 90*60

                    else:
                        parameter.time = time.time() - start

                parameter.numberofconstraintsmodel = numberofconstraints
                parameter.tpr = pywmi.RejectionEngine(model.domain, model.theory, Real(1.0),
                                                      evalation_sample_count).compute_probability(learned_theory)
                parameter.tnr = pywmi.RejectionEngine(model.domain, ~model.theory, Real(1.0),
                                                      evalation_sample_count).compute_probability(~learned_theory)

                parameter.updating()
                equationsfound.append([smt_to_nested(learned_theory)])

            dir_name_for_equaltions = "../output/{}/{}/{}/D:{}S:{}.csv".format(incal,model.name, "theories_learned", i, j)
            with open(dir_name_for_equaltions, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(equationsfound)

    dir_name_results = "../output/{}/{}/{}{}{}.csv".format(incal,model.name, "Results", nbdimensions, nbsamplesize)
    with open(dir_name_results, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(parameter.overallruns)

    return parameter.overallruns


def testingpractical(nbseeds, nbsamplesize, modelinput,incal="smt"):
    parameter = ParameterObserver()
    evalation_sample_count = 1000000
    model = modelinput()
    dir_name_for_equaltions = "../output/{}/{}/{}/".format(incal,model.name, "theories_learned")
    if not os.path.exists(dir_name_for_equaltions):
        os.makedirs(dir_name_for_equaltions)

    for j in nbsamplesize:
        parameter.seednumber = 0
        equationsfound = []
        seeds = random.sample(range(10000000), nbseeds)
        dir_name_for_samplesize = "../output/{}/{}/{}".format(incal,model.name, j)
        if not os.path.exists(dir_name_for_samplesize):
            os.makedirs(dir_name_for_samplesize)
        for k in seeds:

            dir_name_for_seed = "../output/{}/{}/{}/{}".format(incal,model.name, j, k)
            if not os.path.exists(dir_name_for_seed):
                os.makedirs(dir_name_for_seed)
            print(time.asctime(), j, k)
            parameter.seednumber = parameter.seednumber + 1
            parameter.samplesize = j

            parameter.numberofconstrains = len(model.theory.get_atoms())
            data = sample_half_half(model, j, k)

            start = time.time()
            if incal == "smt":
                with ProcessPool() as pool:
                    future = pool.schedule(learn_parameter_free, args=[model, data, k, incal], timeout=60*90)

                try:
                    result = future.result()
                    learned_theory = result[0]
                    numberofconstrains = result[2]
                    parameter.time = time.time() - start
                except TimeoutError:
                    dir_nameO = "../output/{}/{}/{}/{}/".format(incal, model.name, len(data), k)
                    dir_run = "observer{}:{}:".format(len(data), len(model.domain.variables))
                    parameter.time = False
                    learned_theory = loadintermediatresult(dir_nameO, dir_run)
                    numberofconstrains = len(learned_theory.get_atoms())
                    print("Function took longer than seconds")
            elif incal == "milp":
                with ProcessPool() as pool:
                    future = pool.schedule(learn_parameter_free, args=[model, data, k, incal], timeout=90 * 60)

                try:
                    result = future.result()
                    learned_theory = result[0]
                    numberofconstrains = result[2]
                    parameter.time = time.time() - start
                except TimeoutError:
                    dir_nameO = "../output/{}/{}/{}/{}/".format(incal, model.name, len(data), k)
                    dir_run = "observer{}:{}:".format(len(data), j)
                    parameter.time = False
                    learned_theory = loadintermediatresult(dir_nameO, dir_run)
                    numberofconstrains = len(learned_theory.get_atoms())
                    print("Function took longer than seconds")
            elif incal == "smallmilp":
                learned_theory, numberofconstrains, timelimit = smallmilp(model.domain, data, 14)
                if timelimit == True:
                    parameter.time = 90 * 60

                else:
                    parameter.time = time.time() - start
            elif incal == "bigmilp":
                learned_theory, numberofconstrains, timelimit = MILPCS(model.domain, data, 14)
                if timelimit == True:
                    parameter.time = 90 * 60

                else:
                    parameter.time = time.time() - start


            parameter.numberofconstraintsmodel = numberofconstrains

            if learned_theory != None:
                parameter.tpr = pywmi.RejectionEngine(model.domain, model.theory, Real(1.0),
                                                  evalation_sample_count).compute_probability(learned_theory)
                parameter.tnr = pywmi.RejectionEngine(model.domain, ~model.theory, Real(1.0),
                                                  evalation_sample_count).compute_probability(~learned_theory)
                equationsfound.append([smt_to_nested(learned_theory)])
            else:
                parameter.tpr=0
                parameter.tnr=0

            parameter.updating()


        dir_name_for_equaltions = "../output/{}/{}/{}/S:{}.csv".format(incal,model.name, "theories_learned", j)

        with open(dir_name_for_equaltions, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(equationsfound)

    dir_name = "../output/{}/{}/{}{}{}.csv".format(incal,model.name, "Results", model.name, nbsamplesize)
    with open(dir_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(parameter.overallruns)

    return parameter.overallruns


def gen_polytope(domain, n, h, ratio, test_sample_count):
    points = uniform(domain, n)
    convex_hull = ConvexHull(points)
    equations = convex_hull.equations
    inequalities = []
    for i in range(len(equations)):
        inequality = Real(equations[i, -1].item())
        for j in range(len(domain.real_vars)):
            inequality += domain.get_symbol(domain.real_vars[j]) * Real(equations[i, j].item())
        inequalities.append(inequality <= Real(0))
    test_samples = uniform(domain, test_sample_count)
    convex_hull_formula = And(*random.sample(inequalities, h))
    labels = evaluate(domain, convex_hull_formula, test_samples)
    if 1 - ratio <= sum(labels) / len(labels) <= ratio:
        print(sum(labels) / len(labels))
        return convex_hull_formula
    else:
        print("Failed")
        raise RuntimeError()

from genratorsyn import generate_domain
from problem import Problem

def testingsyn(nbseeds, numberofvariables, nbsamplesize, numberofhalfspaces, incal="smt"):
    #parameter = ParameterObserver()
    sample_countev = 1000000
    from genratorsyn import generate_synthetic_dataset
    for a in numberofhalfspaces:
        parameter = ParameterObserver()
        for i in numberofvariables:

            dir_name_for_equaltions = "../output/{}/{}/{}/{}/".format(incal,"syn",a, "theories_learned")
            if not os.path.exists(dir_name_for_equaltions):
                os.makedirs(dir_name_for_equaltions)

            for j in nbsamplesize:
                parameter.seednumber = 0
                equationsfound = []
                seeds = random.sample(range(10000000), nbseeds)
                dir_name_for_samplesize = "../output/{}/{}/{}/{}".format(incal,"syn",a, j)
                if not os.path.exists(dir_name_for_samplesize):
                    os.makedirs(dir_name_for_samplesize)

                for k in seeds:

                    dir_name_for_seed = "../output/{}/{}/{}/{}/{}".format(incal,"syn",a, j, k)
                    if not os.path.exists(dir_name_for_seed):
                        os.makedirs(dir_name_for_seed)
                    print(time.asctime(), i, j, k,a)
                    parameter.seednumber = parameter.seednumber + 1
                    parameter.dimension = i
                    parameter.samplesize = j

                    parameter.numberofconstrains = a

                    domain = generate_domain(0, i)
                    sample_count = 10000

                    s=2*(i-1)
                    formula = None

                    for m in range(1000):
                        try:
                            formula = gen_polytope(domain, s, a, 0.7, sample_count)
                            print("HERE")
                            break
                        except RuntimeError:
                            continue

                    if formula==None:
                        continue


                    synmodel = Problem(domain, formula, "Syn")
                    data=sample(synmodel,j)



                    start = time.time()
                    if incal == "smt":
                        with ProcessPool() as pool:
                            future = pool.schedule(learn_parameter_free, args=[synmodel, data, k,incal,True], timeout=30*60)

                        try:
                            result = future.result()
                            learned_theory = result[0]
                            km = result[1]
                            numberofconstrains = result[2]
                            parameter.time = time.time() - start
                        except TimeoutError:
                            dir_nameO = "../output/{}/{}/{}/{}/{}/".format(incal,"syn",a, len(data), k)
                            dir_run = "observer{}:{}:".format(len(data), i)
                            parameter.time = 90*60
                            learned_theory = loadintermediatresult(dir_nameO, dir_run)
                            numberofconstrains = len(learned_theory.get_atoms())
                            print("Function took longer than seconds")
                    elif incal == "milp":
                        with ProcessPool() as pool:
                            future = pool.schedule(learn_parameter_free, args=[synmodel, data, k,incal,True], timeout=30*60)

                        try:
                            result = future.result()
                            learned_theory = result[0]
                            numberofconstrains = result[2]
                            parameter.time = time.time() - start
                        except TimeoutError:
                            dir_nameO = "../output/{}/{}/{}/{}/{}/".format(incal,"syn",a, len(data), k)
                            dir_run = "observer{}:{}:".format(len(data), i)
                            parameter.time = 90*60
                            learned_theory = loadintermediatresult(dir_nameO, dir_run)
                            numberofconstrains = len(learned_theory.get_atoms())
                            print("Function took longer than seconds")
                    elif incal =="smallmilp":
                        learned_theory, numberofconstrains, timelimit = smallmilp(synmodel.theory.domain, data, 14)
                        if timelimit == True:
                            parameter.time = 90 * 60

                        else:
                            parameter.time = time.time() - start
                    elif incal =="bigmilp":
                        learned_theory, numberofconstrains, timelimit = MILPCS(synmodel.theory.domain, data, 14)
                        if timelimit == True:
                            parameter.time = 90*60

                        else:
                            parameter.time = time.time() - start

                    parameter.numberofconstraintsmodel = numberofconstrains
                    if learned_theory != None:
                        parameter.tpr = pywmi.RejectionEngine(synmodel.domain,synmodel.theory, Real(1.0),
                                                              sample_countev).compute_probability(
                            learned_theory)
                        parameter.tnr = pywmi.RejectionEngine(synmodel.domain, ~synmodel.theory, Real(1.0),
                                        sample_countev).compute_probability(
                            ~learned_theory)
                        equationsfound.append([smt_to_nested(learned_theory)])
                    else:
                        parameter.tpr = 0
                        parameter.tnr = 0

                    parameter.updating()

                dir_name_for_equaltions = "../output/{}/{}/{}/{}/D:{}S:{}.csv".format(incal,"syn",a, "theories_learned", i, j)
                with open(dir_name_for_equaltions, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(equationsfound)

        dir_name_results = "../output/{}/{}/{}/{}{}{}.csv".format(incal,"syn",a, "Results", numberofvariables, nbsamplesize)
        with open(dir_name_results, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(parameter.overallruns)

    return parameter.overallruns



