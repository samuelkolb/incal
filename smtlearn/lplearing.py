from __future__ import print_function

import itertools
import math
import random
import time
from math import sqrt

import matplotlib as mpl  # comment
from pysmt.shortcuts import *

import plotting
from generator import get_sample
from inc_logging import LoggingObserver
from incremental_learner import AllViolationsStrategy, RandomViolationsStrategy, IncrementalObserver, \
    WeightedRandomViolationsStrategy, MaxViolationsStrategy
from k_cnf_smt_learner import KCnfSmtLearner
# from k_dnf_logic_learner import KDNFLogicLearner, GreedyLogicDNFLearner, GreedyMaxRuleLearner
# from k_dnf_smt_learner import KDnfSmtLearner
# from k_dnf_greedy_learner import GreedyMilpRuleLearner
from parameter_free_learner import learn_bottom_up
from problem import Domain, Problem
from smt_check import SmtChecker
from experiments import IncrementalConfig
from parse import smt_to_nested, nested_to_smt

import parse
from smt_print import pretty_print
import json
import os

# from virtual_data import OneClassStrategy

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from lp import LPLearner
import string
import csv
import pywmi
from sklearn.model_selection import KFold, cross_val_score
from lp_problems import cuben, simplexn, polutionreduction, police
from lp_wineproblem import importwine,partioning
from lp_oldfuctions import accuracy


def evaluate_assignment(problem, assignment):
    # substitution = {problem.domain.get_symbol(v): val for v, val in assignment.items()}
    # return problem.theory.substitute(substitution).simplify().is_true()
    return SmtChecker(assignment).check(problem.theory)


def evaluate_assignment2(problem, assignment):
    # substitution = {problem.domain.get_symbol(v): val for v, val in assignment.items()}
    # return problem.theory.substitute(substitution).simplify().is_true()
    return SmtChecker(assignment).check(problem)

def normilisation_rightsideto1(theory):
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
        self.dimensions = 0
        self.samplesize = 0
        self.time = 0
        self.numberofconstrains = 0
        self.numberofconstrainsmodel = 0
        self.overallruns = []
        self.tpr = 0
        self.tnr = 0

    def updating(self):
        self.overallruns.append(
            [self.seednumber, self.dimensions, self.samplesize, self.time, self.numberofconstrains,
             self.numberofconstrainsmodel, self.tpr, self.tnr])


def get_dt_weights(m, data):
    import dt_selection
    dt_weights = [min(d.values()) for d in dt_selection.get_distances(m.domain, data)]
    return dt_weights


def learn_parameter_free(problem, data, seed):
    # feat_x, feat_y = problem.domain.real_vars[:2] needed for plotting observer

    o = TrackingObserver(random.sample(list(range(len(data))), 20))

    def learn_inc(_data, i, _k, _h):

        w = get_dt_weights(problem, data)
        # learner = LPLearner(_h, RandomViolationsStrategy(10))
        # learner = LPLearner(_h, WeightedRandomViolationsStrategy(10,w))
        learner = LPLearner(_h, MaxViolationsStrategy(1, w))

        dir_nameO = "../output/{}/{}/{}/observer{}:{}:{}.csv".format(problem.name, len(data), seed, len(data),
                                                                     len(problem.domain.variables), _h)
        #img_name = "{}_{}_{}_{}_{}_{}_{}".format(learner.name, len(problem.domain.variables), i, _k, _h, len(data),
        #                                         seed)
        #learner.add_observer(plotting.PlottingObserver(problem.domain, data, dir_name, img_name, feat_x, feat_y))
        learner.add_observer(LoggingObserver(dir_nameO, verbose=False))

        learner.add_observer(o)
        if len(o.initials) > 20:
            initial_indices = random.sample(o.initials, 20)

        else:
            initial_indices = o.initials

        learned_theory = learner.learn(problem.domain, data, initial_indices)
        print("Learned theory:\n{}".format(pretty_print(learned_theory)))
        return learned_theory

    return learn_bottom_up(data, learn_inc, 1000000, 1)


def testing(nbseeds, nbdimensions, nbsamplesize, learner, modelinput):
    parameter = ParameterObserver()
    sample_count = 1000000

    for i in nbdimensions:

        model = modelinput(i)
        dir_name_for_equaltions = "../output/{}/{}/".format(model.name, "theories_learned")
        if not os.path.exists(dir_name_for_equaltions):
            os.makedirs(dir_name_for_equaltions)

        for j in nbsamplesize:
            parameter.seednumber = 0
            equationsfound = []
            seeds = random.sample(range(10000000), nbseeds)
            dir_name_for_samplesize = "../output/{}/{}".format(model.name, j)
            if not os.path.exists(dir_name_for_samplesize):
                os.makedirs(dir_name_for_samplesize)

            for k in seeds:

                dir_name_for_seed = "../output/{}/{}/{}".format(model.name, j, k)
                if not os.path.exists(dir_name_for_seed):
                    os.makedirs(dir_name_for_seed)
                print(time.asctime(), i, j, k)
                parameter.seednumber = parameter.seednumber + 1
                parameter.dimensions = i
                parameter.samplesize = j

                parameter.numberofconstrains = len(model.theory.get_atoms())

                data = sample_half_half(model, j, k)

                start = time.time()
                learned_theory, km, numberofconstrains = learner(model, data, k)
                #just insert here the milp one for testing.-> milp as in paper

                parameter.time = time.time() - start
                parameter.numberofconstrainsmodel = numberofconstrains
                # parameter.accucy=accucy(theroy=model,theorylearned=theory,data1=adata)
                parameter.tpr = pywmi.RejectionEngine(model.domain, model.theory, Real(1.0),
                                                      sample_count).compute_probability(learned_theory)
                parameter.tnr = pywmi.RejectionEngine(model.domain, ~model.theory, Real(1.0),
                                                      sample_count).compute_probability(~learned_theory)

                parameter.updating()
                equationsfound.append([smt_to_nested(learned_theory)])

            dir_name_for_equaltions = "../output/{}/{}/D:{}S:{}.csv".format(model.name, "theories_learned", i, j)
            with open(dir_name_for_equaltions, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(equationsfound)

    dir_name_results = "../output/{}/{}{}{}.csv".format(modelinput, "Results", nbdimensions, nbsamplesize)
    with open(dir_name_results, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(parameter.overallruns)

    return parameter.overallruns


def testingpractical(nbseeds, nbsamplesize, learner, modelinput):
    parameter = ParameterObserver()
    sample_count = 1000000
    model = modelinput()
    dir_name_for_equaltions = "../output/{}/{}/".format(model.name, "theories_learned")
    if not os.path.exists(dir_name_for_equaltions):
        os.makedirs(dir_name_for_equaltions)

    for j in nbsamplesize:
        parameter.seednumber = 0
        equationsfound = []
        seeds = random.sample(range(10000000), nbseeds)
        dir_name_for_samplesize = "../output/{}/{}".format(model.name, j)
        if not os.path.exists(dir_name_for_samplesize):
            os.makedirs(dir_name_for_samplesize)
        for k in seeds:

            dir_name_for_seed = "../output/{}/{}/{}".format(model.name, j, k)
            if not os.path.exists(dir_name_for_seed):
                os.makedirs(dir_name_for_seed)
            print(time.asctime(), j, k)
            parameter.seednumber = parameter.seednumber + 1
            parameter.samplesize = j

            parameter.numberofconstrains = len(model.theory.get_atoms())
            data = sample_half_half(model, j, k)

            start = time.time()
            learned_theory, km, numberofconstrains = learner(model, data, k)
            parameter.numberofconstrainsmodel = numberofconstrains
            parameter.time = time.time() - start
            parameter.tpr = pywmi.RejectionEngine(model.domain, model.theory, Real(1.0),
                                                  sample_count).compute_probability(learned_theory)
            parameter.tnr = pywmi.RejectionEngine(model.domain, ~model.theory, Real(1.0),
                                                  sample_count).compute_probability(~learned_theory)

            # parameter.accucy=accucy(theroy=model,theorylearned=theory,size=100,seed=k+1)
            parameter.updating()
            equationsfound.append([smt_to_nested(learned_theory)])

        dir_name_for_equaltions = "../output/{}/{}/S:{}.csv".format(model.name, "theories_learned", j)

        with open(dir_name_for_equaltions, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(equationsfound)

    dir_name = "../output/{}/{}{}{}.csv".format(modelinput, "Results", model.name, nbsamplesize)
    with open(dir_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(parameter.overallruns)

    return parameter.overallruns


def qualifiyingwine(lenght):
    data, head = importwine()
    labeled = partioning(data, 5, 5)
    random.shuffle(labeled)
    # lenght=round(0.8*(len(labeled)))

    train = labeled[:lenght]
    test = labeled[lenght:lenght + 200]

    variables = []
    var_types = {}
    var_domains = {}
    for i in head:
        variables.append(i[1:-1])
        var_types[i[1:-1]] = REAL
        var_domains[i[1:-1]] = (0, 1)
    domain = Domain(variables, var_types, var_domains)
    problem = Problem(domain, 0, "wine")
    theory, km, numberofconstrains = learn_parameter_free(problem, train, 3)
    a = accuracy(theorylearned=theory, data1=test)
    print(a)
    return theory, a


def import_slump():
    with open('/Users/Elias/Desktop/slump_test.data.txt', newline='') as csvfile:
        slumpdata = csv.reader(csvfile, delimiter=',', quotechar='|')
        linecount = 0
        dataoutput = []
        for row in slumpdata:
            if linecount > 0:
                row = [float(x) for x in row]
                row[8], row[1] = row[1], row[8]
                dataoutput.append(row[1:9])
            else:
                row[8], row[1] = row[1], row[8]
                header = row[1:9]
            linecount += 1

    return dataoutput, header



def definingslumpclass(data):
    def normilisation(x, min, max, p):
        return (x - min[p]) / (max[p] - min[p])

    # minn=list(map(min, zip(*data)))
    # maxx=list(map(max, zip(*data)))

    # for i in data:             #aktivate normlisation again
    #   for j in range(1,len(i)):
    #      i[j]=normilisation(i[j],minn,maxx,j)

    for i in data:
        lenght = len(i)
        for j in range(1, lenght):
            for k in range(j, lenght):
                i.append(i[j] * i[k])

    for i in data[:]:
        if (i[0] * 10) >= 10 and (i[0] * 10) <= 40:
            i[0] = 1
        elif (i[0] * 10) >= 50 and (i[0] * 10) <= 90:
            i[0] = 2
        elif (i[0] * 10) >= 100 and (i[0] * 10) <= 150:
            i[0] = 3
        elif (i[0] * 10) >= 160 and (i[0] * 10) <= 210:
            i[0] = 4
        elif (i[0] * 10) >= 220:
            i[0] = 5
        else:
            i[0] = 0

    return data


def creatingvaraiblenames(head):
    head = [h.replace(" ", "") for h in head]
    new = head[:]

    for i in range(1, len(head)):
        for j in range(i, len(head)):
            new.append(head[i] + head[j])
    return new


def mergeslumpdataandhead(head, data):
    outputlist = []
    for i in data:
        outputlist.append(dict(zip(head, i)))

    return outputlist


def classification(cl, data):
    outputlist = []
    for i in data:
        if i["SLUMP(cm)"] == cl:
            del i["SLUMP(cm)"]
            outputlist.append((i, True))
        else:
            del i["SLUMP(cm)"]
            outputlist.append((i, False))
    return outputlist


def slamp(clas):
    postivies = 0

    data, variablenames = import_slump()
    data = definingslumpclass(data)
    variablenames = creatingvaraiblenames(variablenames)
    q = mergeslumpdataandhead(variablenames, data)
    labeled = classification(clas, q)
    # random.shuffle(labeled)

    for i in labeled:
        if i[1]:
            postivies += 1

    variables = []
    var_types = {}
    var_domains = {}

    for i in variablenames[1:]:
        variables.append(i)
        var_types[i] = REAL
        var_domains[i] = (None, None)
    domain = Domain(variables, var_types, var_domains)
    problem = Problem(domain, 0, "slump")
    dir_name_for_equaltions = "../output/{}/{}/".format(problem.name, "theories_learned")
    if not os.path.exists(dir_name_for_equaltions):
        os.makedirs(dir_name_for_equaltions)

    k_fold = KFold(n_splits=5)
    overallruns = []
    foldcount = 1
    equationsfound = []
    for train_indices, test_indices in k_fold.split(labeled):
        train = [labeled[i] for i in train_indices]
        test = [labeled[i] for i in test_indices]

        start = time.time()
        learned_theory, km, hm = learn_parameter_free(problem, train, 2)
        timee = time.time() - start
        overallruns.append([clas, foldcount, timee, accuracy(theorylearned=learned_theory, data1=test),
                            pretty_print(normilisation_rightsideto1(learned_theory))])
        foldcount += 1
        print(overallruns)
        equationsfound.append([smt_to_nested(learned_theory)])

    dir_name_for_equaltions = "../output/{}/{}/class:{}.csv".format(problem.name, "theories_learned", clas)

    with open(dir_name_for_equaltions, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(equationsfound)

    dir_name = "../output/{}/{}{}.csv".format(problem.name, "Results", clas)
    with open(dir_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(overallruns)

    return overallruns, postivies


model = polutionreduction()
data11 = sample_half_half(model, 100)
learned_theory, km, hm = learn_parameter_free(model, data11, 5)
# x=smt_to_nested(learned_theory)
# y=nested_to_smt(x)
# sample_count = 1000000
# tpr = pywmi.RejectionEngine(model.domain, model.theory, Real(1.0), sample_count).compute_probability(learned_theory)
# tnr = pywmi.RejectionEngine(model.domain, ~model.theory, Real(1.0), sample_count).compute_probability(~learned_theory)
# print(tpr, tnr)


# importing the equation results
# with open("/Users/Elias/Documents/GitHub/smtlearn/output/cuben/theories_learned/D:[2]S:[50].csv") as csv_file:
#   csv_reader = csv.reader(csv_file, delimiter=',')
#  line_count = 0
# for row in csv_reader:
#    y=nested_to_smt(row[0])


# print(testing(2,[2],[50],learn_parameter_free,cuben))
# print(testingpractical(2,[50,100],learn_parameter_free,polutionreduction))


random.seed(65)

# print(testing(10,[2,3],[20,30,40,50,100,200,300],learn_parameter_free,cuben))
# print(testing(10,[2,3],[20,30,40,50,100,200,300],learn_parameter_free,simplexn))

# print(testingpractical(10,[20,30,40,50,100,200,300],learn_parameter_free,polutionreduction))
# print(testingpractical(10,[20,30,40,50,100,200,300],learn_parameter_free,police))

# print(testing(10,[2,3],[400,500],learn_parameter_free,cuben))
# print(testing(10,[2,3],[400,500],learn_parameter_free,simplexn))

# print(testingpractical(10,[500,1000,2000],learn_parameter_free,polutionreduction))
# print(testingpractical(10,[500,1000,2000],learn_parameter_free,police))


# start here again
# print(testing(10,[4],[20,30,40,50,100,200,300],learn_parameter_free,cuben))
# print(testing(10,[4],[20,30,40,50,100,200,300],learn_parameter_free,simplexn))

# print(testing(10,[4],[20,30,40,50,100],learn_parameter_free,cuben()))
# print(testing(10,[4],[20,30,40,50,100],learn_parameter_free,simplexn()))


# TODO  5/6 dimensions

# running slump
# for i in [1,3,4,5]:
#   x,y,z=slamp(i)

# print(testing(10,[4],[20,30,40,50,100,200,300],learn_parameter_free,cuben))
# print(testing(10,[4],[20,30,40,50,100,200,300],learn_parameter_free,simplexn))

# print(testing(10,[4],[20,30,40,50,100],learn_parameter_free,cuben()))
# print(testing(10,[4],[20,30,40,50,100],learn_parameter_free,simplexn()))


# x,y=import_slump()
# x=multi(x)
# y=header(y)
# e=[]
# for j in range(len(y)):
#   l=[]
#  for i in x:
#     l.append(i[j])
# e.append(min(l))
