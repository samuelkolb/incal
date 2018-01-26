from __future__ import print_function, division

import fnmatch
import hashlib
import json

import os
import random

import itertools
import pysmt.shortcuts as smt
import re

import time

import generator
import inc_logging
import parse
import problem
from incremental_learner import RandomViolationsStrategy
from k_cnf_smt_learner import KCnfSmtLearner
from k_dnf_smt_learner import KDnfSmtLearner
from parameter_free_learner import learn_bottom_up
from smt_print import pretty_print
from smt_walk import SmtWalker


class OperatorWalker(SmtWalker):
    def __init__(self):
        self.operators = set()

    def walk_and(self, args):
        self.operators.add("&")
        self.walk_smt_multiple(args)

    def walk_or(self, args):
        self.operators.add("|")
        self.walk_smt_multiple(args)

    def walk_plus(self, args):
        self.operators.add("+")
        self.walk_smt_multiple(args)

    def walk_minus(self, left, right):
        self.operators.add("-")
        self.walk_smt_multiple([left, right])

    def walk_times(self, args):
        self.operators.add("*")
        self.walk_smt_multiple(args)

    def walk_not(self, argument):
        self.operators.add("~")
        self.walk_smt_multiple([argument])

    def walk_ite(self, if_arg, then_arg, else_arg):
        self.operators.add("ite")
        self.walk_smt_multiple([if_arg, then_arg, else_arg])

    def walk_pow(self, base, exponent):
        self.operators.add("^")
        self.walk_smt_multiple([base, exponent])

    def walk_lte(self, left, right):
        self.operators.add("<=")
        self.walk_smt_multiple([left, right])

    def walk_lt(self, left, right):
        self.operators.add("<")
        self.walk_smt_multiple([left, right])

    def walk_equals(self, left, right):
        self.operators.add("=")
        self.walk_smt_multiple([left, right])

    def walk_symbol(self, name, v_type):
        pass

    def walk_constant(self, value, v_type):
        pass

    def find_operators(self, formula):
        self.walk_smt(formula)
        return list(self.operators)


class HalfSpaceWalker(SmtWalker):
    def __init__(self):
        self.half_spaces = set()

    def walk_and(self, args):
        self.walk_smt_multiple(args)

    def walk_or(self, args):
        self.walk_smt_multiple(args)

    def walk_plus(self, args):
        self.walk_smt_multiple(args)

    def walk_minus(self, left, right):
        self.walk_smt_multiple([left, right])

    def walk_times(self, args):
        self.walk_smt_multiple(args)

    def walk_not(self, argument):
        self.walk_smt_multiple([argument])

    def walk_ite(self, if_arg, then_arg, else_arg):
        self.walk_smt_multiple([if_arg, then_arg, else_arg])

    def walk_pow(self, base, exponent):
        self.walk_smt_multiple([base, exponent])

    def walk_lte(self, left, right):
        self.half_spaces.add(parse.smt_to_nested(left <= right))

    def walk_lt(self, left, right):
        self.half_spaces.add(parse.smt_to_nested(left < right))

    def walk_equals(self, left, right):
        self.walk_smt_multiple([left, right])

    def walk_symbol(self, name, v_type):
        pass

    def walk_constant(self, value, v_type):
        pass

    def find_half_spaces(self, formula):
        self.walk_smt(formula)
        return list(self.half_spaces)


def import_problem(name, filename):
    target_formula = smt.read_smtlib(filename)

    variables = target_formula.get_free_variables()
    var_names = [str(v) for v in variables]
    var_types = {str(v): v.symbol_type() for v in variables}
    var_domains = {str(v): (0, 200) for v in variables}  # TODO This is a hack

    domain = problem.Domain(var_names, var_types, var_domains)
    target_problem = problem.Problem(domain, target_formula, name)

    return target_problem


def get_cache_name():
    cache = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo"), "cache")
    return os.path.join(cache, "summary.json")


def get_problem_file(problem_id):
    return os.path.join(os.path.dirname(get_cache_name()), "problems", "{}.txt".format(problem_id))


def load():
    name = get_cache_name()
    if not os.path.exists(name):
        with open(name, "w") as f:
            json.dump({"files": {}}, f)

    with open(name, "r") as f:
        flat = json.load(f)
    return flat


def dump(flat):
    name = get_cache_name()
    temp_name = "{}.tmp".format(name)
    with open(temp_name, "w") as f:
        json.dump(flat, f)
    os.rename(temp_name, name)


def scan(full_dir, root_dir):
    matches = []

    for root, dir_names, file_names in os.walk(full_dir):
        for filename in fnmatch.filter(file_names, '*.smt2'):
            full_path = os.path.abspath(os.path.join(root, filename))
            matches.append(re.match("{r}{s}(.*)".format(s=os.path.sep, r=root_dir), full_path).group(1))

    for match in matches:
        print(match)

    flat = load()
    problems = flat["files"]
    counter = 0
    for match in matches:
        if match not in problems:
            print("Importing {}".format(match))
            smt_file = os.path.join(root_dir, match)
            if os.path.getsize(smt_file) / 1024 <= 100:
                imported_problem = import_problem(match, smt_file)
                problems[match] = {
                    "loaded": True,
                    "var_count": len(imported_problem.domain.variables),
                    "file_size": os.path.getsize(match)
                }
                smt.reset_env()
            else:
                problems[match] = {"loaded": False, "reason": "size", "file_size": os.path.getsize(match)}
            counter += 1
            if counter >= 1:
                dump(flat)
                counter = 0

    if counter != 0:
        dump(flat)


def has_equals(_props):
    return "=" in [str(o) for o in _props["operators"]]


def has_conjunctions(_props):
    return "&" in [str(o) for o in _props["operators"]]


def has_disjunctions(_props):
    return "|" in [str(o) for o in _props["operators"]]


def has_connectives(_props):
    return has_conjunctions(_props) or has_disjunctions(_props)


def var_count(_props):
    return _props["var_count"]


def half_space_count(_props):
    return len(_props["half_spaces"])


def file_size(_props):
    return _props["file_size"]


def analyze(root_dir):
    flat = load()
    files = flat["files"]
    if "lookup" not in flat:
        flat["lookup"] = dict()
    lookup = flat["lookup"]

    print(*["mode", "var_count", "half_space_count", "file_size", "name"], sep="\t")
    for name, properties in files.items():
        smt_file = os.path.join(root_dir, name)
        if properties["loaded"] and properties["var_count"] < 10:
            change = False
            if "id" not in properties:
                properties["id"] = hashlib.sha256(smt_file).hexdigest()
                change = True

            if properties["id"] not in lookup:
                lookup[properties["id"]] = name
                change = True

            problem_file = get_problem_file(properties["id"])
            if not os.path.isfile(problem_file):
                with open(problem_file, "w") as f:
                    print(problem.export_problem(import_problem(name, smt_file)), file=f)

            with open(problem_file, "r") as f:
                target_problem = problem.import_problem(json.load(f))

            if "operators" not in properties:
                properties["operators"] = OperatorWalker().find_operators(target_problem.theory)
                change = True

            if "half_spaces" not in properties:
                properties["half_spaces"] = HalfSpaceWalker().find_half_spaces(target_problem.theory)
                change = True

            if change:
                dump(flat)

            if not has_equals(properties) and has_connectives(properties):
                mode = "D" if has_disjunctions(properties) else "C"
                information = [mode, var_count(properties), half_space_count(properties), file_size(properties), name]
                print(*[str(i) for i in information], sep="\t")


def learn_formula(problem_id, domain, h, data, seed, log_dir, learn_all=False, learn_dnf=False):
    initial_size = 20
    violations_size = 10
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    def learn_inc(_data, _k, _h):
        if learn_all is not None and learn_all:
            initial_indices = list(range(len(data)))
        else:
            initial_indices = random.sample(list(range(len(data))), initial_size)
        violations_strategy = RandomViolationsStrategy(violations_size)
        if learn_dnf is not None and learn_dnf:
            learner = KDnfSmtLearner(_k, _h, violations_strategy)
        else:
            learner = KCnfSmtLearner(_k, _h, violations_strategy)
        log_file = os.path.join(log_dir, "{}_{}_{}_{}_{}.learning_log.txt".format(problem_id, len(data), seed, _k, _h))
        learner.add_observer(inc_logging.LoggingObserver(log_file, seed, True, violations_strategy))
        learned_theory = learner.learn(domain, data, initial_indices)
        # learned_theory = Or(*[And(*planes) for planes in hyperplane_dnf])
        print("Learned theory:\n{}".format(pretty_print(learned_theory)))
        return learned_theory

    phi, k, h = learn_bottom_up(data, learn_inc, 1, 1, init_h=h, max_h=h)

    overview = os.path.join(log_dir, "problems.txt")
    if not os.path.isfile(overview):
        flat = {}
    else:
        with open(overview, "r") as f:
            flat = json.load(f)
    if problem_id not in flat:
        flat[problem_id] = {}
    flat[problem_id][len(data)] = {"k": k, "h": h, "seed": seed, "bias": "dnf" if learn_dnf else "cnf"}
    with open(overview, "w") as f:
        json.dump(flat, f)


def learn(sample_count, log_dir, learn_all=False, learn_dnf=False):
    flat = load()
    files = flat["files"]
    ratio_dict = flat["ratios"]
    seed = hash(time.time())
    random.seed(seed)
    for name, props in files.items():
        if props["loaded"] and props["var_count"] < 10 and not has_equals(props) and has_disjunctions(props) and \
                ratio_dict[name]["finite"] and 0.2 <= ratio_dict[name]["ratio"] <= 0.8:
            with open(get_problem_file(props["id"]), "r") as f:
                target_problem = problem.import_problem(json.load(f))
            adapted_problem = adapt_domain_multiple(target_problem, ratio_dict[name]["bounds"])
            samples = generator.get_problem_samples(adapted_problem, sample_count, 1)
            domain = adapted_problem.domain
            learn_formula(props["id"], domain, len(props["half_spaces"]), samples, seed, log_dir, learn_all, learn_dnf)
            print(props["id"], name)


def adapt_domain(target_problem, lb, ub):
    domain = target_problem.domain
    var_domains = {}
    for v in domain.variables:
        var_domains[v] = (lb, ub)
    adapted_domain = problem.Domain(domain.variables, domain.var_types, var_domains)
    return problem.Problem(adapted_domain, target_problem.theory, target_problem.name)


def adapt_domain_multiple(target_problem, new_bounds):
    domain = target_problem.domain
    adapted_domain = problem.Domain(domain.variables, domain.var_types, new_bounds)
    return problem.Problem(adapted_domain, target_problem.theory, target_problem.name)


def ratios():
    flat = load()
    files = flat["files"]
    if "ratios" not in flat:
        flat["ratios"] = dict()
    ratio_dict = flat["ratios"]

    sample_count = 100
    for name, props in files.items():
        if props["loaded"] and props["var_count"] < 10 and not has_equals(props) and has_disjunctions(props):
            with open(get_problem_file(props["id"]), "r") as f:
                target_problem = problem.import_problem(json.load(f))

            bounds_pool = [(-1, 1), (-10, 10), (-100, 100), (-1000, 1000)]
            domain = target_problem.domain
            result = {"finite": False, "samples": sample_count}
            for bounds in itertools.product(*[bounds_pool for _ in range(len(domain.real_vars))]):
                var_bounds = dict(zip(domain.real_vars, bounds))
                current_problem = adapt_domain_multiple(target_problem, var_bounds)
                samples = generator.get_problem_samples(current_problem, sample_count, 1)
                positive_count = len([x for x, y in samples if y])
                is_finite = positive_count not in [0, sample_count]
                if is_finite:
                    ratio = positive_count / sample_count
                    if not result["finite"] or abs(ratio - 0.5) < abs(result["ratio"] - 0.5):
                        result["finite"] = True
                        result["ratio"] = ratio
                        result["bounds"] = var_bounds
                    if abs(ratio - 0.5) <= 0.1:
                        break
            print(result)
            ratio_dict[name] = result
    dump(flat)


def summarize(results_dir, output_type):
    results_file = os.path.join(results_dir, "problems.txt")
    with open(results_file, "r") as f:
        results_flat = json.load(f)

    overview = load()
    lookup = overview["lookup"]

    simple_output = output_type is None

    if simple_output:
        print("name", "sample_size", "total_duration", sep="\t")

    unique_names = set()
    unique_sample_sizes = set()
    duration_table = dict()
    k_table = dict()
    h_table = dict()

    for problem_id in results_flat:
        name = lookup.get(problem_id, problem_id)
        unique_names.add(name)
        for sample_size in results_flat[problem_id]:
            unique_sample_sizes.add(sample_size)
            timed_out = results_flat[problem_id][sample_size].get("time_out", False)
            seed, k, h = (results_flat[problem_id][sample_size][v] for v in ["seed", "k", "h"])
            log_file = "{}_{}_{}_{}_{}.learning_log.txt".format(problem_id, sample_size, seed, k, h)
            log_file_full = os.path.join(results_dir, log_file)
            if not timed_out:
                durations = []
                with open(log_file_full, "r") as f:
                    for line in f:
                        flat = json.loads(line)
                        if flat["type"] == "update":
                            durations.append(flat["selection_time"] + flat["solving_time"])
                duration_table[(name, sample_size)] = sum(durations)
                if simple_output:
                    print(name, sample_size, sum(durations), sep="\t")
            else:
                duration_table[(name, sample_size)] = "({})".format(results_flat[problem_id][sample_size]["time_limit"])
                if simple_output:
                    print(name, sample_size, None, sep="\t")
            k_table[(name, sample_size)] = k
            h_table[(name, sample_size)] = h

    names = list(sorted(unique_names))
    sample_sizes = list(sorted(unique_sample_sizes, key=int))

    if output_type in ["time", "k", "h"]:
        print("", *sample_sizes, sep="\t")

    if output_type == "time":
        for name in names:
            print(name, *[duration_table.get((name, sample_size), "") for sample_size in sample_sizes], sep="\t")

    if output_type == "k":
        for name in names:
            print(name, *[k_table.get((name, sample_size), "") for sample_size in sample_sizes], sep="\t")

    if output_type == "h":
        for name in names:
            print(name, *[h_table.get((name, sample_size), "") for sample_size in sample_sizes], sep="\t")


def load_results(results_dir):
    results_file = os.path.join(results_dir, "problems.txt")
    with open(results_file, "r") as f:
        results_flat = json.load(f)
    return results_flat


def dump_results(results_flat, results_dir):
    results_file = os.path.join(results_dir, "problems.txt")
    with open(results_file, "w") as f:
        json.dump(results_flat, f)


def get_log_messages(results_dir, config, p_id=None, samples=None):
    id_key = "problem_id" if "problem_id" in config else "id"
    sample_key = "sample_size" if "sample_size" in config else "samples"
    problem_id = config[id_key] if p_id is None else p_id
    sample_size = config[sample_key] if samples is None else samples
    seed, k, h = (config[v] for v in ["seed", "k", "h"])
    log_file = "{}_{}_{}_{}_{}.learning_log.txt".format(problem_id, sample_size, seed, k, h)
    log_file_full = os.path.join(results_dir, log_file)
    flats = []
    with open(log_file_full, "r") as f:
        for line in f:
            flats.append(json.loads(line))
    return flats


class TableMaker(object):
    def __init__(self, results_dir, row_key_type, col_key_type, value_type, delimiter=None, data_dir=None):
        self.results_dir = results_dir
        self.row_key_type = row_key_type
        self.col_key_type = col_key_type
        self.value_type = value_type
        self.delimiter = "\t" if delimiter is None else delimiter
        self.data_dir = data_dir
        self.row_keys = None
        self.col_keys = None
        self.table = None
        self.load_table()

    def extract(self, extraction_type, config):
        # noinspection PyCallingNonCallable
        return {
            "id": lambda c: str(c["problem_id"]),
            "name": self.extract_benchmark_name,
            "time": self.extract_time,
            "k": lambda c: int(c[extraction_type]),
            "h": lambda c: int(c[extraction_type]),
            "samples": lambda c: int(c["sample_size"]),
            "l": self.extract_literals,
            "acc": self.extract_accuracy,
        }[extraction_type](config)

    def get_name(self, extraction_type):
        return {
            "id": "Problem ID",
            "name": "Name",
            "time": "Time (s)",
            "k": "k (# terms)",
            "h": "h (# halfspaces)",
            "samples": "# samples",
            "l": "# literals",
            "acc": "accuracy",
        }[extraction_type]

    @staticmethod
    def extract_benchmark_name(config):
        flat = load()
        return str(flat["lookup"][config["problem_id"]])

    def extract_time(self, config):
        timed_out = config.get("time_out", False)
        if not timed_out:
            durations = []
            for message in get_log_messages(self.results_dir, config):
                if message["type"] == "update":
                    durations.append(message["selection_time"] + message["solving_time"])
            return sum(durations)
        else:
            return "({})".format(config["time_limit"])

    def extract_literals(self, config):
        with open(os.path.join(self.data_dir, "{}.txt".format(str(config["problem_id"])))) as f:
            s_problem = generator.import_synthetic_data(json.load(f))
        return s_problem.synthetic_problem.literals

    def extract_accuracy(self, config):
        timed_out = config.get("time_out", False)
        if not timed_out:
            return config["approx_accuracy"]["1000"][0]["acc"]
        else:
            return None

    def load_table(self):
        problems = load_results(self.results_dir)

        unique_row_keys = set()
        unique_col_keys = set()
        table = dict()

        for problem_id in problems:
            for sample_size in problems[problem_id]:
                config = problems[problem_id][sample_size]
                config["problem_id"] = problem_id
                config["sample_size"] = sample_size

                row_key = self.extract(self.row_key_type, config)
                col_key = self.extract(self.col_key_type, config)
                value = self.extract(self.value_type, config)

                unique_row_keys.add(row_key)
                unique_col_keys.add(col_key)
                if (row_key, col_key) not in table:
                    table[(row_key, col_key)] = []
                table[(row_key, col_key)].append(value)

        unique_row_keys = unique_row_keys
        unique_col_keys = unique_col_keys
        row_key_is_int = all(isinstance(k, (int, long)) for k in unique_row_keys)
        self.row_keys = list(sorted(unique_row_keys, key=(int if row_key_is_int else str)))
        col_key_is_int = all(isinstance(k, (int, long)) for k in unique_col_keys)
        self.col_keys = list(sorted(unique_col_keys, key=(int if col_key_is_int else str)))

        self.table = table

        return self

    def __str__(self):
        def get_val(_key):
            if _key not in self.table:
                return ""
            elif len(self.table[_key]) == 1:
                return self.table[_key][0]
            else:
                initial = self.table[_key][0]
                for i in range(1, len(self.table[_key])):
                    initial += self.table[_key][i]
                return initial / len(self.table[_key])

        lines = [self.delimiter.join([""] + [str(k) for k in self.col_keys])]
        for rk in self.row_keys:
            lines.append(self.delimiter.join([str(rk)] + [str(get_val((rk, ck))) for ck in self.col_keys]))
        return "\n".join(lines)

    def plot_table(self, filename=None, aggregate=False, y_min=None, y_max=None, x_min=None, x_max=None):
        import rendering
        import numpy

        def get_val(_key):
            if _key not in self.table:
                return numpy.nan
            else:
                return numpy.nanmean(self.table[_key])

        scatter = rendering.ScatterData("", numpy.array(self.col_keys))
        scatter.y_lim([y_min, y_max])
        scatter.x_lim([x_min, x_max])
        series = [numpy.array([get_val((rk, ck)) for ck in self.col_keys]) for rk in self.row_keys]
        if aggregate:
            series_array = numpy.array(series)
            legend_name = "Average " + self.get_name(self.value_type)
            scatter.add_data(legend_name, numpy.nanmean(series_array, 0), numpy.nanstd(numpy.array(series), 0))
            # std_dev_series =
        else:
            for i in range(len(self.row_keys)):
                scatter.add_data(self.row_keys[i], series[i])
        scatter.plot(filename=filename, lines=True, log_x=False, log_y=False, label_y=self.get_name(self.value_type),
                     label_x=self.get_name(self.col_key_type))
