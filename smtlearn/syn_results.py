import importlib
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy

import tabulate

from syn_experiment import SyntheticExperiment


def mean(series):
    running_sum = 0
    count = 0
    for s in series:
        if s is not None:
            running_sum += s
            count += 1
    return running_sum / count


def std(series):
    return numpy.std(numpy.array(series)).item()


def main():
    for vv in [3]:
        x = [2, 4, 8, 16, 32]
        time_series = []

        for method in ["smt", "smtz3debug"]:
            group = "syn"
            db_name = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "syn_{}.sqlite".format(method))
            os.environ["DB"] = db_name
            print(os.environ["DB"])

            for ss in [20]:
                errors, keys, means, deviations, time_outs = get_results(ss, vv, group, method)
                time_series.append(means)

        print(tabulate.tabulate([x, *time_series]))


def get_results(ss, vv, group, method):
    import autodora.sql_storage as sql
    importlib.reload(sql)
    storage = sql.SqliteStorage()
    experiments = storage.get_experiments(SyntheticExperiment, group)
    run_times_dict = defaultdict(lambda: [])
    time_outs_dict = defaultdict(lambda: 0)
    errors_dict = defaultdict(lambda: 0)
    for e in experiments:
        filename = e["file"]
        parts = filename.split("_")
        if method == "smt":
            v = int(parts[2])
            h = int(parts[3])
        else:
            v = int(parts[1])
            h = int(parts[2])
        runtime = e["@runtime"]
        sample_size = e["sample_size"]
        key = h
        if sample_size == ss and v == vv:
            print(e)

            if e["@error"] is not None:
                errors_dict[key] += 1
                # print(e["@error"])
            elif runtime is None or runtime > 200:
                time_outs_dict[key] += 1
                run_times_dict[key].append(e["@timeout"])
            else:
                run_times_dict[key].append(runtime)
    keys = sorted(run_times_dict.keys())
    means = [mean(run_times_dict[key]) for key in keys]
    deviations = [std(run_times_dict[key]) for key in keys]
    time_outs = [time_outs_dict[key] for key in keys]
    errors = [errors_dict[key] for key in keys]
    sql.database.close()
    return errors, keys, means, deviations, time_outs


if __name__ == '__main__':
    main()
