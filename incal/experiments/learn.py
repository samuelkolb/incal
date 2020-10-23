import glob
import os
import random
import warnings

from .prepare import get_synthetic_db
from incal.util.options import Options, Experiment

from incal.learn import LearnOptions, LearnResults
from incal.util.parallel import run_commands
from .prepare import select_benchmark_files, benchmark_filter, get_benchmark_results_dir


def get_bound_volume(bounds):
    size = 1
    for ub_lb in bounds.values():
        size *= ub_lb[1] - ub_lb[0]
    return size


def rel_ratio(ratio):
    return abs(0.5 - ratio)


def learn_synthetic(
    input_directory,
    output_directory,
    runs,
    sample_size,
    processes,
    time_out,
    learn_options: LearnOptions,
):
    commands = []

    db = get_synthetic_db(input_directory)
    for name in db.getall():
        entry = db.get(name)
        matching_samples = []
        for sample in entry["samples"]:
            if sample["sample_size"] == sample_size and len(matching_samples) < runs:
                matching_samples.append(sample)
        if len(matching_samples) != runs:
            raise RuntimeError(
                "Insufficient samples available, prepare more samples first"
            )

        for sample in matching_samples:
            detail_learn_options = learn_options.copy()
            detail_learn_options.domain = os.path.join(
                input_directory, "{}.density".format(name)
            )
            detail_learn_options.data = os.path.join(
                input_directory, sample["samples_file"]
            )
            detail_learn_options.labels = os.path.join(
                input_directory, sample["labels_file"]
            )

            export_file = "{}{sep}{}.{}.{}.result".format(
                output_directory, name, sample_size, sample["seed"], sep=os.path.sep
            )
            log_file = "{}{sep}{}.{}.{}.log".format(
                output_directory, name, sample_size, sample["seed"], sep=os.path.sep
            )

            if not os.path.exists(os.path.dirname(export_file)):
                os.makedirs(os.path.dirname(export_file))

            commands.append(
                "incal-track {} --export {} --log {}".format(
                    detail_learn_options.print_arguments(), export_file, log_file
                )
            )

    run_commands(commands, processes, time_out)


def learn_benchmark(
    runs, sample_size, processes, time_out, learn_options: LearnOptions
):
    # def filter1(entry):
    #     return "real_variables_count" in entry and entry["real_variables_count"] + entry["bool_variables_count"] <= 10
    #
    # count = 0
    # boolean = 0
    # for name, entry, density_filename in select_benchmark_files(filter1):
    #     if entry["bool_variables_count"] > 0:
    #         boolean += 1
    #     count += 1
    #
    # print("{} / {}".format(boolean, count))
    #
    # count = 0
    # boolean = 0
    # for name, entry, density_filename in select_benchmark_files(benchmark_filter):
    #     if entry["bool_variables_count"] > 0:
    #         boolean += 1
    #     count += 1
    #
    # print("{} / {}".format(boolean, count))

    def learn_filter(_e):
        return benchmark_filter(_e) and "samples" in _e

    count = 0
    problems_to_learn = []
    for name, entry, density_filename in select_benchmark_files(learn_filter):
        if len(entry["bounds"]) > 0:
            best_ratio = min(rel_ratio(t[1]) for t in entry["bounds"])
            if best_ratio <= 0.3:
                qualifying = [
                    t
                    for t in entry["bounds"]
                    if rel_ratio(t[1]) <= 0.3
                    and abs(rel_ratio(t[1]) - best_ratio) <= best_ratio / 5
                ]
                selected = sorted(qualifying, key=lambda x: get_bound_volume(x[0]))[0]
                print(
                    name,
                    "\n",
                    rel_ratio(selected[1]),
                    best_ratio,
                    selected[0],
                    entry["bool_variables_count"],
                )
                count += 1
                selected_samples = [
                    s
                    for s in entry["samples"]
                    if s["bounds"] == selected[0] and s["sample_size"] >= sample_size
                ]
                if len(selected_samples) < runs:
                    raise RuntimeError(
                        "Insufficient number of data set available ({} of {})".format(
                            len(selected_samples), runs
                        )
                    )
                elif len(selected_samples) > runs:
                    selected_samples = selected_samples[:runs]
                for selected_sample in selected_samples:
                    problems_to_learn.append((name, density_filename, selected_sample))

    commands = []
    for name, density_filename, selected_sample in problems_to_learn:
        detail_learn_options = learn_options.copy()
        detail_learn_options.domain = density_filename
        detail_learn_options.data = selected_sample["samples_filename"]
        detail_learn_options.labels = selected_sample["labels_filename"]
        export_file = "{}{sep}{}.{}.{}.result".format(
            get_benchmark_results_dir(),
            name,
            selected_sample["sample_size"],
            selected_sample["seed"],
            sep=os.path.sep,
        )
        log_file = "{}{sep}{}.{}.{}.log".format(
            get_benchmark_results_dir(),
            name,
            selected_sample["sample_size"],
            selected_sample["seed"],
            sep=os.path.sep,
        )
        if not os.path.exists(os.path.dirname(export_file)):
            os.makedirs(os.path.dirname(export_file))
        commands.append(
            "incal-track {} --export {} --log {}".format(
                detail_learn_options.print_arguments(), export_file, log_file
            )
        )

    run_commands(commands, processes, time_out)


def get_experiment(res_path=None):
    def import_handler(parameters_dict, results_dict, config_dict):
        for key, entry in parameters_dict.items():
            if isinstance(entry, str):
                index = entry.find("res/")
                if index >= 0:
                    parameters_dict[key] = res_path + os.path.sep + entry[index + 4 :]

    config = Options()
    config.add_option("export", str)
    return Experiment(
        LearnOptions(), LearnResults(), config, import_handler if res_path else None
    )


def track():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        experiment = get_experiment()
        experiment.import_from_command_line()
        experiment.save(experiment.config.export)
        experiment.execute()
        experiment.save(experiment.config.export)
