import argparse
import os
from glob import glob

from autodora.observe import ProgressObserver
from autodora.observers.telegram_observer import TelegramObserver
from autodora.runner import CommandLineRunner, PrintObserver
from autodora.sql_storage import SqliteStorage
from autodora.trajectory import product

try:
    from telegram.ext import Updater
    from telegram import ParseMode
except ImportError:
    Updater = None
    ParseMode = None

from .learn_experiment import LearnExperiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dirs",
        nargs="+",
        type=str,
        help="The directories to load files from to run experiments on",
    )

    parser.add_argument("--learners", nargs="+", type=str, help="The learners to use")
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Performs a dry run, only printing out experiments instead of running them",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="The number of concurrent processes to use for running experiments.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="The number of seconds to run each experiment for.",
    )

    args = parser.parse_args()

    file_settings = dict()
    log_files = dict()
    storage = SqliteStorage()
    processes = args.processes or 1

    dispatcher = ProgressObserver()
    dispatcher.add_observer(PrintObserver())
    try:
        dispatcher.add_observer(TelegramObserver())
    except RuntimeError:
        pass

    for directory in args.dirs:
        directory = os.path.abspath(directory)
        dir_files = []
        for filename in glob(os.path.join(directory, "*.json")):
            basename = ".".join(filename.split(".")[:-1])
            for data_filename in glob(f"{basename}*.data.npy"):
                label_filename = data_filename.replace("data", "labels")
                dir_files.append((filename, data_filename, label_filename))

        name = os.path.basename(directory)
        file_settings[name] = []
        log_files[name] = os.path.join(directory, "run_exp.log")
        for f, d, l in dir_files:
            file_settings[name].append(
                {"formula_filename": f, "data_filename": d, "labels_filename": l}
            )

    learners_settings = {"learner": args.learners or ["cnf."]}

    for key, values in file_settings.items():
        trajectory = LearnExperiment.explore(key, product(values, learners_settings))
        if args.dry:
            print(*trajectory.experiments, sep="\n")
        else:
            CommandLineRunner(
                trajectory,
                storage,
                processes=processes,
                timeout=args.timeout,
                observer=dispatcher,
            ).run()


if __name__ == "__main__":
    main()
