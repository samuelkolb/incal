from datetime import datetime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .experiment import Experiment


class Observer:
    def __init__(self):
        self.observers = []

    def add_observer(self, observer):
        self.observers.append(observer)

    def remove_observer(self, observer):
        self.observers.remove(observer)


def dispatch(func):
    def modified(self, *args, **kwargs):
        for observer in self.observers:
            getattr(observer, func.__name__)(*args, **kwargs)
    return modified


class ProgressObserver(Observer):
    def __init__(self, auto_load=True):
        super().__init__()
        self.auto_load = auto_load

    @dispatch
    def run_started(self, platform, name, run_count, run_date, experiment_count):
        # type: (str, str, int, datetime, int) -> None
        raise NotImplementedError()

    @dispatch
    def experiment_started(self, index, experiment):
        # type: (int, Experiment) -> None
        raise NotImplementedError()

    @dispatch
    def experiment_finished(self, index, experiment):
        # type: (int, Experiment) -> None
        raise NotImplementedError()

    @dispatch
    def experiment_interrupted(self, index, experiment):
        # type: (int, Experiment) -> None
        raise NotImplementedError()

    @dispatch
    def experiment_failed(self, index, experiment):
        # type: (int, Experiment) -> None
        raise NotImplementedError()

    @dispatch
    def run_finished(self, platform, name, run_count, run_date):
        # type: (str, str, int, datetime) -> None
        raise NotImplementedError()



