import importlib
from typing import List, TYPE_CHECKING, Optional, Type

if TYPE_CHECKING:
    from .experiment import Experiment


class Storage(object):
    def save(self, experiment):
        raise NotImplementedError()

    def get_experiment(self, cls, identifier):
        # type: (Type, int) -> Experiment
        raise NotImplementedError()

    def get_experiments(self, cls, group=None):
        # type: (Type, Optional[str]) -> List[Experiment]
        raise NotImplementedError()

    def remove(self, group):
        raise NotImplementedError()

    def get_groups(self):
        # type: () -> List[str]
        raise NotImplementedError()

    def get_new_run(self):
        # type: () -> int
        raise NotImplementedError()



def export_storage(storage):
    from .sql_storage import SqliteStorage
    if isinstance(storage, SqliteStorage):
        return "sqlite"
    else:
        raise ValueError("Could not export storage {storage}".format(storage=storage))


def import_storage(storage_string):
    if storage_string == "sqlite":
        from .sql_storage import SqliteStorage
        return SqliteStorage()
    else:
        raise ValueError("Could not import storage {storage_string}".format(storage_string=storage_string))


def full_class_name(cls):
    # Inspired by https://stackoverflow.com/a/2020083
    module = cls.__module__
    if module is None or module == str.__class__.__module__:
        return cls.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + cls.__name__


def str_to_class(n):
    # Inspired by https://stackoverflow.com/a/13808375
    parts = n.split(".")
    module_name = ".".join(parts[:-1])
    class_name = parts[-1]

    m = importlib.import_module(module_name)
    return getattr(m, class_name)
