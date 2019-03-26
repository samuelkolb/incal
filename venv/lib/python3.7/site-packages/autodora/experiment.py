import sys
import time
from datetime import datetime
from typing import Union, Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from storage import Storage

from .trajectory import Trajectory


class Parameter(object):
    def __init__(self, p_type, default=None, description=None, name=None, arg_name=None):
        self.name = name
        self.p_type = p_type
        self.default = default
        self.description = description
        self.specific_arg_name = arg_name

    @property
    def arg_name(self):
        return self.specific_arg_name or self.name


class Result(Parameter):
    pass


class Config(Parameter):
    pass


class Group(object):
    def __init__(self, name):
        self.name = name
        self.parameters = dict()  # type: Dict[str, Parameter]
        self.values = dict()  # type: Dict[str, Any]

    def add_parameter(self, name, p_type, default=None, description=None, arg_name=None):
        self.add(Parameter(p_type, default, description, name, arg_name))

    def add(self, parameter):
        self.parameters[parameter.name] = parameter

    def set_values(self, **kwargs):
        for key, value in kwargs.items():
            self.set_value(key, value)

    def set_value(self, name, value):
        if name not in self.parameters:
            raise ValueError("No parameter called {name}".format(name=name))
        if not isinstance(value, self.parameters[name].p_type):
            raise ValueError("Unexpected type for value {value}".format(value=value))
        self.values[name] = value

    def __setitem__(self, key, value):
        self.set_value(key, value)

    def __getitem__(self, item):
        return self.values[item] if item in self.values else self.parameters[item].default

    def add_arguments(self, parser, prefix=None):
        for name, parameter in self.parameters.items():
            parser.add_argument(
                "--{}{}".format(prefix or '', parameter.arg_name),
                type=parameter.p_type,
                default=parameter.default,
                help=parameter.description
            )

    def parse_arguments(self, args, prefix=None):
        for name, parameter in self.parameters.items():
            self.set_value(name, getattr(args, "{}{}".format(prefix or '', parameter.arg_name)))

    def get_arguments(self, prefix=None):
        return " ".join(
            "--{} {}".format("{}{}".format(prefix or '', self.parameters[name].arg_name), value)
            for name, value in self.values.items() if value is not None
        )

    def copy(self):
        group = self.__new__(self.__class__)
        group.parameters = dict(self.parameters)
        group.values = dict(self.values)
        return group

    def __str__(self):
        values = ", ".join("{}: {}".format(key, self.values.get(key, '')) for key in self.parameters)
        return "{}={{{}}}".format(self.name, values)


class Derived(object):
    def __init__(self, callback, cache):
        self.callback = callback
        self.cache = cache

    def __call__(self, *args, **kwargs):
        return self.callback(*args, **kwargs)


def derived(cache):
    def real(func):
        return Derived(func, cache)
    return real


class Experiment(object):
    def __init__(self, group, storage=None, identifier=None):
        # type: (str, Optional[Storage], Optional[int]) -> None
        self.derived_callbacks = dict()
        self.storage = storage
        self.identifier = identifier

        self.group = group

        self.config = Group("config")
        self.config.add_parameter("@timeout", int, None, "The timeout value set for this experiment")
        self.config.add_parameter("@run.count", int, None, "The run count (for local storage)")
        self.config.add_parameter("@run.computer", str, None, "The computer name the run was performed on")
        self.config.add_parameter("@run.date", datetime, None, "The date when the run was instantiated")

        self.parameters = Group("parameters")

        self.result = Group("result")
        self.result.add_parameter("@start_time", datetime, None, "When this experiment was started")
        self.result.add_parameter("@end_time", datetime, None, "When this experiment was started")
        self.result.add_parameter("@runtime", float, None, "How long the experiment took to execute (perf time)")
        self.result.add_parameter("@runtime_wall", float, None,
                                  "How long the experiment took to execute (wall clock time)")
        self.result.add_parameter("@runtime_process", float, None,
                                  "How long the experiment took to execute (process time)")

        self.derived_callbacks["@completed"] = Derived(self.is_completed, False)

        self.derived = dict()

        for key, value in self.__class__.__dict__.items():
            if isinstance(value, Parameter):
                if value.name is None:
                    value.name = key
                if isinstance(value, Config):
                    self.config.add(value)
                elif isinstance(value, Result):
                    self.result.add(value)
                else:
                    self.parameters.add(value)
            elif isinstance(value, Derived):
                if key.startswith("derived_"):
                    key = key[8:]
                self.derived_callbacks[key] = value

    def is_completed(self):
        return self["@end_time"] is not None

    def get_derived(self, name):
        if name in self.derived:
            return self.derived[name]

        if name in self.derived_callbacks:
            try:
                result = self.derived_callbacks[name](self)
            except TypeError:
                result = self.derived_callbacks[name]()

            if self.derived_callbacks[name].cache:
                self.derived[name] = result
            return result

        raise ValueError("There is no derived attribute with the name {name}".format(name=name))

    def __getitem__(self, item):
        return self.get(item)

    def __setitem__(self, key, value):
        self.set(key, value)

    def get(self, name: Union[str, Parameter]):
        if isinstance(name, Parameter):
            name = name.name
        parts = name.split(".", 1)
        if parts[0] == "par" or parts[0] == "parameter":
            return self.parameters[parts[1]]
        elif parts[0] == "res" or parts[0] == "result":
            return self.result[parts[1]]
        elif parts[0] == "conf" or parts[0] == "config":
            return self.config[parts[1]]
        elif parts[0] == "derived":
            return self.get_derived(parts[1])
        else:
            name = ".".join(parts)
            results = []
            if name in self.config.parameters:
                results.append(self.config[name])
            elif name in self.parameters.parameters:
                results.append(self.parameters[name])
            elif name in self.result.parameters:
                results.append(self.result[name])
            elif name in self.derived_callbacks:
                results.append(self.get_derived(name))
            if len(results) == 1:
                return results[0]
            elif len(results) > 1:
                raise ValueError("Multiple entries found for the name {name}, please use parameter.{name}, "
                                 "result.{name}, config.{name} or derived.{name} to disambiguate".format(name=name))
        raise ValueError("No entry found for the name {name}".format(name=name))

    def set(self, name, value):
        if isinstance(name, Parameter):
            name = name.name

        parts = name.split(".", 1)
        if parts[0] == "par" or parts[0] == "parameter":
            self.parameters[parts[1]] = value
        elif parts[0] == "res" or parts[0] == "result":
            self.result[parts[1]] = value
        elif parts[0] == "conf" or parts[0] == "config":
            self.config[parts[1]] = value
        elif parts[0] == "derived":
            self.get_derived(parts[1])
        else:
            name = ".".join(parts)
            results = []
            if name in self.config.parameters:
                results.append(self.config)
            elif name in self.parameters.parameters:
                results.append(self.parameters)
            elif name in self.result.parameters:
                results.append(self.result)

            if len(results) == 1:
                results[0][name] = value
            elif len(results) > 1:
                raise ValueError("Multiple entries found for the name {name}, please use parameter.{name}, "
                                 "result.{name}, config.{name} or derived.{name} to disambiguate".format(name=name))
            else:
                raise ValueError("No entry found for the name {name}".format(name=name))

    def run(self, auto_save=True):
        try:
            self.result["@start_time"] = datetime.now()
            if auto_save:
                self.save()
            start = time.perf_counter()
            start_process = time.process_time()
            start_wall = time.time()
            self.run_internal()
            runtime = time.perf_counter() - start
            runtime_process = time.process_time() - start_process
            runtime_wall = time.time() - start_wall
            self.result["@end_time"] = datetime.now()
            self.result["@runtime"] = runtime
            self.result["@runtime_process"] = runtime_process
            self.result["@runtime_wall"] = runtime_wall
            if auto_save and self.storage:
                self.save()
            return self
        except KeyboardInterrupt:
            return self

    def run_internal(self):
        raise NotImplementedError()

    def save(self, storage=None):
        if storage:
            storage.save(self)
        else:
            if not self.storage:
                raise ValueError("No storage specified")
            self.storage.save(self)

    def fresh_copy(self):
        return self.storage.get_experiment(self.__class__, self.identifier)

    def __str__(self):
        return "EXP(group={}, id={}, {}, {}, {}, derived={})"\
            .format(self.group or '', self.identifier or '', self.config, self.parameters, self.result, self.derived)

    def add_arguments(self, parser, prefix=None):
        self.config.add_arguments(parser, "{}conf.".format(prefix or ''))
        self.parameters.add_arguments(parser, "{}par.".format(prefix or ''))
        self.result.add_arguments(parser, "{}res.".format(prefix or ''))

    def parse_arguments(self, args, prefix=None):
        self.config.parse_arguments(args, "{}conf.".format(prefix or ''))
        self.parameters.parse_arguments(args, "{}par.".format(prefix or ''))
        self.result.parse_arguments(args, "{}res.".format(prefix or ''))

    def get_arguments(self, prefix=None):
        strings = [
            self.config.get_arguments("{}conf.".format(prefix or '')),
            self.parameters.get_arguments("{}par.".format(prefix or '')),
            self.result.get_arguments("{}res.".format(prefix or '')),
        ]
        return " ".join(strings)

    @classmethod
    def explore(cls, name: str, settings: Union[List[Dict[str, Any]], Dict[str, List]]):
        trajectory = Trajectory(name)
        trajectory.explore(cls, settings)
        return trajectory

    @classmethod
    def enable_cli(cls):
        if cls.__module__ == "__main__":
            cls.run_cli()

    @classmethod
    def run_cli(cls):
        from .cli import parse_cli
        parse_cli(cls)

    def __getstate__(self):
        return {
            "group": self.group,
            "identifier": self.identifier,
            "storage": self.storage,
            "config": self.config,
            "parameters": self.parameters,
            "result": self.result,
            "derived": self.derived,
        }

    def __setstate__(self, state):
        self.__init__(state["group"], state["storage"], state["identifier"])
        self.config = state["config"]
        self.parameters = state["parameters"]
        self.result = state["result"]
        self.derived = state["derived"]
