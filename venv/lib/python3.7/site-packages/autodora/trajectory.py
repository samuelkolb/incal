from typing import Dict, Any, List


def flatten(dict_settings):
    common_length = None
    for val in dict_settings.values():
        if not common_length:
            common_length = len(val)
        elif len(val) != common_length:
            raise ValueError("All value assignments must have the same length")
    return [{k: v[i] for k, v in dict_settings.items()} for i in range(common_length)]


def product(*args):
    args = [flatten(arg) if isinstance(arg, dict) else arg for arg in args]
    result = args[0]
    for arg in args[1:]:
        result = [dict(**d1, **d2) for d1 in result for d2 in arg]
    return result


class Trajectory(object):
    def __init__(self, name):
        self.experiments = []
        self.name = name

    def add(self, experiment):
        self.experiments.append(experiment)

    def explore(self, cls, settings: List[Dict[str, Any]]):
        if isinstance(settings, dict):
            common_length = None
            for val in settings.values():
                if not common_length:
                    common_length = len(val)
                elif len(val) != common_length:
                    raise ValueError("All value assignments must have the same length")
            if common_length is None:
                settings = [{}]
            else:
                settings = [{k: v[i] for k, v in settings.items()} for i in range(common_length)]

        for setting in settings:
            experiment = cls(self.name)
            for key, value in setting.items():
                experiment[key] = value

            self.add(experiment)
