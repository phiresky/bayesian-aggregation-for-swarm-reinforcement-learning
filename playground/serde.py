"""params serialization / deserialization"""
import json
from typing import Any

import yaml
from typedload import datadumper, dataloader

dumper = datadumper.Dumper(hidedefault=False)

loader = dataloader.Loader()


def ser_y(
    loader,
    value,
):
    return yaml.dump(value)


def deser_y(loader, value, type_):
    return yaml.load(value, Loader=yaml.FullLoader)


loader.handlers.append((lambda t: issubclass(t, yaml.YAMLObject), deser_y))
dumper.handlers.append((lambda t: isinstance(t, yaml.YAMLObject), ser_y))

# todo: replace typedload library


def to_dict(params: Any) -> dict:
    return dumper.dump(params)


def serialize(params: Any) -> str:
    return json.dumps(
        to_dict(params),
        indent="\t",
    )


def from_dict(data: dict, type: Any):
    return loader.load(data, type)


def deserialize(data: str, type: Any):
    return from_dict(json.loads(data), type)
