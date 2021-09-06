"""python is a stupid language without non-shitty sum types (tagged unions)"""
from typing import Literal, Union


class SumTypeMeta(type):
    def __init__(self, outername, bases, clsdict):
        if outername != "SumType":
            inner_classes = []
            inner_class_names = []
            for name, cls in clsdict.items():
                if type(cls) == type:
                    assert (
                        name != "Type"
                    ), f"Sum type {outername} must have at least two variants"

                    type_value = cls._field_defaults.get("type", None)
                    assert (
                        type_value == name
                    ), f"{outername}.{name}.type must always be set to '{name}', but is {type_value}"

                    type_type = cls.__annotations__["type"]
                    assert (
                        type_type == Literal[name]
                    ), f"{outername}.{name}.type must be of type Literal['{name}'], but is {type_type}"
                    inner_classes.append(cls)
                    inner_class_names.append(name)

            typevar = clsdict.get("Type", None)
            joinednames = ", ".join(inner_class_names)
            typeerr = f"{outername}.Type must be Union[{joinednames}], is {typevar}"
            assert typevar is not None, typeerr
            assert typevar.__origin__ == Union, typeerr
            assert typevar.__args__ == tuple(inner_classes), typeerr
        super(SumTypeMeta, self).__init__(outername, bases, clsdict)


class SumType(metaclass=SumTypeMeta):
    pass
