"""compare the params of different runs in a plot"""

import json
from pathlib import Path
from pprint import pprint
from typing import Any, List, NamedTuple

import deepdiff
from tap import Tap


class Arg(Tap):
    path: str


args = Arg().parse_args()

config = json.loads(Path(args.path).read_text())


class Group(NamedTuple):
    config: Any
    members: List[Any]
    diff: Any
    diff_to: str


have_groups = []
for k, v in config["groups"].items():
    for dir in v:
        config = json.loads((Path(dir) / "full_params.json").read_text())
        del config["runname"]
        found = False
        best_match = None
        for potential in have_groups:
            diff = deepdiff.DeepDiff(potential.config, config)
            if diff == {}:
                potential.members.append((k, dir))
                found = True
                break
            if best_match is None or len(diff.to_json()) < len(best_match[0].to_json()):
                best_match = (diff, potential.members[0])
        if not found:
            have_groups.append(
                Group(
                    config=config,
                    members=[(k, dir)],
                    diff=best_match[0] if best_match else config,
                    diff_to=best_match[1] if best_match else None,
                )
            )

print(f"{len(have_groups)} config groups:")
for group in have_groups:
    print("members:")
    pprint(group.members)
    print(f"diff from {group.diff_to} to {group.members[0][0]}")
    pprint(group.diff)
