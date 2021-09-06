# %%
import asyncio
import json
import sys
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
)

import numpy as np
import plotly
import plotly.graph_objs as go
import tap
from PIL import ImageColor
from plotly.missing_ipywidgets import FigureWidget
from tqdm import tqdm

from playground.train import runname_with_time


class Tbout(NamedTuple):
    steps: np.ndarray
    values: np.ndarray
    runsdir: Path


class Args(tap.Tap):
    runs: List[str]
    mean: Literal["median", "mean"] = "median"
    err_band: Literal["none", "75", "std"] = "75"
    show_all_runs: bool = False
    title: str
    no_title: bool = False
    max_steps: int = int(1e9)
    skip_min_iters: int = 0
    tb_variable: str = "eval/mean_reward"
    scale: Literal["linear", "log"] = "linear"
    top: Optional[float] = None

    def __getstate__(self):
        return self.as_dict()

    def __setstate__(self, d):
        self.__init__()
        self.from_dict(d)


def read_tensorboard(runsdir: Path, args: Args):
    from tensorboard.backend.event_processing import event_accumulator

    g = list(runsdir.glob("PPO_1/events.out.tfevents*"))
    if len(g) == 0:
        print(f"warn: given dir empty: {runsdir}")
        return None
    (eventsfile,) = g
    acc = event_accumulator.EventAccumulator(str(eventsfile))
    acc.Reload()
    try:
        scalars = acc.Scalars(args.tb_variable)
    except KeyError as e:
        print(runsdir, e)
        return None
    steps, values = zip(*[(s.step, s.value) for s in scalars])
    return Tbout(
        steps=np.array(steps, dtype=np.float64),
        values=np.array(values, dtype=np.float64),
        runsdir=runsdir,
    )


def expand_globs(runs: List[str]):
    groups: dict[str, list[str]] = {}
    target = None
    for run in tqdm(runs):
        if run.endswith(":"):
            category = run.removesuffix(":")
            target = groups.setdefault(category, [])
        else:
            if target is None:
                raise Exception("must start with category:")
            if run.startswith("!"):
                for f in Path(".").glob(run.removeprefix("!")):
                    if str(f) in target:
                        target.remove(str(f))
            else:
                target.extend([str(p) for p in Path(".").glob(run)])
    print("groups:")
    for group, vals in list(groups.items()):
        print(f"{group}: {len(vals)} runs")
        if len(vals) == 0:
            del groups[group]
    return groups


async def read_groups(runs: Dict[str, str], args: Args) -> dict[str, list[Tbout]]:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    promises = {}
    groups: dict[str, list[Tbout]] = {}

    with ProcessPoolExecutor() as pool:

        for groupname, runs in tqdm(runs.items()):
            promises[groupname] = [
                asyncio.wrap_future(pool.submit(read_tensorboard, Path(run), args))
                for run in runs
            ]
        for group in promises:
            out = await asyncio.gather(*promises[group])
            groups[group] = [g for g in out if g is not None]

    return groups


def agg_gauss(rows, outx):
    x, y = np.concatenate(rows, axis=1)
    # from sklearn.linear_model import BayesianRidge
    from sklearn.gaussian_process import GaussianProcessRegressor

    model = GaussianProcessRegressor()
    x = x[..., None]  # add axis since we have a single "feature"
    outx = outx[..., None]
    model.fit(X=x, y=y)
    y_mean, y_std = model.predict(outx, return_std=True)
    print("std", y_std[0:10])
    return y_mean, y_mean - y_mean * y_std, y_mean + y_mean * y_std


def agg_simple(rows: list[tuple[np.ndarray, np.ndarray]], outx, args: Args):
    # print("rows", rows)
    y = np.array([y for x, y in rows])
    print("yshape", y.shape)
    if args.mean == "median":
        y_mean = np.median(y, axis=0)
    elif args.mean == "mean":
        y_mean = np.mean(y, axis=0)
    else:
        raise Exception("unknown mean")
    if args.err_band == "75":
        lower = np.percentile(y, 25, axis=0)
        upper = np.percentile(y, 75, axis=0)
    elif args.err_band == "std":
        y_std = y.std(axis=0)
        lower = y_mean - y_std
        upper = y_mean + y_std
    elif args.err_band == "none":
        return y_mean, None, None
    else:
        raise Exception("unknown band")
    return y_mean, lower, upper
    pass


colors = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]
colors = [",".join(map(str, ImageColor.getrgb(color))) for color in colors]
T = TypeVar


def find_index(l: Iterable[int], fn: Callable[[int], bool]):
    for i, e in enumerate(l):
        if fn(e):
            return i


def top_n(rows: list[tuple[np.ndarray, np.ndarray]], top: Optional[float]):
    if not top:
        return rows
    rows.sort(key=lambda r: -np.mean(r[1]))
    if top < 1:
        return rows[0 : round(top * len(rows))]
    else:
        return rows[0:top]


def get_step_idx(row: Tbout, max_steps):
    return find_index(row.steps, lambda e: e > max_steps) or int(1e9)


def print_median_run(rows: list[Tbout]):
    rows = [(row.runsdir, np.mean(row.values)) for row in rows]
    rows.sort(key=lambda row: row[1])
    medrow = rows[len(rows) // 2]
    print(f"median run is {medrow[0]} btw with score {medrow[1]:.2g}")
    print(f"max run is {rows[-1][0]} with score {rows[-1][1]}")
    print("mean scores", ", ".join([f"{row[1]:.2g}" for row in rows]))


def process_group(group: str, rows: list[Tbout], color, args: Args):
    if len(rows) == 0:
        raise Exception(f"group empty: {group}")
    # find the index in the steps array that corresponds to max_steps
    minlenrow1 = min(rows, key=lambda row: get_step_idx(row, args.max_steps))
    minlen1 = get_step_idx(minlenrow1, args.max_steps)
    minlenrow = min(
        rows,
        key=lambda r: len(r.steps) if len(r.steps) >= args.skip_min_iters else 9999999,
    )
    minlen2 = len(minlenrow.steps)
    maxlen2 = len(max(rows, key=lambda r: len(r.steps)).steps)
    if minlen2 < maxlen2:
        print(
            f"warning: cutting of group {group} at min len {minlen2} (max len {maxlen2}) cause of {minlenrow.runsdir}"
        )
    print(
        f"{group}: {minlen1=}, {minlen2=} (row={minlenrow[2]}), max step = {minlenrow[0][-1]}"
    )
    minlen = min(minlen1, minlen2)
    firstx = None
    firstrunname = rows[0].runsdir
    rows2 = []
    print_median_run(rows)
    for i, row in enumerate(rows):
        if len(row.steps) < args.skip_min_iters:
            continue
        # minlen = min(len(x), len(rx))
        rx: np.ndarray = row.steps[0:minlen]
        ry: np.ndarray = row.values[0:minlen]
        if firstx is None:
            firstx = rx
        if (firstx != rx).all():
            print(
                f"a row of group {group} does not match: [{firstx[0:3]}, ...len {len(firstx)}] (runname={firstrunname}) != [{rx[0:3]}, ...len {len(rx)}] ({row.runsdir})"
            )
        else:
            rows2.append((rx, ry))
    # sample, x, y

    if len(rows2) == 0:
        raise Exception(f"{group=} empty")
    rows2 = top_n(rows2, args.top)
    outx = rows2[0][0]

    mean, stdmin, stdmax = agg_simple(rows2, outx, args)
    # print("outx", outx[0:10])
    # print("mean", mean[0:10])
    # print("stdmin", stdmin[0:10])
    # print("stdmax", stdmax[0:10])
    # print("ys[9]", [row[1][9] for row in rows])
    series = []
    series.append(
        go.Scatter(
            name=f"{group} (n={len(rows2)})",
            x=outx,
            y=mean,
            mode="lines",
            line=dict(color=f"rgba({color},1)"),
        )
    )
    if args.show_all_runs:
        for i, (rx, ry) in enumerate(rows2):
            series.append(
                go.Scatter(
                    name=f"{group}[{i}]",
                    x=rx,
                    y=ry,
                    line=dict(color=f"rgba({color},0.1)"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    if stdmin is not None:
        series.append(
            go.Scatter(
                name=group + ".std",
                x=np.concatenate([outx, outx[::-1]]),
                y=np.concatenate([stdmax, stdmin[::-1]]),
                fill="toself",
                fillcolor=f"rgba({color}, 0.3)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
        )
    return series


def cmp_series(a: FigureWidget) -> int:
    if a.name.endswith(".std"):
        return -2
    if "Max Agg" in a.name or "TRL, Mean Agg" in a.name:
        return -1
    return 0


async def create_figure(args: Args):
    inverted = False
    if args.scale == "log" and args.tb_variable == "eval/mean_reward":
        args.tb_variable = "eval/mean_neg_reward"
        inverted = True
    run_groups = expand_globs(args.runs)
    groups = await read_groups(run_groups, args)
    if len(groups) > len(colors):
        raise Exception(f"not enough colors")
    series = []
    for color, (group, rows) in zip(colors, groups.items()):
        series.extend(process_group(group, rows, color, args))

    series = sorted(series, key=cmp_series)
    layout = dict(
        title=None if args.no_title else args.title,
        xaxis=dict(title="Agent Steps"),
        yaxis=dict(
            title="Avg. Reward",
            type=args.scale,
            tickformat=".1r" if args.scale == "log" else None,
            autorange="reversed" if inverted else True,
        ),
    )
    if args.no_title:
        layout["width"] = 600
        layout["height"] = 300
    figure = go.Figure(
        series,
        layout,
    )
    figure.layout.font.family = "Latin Modern Roman"
    figure.layout.font.color = "black"
    if args.no_title:
        figure.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return figure, groups


async def save_plot(args: Args):
    name = runname_with_time(args.title)
    figure, groups = await create_figure(args)
    meta = json.dumps(
        {
            "argv": sys.argv,
            "args": args.get_reproducibility_info(),
            "groups": {
                group: [str(ele.runsdir) for ele in eles]
                for group, eles in groups.items()
            },  # run_groups,
            "plotly": figure,
        },
        cls=plotly.utils.PlotlyJSONEncoder,
        indent="\t",
    )
    Path(f"plots/{name}.json").write_text(meta)
    for ext in ["pdf", "svg"]:
        plot_path = f"plots/{name}.{ext}"
        figure.write_image(plot_path)
    return plot_path


async def run():
    args = Args().parse_args()
    await save_plot(args)


async def reproduce(jsonpath: str):
    args = Args().parse_args(json.load(open(jsonpath))["argv"][1:])
    await save_plot(args)


if __name__ == "__main__":
    asyncio.run(run())
# %%
