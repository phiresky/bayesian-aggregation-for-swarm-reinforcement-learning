import shutil
import time
from pathlib import Path

from tqdm import tqdm


def move_uninteresting(dir: Path):
    tfdir = dir / "PPO_1"
    if not tfdir.exists():
        interesting = False
    else:
        (eventsfile,) = tfdir.glob("events.out.tfevents*")

        interesting = is_interesting(eventsfile)
        if not interesting:
            if eventsfile.stat().st_mtime > time.time() - 2 * 60 * 60:
                print(f"recent uninteresting dir {eventsfile}, skipping")
                return
    if not interesting:
        tgtdir = "runs-bad"
        print(f"moving {dir} to {tgtdir}")
        shutil.move(str(dir), tgtdir)


def is_interesting(eventsfile: Path):

    from tensorboard.backend.event_processing import event_accumulator

    acc = event_accumulator.EventAccumulator(str(eventsfile))
    acc.Reload()
    scalars = acc.Tags()["scalars"]
    if len(scalars) == 0:
        return False
    if "eval/mean_reward" not in scalars:
        return False
    if len(acc.Scalars("eval/mean_reward")) < 3:
        return False
    return True


def dirs():
    for dir in sorted(Path("runs/").glob("*/")):
        if str(dir) < "runs/2021-07-01":  # already handled
            continue
        yield dir


def main():
    import multiprocessing

    with multiprocessing.Pool() as p:
        for _ in tqdm(p.imap(move_uninteresting, dirs())):
            pass


if __name__ == "__main__":
    main()
