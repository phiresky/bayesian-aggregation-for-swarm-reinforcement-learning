import random
import sys
from io import TextIOWrapper
from pathlib import Path


class TeeStdoutToFile(object):
    def __init__(self, file: TextIOWrapper):
        self.file = file
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


class TeeStderrToFile(object):
    def __init__(self, file: TextIOWrapper):
        self.file = file
        self.stderr = sys.stderr
        sys.stderr = self

    def __del__(self):
        sys.stderr = self.stderr
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stderr.write(data)

    def flush(self):
        self.file.flush()
        self.stderr.flush()


def tee_output_to_file(name: Path):
    file = name.open("w")
    print(f"logging to file {name}")
    TeeStdoutToFile(file), TeeStderrToFile(file)


def create_git_tag(git_tag_name: str):
    import subprocess
    import time

    e = Exception()
    maxretry = 10
    commitid = None
    output2 = None
    # this can fail if another instance of the program is started at the same time with a slow file system (e.g. on clister)
    # so retry it a few times
    for retry in range(0, maxretry):
        try:
            # create a git "stash" without pushing it on the stash stack
            # this basically just creates and returns a commit hash that contains all the dirty changes of the repo
            # includes all changed files, but not newly created / untracked files (!)
            commitid = subprocess.check_output(
                ["git", "stash", "create"], encoding="ascii", stderr=subprocess.STDOUT
            ).strip()
            if len(commitid) == 0:
                # repo is not dirty, the tag is just the current commit
                commitid = "HEAD"

            output2 = subprocess.check_output(
                [
                    "git",
                    "tag",
                    "--annotate",
                    "--message",
                    f"repo state for run {git_tag_name}",
                    git_tag_name,
                    commitid,
                ],
                encoding="ascii",
                stderr=subprocess.STDOUT,
            )

            print(f"created git tag {git_tag_name} as {commitid}")
            return
        except subprocess.CalledProcessError as ee:
            # most likely reason is that repo is locked (index.lock exists)
            print(
                f"could not create commit, waiting 1-60s and retrying ({retry}/{maxretry})"
            )
            time.sleep(random.uniform(1, 60))
            e = ee
    print(f"even after 10 retries: {commitid};; {output2};; {e}")
    print("no git tag created :( ignoring")
    # raise e


def setup_logging(run_dir: Path):
    tee_output_to_file(run_dir / "output.log")
    create_git_tag(str(run_dir))
