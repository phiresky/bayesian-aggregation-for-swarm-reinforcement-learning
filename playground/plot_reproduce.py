import asyncio
import sys

from .plot import reproduce

path = sys.argv[1]
asyncio.run(reproduce(path))
