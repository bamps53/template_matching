import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f'Elapsed time for {name:<20}: {end - start:.4f}')