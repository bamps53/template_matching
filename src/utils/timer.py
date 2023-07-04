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


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds to run.")
        return result
    return wrapper
