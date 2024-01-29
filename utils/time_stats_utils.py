import time

from data import DATA_DIR


def time_stats_decorator(category: str = "other"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            res = func(*args, **kwargs)
            with open(DATA_DIR / "time_stats.csv", "a") as f:
                f.write(f'{category},{func.__name__},"{str(args)}","{str(kwargs)}",{time.time() - start_time}\n')
            return res
        return wrapper

    return decorator
