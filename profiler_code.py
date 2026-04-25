import cProfile
import pstats
import io
from functools import wraps


def profile_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()  # start profiling

        result = func(*args, **kwargs)

        profiler.disable()  # stop profiling

        output = io.StringIO()  # in-memory text buffer
        stats = pstats.Stats(profiler, stream=output)
        stats.sort_stats("cumulative")  # sort by total time spent in function calls
        stats.print_stats(10)  # show top 10 entries

        print(f"Profiling results for {func.__name__}:")
        print(output.getvalue())

        return result

    return wrapper


@profile_function
def slow_function():
    total = 0
    for i in range(1000000):
        total += i
    return total


slow_function()