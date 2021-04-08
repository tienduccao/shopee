from memory_profiler import memory_usage


MAX_MEMORY = 1000


def mem_check(func):
    def wrapper(*arg, **kwargs):
        max_mem, function_value = memory_usage(proc=(func, arg, kwargs), max_usage=True, retval=True)
        print(f"Max memory usage of {func.__name__} is {max_mem} MB")
        if max_mem > MAX_MEMORY:
            raise Exception(f'Max Memory Error {max_mem} vs {MAX_MEMORY}')
        return function_value

    return wrapper


@mem_check
def my_func(k):
    a = [1] * (k ** 6)
    b = [2] * (2 * k ** 6)
    del b
    return len(a)


print(my_func(15))
print(my_func(20))
