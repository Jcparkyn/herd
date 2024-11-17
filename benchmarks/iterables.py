import itertools


def custom_range(start, stop):
    current = start
    while current < stop:
        yield current
        current += 1


def step(iter, n):
    for i, x in enumerate(iter):
        if i % n == 0:
            yield x


def ncycles(iterable, n):
    "Returns the sequence elements n times."
    return itertools.chain.from_iterable(itertools.repeat(tuple(iterable), n))


l = custom_range(0, 10000000)
l = map(lambda x: -x, l)
l = ncycles(l, 2)
l = step(l, 3000000)
l = list(l)

print(l)
assert len(l) == 7
assert l[-1] == -8000000
