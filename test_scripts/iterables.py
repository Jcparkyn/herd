import itertools


def step(iter, n):
    for i, x in enumerate(iter):
        if i % n == 0:
            yield x


iter2 = step(map(lambda x: -x, range(0, 100000000)), 10000000)

print(list(iter2))
