function* map(fn, iter) {
  for (const x of iter) {
    yield fn(x);
  }
}

function* customRange(start, stop) {
  let current = start;
  while (current < stop) {
    yield current;
    current++;
  }
}

function* step(iter, n) {
  let i = 0;
  for (const x of iter) {
    if (i % n === 0) {
      yield x;
    }
    i++;
  }
}

function* ncycles(iterable, n) {
  const cached = Array.from(iterable);
  for (let i = 0; i < n; i++) {
    yield* cached;
  }
}

let l = customRange(0, 10000000);
l = map(x => -x, l);
l = ncycles(l, 2);
l = step(l, 3000000);
l = Array.from(l);

console.log(l);
console.assert(l.length === 7, `Expected length 7, got ${l.length}`);
console.assert(l[l.length - 1] === -8000000, `Expected -8000000, got ${l[l.length - 1]}`);