# Herd

Herd is a simple interpreted programming language where everything is a value.

**Disclaimer: This is a hobby language, and you probably shouldn't use it for anything important.**

## What makes Herd special?

In Herd, everything is pass-by-value, including lists and dicts.
This means that when you pass a list or dict to a function, **you can guarantee that the function won't modify your copy**.

*But isn't that just immutability?* Not quite, because lists and dicts in Herd are actually mutable.
This means you can modify them locally just like you would in an imperative language, but there will be no side-effects on other copies of the value.

## How does it work?

All reference types in Herd (e.g. strings, lists, dicts) use reference counting. Whenever you make a copy of a reference type, the reference count is incremented. Whenever you _modify_ a reference type, one of two things happen:
- If there's only one reference to the value, it can be modified in-place without making any copies. This is because the code doing the modification must be the only reference, so no other code will be able to observe the modification.
- If there's more than one reference to the value, the language makes a shallow copy of it with the modification applied. The copy now has a reference count of one, so subsequent modifications usually don't need to allocate.

There's one very convenient consequence of everything being a value: _Reference cycles are impossible!_ This means the reference counting system doesn't need cycle detection, and can also be used as a garbage collector.

## Language tour

### Hello world

Use the `println` function to print to the console. Strings are defined using single quotes.

```dart
println 'Hello world!';
```

### Variables

```dart
x = 1; // define an immutable variable
var y = 2; // define a mutable variable
set y = 3; // modify a mutable variable
```

### Types

Values in herd have the following types:

- `()` the unit type, representing no value.
- `bool` boolean values: `true` and `false`.
- `number` 64-bit floating point number, e.g. `42`, `-7`, `2.6`.
- `string` a text string, e.g. `'hello'`.
- `list` an ordered collection of values, e.g. `[1, 2, 3]`.
- `dict` a collection of key-value pairs, e.g. `{ x: 1, y: 2 }`.
- `function` a function value, e.g. `\x y\ x + y`.

### Lists and dicts

```dart
var list = ['a', 'b', 'c'];
set list.[1] = 'z';
println list; // prints ['a', 'z', 'c']
```

```dart
var dict = { x: 1, y: 2 };
set dict.x = 3;
println dict; // prints { x: 3, y: 2 }
```

You can use shorthand when creating dicts if the key name matches the variable name:

```dart
x = 10;
y = 20;
dict = { x, y }; // equivalent to { x: x, y: y }
```

### Blocks

In herd, blocks are defined using parentheses. If the last expression in the block is not terminated with a semicolon, its value becomes the value of the block.

```dart
x = (
  a = 2;
  b = 3;
  a + b
);
println x; // prints 5
```

### Functions

All functions in herd are defined using the anonymous function syntax:

```dart
// define function taking two parameters
multiply1 = \a b\ (
  return a * b;
);
// call the function
x = multiply1 3 4;
println x; // prints 12
```

The body of the function is just a block expression, so you can omit the `return` statement if you want:

```dart
multiply2 = \a b\ (a * b);
```

You can also omit the parentheses for simple expressions:

```dart
multiply3 = \a b\ a * b;
```

And there's an even shorter syntax for single-parameter functions:

```dart
increment = \(_ + 1);
```

Functions with no parameters can be defined using `\\`, and called by passing a unit value `()`:

```dart
getRandomNumber = \\ Math.randomInt 0 100;
println (getRandomNumber ()); // prints a random number
```

### Values and mutability

In herd, _everything_ is immutable unless declared with `var`. This includes lists and complex objects that would be mutable in other languages.

```dart
list = ['a', 'b']; // list is immutable, because it wasn't defined with var.
set list.[0] = 'A'; // Not allowed
```

Each copy of a value is a _distinct copy_, and modifications to one variable won't modify other variables - even for lists and dicts!

```dart
var list1 = ['a', 'b'];
var list2 = list1; // make a copy
set list2.[0] = 'c'; // modify list2
println list1; // prints ['a', 'b']
println list2; // prints ['c', 'b']
```

This also applies when passing values to functions:

```dart
doubleEveryItem = \var list\ (
  for [i, x] in List.enumerate list do (
    set list.[i] = x * 2;
  )
);
list1 = [3, 4];
list2 = doubleEveryItem list1;
println list1; // prints [3, 4]
println list2; // prints [6, 8]
```

### Pattern matching

Use `!` to destructure lists and dicts:
```dart
list = [1, 2];
![a, b] = list; // destructure list
println a; // prints 1
println b; // prints 2

dict = { x: 10, y: 20 };
!{ x, y } = dict; // destructure dict
println x; // prints 10
println y; // prints 20
```

You can also use pattern matching in `switch` expressions to handle multiple cases:

```dart
x = [1, 2, 3];
y = switch x on {
  [] => 'Empty list',
  [1, ...rest] => 'Begins with 1, then ' ++ (toString rest),
  [a] => 'Single item: ' ++ (toString a),
  _ => 'Something else',
};
println y; // prints 'Begins with 1, then [2, 3]'
```

You can use `var` to destructure to a mutable variable, or `set` to modify an existing variable:

```dart
list = [1, 2];
[a, var b] = list; // a is immutable, b is mutable
set b = b + 10;
println [list, a, b]; // prints [[1, 2], 1, 12]
```

### The pipe operator

The pipe operator `|` can be used to chain function calls in a more readable way:

```dart
x = [1, 2, 3]
  | List.map \(_ * 2) // double each item
  | List.filter \(_ > 3); // keep only items greater than 3
println x; // prints [4, 6]
```

You can combine the pipe operator with `set` to modify variables in-place using `|=`:

```dart
var x = [1, 2];
set x |= List.push 3; // equivalent to set x = x | List.push 3;
println x; // prints [1, 2, 3]
```

### Modules and imports

Export code from a file by returning it at the end of the file:

```dart
// in file1.herd
x = 42;
double = \n\ n * 2;
return { x, double };
```

Import code from other files using the `import` function:

```dart
// in file2.herd
!{ x, double } = import 'file1.herd';
println (double x); // prints 84
```

You can also import all code from a file:

```dart
// in file2.herd
File1 = import 'file1.herd';
println (File1.double File1.x); // prints 84
```

### Standard library

Standard library modules are already imported for you, and can be accessed from the imported modules:

- `System` - system functions for getting current time, program args, etc.
- `IO` - file input/output functions.
- `List` - list utility functions.
- `Dict` - dict utility functions.
- `Math` - mathematical functions and constants.
- `Bitwise` - bitwise operations on integers.
- `Random` - random number generation.
- `String` - string utility functions.
- `File` - file system utility functions.
- `Parallel` - multithreading utilities.

Some very commonly used functions are also available globally:
- `not`
- `range`
- `assert`
- `toString`
- `print`
- `println`
- `printf`
- `len`

### Multithreading

Herd has built-in support for multithreading using the `Parallel` standard library module.

Use `Parallel.parallelMap` to map a function over a list in parallel:
```dart
list = [1, 2, 3];
// square each item in parallel
squared = Parallel.parallelMap list \(_ * _);
println squared; // prints [1, 4, 9]
```

Use `Parallel.parallelRun` to run multiple functions in parallel and wait for all of them to complete:
```dart
getOrder = \id\ (...); // TODO
getCatalog = \\ (...); // TODO
orderId = 123;
// Run both functions in parallel
![order, catalog] = Parallel.parallelRun [
  \\ getOrder orderId,
  \\ getCatalog (),
];
```

In herd, it is _impossible_ to create a data race, because any mutations only affect the current thread.
You can safely pass complex data structures between threads without worrying about synchronization.
However (as with the rest of herd), each function will have its own copy of the data, so the only way to communicate between threads is via the return value.

## Design choices

**Dynamic typing**
I'm not generally a fan of dynamic typing, but:
- I wanted to learn how to build an interpreter.
- I wanted the language to be as simple as possible.
- I don't think this concept has ever been explored in a dynamic language before (excluding Matlab and R, which have large caveats). Swift is the closest equivalent, but it's statically typed and much more complex.

**User-defined types**
Herd currently has a very simple type system, with no user-defined types.
If I wanted to turn this into a production-ready language, I'd probably add Julia-esque structs and multiple dispatch (following the same immutability guarantees as the rest of the language).

**Semicolons**
This is mostly just because I'm too lazy to write a whitespace-sensitive parser.

**var, set, and =**
I took a different approach with the syntax here than most other languages I've seen, in particular that the default for `=` is to define immutable variables, and mutating them requires a dedicated `set` keyword.
1. In most programs, the majority of variables can be immutable.
2. Programs define immutable variables more often than they mutate variables.

So I made immutable variable definition the "default", and gave keywords to the other two operations.

**Function syntax**
Just trying out something different which is a bit more concise than most other languages.
This is sort of a hybrid between Rust's closure syntax (but with `\` instead of `|`) and ML-style function calls (`f x y`) which require a bit less punctuation.

**Currying (or lack thereof)**
Currying has a lot of nice properties for language implementation and reasoning, but personally I don't think it's a good fit for herd and would make the language less approachable.
- With currying, we can't produce a good error message when the user calls a function with the wrong number of arguments.
- Currying is incompatible with varargs functions (which I admittedly haven't implemented yet).
- It's still possible to explicitly partially apply a function using the `\(func x _)` syntax.


## Performance

Herd uses a very naive single-pass JIT compiler to convert code to machine code at runtime, with cranelift for generating the final optimized machine code. This results in surprisingly good performance - not in the same league as modern JS runtimes, but competitive or faster than many interpreted languages (e.g. CPython).

Values in Herd are represented using NaN-boxing, so primitive types (number, bool, unit) can be stored without any heap allocation. I chose this over tagged pointers, because it makes it much easier to get good numerical performance (at least for 64-bit floats) without complex inlining logic.

The current biggest performance gaps in herd are:
1. Atomic reference counting overhead, particurly for array/list mutations in hot loops. A lot of this could be removed by lifting the atomic operations out of loops, so the hot code can safely mutate its owned copy.
2. The single-pass JIT compiler adds some startup overhead by compiling all imported code, especially as the standard library and user codebase get larger. Cranelift is still pretty fast though, so this isn't a major problem.
3. The single-pass JIT also limits the optimizations that we can do, particularly around type specialization. A more advanced tracing JIT could get a bit of extra performance, but with a big complexity cost.

Here are some benchmark numbers on an i5-13600KF, comparing herd to CPython 3.11 and JavaScript (Node.js 18.6) on a selection of scripts (see `./benchmarks` for the full code):

| Benchmark     | herd          | js               | python            |
|---------------|---------------|------------------|-------------------|
| binarytrees   | 1285.1ms      | 789.1ms (-38.6%) | 1992.4ms (55.0%)  |
| binarytrees-m | 211.1ms       |                  |                   |
| helloworld    | 25.9ms        | 53.4ms (106.1%)  | 26.8ms (3.6%)     |
| iterables     | 630.6ms       |                  | 1360.9ms (115.8%) |
| mandelbrot    | 485.6ms       | 146.1ms (-69.9%) | 3109.8ms (540.3%) |
| nbody         | 2998.1ms      | 86.0ms (-97.1%)  | 1898.2ms (-36.7%) |