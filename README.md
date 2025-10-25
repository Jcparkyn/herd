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

### Lists and dicts

```dart
var arr = ['a', 'b', 'c'];
set arr.[1] = 'z';
println arr; // prints ['a', 'z', 'c']
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

Standard library modules can be imported using the `@` prefix:

```dart
List = import '@list';
```

Many standard library modules are already imported for you, e.g. `System`, `IO`, `List`, `Dict`, `Math`, `String`.
