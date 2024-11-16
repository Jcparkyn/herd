# Herd

Herd is a simple interpreted programming language where everything is a value.

**Disclaimer: This is a hobby language, and you probably shouldn't use it for anything important.**

## What makes Herd special?

In Herd, everything is pass-by-value, including arrays and dicts.
This means that when you pass an array or dict to a function, **you can guarantee that the function won't modify your copy**.

*But isn't that just immutability?* Not quite, because arrays and dicts in herd are actually mutable.
This means you can modify them locally just like you would in an imperative language, but there will be no side-effects on other copies of the value.

## How does it work?

All reference types in Herd (e.g. strings, lists, dicts) use reference counting. Whenever you make a copy of a reference type, the reference count is incremented. Whenever you _modify_ a reference type, one of two things happen:
- If there's only one reference to the value, it can be modified in-place without making any copies. This is because the code doing the modification must be the only reference, so no other code will be able to observe the modification.
- If there's more than one reference to the value, the language makes a shallow copy of it with the modification applied. The copy now has a reference count of one, so subsequent modifications usually don't need to allocate.

There's one very convenient consequence of everything being a value: _Reference cycles are impossible!_ This means the reference counting system doesn't need cycle detection, and can also be used as a garbage collector.