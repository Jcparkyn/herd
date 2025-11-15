use std::collections::HashMap;
use std::fmt::Display;
use std::path::PathBuf;

use herd::analysis::Analyzer;
use herd::jit::{self, ModuleLoader, VmContext};
use herd::lang::ProgramParser;
use herd::rc::Weak;
use herd::value64::{Boxable, RcTrackList, Value64, RC_TRACKER};

fn reset_tracker() {
    fn reset_list<T: Boxable>(list: &mut RcTrackList<T>) {
        list.0.retain_mut(|(rc, count)| {
            if rc.strong_count() == 0 {
                return false;
            }
            *count = rc.strong_count();
            return true;
        });
    }
    RC_TRACKER.with(|tracker| {
        let mut tracker = tracker.borrow_mut();
        reset_list(&mut tracker.lists);
        reset_list(&mut tracker.dicts);
        reset_list(&mut tracker.strings);
        reset_list(&mut tracker.lambdas);
    });
}

fn assert_rcs_dropped() {
    fn assert_rc_dropped<T: Display>(rc: &Weak<T>, expected_count: usize) {
        assert_eq!(
            expected_count,
            rc.strong_count(),
            "This Rc still has {} references but should have {}: {}",
            rc.strong_count(),
            expected_count,
            rc.upgrade().unwrap()
        );
    }
    RC_TRACKER.with(|tracker| {
        let tracker = tracker.borrow_mut();
        for (l, c) in tracker.lists.0.iter() {
            assert_rc_dropped(l, *c);
        }
        for (d, c) in tracker.dicts.0.iter() {
            assert_rc_dropped(d, *c);
        }
        for (s, c) in tracker.strings.0.iter() {
            assert_rc_dropped(s, *c);
        }
        for (l, c) in tracker.lambdas.0.iter() {
            assert_rc_dropped(l, *c);
        }
    });
}

#[test]
fn add() {
    let program = r#"
        return 1 + 2;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"3");
    assert_rcs_dropped();
}

#[test]
fn sub() {
    let program = r#"
        return 1 - 2;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"-1");
    assert_rcs_dropped();
}

#[test]
fn bodmas() {
    let program = r#"
        return 6 + (1 - 2) * 3;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"3");
    assert_rcs_dropped();
}

#[test]
fn array_literal() {
    let program = r#"
        return [1, 2, 3];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[1, 2, 3]");
    assert_rcs_dropped();
}

#[test]
fn nested_array_literal() {
    let program = r#"
        return [[1, 2], [3]];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[[1, 2], [3]]");
    assert_rcs_dropped();
}

#[test]
fn variables() {
    let program = r#"
        return (
            a = 1;
            b = 2;
            a + b
        );
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"3");
    assert_rcs_dropped();
}

#[test]
fn equals() {
    let program = r#"
        return [
            0 == 1, 1 == 1, 1 == (), () == (),
            [0] == [1], [1] == [1]
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[false, true, false, true, false, true]");
    assert_rcs_dropped();
}

#[test]
fn cmp_lt() {
    // TODO: should we panic for non-numbers?
    let program = r#"
        return [
            0 < 1, 1 < 1, 1 < (), () < (),
            0 <= 1, 1 <= 1, 1 <= (), () <= (),
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[true, false, false, false, true, true, false, false]");
    assert_rcs_dropped();
}

#[test]
fn cmp_gt() {
    let program = r#"
        return [
            0 > 1, 1 > 1, 1 > (), () > (),
            0 >= 1, 1 >= 1, 1 >= (), () >= (),
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[false, false, false, false, false, true, false, false]");
    assert_rcs_dropped();
}

#[test]
fn blocks() {
    let program = r#"
        x = (a = 'A'; a);
        y = (a = 'B'; [a, x]);
        var z = 'Z';
        (
            set z = [z, 'Y'];
        );
        return [x, y, z];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"['A', ['B', 'A'], ['Z', 'Y']]");
    assert_rcs_dropped();
}

#[test]
fn index_list() {
    let program = r#"
        return ['a', 'b', 'c'].[1];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"'b'");
    assert_rcs_dropped();
}

#[test]
fn index_dict() {
    let program = r#"
        dict = {a: 'A', b: 'B'};
        return [dict, dict.a, dict.b];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[{a: 'A', b: 'B'}, 'A', 'B']");
    assert_rcs_dropped();
}

#[test]
fn index_dict_drop() {
    let program = r#"
        dict = {a: [1], b: [2]};
        return dict.a;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[1]");
    assert_rcs_dropped();
}

#[test]
fn index_dict_parameter() {
    let main_program = r#"
        update = \x\ { val: x.val };
        var state = { val: [42] };
        set state = update state;
        return state;
    "#;

    let result = eval_snapshot_str(main_program);
    insta::assert_snapshot!(result, @"{val: [42]}");
    assert_rcs_dropped();
}

#[test]
fn if_else() {
    let program = r#"
        return [
            if 1 == 1 then 1 else 0,
            if 1 == 1 then 1,
            if 1 == 0 then 1 else 0,
            if 1 == 0 then 1,
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[1, 1, 0, ()]");
    assert_rcs_dropped();
}

#[test]
fn if_else_else_if() {
    let program = r#"
        f = \n\ if n == 1 then '1' else if n == 2 then '2' else 'unknown';
        return [
            f 1,
            f 2,
            f 3,
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"['1', '2', 'unknown']");
    assert_rcs_dropped();
}

#[test]
fn logic_and() {
    let program = r#"
        return [
            true and true,
            true and false,
            false and true,
            false and false,
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[true, false, false, false]");
    assert_rcs_dropped();
}

#[test]
fn logic_and_reference() {
    let program = r#"
        return [
            [1] and 'abc',
            '' and 'false',
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[true, false]");
    assert_rcs_dropped();
}

#[test]
fn logic_and_short_circuit_truthy() {
    let program = r#"
        var marker = 0;
        result = 42 and (set marker = 1; true);
        return [result, marker];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[true, 1]");
    assert_rcs_dropped();
}

#[test]
fn logic_and_short_circuit_falsy() {
    let program = r#"
        var marker = 0;
        result = '' and (set marker = 1; true);
        return [result, marker];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[false, 0]");
    assert_rcs_dropped();
}

#[test]
fn logic_or_short_circuit_truthy() {
    let program = r#"
        var marker = 0;
        result = 42 or (set marker = 1; false);
        return [result, marker];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[true, 0]");
    assert_rcs_dropped();
}

#[test]
fn logic_or_short_circuit_falsy() {
    let program = r#"
        var marker = 0;
        result = '' or (set marker = 1; true);
        return [result, marker];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[true, 1]");
    assert_rcs_dropped();
}

#[test]
fn logic_or_reference() {
    let program = r#"
        return [
            [1] or 'abc',
            '' or [],
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[true, false]");
    assert_rcs_dropped();
}

#[test]
fn logic_or() {
    let program = r#"
        return [
            true or true,
            true or false,
            false or true,
            false or false,
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[true, true, true, false]");
    assert_rcs_dropped();
}

#[test]
fn for_in_loop() {
    let program = r#"
        var result = [];
        for x in ['a', 'b', 'c'] do (
            set result |= push x;
        )
        return result;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"['a', 'b', 'c']");
    assert_rcs_dropped();
}

#[test]
fn for_in_loop_scope() {
    let program = r#"
        var result = [];
        for [x, y] in [['a', 'b']] do (
            set result |= push x;
            set result |= push y;
        )
        x = 1;
        y = 2;
        return result;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"['a', 'b']");
    assert_rcs_dropped();
}

#[test]
fn while_loop() {
    let program = r#"
        var sum = 1;
        while sum < 10 do (
            set sum = sum * 2;
        )
        return sum;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"16");
    assert_rcs_dropped();
}

#[test]
fn user_functions() {
    let program = r#"
        square = \a\ a * a;
        return square 3;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"9");
    assert_rcs_dropped();
}

#[test]
fn user_functions_2() {
    let program = r#"
        mul = \a b\ a * b;
        return [
            mul 3 4,
            mul 5 6,
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[12, 30]");
    assert_rcs_dropped();
}

#[test]
fn builtin_range() {
    let program = r#"
        return range -1 3;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[-1, 0, 1, 2]");
    assert_rcs_dropped();
}

#[test]
fn builtin_not() {
    let program = r#"
        return [not true, not 1, not false];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[false, false, true]");
    assert_rcs_dropped();
}

#[test]
fn early_return_if() {
    let program = r#"
        f = \a\ [if a then 'A' else (return 'B';)];
        return [f true, f false];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[['A'], 'B']");
    assert_rcs_dropped();
}

#[test]
fn string_literal() {
    let program = r#"
        return ['hello', 'world'];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @r#"['hello', 'world']"#);
    assert_rcs_dropped();
}

#[test]
fn string_interning() {
    let program = r#"
        return ['hello', 'hello'];
    "#;
    let result = eval_snapshot(program, HashMap::new());
    insta::assert_snapshot!(result.to_string(), @r#"['hello', 'hello']"#);
    let list = result.try_into_list().unwrap();
    // This is a dirty way to check that the pointers are equal.
    assert_eq!(list.values[0].bits(), list.values[1].bits());
    drop(list);
    assert_rcs_dropped();
}

#[test]
fn dict_literal() {
    let program = r#"
        dict = {a: 'A', b: 'B'};
        return [dict, dict.a, dict.b];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[{a: 'A', b: 'B'}, 'A', 'B']");
    assert_rcs_dropped();
}

#[test]
fn dict_literal_shorthand() {
    let program = r#"
        a = 'A';
        dict = {a, b: 'B'};
        return dict;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"{a: 'A', b: 'B'}");
    assert_rcs_dropped();
}

#[test]
fn array_assign() {
    let program = r#"
        a = ['1', '2', '3'];
        var b = a;
        set b.[0] = '4';
        set b.[1] = '5';
        return [a, b];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[['1', '2', '3'], ['4', '5', '3']]");
    assert_rcs_dropped();
}

#[test]
fn dict_assign() {
    let program = r#"
        a = {x: 1, y: 2};
        var b = a;
        set b.x = 4;
        return [a, b];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[{x: 1, y: 2}, {x: 4, y: 2}]");
    assert_rcs_dropped();
}

#[test]
fn nested_assign() {
    let program = r#"
        a = [{x: 1, y: 2}, 3];
        var b = a;
        set b.[0].x = 4;
        set b.[0].y = b.[0].y + 1;
        set b.[1] = 5;
        return [a, b];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[[{x: 1, y: 2}, 3], [{x: 4, y: 3}, 5]]");
    assert_rcs_dropped();
}

#[test]
fn lambda_func() {
    let program = r#"
        f = \x y\ x + 2 * y;
        return f 2 3;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"8");
    assert_rcs_dropped();
}

#[test]
fn call_order() {
    let program = r#"
        f = \a\ not a;
        return (f (f true));
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"true");
    assert_rcs_dropped();
}

#[test]
fn func_with_captures() {
    let program = r#"
        a = 'a';
        f = \x\ [x, a];
        return f 'b';
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"['b', 'a']");
    assert_rcs_dropped();
}

#[test]
fn multiple_funcs() {
    let program = r#"
        f = \a\ a + 1;
        g = \b\ (f b) * 2;
        return g 6;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"14");
    assert_rcs_dropped();
}

#[test]
fn func_with_locals() {
    let program = r#"
        a = 6;
        f = \x\ (
            b = x + 1;
            b * 2
        );
        return f 2;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"6");
    assert_rcs_dropped();
}

#[test]
fn func_with_pattern_matching() {
    let program = r#"
        a = 6;
        f = \[x, y]\ (
            x + y * 2
        );
        return f [3, 4];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"11");
    assert_rcs_dropped();
}

#[test]
fn simple_recursion() {
    let program = r#"
        fac = \n\ if n < 2 then 1 else (n * (fac (n - 1)));
        return fac 5;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"120");
    assert_rcs_dropped();
}

#[test]
fn pattern_assignment_simple() {
    let program = r#"
        ![a, b] = [1, 2];
        return [b, a];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[2, 1]");
    assert_rcs_dropped();
}

#[test]
fn pattern_assignment_set() {
    let program = r#"
        var a = 3;
        ![set a, b] = [1, 2];
        return [b, a];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[2, 1]");
    assert_rcs_dropped();
}

#[test]
fn pattern_assignment_nested() {
    let program = r#"
        ![[a, b], c] = [[1, 2], 3];
        return [a, b, c];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[1, 2, 3]");
    assert_rcs_dropped();
}

#[test]
fn pattern_assignment_dict() {
    let program = r#"
        !{a, b: [b0, b1]} = {a: 1, b: [2, 3], c: 4};
        return [a, b0, b1];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[1, 2, 3]");
    assert_rcs_dropped();
}

#[test]
fn string_concat() {
    let program = r#"
        return [
            'hello ' ++ 'world',
            'hello ' ++ 'world' ++ '!',
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"['hello world', 'hello world!']");
    assert_rcs_dropped();
}

#[test]
fn list_concat() {
    let program = r#"
        return ['hello', 'world'] ++ ['!'];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"['hello', 'world', '!']");
    assert_rcs_dropped();
}

#[test]
fn match_expression() {
    let program = r#"
        a = [2, 3];
        b = switch a on {
            [] => 0,
            [x, y] => x + y,
        };
        c = switch [] on {
            [] => 0,
            [x, y] => x + y,
        };
        return [b, c];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[5, 0]");
    assert_rcs_dropped();
}

#[test]
fn match_empty_dict() {
    let program = r#"
        x = {};
        return switch x on {
            {} => 'empty_dict', 
        };
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"'empty_dict'");
    assert_rcs_dropped();
}

#[test]
fn match_dict() {
    let program = r#"
        f = \x\ switch x on {
            () => 'nil',
            [] => 'empty_list',
            {a, b: 0} => ['dict0', a],
            {a, b: 1} => ['dict1', a],
            {a, b: [2, c]} => ['dict2', a, c],
            {} => 'any_dict',
        };
        return [
            f (),
            f [],
            f {a: 'A'},
            f {a: 'A', b: 0},
            f {a: 'A', b: 1},
            f {a: 'A', b: [2, 3]},
            f {},
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"['nil', 'empty_list', 'any_dict', ['dict0', 'A'], ['dict1', 'A'], ['dict2', 'A', 3], 'any_dict']");
    assert_rcs_dropped();
}

#[test]
fn match_constant() {
    let program = r#"
        f = \x\ switch x on {
            () => 'nil',
            [] => 'empty',
            'foo' => 'foo',
            'bar' => 'bar',
            [a] => 'singleton',
            [0, a] => 'zero',
            [1, a] => 'one',
            [_, a] => a,
        };
        return [
            f (),
            f [],
            f 'foo',
            f 'bar',
            f [42],
            f [0, 1],
            f [1, 2],
            f [3, 4],
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"['nil', 'empty', 'foo', 'bar', 'singleton', 'zero', 'one', 4]");
    assert_rcs_dropped();
}

#[test]
fn binary_trees() {
    let program = r#"
        makeTree = \d\ if d > 0 then (
            [makeTree (d - 1), makeTree (d - 1)]
        ) else (
            [(), ()]
        );
        checkTree = \node\ switch node on {
            [(), ()] => 1,
            [l, r] => 1 + (checkTree l) + (checkTree r),
        };
        return [
            checkTree (makeTree 0),
            checkTree (makeTree 1),
            checkTree (makeTree 2),
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[1, 3, 7]");
    assert_rcs_dropped();
}

#[test]
fn mutual_recursion() {
    let program = r#"
        isEven_ = \isOdd n\ if n == 0 then true else (isOdd (n - 1));
        isOdd = \n\ if n == 0 then false else (isEven_ isOdd (n - 1));
        isEven = \n\ isEven_ isOdd n;
        return [
            isEven 0,
            isEven 1,
            isEven 5,
            isEven 6,
            isOdd 0,
            isOdd 1,
            isOdd 5,
            isOdd 6,
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[true, false, false, true, false, true, true, false]");
    assert_rcs_dropped();
}

#[test]
fn simple_imports() {
    let main_program = r#"
        foo = import 'foo.herd';
        return foo.fn1 41;
    "#;

    let foo_program = r#"
        fn1 = \a\ a + 1;
        return { fn1 };
    "#;

    let modules = HashMap::from([("foo.herd".to_string(), foo_program.to_string())]);
    let result = eval_snapshot_str_modules(main_program, modules);
    insta::assert_snapshot!(result, @"42");
    assert_rcs_dropped();
}

#[test]
fn stdlib_imports() {
    let main_program = r#"
        list = import '@list';
        return list.push [1, 2] 3;
    "#;

    let result = eval_snapshot_str(main_program);
    insta::assert_snapshot!(result, @"[1, 2, 3]");
    assert_rcs_dropped();
}

#[test]
fn stdlib_map() {
    let main_program = r#"
        list = import '@list';
        return [1, 2] | list.map \(_ + 1);
    "#;

    let result = eval_snapshot_str(main_program);
    insta::assert_snapshot!(result, @"[2, 3]");
    assert_rcs_dropped();
}

#[test]
fn stdlib_filter() {
    let main_program = r#"
        list = import '@list';
        return [1, 2, 3] | list.filter \(_ != 2);
    "#;

    let result = eval_snapshot_str(main_program);
    insta::assert_snapshot!(result, @"[1, 3]");
    assert_rcs_dropped();
}

#[test]
fn stdlib_reverse() {
    let main_program = r#"
        list = import '@list';
        return [1, 2, 3] | list.reverse;
    "#;

    let result = eval_snapshot_str(main_program);
    insta::assert_snapshot!(result, @"[3, 2, 1]");
    assert_rcs_dropped();
}

#[test]
fn stdlib_slice() {
    let main_program = r#"
        list = [1, 2, 3];
        return [
            List.slice list 0 1,
            List.slice list () 1,
            List.slice list 1 (),
            List.slice list 1 2,
            List.slice list 2 5,
            List.slice list -2 -1,
            List.slice list -2 (),
        ];
    "#;

    let result = eval_snapshot_str(main_program);
    insta::assert_snapshot!(result, @"[[1], [1], [2, 3], [2], [3], [2], [2, 3]]");
    assert_rcs_dropped();
}

#[test]
fn stdlib_bitwise_xor() {
    let main_program = r#"
        !{xor} = import '@bitwise';
        return [
            xor 1 2,
            xor 7 9,
            xor 0.5 1.5,
        ];
    "#;

    let result = eval_snapshot_str(main_program);
    insta::assert_snapshot!(result, @"[3, 14, 1]");
    assert_rcs_dropped();
}

#[test]
fn stdlib_bitwise_and() {
    let main_program = r#"
        !{ bitwiseAnd } = import '@bitwise';
        return [
            bitwiseAnd 43 27,
            bitwiseAnd 0 0,
            bitwiseAnd 13.5 7.5,
        ];
    "#;

    let result = eval_snapshot_str(main_program);
    insta::assert_snapshot!(result, @"[11, 0, 5]");
    assert_rcs_dropped();
}

#[test]
fn stdlib_sort() {
    let main_program = r#"
        !{ sort } = import '@list';
        return sort [1, 'dog', 3, 2, 'zebra', 'cat'];
    "#;

    let result = eval_snapshot_str(main_program);
    insta::assert_snapshot!(result, @"['cat', 'dog', 'zebra', 1, 2, 3]");
    assert_rcs_dropped();
}

#[test]
fn parallel_map() {
    let main_program = r#"
        inputs = range 0 10002;
        results = Parallel.parallelMap inputs (\x\ x - 5000);
        var sum = 0;
        for x in results do (
            set sum = sum + x;
        )
        return sum;
    "#;

    let result = eval_snapshot_str(main_program);
    insta::assert_snapshot!(result, @"5001");
    assert_rcs_dropped();
}

fn eval_snapshot(program: &str, modules: HashMap<String, String>) -> Value64 {
    let parser = ProgramParser::new();
    let prelude_ast = parser.parse(include_str!("../src/prelude.herd")).unwrap();
    let mut program_ast = parser.parse(program).unwrap();
    program_ast.splice(0..0, prelude_ast);
    let mut analyzer = Analyzer::new();
    analyzer.analyze_statements(&mut program_ast).unwrap();

    let module_loader = TestModuleLoader { modules };
    let mut jit = jit::JIT::new(Box::new(module_loader));
    let src_path = PathBuf::new();
    let main_func = jit
        .compile_program_as_function(&program_ast, &src_path)
        .unwrap();
    reset_tracker();
    let vmc = VmContext::new(jit, vec![]);
    let result = unsafe { vmc.run_func(main_func, vec![]) };

    result
}

fn eval_snapshot_str(program: &str) -> String {
    eval_snapshot(program, HashMap::new()).to_string()
}

fn eval_snapshot_str_modules(program: &str, modules: HashMap<String, String>) -> String {
    eval_snapshot(program, modules).to_string()
}

struct TestModuleLoader {
    modules: HashMap<String, String>,
}

impl ModuleLoader for TestModuleLoader {
    fn load(&self, path: &str) -> std::io::Result<String> {
        match self.modules.get(path) {
            Some(source) => Ok(source.clone()),
            None => Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Module not found: {}", path),
            )),
        }
    }
}
