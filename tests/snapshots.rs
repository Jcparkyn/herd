use std::fmt::Display;
use std::rc::Weak;

use bovine::analysis::Analyzer;
use bovine::jit;
use bovine::lang::ProgramParser;
use bovine::value64::{Value64, RC_TRACKER};

fn reset_tracker() {
    RC_TRACKER.with(|tracker| {
        let mut tracker = tracker.borrow_mut();
        tracker.lists.0.clear();
    });
}

fn assert_rcs_dropped() {
    fn assert_rc_dropped<T: Display>(rc: &Weak<T>) {
        assert_eq!(
            0,
            rc.strong_count(),
            "This Rc still has {} references: {}",
            rc.strong_count(),
            rc.upgrade().unwrap()
        );
    }
    RC_TRACKER.with(|tracker| {
        let tracker = tracker.borrow_mut();
        for l in tracker.lists.0.iter() {
            assert_rc_dropped(&l);
        }
        for d in tracker.dicts.0.iter() {
            assert_rc_dropped(&d);
        }
        for s in tracker.strings.0.iter() {
            assert_rc_dropped(&s);
        }
        // TODO
        // for l in tracker.lambdas.0.iter() {
        //     assert_rc_dropped(&l);
        // }
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
        x = (a = 1; a);
        y = (a = 2; a + x);
        var z = 1;
        (
            set z = z + 1;
        );
        return [x, y, z];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[1, 3, 2]");
    assert_rcs_dropped();
}

#[test]
fn index_list() {
    let program = r#"
        return [1, 2, 3].[1];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"2");
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
        var sum = 0;
        for x in [1, 2, 3] do (
            set sum = sum + x;
        )
        return sum;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"6");
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
        f = \a\ [if a then 1 else (return 0;)];
        return [f true, f false];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[[1], 0]");
    // assert_rcs_dropped();
}

#[test]
fn string_literal() {
    let program = r#"
        return ['hello', 'world'];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @r#"['hello', 'world']"#);
}

#[test]
fn string_interning() {
    let program = r#"
        return ['hello', 'hello'];
    "#;
    let result = eval_snapshot(program);
    insta::assert_snapshot!(result.to_string(), @r#"['hello', 'hello']"#);
    let list = result.try_into_list().unwrap();
    // This is a dirty way to check that the pointers are equal.
    assert_eq!(list.values[0].bits(), list.values[1].bits());
}

#[test]
fn dict_literal() {
    let program = r#"
        dict = [a: 1, b: 2];
        return [dict, dict.a, dict.b];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[[a: 1, b: 2], 1, 2]");
}

#[test]
fn array_assign() {
    let program = r#"
        a = [1, 2];
        var b = a;
        set b.[0] = 4;
        return [a, b];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[[1, 2], [4, 2]]");
}

#[test]
fn dict_assign() {
    let program = r#"
        a = [x: 1, y: 2];
        var b = a;
        set b.x = 4;
        return [a, b];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[[x: 1, y: 2], [x: 4, y: 2]]");
}

#[test]
fn nested_assign() {
    let program = r#"
        a = [[x: 1, y: 2], 3];
        var b = a;
        set b.[0].x = 4;
        set b.[1] = 5;
        return [a, b];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[[[x: 1, y: 2], 3], [[x: 4, y: 2], 5]]");
}

#[test]
fn lambda_func() {
    let program = r#"
        f = \x y\ x + 2 * y;
        return f 2 3;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"8");
}

#[test]
fn func_with_captures() {
    let program = r#"
        a = 6;
        f = \x\ [x, a];
        return f 2;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[2, 6]");
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
}

#[test]
fn simple_recursion() {
    let program = r#"
        fac = \n\ if n < 2 then 1 else (n * (fac (n - 1)));
        return fac 5;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"120");
}

#[test]
fn pattern_assignment_simple() {
    let program = r#"
        ![a, b] = [1, 2];
        return [b, a];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[2, 1]");
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
}

#[test]
fn pattern_assignment_nested() {
    let program = r#"
        ![[a, b], c] = [[1, 2], 3];
        return [a, b, c];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[1, 2, 3]");
}

#[test]
fn match_expression() {
    let program = r#"
        a = [2, 3];
        b = switch a {
            [] => 0,
            [x, y] => x + y,
        };
        c = switch [] {
            [] => 0,
            [x, y] => x + y,
        };
        return [b, c];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[5, 0]");
}

#[test]
fn match_constant() {
    let program = r#"
        f = \x\ switch x {
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
}

fn eval_snapshot(program: &str) -> Value64 {
    let parser = ProgramParser::new();
    let prelude_ast = parser.parse(include_str!("../src/prelude.bovine")).unwrap();
    let mut program_ast = parser.parse(program).unwrap();
    program_ast.splice(0..0, prelude_ast);
    let mut analyzer = Analyzer::new();
    analyzer.analyze_statements(&mut program_ast).unwrap();

    let mut jit = jit::JIT::new();
    let main_func = jit.compile_program_as_function(&program_ast).unwrap();
    reset_tracker();
    let result = unsafe { jit.run_func(main_func, Value64::NIL) };

    result
}

fn eval_snapshot_str(program: &str) -> String {
    eval_snapshot(program).to_string()
}
