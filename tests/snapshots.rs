use bovine::analysis::Analyzer;
use bovine::jit;
use bovine::lang::ProgramParser;
use bovine::value64::Value64;

#[test]
fn add() {
    let program = r#"
        main = \\ 1 + 2;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"3");
}

#[test]
fn sub() {
    let program = r#"
        main = \\ 1 - 2;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"-1");
}

#[test]
fn bodmas() {
    let program = r#"
        main = \\ 6 + (1 - 2) * 3;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"3");
}

#[test]
fn array_literal() {
    let program = r#"
        main = \\ [1, 2, 3];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[1, 2, 3]");
}

#[test]
fn variables() {
    let program = r#"
        main = \\ (
            a = 1;
            b = 2;
            a + b
        );
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"3");
}

#[test]
fn equals() {
    let program = r#"
        main = \\ [
            0 == 1, 1 == 1, 1 == (), () == (),
            [0] == [1], [1] == [1]
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[false, true, false, true, false, true]");
}

#[test]
fn cmp_lt() {
    let program = r#"
        main = \\ [
            0 < 1, 1 < 1, 1 < (), () < (),
            0 <= 1, 1 <= 1, 1 <= (), () <= (),
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[true, false, false, false, true, true, false, false]");
}

#[test]
fn cmp_gt() {
    let program = r#"
        main = \\ [
            0 > 1, 1 > 1, 1 > (), () > (),
            0 >= 1, 1 >= 1, 1 >= (), () >= (),
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[false, false, false, false, false, true, false, false]");
}

#[test]
fn index_list() {
    let program = r#"
        main = \\ [1, 2, 3].[1];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"2");
}

#[test]
fn if_else() {
    let program = r#"
        main = \\ [
            if 1 == 1 then 1 else 0,
            if 1 == 1 then 1,
            if 1 == 0 then 1 else 0,
            if 1 == 0 then 1,
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[1, 1, 0, ()]");
}

#[test]
fn logic_and() {
    let program = r#"
        main = \\ [
            true and true,
            true and false,
            false and true,
            false and false,
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[true, false, false, false]");
}

#[test]
fn logic_or() {
    let program = r#"
        main = \\ [
            true or true,
            true or false,
            false or true,
            false or false,
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[true, true, true, false]");
}

#[test]
fn for_in_loop() {
    let program = r#"
        main = \\ (
            var sum = 0;
            for x in [1, 2, 3] do (
                set sum = sum + x;
            )
            sum
        );
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"6");
}

#[test]
fn while_loop() {
    let program = r#"
        main = \\ (
            var sum = 1;
            while sum < 10 do (
                set sum = sum * 2;
            )
            sum
        );
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"16");
}

#[test]
fn user_functions() {
    let program = r#"
        square = \a\ a * a;
        main = \\ square 3;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"9");
}

#[test]
fn builtin_range() {
    let program = r#"
        main = \\ range -1 3;
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[-1, 0, 1, 2]");
}

#[test]
fn builtin_not() {
    let program = r#"
        main = \\ [not true, not 1, not false];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[false, false, true]");
}

#[test]
fn early_return_if() {
    let program = r#"
        f = \a\ [if a then 1 else (return 0;)];
        main = \\ [
            f true,
            f false,
        ];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[[1], 0]");
}

#[test]
fn string_literal() {
    let program = r#"
        main = \\ ['hello', 'world'];
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @r#"['hello', 'world']"#);
}

#[test]
fn string_interning() {
    let program = r#"
        main = \\ ['hello', 'hello'];
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
        main = \\ (
            dict = [a: 1, b: 2];
            [dict, dict.a, dict.b]
        );
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[[a: 1, b: 2], 1, 2]");
}

#[test]
fn array_assign() {
    let program = r#"
        main = \\ (
            a = [1, 2];
            var b = a;
            set b.[0] = 4;
            [a, b]
        );
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[[1, 2], [4, 2]]");
}

#[test]
fn dict_assign() {
    let program = r#"
        main = \\ (
            a = [x: 1, y: 2];
            var b = a;
            set b.x = 4;
            [a, b]
        );
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[[x: 1, y: 2], [x: 4, y: 2]]");
}

#[test]
fn nested_assign() {
    let program = r#"
        main = \\ (
            a = [[x: 1, y: 2], 3];
            var b = a;
            set b.[0].x = 4;
            set b.[1] = 5;
            [a, b]
        );
    "#;
    let result = eval_snapshot_str(program);
    insta::assert_snapshot!(result, @"[[[x: 1, y: 2], 3], [[x: 4, y: 2], 5]]");
}

fn eval_snapshot(program: &str) -> Value64 {
    let parser = ProgramParser::new();
    let prelude_ast = parser.parse(include_str!("../src/prelude.bovine")).unwrap();
    let mut program_ast = parser.parse(program).unwrap();
    program_ast.splice(0..0, prelude_ast);
    let mut analyzer = Analyzer::new();
    analyzer.analyze_statements(&mut program_ast).unwrap();

    let mut jit = jit::JIT::new();
    jit.compile_program(&program_ast).unwrap();

    let main_func = jit
        .get_func_id("main")
        .expect("Main function should be defined");
    let result = unsafe { jit.run_func(main_func, Value64::NIL) };

    result
}

fn eval_snapshot_str(program: &str) -> String {
    eval_snapshot(program).to_string()
}
