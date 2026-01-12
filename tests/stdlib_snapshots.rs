mod common;
use common::snapshot_helpers::{assert_rcs_dropped, eval};

#[test]
fn stdlib_imports() {
    let main_program = r#"
        list = import '@list';
        return list.push [1, 2] 3;
    "#;

    let result = eval(main_program).prelude(true).expect_ok_string();
    insta::assert_snapshot!(result, @"[1, 2, 3]");
    assert_rcs_dropped();
}

#[test]
fn stdlib_map() {
    let main_program = r#"
        list = import '@list';
        return [1, 2] | list.map \(_ + 1);
    "#;

    let result = eval(main_program).prelude(true).expect_ok_string();
    insta::assert_snapshot!(result, @"[2, 3]");
    assert_rcs_dropped();
}

#[test]
fn stdlib_filter() {
    let main_program = r#"
        list = import '@list';
        return [1, 2, 3] | list.filter \(_ != 2);
    "#;

    let result = eval(main_program).prelude(true).expect_ok_string();
    insta::assert_snapshot!(result, @"[1, 3]");
    assert_rcs_dropped();
}

#[test]
fn stdlib_reverse() {
    let main_program = r#"
        list = import '@list';
        return [1, 2, 3] | list.reverse;
    "#;

    let result = eval(main_program).prelude(true).expect_ok_string();
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

    let result = eval(main_program).prelude(true).expect_ok_string();
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

    let result = eval(main_program).prelude(true).expect_ok_string();
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

    let result = eval(main_program).prelude(true).expect_ok_string();
    insta::assert_snapshot!(result, @"[11, 0, 5]");
    assert_rcs_dropped();
}

#[test]
fn stdlib_sort() {
    let main_program = r#"
        !{ sort } = import '@list';
        return sort [1, 'dog', 3, 2, 'zebra', 'cat'];
    "#;

    let result = eval(main_program).prelude(true).expect_ok_string();
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

    let result = eval(main_program).prelude(true).expect_ok_string();
    insta::assert_snapshot!(result, @"5001");
    assert_rcs_dropped();
}

#[test]
fn parallel_map_error() {
    let main_program = r#"
        inputs = range 0 2;
        results = Parallel.parallelMap inputs (\x\ x + '1');
        return results;
    "#;

    let result = eval(main_program).prelude(true).expect_err_string();
    insta::assert_snapshot!(result, @r###"
    Error: Expected an f64, found '1'
    at someMethod (main.herd:3:56)
    Caused another error: Error in parallel map
    at someMethod (@parallel:1:23)
    at someMethod (main.herd:3:19)
    "###);
}

#[test]
fn stdlib_sort_error() {
    let main_program = r#"
        !{ sort } = import '@list';
        return sort { a: 1, b: 2 };
    "#;

    let result = eval(main_program).expect_err_string();
    insta::assert_snapshot!(result, @r###"
    Error: Expected a list, got {a: 1, b: 2}
    at someMethod (@list:11:14)
    at someMethod (main.herd:3:16)
    "###);
}

#[test]
fn stdlib_slice_error() {
    let main_program = r#"
        list = [1, 2, 3];
        return List.slice list 'a' 1;
    "#;

    let result = eval(main_program).prelude(true).expect_err_string();
    insta::assert_snapshot!(result, @r###"
    Error: Expected a number, got 'a'
    at someMethod (@list:43:25)
    at someMethod (main.herd:3:16)
    "###);
}

#[test]
fn stdlib_map_error() {
    let main_program = r#"
        return [1, 2] | List.map 123;
    "#;

    let result = eval(main_program).prelude(true).expect_err_string();
    insta::assert_snapshot!(result, @r###"
    Error: Tried to call something that isn't a function: 123
    at someMethod (@list:16:19)
    at someMethod (main.herd:2:25)
    "###);
}

#[test]
fn stdlib_import_error_non_existent() {
    let main_program = r#"
        return import '@nonexistent_module';
    "#;

    let result = eval(main_program).expect_err_string();
    insta::assert_snapshot!(result, @r###"
    Error: Error importing module '@nonexistent_module': MissingStdLibModule("@nonexistent_module")
    at someMethod (main.herd:2:16)
    "###);
}

#[test]
fn stdlib_import_error_malformed_path() {
    let main_program = r#"
        return import 'malformed/path';
    "#;

    let result = eval(main_program).expect_err_string();
    insta::assert_snapshot!(result, @r###"
    Error: Error importing module 'malformed/path': File(Custom { kind: NotFound, error: "Module not found: malformed/path" })
    at someMethod (main.herd:2:16)
    "###);
}
