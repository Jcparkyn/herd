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
