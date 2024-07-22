use interpreter::{analyze_statement, analyze_statements, Interpreter, InterpreterError};
use lalrpop_util::{lalrpop_mod, ParseError};
use std::{
    collections::HashSet,
    env,
    io::{stdin, stdout, Write},
};

mod ast;
mod interpreter;
mod parse_helpers;

lalrpop_mod!(
    #[allow(clippy::ptr_arg)]
    #[rustfmt::skip]
    pub lang
);

fn main() {
    let args: Vec<String> = env::args().collect();

    if let [_, path] = &args[..] {
        let program = match std::fs::read_to_string(path) {
            Ok(program) => program,
            Err(err) => {
                println!("Error while reading file: {}", err);
                return;
            }
        };
        let parser = lang::ProgramParser::new();
        let mut interpreter = Interpreter::new();
        let program_ast = parser.parse(&program);
        match program_ast {
            Err(err) => {
                println!("Error while parsing: {}", err);
            }
            Ok(mut program) => {
                analyze_statements(&mut program, &mut HashSet::new());
                for statement in program {
                    match interpreter.execute(&statement) {
                        Ok(()) => {}
                        Err(InterpreterError::Return(_)) => {
                            return println!(
                                "Error: You can only use return statements inside a function."
                            )
                        }
                        Err(err) => return println!("Error while evaluating: {}", err),
                    }
                }
            }
        }
    } else {
        run_repl();
    }
}

fn run_repl() {
    let parser = lang::StatementParser::new();
    let mut interpreter = Interpreter::new();
    loop {
        print!("> ");
        match stdout().flush() {
            Err(err) => {
                println!("Error while flushing: {}", err);
                break;
            }
            _ => {}
        }
        let mut buffer = String::new();
        loop {
            if let Err(err) = stdin().read_line(&mut buffer) {
                println!("Error while reading: {}", err);
                return;
            }
            let ast_result = parser.parse(&buffer);
            match ast_result {
                Err(ParseError::UnrecognizedEof { .. }) => {}
                Err(err) => {
                    println!("Error while parsing: {}", err);
                    break;
                }
                Ok(mut statement) => {
                    let mut deps = HashSet::from_iter(interpreter.list_globals().cloned());
                    analyze_statement(&mut statement, &mut deps);
                    println!("ast: {:?}", statement);
                    match interpreter.execute(&statement) {
                        Ok(()) => {}
                        Err(InterpreterError::Return(value)) => {
                            return println!("Return value: {value}")
                        }
                        Err(err) => println!("Error while evaluating: {}", err),
                    }
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{interpreter, lang};
    use rstest::rstest;

    #[rstest(
    exp,
    expected,
    case("1 + 2", 3),
    case("2 * 2", 4),
    case("3 - 9", -6),
    case("-5 * -7", 35),
    case("100 * 0", 0),
    case("8 * 2 + 3", 19),
    case("6 * (2 + 3)", 30),
    case("-100 - (-1000)", 900),
    case("150 / (20 - 5) * 3", 30),
    case("150 / ((20 - 5) * 5)", 2),
    case("10 * 15 / ((20 - 5) * 5)", 2),
    case("(100 + 50) / ((20 - 5) * 5)", 2),
    case("100 + 150 / ((20 - 5) * 5)", 102),
    )]
    fn when_expression_evaluated_then_correct_value_returned(exp: &str, expected: i32) {
        use crate::interpreter::Value;

        let expr = lang::ExprParser::new().parse(exp).unwrap();
        assert_eq!(
            interpreter::Interpreter::new().eval(&expr).unwrap(),
            Value::Number(expected as f64)
        );
    }
}
