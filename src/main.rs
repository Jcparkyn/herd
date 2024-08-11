use analysis::Analyzer;
use interpreter::{Interpreter, InterpreterError};
use lalrpop_util::{lalrpop_mod, ParseError};
use std::{
    env,
    io::{stdin, stdout, Write},
};

mod analysis;
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
                let mut analyzer = Analyzer::new();
                let analyze_result = analyzer.analyze_statements(&mut program);
                #[cfg(debug_assertions)]
                println!("ast: {:#?}", program);
                match analyze_result {
                    Ok(()) => {}
                    Err(errs) => {
                        print_errors(errs);
                        return;
                    }
                }
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
    let parser = lang::ProgramParser::new();
    let mut interpreter = Interpreter::new();
    let mut analyzer = Analyzer::new();
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
                Ok(mut statements) => {
                    match analyzer.analyze_statements(&mut statements) {
                        Ok(()) => {}
                        Err(errs) => {
                            print_errors(errs);
                            break;
                        }
                    }
                    println!("ast: {:?}", statements);
                    for statement in statements {
                        match interpreter.execute(&statement) {
                            Ok(()) => {}
                            Err(InterpreterError::Return(value)) => {
                                return println!("Return value: {value}")
                            }
                            Err(err) => println!("Error while evaluating: {}", err),
                        }
                    }
                    break;
                }
            }
        }
    }
}

fn print_errors(errs: Vec<analysis::AnalysisError>) {
    println!("Errors while analyzing:");
    for err in errs {
        println!("\t{}", err);
    }
}
