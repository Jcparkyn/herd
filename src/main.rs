use std::fmt::Debug;

use analysis::{AnalysisError, Analyzer};
use clap::Parser;
use interpreter::{Interpreter, InterpreterError};
use lalrpop_util::{lalrpop_mod, ParseError};
use lines::Lines;
use mimalloc::MiMalloc;
use pos::Spanned;
use rustyline::error::ReadlineError;
use rustyline::highlight::MatchingBracketHighlighter;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{
    Cmd, Completer, Config, Editor, EventHandler, Helper, Highlighter, Hinter, KeyCode, KeyEvent,
    Modifiers, Movement, Result,
};
use value64::Callable;

mod analysis;
mod ast;
mod interpreter;
mod jit;
mod lines;
mod parse_helpers;
mod pos;
mod value64;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

lalrpop_mod!(
    #[allow(clippy::ptr_arg)]
    #[rustfmt::skip]
    pub lang
);

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg()]
    file: Option<String>,

    #[arg(long)]
    ast: bool,
}

fn main() {
    let args = Args::parse();

    if let Some(path) = args.file {
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
        let lines = Lines::new(program.clone().into_bytes());
        match program_ast {
            Err(err) => {
                println!("Error while parsing: {}", err);
            }
            Ok(mut program) => {
                let mut analyzer = Analyzer::new();
                let analyze_result = analyzer.analyze_statements(&mut program);
                if args.ast {
                    println!("ast: {:#?}", program);
                }
                match analyze_result {
                    Ok(()) => {}
                    Err(errs) => {
                        print_analysis_errors(errs, lines);
                        return;
                    }
                }
                for statement in program {
                    match interpreter.execute(&statement) {
                        Ok(()) => {}
                        Err(err) => {
                            let formatter = InterpreterErrorFormatter {
                                err: &err,
                                lines: &lines,
                            };
                            eprintln!("Error: {}", formatter);
                            return;
                        }
                    }
                }
            }
        }
    } else {
        run_repl(args);
    }
}

fn run_repl(args: Args) {
    let rl_config = Config::builder()
        .history_ignore_space(true)
        .auto_add_history(true)
        .indent_size(4)
        .build();
    let helper = ReplInputValidator {
        parser: lang::ProgramParser::new(),
        highlighter: MatchingBracketHighlighter::new(),
    };
    let mut rl = Editor::with_config(rl_config).unwrap();
    rl.set_helper(Some(helper));
    rl.bind_sequence(
        KeyEvent(KeyCode::Tab, Modifiers::NONE),
        EventHandler::Simple(Cmd::Indent(Movement::ForwardChar(0))),
    );
    let parser = lang::ProgramParser::new();
    let mut interpreter = Interpreter::new();
    let mut analyzer = Analyzer::new();
    loop {
        let input = match rl.readline("> ") {
            Ok(input) => input,
            Err(ReadlineError::Interrupted) => return,
            Err(err) => {
                println!("Error: {err}");
                return;
            }
        };

        let ast_result = parser.parse(&input);
        let lines = Lines::new(input.clone().into_bytes());
        let mut statements = match ast_result {
            Err(err) => {
                println!("Error while parsing: {}", err);
                continue;
            }
            Ok(s) => s,
        };
        let analyze_result = analyzer.analyze_statements(&mut statements);
        if args.ast {
            println!("ast: {:?}", statements);
        }
        match analyze_result {
            Ok(()) => {}
            Err(errs) => {
                print_analysis_errors(errs, lines);
                continue;
            }
        }

        for statement in statements {
            match interpreter.execute(&statement) {
                Ok(()) => {}
                Err(err) => {
                    let formatter = InterpreterErrorFormatter {
                        err: &err,
                        lines: &lines,
                    };
                    eprintln!("Error: {}", formatter);
                    return;
                }
            }
        }
    }
}

fn print_analysis_errors(errs: Vec<Spanned<AnalysisError>>, lines: Lines) {
    println!("Errors while analyzing:");
    for err in errs {
        let location = lines.location(err.span.start).unwrap();
        println!("\t{} (at {})", err.value, location);
    }
}

#[derive(Completer, Helper, Hinter, Highlighter)]
struct ReplInputValidator {
    parser: lang::ProgramParser,
    #[rustyline(Highlighter)]
    highlighter: MatchingBracketHighlighter,
}

impl Validator for ReplInputValidator {
    fn validate(&self, ctx: &mut ValidationContext<'_>) -> Result<ValidationResult> {
        let parse_result = self.parser.parse(ctx.input());
        match parse_result {
            Err(ParseError::UnrecognizedEof { .. }) => Ok(ValidationResult::Incomplete),
            _ => Ok(ValidationResult::Valid(None)),
        }
    }
}

struct InterpreterErrorFormatter<'a> {
    err: &'a Spanned<InterpreterError>,
    lines: &'a Lines,
}

impl<'a> std::fmt::Display for InterpreterErrorFormatter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt_runtime_error(f, &self.err, &self.lines, true)
    }
}

fn fmt_runtime_error(
    f: &mut std::fmt::Formatter<'_>,
    err: &Spanned<InterpreterError>,
    lines: &Lines,
    outer: bool,
) -> std::fmt::Result {
    use InterpreterError::*;
    match &err.value {
        KeyNotExists(name) => writeln!(f, "Field {} doesn't exist", name),
        IndexOutOfRange { list_len, accessed } => writeln!(
            f,
            "Cant access index {} of an list with {} elements",
            accessed, list_len
        ),
        WrongArgumentCount { expected, supplied } => writeln!(
            f,
            "Wrong number of arguments for function. Expected {expected}, got {supplied}"
        ),
        WrongType { message } => writeln!(f, "{}", message),
        Return(_) => writeln!(f, "You can only use return statements inside a function"),
        PatternMatchFailed { message } => writeln!(f, "Unsuccessful pattern match: {}", message),
        FunctionCallFailed { function, inner } => {
            fmt_runtime_error(f, &inner, lines, false)?;
            let inner_location = lines.location(inner.span.start).unwrap();
            write!(f, "\tat ")?;
            match function {
                Callable::Lambda(l) => writeln!(
                    f,
                    "{} ({})",
                    l.self_name.as_deref().unwrap_or("<lambda>"),
                    inner_location
                ),
                Callable::Builtin(b) => writeln!(f, "{} (built-in function)", b),
            }
        }
    }?;
    if outer {
        let location = lines.location(err.span.start).unwrap();
        writeln!(f, "\tat {}", location)?;
    };
    Ok(())
}
