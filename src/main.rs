use analysis::Analyzer;
use clap::Parser;
use interpreter::Interpreter;
use lalrpop_util::{lalrpop_mod, ParseError};
use rustyline::error::ReadlineError;
use rustyline::highlight::MatchingBracketHighlighter;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{
    Cmd, Completer, Config, Editor, EventHandler, Helper, Highlighter, Hinter, KeyCode, KeyEvent,
    Modifiers, Movement, Result,
};

mod analysis;
mod ast;
mod interpreter;
mod parse_helpers;
mod pos;
mod value;

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
                        print_errors(errs);
                        return;
                    }
                }
                for statement in program {
                    match interpreter.execute(&statement) {
                        Ok(()) => {}
                        Err(err) => {
                            println!("Error while evaluating (at {:?}): {}", err.span, err.value);
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
                print_errors(errs);
                continue;
            }
        }

        for statement in statements {
            match interpreter.execute(&statement) {
                Ok(()) => {}
                Err(err) => {
                    println!("Error while evaluating (at {:?}): {}", err.span, err.value);
                    return;
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
