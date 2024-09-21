use std::fmt::Debug;
use std::mem;

use bovine::analysis::{AnalysisError, Analyzer};
use bovine::ast;
use bovine::ast::Expr;
use bovine::interpreter::{Interpreter, InterpreterError};
use bovine::jit;
use bovine::lines::{Lines, Location};
use bovine::pos::Spanned;
use bovine::value64::Value64;
use clap::Parser;
use lalrpop_util::{lalrpop_mod, ParseError};
use mimalloc::MiMalloc;
use rustyline::error::ReadlineError;
use rustyline::highlight::MatchingBracketHighlighter;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{
    Cmd, Completer, Config, Editor, EventHandler, Helper, Highlighter, Hinter, KeyCode, KeyEvent,
    Modifiers, Movement,
};

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

    #[arg(long)]
    jit: bool,
}

fn main() {
    let args = Args::parse();

    if let Some(ref path) = args.file {
        run_file(path, &args);
    } else {
        run_repl(args);
    }
}

fn run_file(path: &str, args: &Args) {
    let program_str = match std::fs::read_to_string(path) {
        Ok(program) => program,
        Err(err) => {
            println!("Error while reading file: {}", err);
            return;
        }
    };
    let lines = Lines::new(program_str.clone().into_bytes());
    let parser = lang::ProgramParser::new();
    let mut program = match parser.parse(&program_str) {
        Err(err) => {
            println!("Error while parsing: {}", err);
            return;
        }
        Ok(program) => program,
    };
    let prelude_ast = parser.parse(include_str!("prelude.bovine")).unwrap();
    program.splice(0..0, prelude_ast);
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
    if args.jit {
        let mut jit = jit::JIT::new();
        for statement in program {
            let func = match statement.value {
                ast::Statement::PatternAssignment(_, rhs) => match rhs.value {
                    Expr::Lambda(f) => f,
                    _ => panic!("Only function definitions are allowed at the top level"),
                },
                ast::Statement::Expression(e) => match e.value {
                    // Ignore main () call, for compatibility with tree-walker
                    Expr::Call { .. } => continue,
                    _ => panic!("Only function definitions are allowed at the top level"),
                },
                _ => panic!("Only function definitions are allowed at the top level"),
            };
            match jit.compile_func(&func) {
                Ok(_) => {}
                Err(err) => {
                    println!("Error while compiling function: {:?}", err);
                    return;
                }
            }
        }
        let main_func = jit
            .get_func_code("main")
            .expect("Main function should be defined");
        let result = unsafe { run_func(main_func, Value64::NIL) };
        println!("Result: {}", result);
    } else {
        let mut interpreter = Interpreter::new();
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

    let mut prelude_ast = parser.parse(include_str!("prelude.bovine")).unwrap();
    analyzer.analyze_statements(&mut prelude_ast).unwrap();
    for stmt in prelude_ast {
        interpreter.execute(&stmt).unwrap();
    }
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
    fn validate(&self, ctx: &mut ValidationContext<'_>) -> rustyline::Result<ValidationResult> {
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
            "Cant access index {} of a list with {} elements",
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
            // Spans for prelude functions can be outside of the original source file.
            // We need to also store the file for each span.
            let inner_location = lines.location(inner.span.start).unwrap_or(Location {
                ..Default::default()
            });
            write!(f, "\tat ")?;
            writeln!(
                f,
                "{} ({})",
                function.self_name.as_deref().unwrap_or("<lambda>"),
                inner_location
            )
        }
    }?;
    if outer {
        let location = lines.location(err.span.start).unwrap();
        writeln!(f, "\tat {}", location)?;
    };
    Ok(())
}

unsafe fn run_func(func_ptr: *const u8, input: Value64) -> Value64 {
    // Cast the raw pointer to a typed function pointer. This is unsafe, because
    // this is the critical point where you have to trust that the generated code
    // is safe to be called.
    let code_fn = mem::transmute::<_, extern "C" fn(Value64) -> Value64>(func_ptr);
    // And now we can call it!
    code_fn(input)
}

// fn add_stdlib(program: &mut Vec<SpannedStatement>) {
//     fn build

//     fn build_std_func(func: BuiltInFunction, arg_count: usize) -> SpannedStatement {
//         let zero_span = Span::new(0, 0);
//         let rhs: Expr = Expr::Lambda(LambdaExpr::new(params, body));
//         let stmt = ast::Statement::PatternAssignment(
//             zero_span.wrap(ast::MatchPattern::Declaration(
//                 ast::VarRef::new(func.to_string()),
//                 ast::DeclarationType::Const,
//             )),
//             zero_span.wrap(rhs),
//         );
//         Spanned::new(Span::new(0, 0), stmt)
//     }
//     program.insert(0, build_std_func(BuiltInFunction::Len));
// }
