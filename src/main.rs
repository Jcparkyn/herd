use herd::rc::Rc;
use std::fmt::Debug;
use std::path::{Path, PathBuf};

use clap::Parser;
use herd::analysis::{AnalysisError, Analyzer};
use herd::jit::{self, DefaultModuleLoader};
use herd::lang;
use herd::lang::ProgramParser;
use herd::lines::Lines;
use herd::pos::Spanned;
use herd::value64::Value64;
use lalrpop_util::ParseError;
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

    if let Some(ref path_str) = args.file {
        let path = PathBuf::from(path_str);
        run_file(&path, &args);
    } else {
        run_repl(args);
    }
}

fn run_file(path: &Path, args: &Args) {
    let program_str = match std::fs::read_to_string(path) {
        Ok(program) => program,
        Err(err) => {
            println!("Error while reading file: {}", err);
            return;
        }
    };
    let lines = Lines::new(program_str.clone().into_bytes());
    let parser = ProgramParser::new();
    let mut program = match parser.parse(&program_str) {
        Err(err) => {
            println!("Error while parsing: {}", err);
            return;
        }
        Ok(program) => program,
    };
    let prelude_ast = parser.parse(include_str!("prelude.herd")).unwrap();
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
    let module_loader = DefaultModuleLoader {
        base_path: path.parent().unwrap().to_path_buf(),
    };
    let mut jit = jit::JIT::new(Box::new(module_loader));
    // TODO: Make it impossible to import from the root script, to prevent cycles
    // jit.modules.insert(path.canonicalize().unwrap(), None);
    let main_func = match jit.compile_program_as_function(&program, path) {
        Ok(id) => id,
        Err(err) => {
            println!("Error while compiling function: {:?}", err);
            return;
        }
    };
    let result = unsafe { jit.run_func(main_func, vec![]) };
    println!("Result: {}", result);
}

struct ReplResponseValues {
    retval: Value64,
    globals: Vec<(String, Value64)>,
}

fn get_repl_globals(return_val: &Value64) -> ReplResponseValues {
    let mut globals = vec![];
    let return_dict = return_val.as_dict().unwrap();
    let retval_key = Value64::from_string(Rc::new("<returnval>".to_string()));
    for (k, v) in return_dict.values.iter() {
        if k != &retval_key {
            globals.push((k.as_string().unwrap().clone(), v.clone()));
        }
    }
    let retval = return_dict.values.get(&retval_key).unwrap().clone();
    ReplResponseValues { retval, globals }
}

fn run_repl(args: Args) {
    let rl_config = Config::builder()
        .history_ignore_space(true)
        .auto_add_history(true)
        .indent_size(4)
        .build();
    let helper = ReplInputValidator {
        parser: lang::ReplProgramParser::new(),
        highlighter: MatchingBracketHighlighter::new(),
    };
    let mut rl = Editor::with_config(rl_config).unwrap();
    rl.set_helper(Some(helper));
    rl.bind_sequence(
        KeyEvent(KeyCode::Tab, Modifiers::NONE),
        EventHandler::Simple(Cmd::Indent(Movement::ForwardChar(0))),
    );
    let parser = lang::ReplProgramParser::new();
    let mut analyzer = Analyzer::new();

    let mut prelude_ast = parser.parse(include_str!("prelude.herd")).unwrap();
    analyzer.analyze_statements(&mut prelude_ast).unwrap();

    let current_dir = std::env::current_dir().unwrap();
    let module_loader = DefaultModuleLoader {
        base_path: current_dir.clone(),
    };
    let mut jit = jit::JIT::new(Box::new(module_loader));

    let prelude_func = jit
        .compile_repl_as_function(&prelude_ast, &current_dir, &[])
        .unwrap();
    let prelude_return = unsafe { jit.run_func(prelude_func, vec![]) };
    let mut globals = get_repl_globals(&prelude_return).globals;
    // Keep track of previous number of lines entered, so new code has unique line counts.
    let mut line_count = 0usize;

    loop {
        let input = match rl.readline("> ") {
            Ok(input) => "\n".repeat(line_count) + &input,
            Err(ReadlineError::Interrupted) => return,
            Err(err) => {
                println!("Error: {err}");
                return;
            }
        };

        let ast_result = parser.parse(&input);
        let lines = Lines::new(input.clone().into_bytes());
        line_count = lines.line_count();
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

        let (names, values): (Vec<String>, Vec<Value64>) = globals.iter().cloned().unzip();
        let func = jit
            .compile_repl_as_function(&statements, &current_dir, &names)
            .unwrap();

        let func_return = unsafe { jit.run_func(func, values) };
        if func_return.is_error() {
            // TODO: We need to roll back the analyzer state here.
            println!("Error!");
        } else {
            let response = get_repl_globals(&func_return);
            if !response.retval.is_nil() {
                println!("Result: {}", response.retval);
            }
            globals = response.globals;
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
    parser: lang::ReplProgramParser,
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
