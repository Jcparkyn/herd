#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt::Display;
use std::path::PathBuf;

use herd::analysis::Analyzer;
use herd::error::HerdError;
use herd::jit::{self, ModuleLoader, VmContext};
use herd::lang::ProgramParser;
use herd::prelude::PRELUDE;
use herd::rc::Weak;
use herd::value64::{Boxable, RC_TRACKER, RcTrackList, Value64};

pub fn reset_tracker() {
    fn reset_list<T: Boxable>(list: &mut RcTrackList<T>) {
        list.0.retain_mut(|(rc, count)| {
            if rc.strong_count() == 0 {
                return false;
            }
            *count = rc.strong_count();
            return true;
        });
    }
    RC_TRACKER.with(|tracker| {
        let mut tracker = tracker.borrow_mut();
        reset_list(&mut tracker.lists);
        reset_list(&mut tracker.dicts);
        reset_list(&mut tracker.strings);
        reset_list(&mut tracker.lambdas);
    });
}

#[track_caller] // Report the location of the test failure, not inside this function
pub fn assert_rcs_dropped() {
    fn assert_rc_dropped<T: Display>(rc: &Weak<T>, expected_count: usize) {
        assert_eq!(
            expected_count,
            rc.strong_count(),
            "This Rc still has {} references but should have {}: {}",
            rc.strong_count(),
            expected_count,
            rc.upgrade().unwrap()
        );
    }
    RC_TRACKER.with(|tracker| {
        let tracker = tracker.borrow_mut();
        for (l, c) in tracker.lists.0.iter() {
            assert_rc_dropped(l, *c);
        }
        for (d, c) in tracker.dicts.0.iter() {
            assert_rc_dropped(d, *c);
        }
        for (s, c) in tracker.strings.0.iter() {
            assert_rc_dropped(s, *c);
        }
        for (l, c) in tracker.lambdas.0.iter() {
            assert_rc_dropped(l, *c);
        }
    });
}

pub struct EvalConfig<'a> {
    pub program: &'a str,
    pub modules: HashMap<String, String>,
}

impl EvalConfig<'_> {
    pub fn with_module(mut self, path: impl Into<String>, source: impl Into<String>) -> Self {
        self.modules.insert(path.into(), source.into());
        self
    }

    pub fn eval(&self) -> Result<Value64, HerdError> {
        eval_snapshot(self.program, self.modules.clone())
    }

    #[track_caller]
    pub fn expect_ok(&self) -> Value64 {
        self.eval().expect("The program should return successfully")
    }

    #[track_caller]
    pub fn expect_ok_string(&self) -> String {
        self.expect_ok().to_string()
    }

    #[track_caller]
    pub fn expect_err(&self) -> HerdError {
        self.eval().expect_err("The program should return an error")
    }

    #[track_caller]
    pub fn expect_err_string(&self) -> String {
        let error = self.expect_err();
        error_to_string(&error, 0)
    }
}

pub fn eval(program: &str) -> EvalConfig<'_> {
    EvalConfig {
        program,
        modules: HashMap::new(),
    }
}

fn error_to_string(err: &HerdError, indent: usize) -> String {
    let mut result = String::new();

    if let Some(inner) = &err.inner {
        result.push_str(&error_to_string(inner, indent));
    }

    let indent_str = " ".repeat(indent);
    let pos_str = match err.pos {
        Some(pos) => pos.to_string(),
        None => "[internal method]".to_string(),
    };
    result.push_str(&format!("{}At {}: {}\n", indent_str, pos_str, err.message));

    result
}

pub fn eval_snapshot(
    program: &str,
    modules: HashMap<String, String>,
) -> Result<Value64, HerdError> {
    let parser = ProgramParser::new();
    let prelude_ast = parser.parse(PRELUDE).unwrap();
    let mut program_ast = parser.parse(program).unwrap();
    program_ast.splice(0..0, prelude_ast);
    let mut analyzer = Analyzer::new();
    analyzer.analyze_statements(&mut program_ast).unwrap();

    let module_loader = TestModuleLoader { modules };
    let mut jit = jit::JIT::new(Box::new(module_loader));
    let src_path = PathBuf::new();
    let main_func = jit
        .compile_program_as_function(&program_ast, &src_path)
        .unwrap();
    reset_tracker();
    let vmc = VmContext::new(jit, vec![]);
    let result = unsafe { vmc.run_func(main_func, vec![]) };
    result
}

pub fn eval_snapshot_str(program: &str) -> String {
    eval_snapshot(program, HashMap::new()).unwrap().to_string()
}

struct TestModuleLoader {
    modules: HashMap<String, String>,
}

impl ModuleLoader for TestModuleLoader {
    fn load(&self, path: &str) -> std::io::Result<String> {
        match self.modules.get(path) {
            Some(source) => Ok(source.clone()),
            None => Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Module not found: {}", path),
            )),
        }
    }
}
