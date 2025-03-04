use std::{collections::HashSet, fmt::Display};

use crate::{
    ast::{
        Block, DeclarationType, Expr, MatchPattern, SpannedExpr, SpannedStatement, Statement,
        VarRef,
    },
    pos::{Span, Spanned},
    rc::Rc,
};

#[derive(Debug)]
pub enum AnalysisError {
    VariableAlreadyDefined(String),
    VariableNotDefined(String),
    AssignToConst(String),
}

type SpError = Spanned<AnalysisError>;

use AnalysisError::*;

impl Display for AnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VariableAlreadyDefined(name) => {
                write!(f, "Variable {} is already defined", name)
            }
            VariableNotDefined(name) => {
                write!(f, "Variable {} is not defined", name)
            }
            AssignToConst(name) => {
                write!(
                    f,
                    "Cannot modify variable \"{}\", because it's a constant. Try declaring it with \"var\" to make it mutable (e.g. var {} = ...).",
                    name, name
                )
            }
        }
    }
}

pub struct Analyzer {
    var_analyzer: VariableAnalyzer,
}

impl Analyzer {
    pub fn new() -> Analyzer {
        Analyzer {
            var_analyzer: VariableAnalyzer::new(),
        }
    }

    pub fn analyze_statements(
        &mut self,
        stmts: &mut [SpannedStatement],
    ) -> Result<(), Vec<SpError>> {
        let initial_var_count = self.var_analyzer.vars.len();
        // Make sure we don't drop any current globals
        let mut deps = HashSet::from_iter(self.var_analyzer.list_vars().cloned());
        analyze_statements_liveness(stmts, &mut deps);
        self.var_analyzer.analyze_stamements(stmts);

        if self.var_analyzer.errors.is_empty() {
            return Ok(());
        } else {
            // Remove any globals that were added
            self.var_analyzer.vars.truncate(initial_var_count);
            Err(std::mem::replace(&mut self.var_analyzer.errors, vec![]))
        }
    }
}

struct LocalVar {
    name: String,
    depth: usize,
    mutable: bool,
}

struct VariableAnalyzer {
    vars: Vec<LocalVar>,
    depth: usize,
    errors: Vec<SpError>,
}

impl VariableAnalyzer {
    fn new() -> VariableAnalyzer {
        VariableAnalyzer {
            vars: vec![],
            depth: 0,
            errors: vec![],
        }
    }

    fn list_vars(&self) -> impl Iterator<Item = &String> + '_ {
        self.vars.iter().map(|v| &v.name)
    }

    fn push_var(&mut self, var: &mut VarRef, mutable: bool) {
        var.slot = self.vars.len() as u32;
        self.vars.push(LocalVar {
            name: var.name.clone(),
            depth: self.depth,
            mutable,
        });
    }

    fn push_err(&mut self, err: AnalysisError, span: Span) {
        self.errors.push(Spanned::new(span, err));
    }

    fn analyze_stamements(&mut self, stmts: &mut [SpannedStatement]) {
        for stmt in stmts {
            self.analyze_statement(&mut stmt.value);
        }
    }

    fn analyze_statement(&mut self, stmt: &mut Statement) {
        match stmt {
            Statement::PatternAssignment(pattern, rhs) => {
                self.analyze_expr(rhs);
                self.analyze_pattern(&mut pattern.value, pattern.span);
            }
            Statement::Expression(expr) => self.analyze_expr(expr),
            Statement::Return(expr) => self.analyze_expr(expr),
        }
    }

    fn analyze_pattern(&mut self, pattern: &mut MatchPattern, span: Span) {
        match pattern {
            MatchPattern::Declaration(var, decl_type) => {
                if let Some(_) = self.get_slot(&var.name) {
                    self.push_err(VariableAlreadyDefined(var.name.clone()), span);
                }
                self.push_var(var, *decl_type == DeclarationType::Mutable);
            }
            MatchPattern::Assignment(target) => {
                if let Some(slot) = self.get_slot(&target.var.name) {
                    target.var.slot = slot;
                    if !self.vars[slot as usize].mutable {
                        self.push_err(AssignToConst(target.var.name.clone()), span);
                    }
                } else {
                    self.push_err(VariableNotDefined(target.var.name.clone()), span);
                }
                for index in target.path.iter_mut() {
                    self.analyze_expr(index);
                }
            }
            MatchPattern::SimpleList(parts) => {
                for part in parts {
                    self.analyze_pattern(part, span);
                }
            }
            MatchPattern::SpreadList(pattern) => {
                for part in pattern.all_parts_mut() {
                    self.analyze_pattern(part, span);
                }
            }
            MatchPattern::Discard => {}
            MatchPattern::Constant(_) => {}
            MatchPattern::Dict(dict) => {
                for (_key, pattern) in dict.entries.iter_mut() {
                    self.analyze_pattern(pattern, span);
                }
            }
        }
    }

    fn analyze_expr(&mut self, expr: &mut SpannedExpr) {
        let span = expr.span;
        match &mut expr.value {
            Expr::Number(_) => {}
            Expr::Bool(_) => {}
            Expr::String(_) => {}
            Expr::Nil => {}
            Expr::Variable(v) => {
                if let Some(var_slot) = self.get_slot(&v.name) {
                    v.slot = var_slot;
                } else {
                    self.push_err(VariableNotDefined(v.name.clone()), span);
                }
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.analyze_expr(condition);
                self.analyze_expr(then_branch);
                if let Some(else_branch) = else_branch {
                    self.analyze_expr(else_branch);
                }
            }
            Expr::Match(m) => {
                self.analyze_expr(&mut m.condition);
                for (pattern, body) in m.branches.iter_mut() {
                    self.push_scope();
                    self.analyze_pattern(&mut pattern.value, span);
                    self.analyze_expr(body);
                    self.pop_scope();
                }
            }
            Expr::Op { op: _, lhs, rhs } => {
                self.analyze_expr(lhs);
                self.analyze_expr(rhs);
            }
            Expr::Block(b) => self.analyze_block(b),
            Expr::Call { callee, args } => {
                self.analyze_expr(callee);
                for arg in args {
                    self.analyze_expr(arg);
                }
            }
            Expr::CallBuiltin { callee: _, args } => {
                for arg in args {
                    self.analyze_expr(arg);
                }
            }
            Expr::Lambda(l) => {
                let mut lambda_analyzer = VariableAnalyzer::new();
                for param in Rc::get_mut(&mut l.params).unwrap() {
                    lambda_analyzer.analyze_pattern(&mut param.value, span)
                }
                for capture in &mut l.potential_captures {
                    if let Some(slot) = self.get_slot(&capture.name) {
                        capture.slot = slot;
                    } else {
                        self.push_err(VariableNotDefined(capture.name.clone()), span);
                    }
                    lambda_analyzer.vars.push(LocalVar {
                        name: capture.name.to_string(),
                        depth: 0,
                        mutable: false,
                    });
                }
                if let Some(name) = &l.name {
                    lambda_analyzer.vars.push(LocalVar {
                        name: name.to_string(),
                        depth: 0,
                        mutable: false,
                    });
                }
                lambda_analyzer.analyze_expr(&mut Rc::get_mut(&mut l.body).unwrap());
                for err in lambda_analyzer.errors {
                    self.errors.push(err);
                }
            }
            Expr::Dict(entries) => {
                for (key, value) in entries {
                    self.analyze_expr(key);
                    self.analyze_expr(value);
                }
            }
            Expr::List(entries) => {
                for entry in entries {
                    self.analyze_expr(entry);
                }
            }
            Expr::GetIndex(lhs, index) => {
                self.analyze_expr(index);
                self.analyze_expr(lhs);
            }
            Expr::ForIn { iter, var, body } => {
                self.analyze_expr(iter);
                self.push_scope();
                self.analyze_pattern(&mut var.value, span);
                self.analyze_expr(body);
                self.pop_scope();
            }
            Expr::While { condition, body } => {
                self.analyze_expr(condition);
                self.analyze_expr(body)
            }
            Expr::Import { .. } => {}
        }
    }

    fn analyze_block(&mut self, block: &mut Block) {
        self.push_scope();
        for stmt in block.statements.iter_mut() {
            self.analyze_statement(&mut stmt.value);
        }
        if let Some(expr) = &mut block.expression {
            self.analyze_expr(expr);
        }
        self.pop_scope();
    }

    fn get_slot(&self, name: &str) -> Option<u32> {
        for (i, var) in self.vars.iter().enumerate().rev() {
            if var.name == name {
                return Some(i as u32);
            }
        }
        None
    }

    fn push_scope(&mut self) {
        self.depth += 1;
    }

    fn pop_scope(&mut self) {
        self.depth -= 1;
        self.vars.retain(|var| var.depth <= self.depth);
    }
}

fn analyze_statements_liveness(stmts: &mut [SpannedStatement], deps: &mut HashSet<String>) {
    for stmt in stmts.iter_mut().rev() {
        analyze_statement_liveness(&mut stmt.value, deps);
    }
}

fn analyze_statement_liveness(stmt: &mut Statement, deps: &mut HashSet<String>) {
    match stmt {
        Statement::PatternAssignment(pattern, rhs) => {
            analyze_pattern_liveness(&mut pattern.value, deps);
            analyze_expr_liveness(&mut rhs.value, deps)
        }
        Statement::Expression(expr) => {
            analyze_expr_liveness(&mut expr.value, deps);
        }
        Statement::Return(expr) => {
            // TODO: this breaks the REPL
            // deps.clear(); // No variables can be used after a return
            analyze_expr_liveness(&mut expr.value, deps)
        }
    }
}

fn analyze_pattern_liveness(pattern: &mut MatchPattern, deps: &mut HashSet<String>) {
    match pattern {
        MatchPattern::Declaration(var, _) => {
            deps.remove(&var.name);
        }
        MatchPattern::Assignment(target) => {
            if target.path.is_empty() {
                deps.remove(&target.var.name);
            } else {
                for index in target.path.iter_mut().rev() {
                    analyze_expr_liveness(&mut index.value, deps);
                }
            }
        }
        MatchPattern::SimpleList(parts) => {
            for part in parts {
                analyze_pattern_liveness(part, deps);
            }
        }
        MatchPattern::SpreadList(pattern) => {
            for part in pattern.all_parts_mut() {
                analyze_pattern_liveness(part, deps);
            }
        }
        MatchPattern::Discard => {}
        MatchPattern::Constant(_) => {}
        MatchPattern::Dict(dict) => {
            for (_key, pattern) in dict.entries.iter_mut() {
                analyze_pattern_liveness(pattern, deps);
            }
        }
    }
}

fn analyze_block_liveness(block: &mut Block, deps: &mut HashSet<String>) {
    // deps: Variables that may be depended on at the current execution point.
    // Code is processed in reverse order, starting with the final expression.
    if let Some(expr) = &mut block.expression {
        analyze_expr_liveness(&mut expr.value, deps);
    }
    analyze_statements_liveness(&mut block.statements, deps);
}

fn analyze_expr_liveness(expr: &mut Expr, deps: &mut HashSet<String>) {
    // deps: Variables that may be depended on at the current execution point.
    // Code is processed in reverse order.
    match expr {
        Expr::Number(_) => {}
        Expr::Bool(_) => {}
        Expr::String(_) => {}
        Expr::Nil => {}
        Expr::Variable(v) => {
            // If is_final was already cleared, don't set it.
            // This is required for loops, which are analyzed twice.
            v.is_final = v.is_final && deps.insert(v.name.to_string());
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let mut deps_else = deps.clone(); // need to clone even if else is empty, in case deps got removed in if branch.
            if let Some(else_branch) = else_branch {
                analyze_expr_liveness(&mut else_branch.value, &mut deps_else);
            }
            analyze_expr_liveness(&mut then_branch.value, deps);
            // Deps at start of expression are union of both blocks (because we don't know which branch will be taken).
            for dep in deps_else {
                deps.insert(dep);
            }
            analyze_expr_liveness(&mut condition.value, deps);
        }
        Expr::Match(m) => {
            let deps_original = deps.clone();
            for (pattern, body) in m.branches.iter_mut() {
                let mut deps_branch = deps_original.clone();
                analyze_expr_liveness(&mut body.value, &mut deps_branch);
                analyze_pattern_liveness(&mut pattern.value, &mut deps_branch);
                for dep in deps_branch {
                    deps.insert(dep);
                }
            }
            analyze_expr_liveness(&mut m.condition.value, deps);
        }
        Expr::Op { op: _, lhs, rhs } => {
            analyze_expr_liveness(&mut rhs.value, deps);
            analyze_expr_liveness(&mut lhs.value, deps);
        }
        Expr::Block(block) => {
            analyze_block_liveness(block, deps);
        }
        Expr::Call { callee, args } => {
            for arg in args.iter_mut().rev() {
                analyze_expr_liveness(&mut arg.value, deps);
            }
            analyze_expr_liveness(&mut callee.value, deps);
        }
        Expr::CallBuiltin { callee: _, args } => {
            for arg in args.iter_mut().rev() {
                analyze_expr_liveness(&mut arg.value, deps);
            }
        }
        Expr::Lambda(l) => {
            // TODO assuming no shared references to body at this point.
            let mut_body = Rc::get_mut(&mut l.body).unwrap();
            // analyze lambda body, in a separate scope.
            let mut lambda_deps = HashSet::new();
            analyze_expr_liveness(&mut mut_body.value, &mut lambda_deps);
            for pattern in Rc::get_mut(&mut l.params).unwrap().iter_mut().rev() {
                analyze_pattern_liveness(&mut pattern.value, &mut lambda_deps)
            }
            if let Some(name) = &l.name {
                lambda_deps.remove(name);
            }
            for dep in lambda_deps {
                l.potential_captures.push(VarRef::new(dep.clone()));
                deps.insert(dep);
            }
        }
        Expr::Dict(entries) => {
            for (k, v) in entries.iter_mut().rev() {
                analyze_expr_liveness(&mut v.value, deps);
                analyze_expr_liveness(&mut k.value, deps);
            }
        }
        Expr::List(elements) => {
            for e in elements.iter_mut().rev() {
                analyze_expr_liveness(&mut e.value, deps);
            }
        }
        Expr::GetIndex(lhs_expr, index_expr) => {
            analyze_expr_liveness(&mut lhs_expr.value, deps);
            analyze_expr_liveness(&mut index_expr.value, deps);
        }
        Expr::ForIn { var, iter, body } => {
            // TODO (I think) this doesn't account for the fact that variables declared in the loop (including the loop variable)
            // are cleared each loop. We could be a bit more aggressive here.
            let mut deps_last_loop = deps.clone();
            analyze_expr_liveness(&mut body.value, &mut deps_last_loop);
            let mut deps_other_loops = deps_last_loop.clone();
            analyze_pattern_liveness(&mut var.value, &mut deps_other_loops); // var can't persist between loops.
            analyze_expr_liveness(&mut body.value, &mut deps_other_loops);
            // final dependency set is union of 0 loops, 1 loop, and >1 loops.
            for dep in deps_last_loop {
                deps.insert(dep);
            }
            for dep in deps_other_loops {
                deps.insert(dep);
            }
            analyze_expr_liveness(&mut iter.value, deps);
            analyze_pattern_liveness(&mut var.value, deps);
        }
        Expr::While { condition, body } => {
            // TODO: check this logic
            let mut deps_last_loop = deps.clone();
            analyze_expr_liveness(&mut body.value, &mut deps_last_loop);
            analyze_expr_liveness(&mut condition.value, deps);
            let mut deps_other_loops = deps_last_loop.clone();
            analyze_expr_liveness(&mut body.value, &mut deps_other_loops);
            analyze_expr_liveness(&mut condition.value, deps);
            // final dependency set is union of 0 loops, 1 loop, and >1 loops.
            for dep in deps_last_loop {
                deps.insert(dep);
            }
            for dep in deps_other_loops {
                deps.insert(dep);
            }
        }
        Expr::Import { .. } => {}
    }
}
