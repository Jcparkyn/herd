use std::{collections::HashSet, fmt::Display, rc::Rc};

use crate::ast::{Block, BuiltInFunction, Expr, LambdaExpr, MatchPattern, Statement, VarRef};

pub enum AnalysisError {
    VariableAlreadyDefined(String),
    VariableNotDefined(String),
    InvalidImplicitLambda,
}

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
            InvalidImplicitLambda => {
                write!(f, "Using _ as a variable creates an implicit lambda function, but these are only allowed inside a {{}} expression with no statements. For example, {{_ + 1}}.")
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
        stmts: &mut [Statement],
    ) -> Result<(), Vec<AnalysisError>> {
        let mut errors = Vec::new();
        for stmt in stmts.iter_mut() {
            for expr in statement_sub_exprs_mut(stmt) {
                rewrite_implicit_lambdas(expr, false, &mut errors);
            }
        }
        if !errors.is_empty() {
            return Err(errors);
        }

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
}

struct VariableAnalyzer {
    vars: Vec<LocalVar>,
    depth: usize,
    errors: Vec<AnalysisError>,
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

    fn push_var(&mut self, var: &mut VarRef) {
        var.slot = self.vars.len() as u32;
        self.vars.push(LocalVar {
            name: var.name.clone(),
            depth: self.depth,
        });
    }

    fn analyze_stamements(&mut self, stmts: &mut [Statement]) {
        for stmt in stmts {
            self.analyze_statement(stmt);
        }
    }

    fn analyze_statement(&mut self, stmt: &mut Statement) {
        match stmt {
            Statement::PatternAssignment(pattern, rhs) => {
                self.analyze_expr(rhs);
                self.analyze_pattern(pattern);
            }
            Statement::Expression(expr) => self.analyze_expr(expr),
            Statement::Return(expr) => self.analyze_expr(expr),
        }
    }

    fn analyze_pattern(&mut self, pattern: &mut MatchPattern) {
        match pattern {
            MatchPattern::Declaration(var) => {
                if let Some(_) = self.get_slot(&var.name) {
                    self.errors.push(VariableAlreadyDefined(var.name.clone()));
                }
                self.push_var(var);
            }
            MatchPattern::Assignment(target) => {
                if let Some(slot) = self.get_slot(&target.var.name) {
                    target.var.slot = slot;
                } else {
                    self.errors
                        .push(VariableNotDefined(target.var.name.clone()));
                }
                for index in target.path.iter_mut() {
                    self.analyze_expr(index);
                }
            }
            MatchPattern::SimpleArray(parts) => {
                for part in parts {
                    self.analyze_pattern(part);
                }
            }
            MatchPattern::SpreadArray(pattern) => {
                for part in pattern.all_parts_mut() {
                    self.analyze_pattern(part);
                }
            }
            MatchPattern::Discard => {}
        }
    }

    fn analyze_expr(&mut self, expr: &mut Expr) {
        match expr {
            Expr::Number(_) => {}
            Expr::Bool(_) => {}
            Expr::String(_) => {}
            Expr::Nil => {}
            Expr::Variable(v) => {
                if let Some(var_slot) = self.get_slot(&v.name) {
                    v.slot = var_slot;
                } else {
                    self.errors.push(VariableNotDefined(v.name.clone()));
                }
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.analyze_expr(condition);
                self.analyze_block(then_branch);
                if let Some(else_branch) = else_branch {
                    self.analyze_block(else_branch);
                }
            }
            Expr::Op { op: _, lhs, rhs } => {
                self.analyze_expr(lhs);
                self.analyze_expr(rhs);
            }
            Expr::Block(b) => self.analyze_block(b),
            Expr::Call { callee, args } => {
                for arg in args {
                    self.analyze_expr(arg);
                }
                self.analyze_expr(callee);
            }
            Expr::BuiltInFunction(_) => {}
            Expr::Lambda(l) => {
                let mut lambda_analyzer = VariableAnalyzer::new();
                for param in &l.params {
                    lambda_analyzer.vars.push(LocalVar {
                        name: param.to_string(),
                        depth: 0,
                    });
                }
                for capture in &mut l.potential_captures {
                    if let Some(slot) = self.get_slot(&capture.name) {
                        capture.slot = slot;
                    } else {
                        self.errors.push(VariableNotDefined(capture.name.clone()));
                    }
                    lambda_analyzer.vars.push(LocalVar {
                        name: capture.name.to_string(),
                        depth: 0,
                    });
                }
                if let Some(name) = &l.name {
                    lambda_analyzer.vars.push(LocalVar {
                        name: name.to_string(),
                        depth: 0,
                    });
                }
                lambda_analyzer.analyze_block(Rc::get_mut(&mut l.body).unwrap());
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
            Expr::Array(entries) => {
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
                self.push_var(var);
                self.analyze_block(body);
                self.vars.pop();
            }
        }
    }

    fn analyze_block(&mut self, block: &mut Block) {
        self.depth += 1;
        for stmt in block.statements.iter_mut() {
            self.analyze_statement(stmt);
        }
        if let Some(expr) = &mut block.expression {
            self.analyze_expr(expr);
        }
        self.depth -= 1;
        self.vars.retain(|var| var.depth <= self.depth);
    }

    fn get_slot(&self, name: &str) -> Option<u32> {
        for (i, var) in self.vars.iter().enumerate().rev() {
            if var.name == name {
                return Some(i as u32);
            }
        }
        None
    }
}

fn analyze_statements_liveness(stmts: &mut [Statement], deps: &mut HashSet<String>) {
    for stmt in stmts.iter_mut().rev() {
        analyze_statement_liveness(stmt, deps);
    }
}

fn analyze_statement_liveness(stmt: &mut Statement, deps: &mut HashSet<String>) {
    match stmt {
        Statement::PatternAssignment(pattern, rhs) => {
            if let MatchPattern::Declaration(var) = pattern {
                if let Expr::Lambda(l) = rhs.as_mut() {
                    l.name = Some(var.name.clone());
                }
            }
            analyze_pattern_liveness(pattern, deps);
            analyze_expr_liveness(rhs, deps)
        }
        Statement::Expression(expr) => {
            analyze_expr_liveness(expr, deps);
        }
        Statement::Return(expr) => analyze_expr_liveness(expr, deps),
    }
}

fn analyze_pattern_liveness(pattern: &mut MatchPattern, deps: &mut HashSet<String>) {
    match pattern {
        MatchPattern::Declaration(var) => {
            deps.remove(&var.name);
        }
        MatchPattern::Assignment(target) => {
            if target.path.is_empty() {
                deps.remove(&target.var.name);
            } else {
                for index in target.path.iter_mut().rev() {
                    analyze_expr_liveness(index, deps);
                }
            }
        }
        MatchPattern::SimpleArray(parts) => {
            for part in parts {
                analyze_pattern_liveness(part, deps);
            }
        }
        MatchPattern::SpreadArray(pattern) => {
            for part in pattern.all_parts_mut() {
                analyze_pattern_liveness(part, deps);
            }
        }
        MatchPattern::Discard => {}
    }
}

fn analyze_block_liveness(block: &mut Block, deps: &mut HashSet<String>) {
    // deps: Variables that may be depended on at the current execution point.
    // Code is processed in reverse order, starting with the final expression.
    if let Some(expr) = &mut block.expression {
        analyze_expr_liveness(expr, deps);
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
            if let Some(builtin) = BuiltInFunction::from_name(&v.name) {
                *expr = Expr::BuiltInFunction(builtin);
                return;
            }
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
                analyze_block_liveness(else_branch, &mut deps_else);
            }
            analyze_block_liveness(then_branch, deps);
            // Deps at start of expression are union of both blocks (because we don't know which branch will be taken).
            for dep in deps_else {
                deps.insert(dep);
            }
            analyze_expr_liveness(condition, deps);
        }
        Expr::Op { op: _, lhs, rhs } => {
            analyze_expr_liveness(rhs, deps);
            analyze_expr_liveness(lhs, deps);
        }
        Expr::Block(block) => {
            analyze_block_liveness(block, deps);
        }
        Expr::BuiltInFunction(_) => {}
        Expr::Call { callee, args } => {
            analyze_expr_liveness(callee, deps);
            for arg in args.iter_mut().rev() {
                analyze_expr_liveness(arg, deps);
            }
        }
        Expr::Lambda(l) => {
            // TODO assuming no shared references to body at this point.
            let mut_body = Rc::get_mut(&mut l.body).unwrap();
            // analyze lambda body, in a separate scope.
            let mut lambda_deps = HashSet::new();
            analyze_block_liveness(mut_body, &mut lambda_deps);
            for p in &l.params {
                lambda_deps.remove(p);
            }
            if let Some(name) = &l.name {
                lambda_deps.remove(name);
            }
            for dep in lambda_deps {
                l.potential_captures.push(VarRef::new(dep));
            }
        }
        Expr::Dict(entries) => {
            for (_, v) in entries.iter_mut().rev() {
                analyze_expr_liveness(v, deps);
            }
        }
        Expr::Array(elements) => {
            for e in elements.iter_mut().rev() {
                analyze_expr_liveness(e, deps);
            }
        }
        Expr::GetIndex(lhs_expr, index_expr) => {
            analyze_expr_liveness(lhs_expr, deps);
            analyze_expr_liveness(index_expr, deps);
        }
        Expr::ForIn { var, iter, body } => {
            // TODO (I think) this doesn't account for the fact that variables declared in the loop (including the loop variable)
            // are cleared each loop. We could be a bit more aggressive here.
            let mut deps_last_loop = deps.clone();
            analyze_block_liveness(body, &mut deps_last_loop);
            let mut deps_other_loops = deps_last_loop.clone();
            deps_other_loops.remove(&var.name); // var can't persist between loops.
            analyze_block_liveness(body, &mut deps_other_loops);
            // final dependency set is union of 0 loops, 1 loop, and >1 loops.
            for dep in deps_last_loop {
                deps.insert(dep);
            }
            for dep in deps_other_loops {
                deps.insert(dep);
            }
            analyze_expr_liveness(iter, deps);
            deps.remove(&var.name);
        }
    }
}

// Returns true if the expression contains an _ identifier (excluding nested blocks).
fn rewrite_implicit_lambdas(
    expr: &mut Expr,
    inside_block: bool,
    errors: &mut Vec<AnalysisError>,
) -> bool {
    match expr {
        Expr::Block(b) => {
            let mut contains = false;
            let valid_block = b.statements.is_empty();
            for e in block_sub_exprs_mut(b) {
                contains |= rewrite_implicit_lambdas(e, valid_block, errors);
            }
            if valid_block && contains {
                let block = std::mem::replace(b, Block::empty());
                *expr = Expr::Lambda(LambdaExpr::new(vec!["_".to_string()], Rc::new(block)))
            }
            return false;
        }
        Expr::Variable(v) => {
            if v.name == "_" {
                if inside_block {
                    return true;
                } else {
                    errors.push(AnalysisError::InvalidImplicitLambda);
                }
            }
            return false;
        }
        Expr::Lambda(l) => {
            // TODO assuming no shared references to body at this point.
            let mut_body = Rc::get_mut(&mut l.body).unwrap();
            for e in block_sub_exprs_mut(mut_body) {
                rewrite_implicit_lambdas(e, false, errors);
            }
            return false;
        }
        _ => {
            let mut result = false;
            for sub_expr in expr_sub_exprs_mut(expr).into_iter().rev() {
                result |= rewrite_implicit_lambdas(sub_expr, inside_block, errors);
            }
            result
        }
    }
}

fn statement_sub_exprs_mut(stmt: &mut Statement) -> Vec<&mut Box<Expr>> {
    match stmt {
        Statement::PatternAssignment(_, rhs) => vec![rhs], // TODO lhs
        Statement::Expression(expr) => vec![expr],
        Statement::Return(expr) => vec![expr],
    }
}

fn block_sub_exprs_mut(block: &mut Block) -> Vec<&mut Box<Expr>> {
    let mut exprs = Vec::new();
    for stmt in &mut block.statements {
        exprs.append(&mut statement_sub_exprs_mut(stmt));
    }
    if let Some(expr) = &mut block.expression {
        exprs.push(expr);
    }
    exprs
}

fn expr_sub_exprs_mut(expr: &mut Expr) -> Vec<&mut Box<Expr>> {
    let mut exprs = Vec::new();
    match expr {
        Expr::Number(_) => {}
        Expr::Bool(_) => {}
        Expr::String(_) => {}
        Expr::Nil => {}
        Expr::Variable(_) => {}
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            exprs.push(condition);
            exprs.append(&mut block_sub_exprs_mut(then_branch));
            if let Some(else_branch) = else_branch {
                exprs.append(&mut block_sub_exprs_mut(else_branch));
            }
        }
        Expr::Op { op: _, lhs, rhs } => {
            exprs.push(lhs);
            exprs.push(rhs);
        }
        Expr::Block(b) => {
            exprs.append(&mut block_sub_exprs_mut(b));
        }
        Expr::BuiltInFunction(_) => {}
        Expr::Call { callee, args } => {
            exprs.push(callee);
            for arg in args {
                exprs.push(arg);
            }
        }
        Expr::Lambda(_) => panic!("TODO"),
        Expr::Dict(entries) => {
            for (k, v) in entries {
                // TODO correct order
                exprs.push(k);
                exprs.push(v);
            }
        }
        Expr::Array(elements) => {
            for e in elements {
                exprs.push(e);
            }
        }
        Expr::GetIndex(lhs_expr, index_expr) => {
            exprs.push(lhs_expr);
            exprs.push(index_expr);
        }
        Expr::ForIn { var: _, iter, body } => {
            exprs.push(iter);
            exprs.append(&mut block_sub_exprs_mut(body));
        }
    }
    exprs
}
