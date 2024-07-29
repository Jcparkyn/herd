use std::{collections::HashSet, rc::Rc};

use crate::{
    ast::{Block, Expr, Statement, VarRef},
    interpreter::BUILTIN_FUNCTIONS,
};

pub fn analyze_statements(stmts: &mut [Statement], deps: &mut HashSet<String>) {
    analyze_statements_liveness(stmts, deps);
    // println!("ast: {:#?}", stmts);
    let mut var_analyzer = VariableAnalyzer::new();
    for stmt in stmts {
        var_analyzer.analyze_statement(stmt);
    }
}

struct LocalVar {
    name: String,
    depth: usize,
}

struct VariableAnalyzer {
    vars: Vec<LocalVar>,
    depth: usize,
}

impl VariableAnalyzer {
    fn new() -> VariableAnalyzer {
        VariableAnalyzer {
            vars: Vec::new(),
            depth: 0,
        }
    }

    fn push_var(&mut self, var: &mut VarRef) {
        var.slot = self.vars.len() as u32;
        self.vars.push(LocalVar {
            name: var.name.clone(),
            depth: self.depth,
        });
    }

    fn analyze_statement(&mut self, stmt: &mut Statement) {
        match stmt {
            Statement::Declaration(var, rhs) => {
                self.analyze_expr(rhs);
                self.push_var(var);
            }
            Statement::Assignment(target, rhs) => {
                self.analyze_expr(rhs);
                if let Some(slot) = self.get_slot(&target.var.name) {
                    target.var.slot = slot;
                } else {
                    panic!("Variable {} not found", target.var.name);
                }
                for index in target.path.iter_mut() {
                    self.analyze_expr(index);
                }
            }
            Statement::Expression(expr) => self.analyze_expr(expr),
            Statement::Return(expr) => self.analyze_expr(expr),
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
                    panic!("Variable {} not found", v.name);
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
                        panic!("Capture variable {} not found", capture.name);
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
            }
            Expr::Dict(entries) => {
                for (_, value) in entries {
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
        Statement::Declaration(var, rhs) => {
            if let Expr::Lambda(l) = rhs.as_mut() {
                l.name = Some(var.name.clone());
            }
            deps.remove(&var.name);
            analyze_expr_liveness(rhs, deps);
        }
        Statement::Assignment(target, rhs) => {
            if target.path.is_empty() {
                deps.remove(&target.var.name);
            } else {
                for index in target.path.iter_mut().rev() {
                    analyze_expr_liveness(index, deps);
                }
            }
            analyze_expr_liveness(rhs, deps);
        }
        Statement::Expression(expr) => {
            analyze_expr_liveness(expr, deps);
        }
        Statement::Return(expr) => analyze_expr_liveness(expr, deps),
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
            if let Some(builtin) = BUILTIN_FUNCTIONS.get(&v.name) {
                *expr = Expr::BuiltInFunction(*builtin);
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
