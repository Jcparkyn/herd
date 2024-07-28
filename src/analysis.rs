use std::{collections::HashSet, rc::Rc};

use crate::ast::{Block, Expr, Statement};

pub fn analyze_statements(stmts: &mut [Statement], deps: &mut HashSet<String>) {
    analyze_statements_liveness(stmts, deps);
}

fn analyze_statements_liveness(stmts: &mut [Statement], deps: &mut HashSet<String>) {
    for stmt in stmts.iter_mut().rev() {
        analyze_statement_liveness(stmt, deps);
    }
}

fn analyze_statement_liveness(stmt: &mut Statement, deps: &mut HashSet<String>) {
    match stmt {
        Statement::Declaration(name, rhs) => {
            deps.remove(name);
            analyze_expr_liveness(rhs, deps);
        }
        Statement::Assignment(target, rhs) => {
            if target.path.is_empty() {
                deps.remove(&target.var);
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
        Expr::Variable {
            name,
            is_final,
            slot: _,
        } => {
            // If is_final was already cleared, don't set it.
            // This is required for loops, which are analyzed twice.
            *is_final = *is_final && deps.insert(name.to_string());
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
        Expr::Lambda {
            body,
            potential_captures,
            params,
        } => {
            // TODO assuming no shared references to body at this point.
            let mut_body = Rc::get_mut(body).unwrap();
            // analyze lambda body, in a separate scope.
            let mut lambda_deps = HashSet::new();
            analyze_block_liveness(mut_body, &mut lambda_deps);
            for p in params {
                lambda_deps.remove(p);
            }
            for dep in lambda_deps {
                (*potential_captures).push(dep);
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
            deps_other_loops.remove(var); // var can't persist between loops.
            analyze_block_liveness(body, &mut deps_other_loops);
            // final dependency set is union of 0 loops, 1 loop, and >1 loops.
            for dep in deps_last_loop {
                deps.insert(dep);
            }
            for dep in deps_other_loops {
                deps.insert(dep);
            }
            analyze_expr_liveness(iter, deps);
            deps.remove(var);
        }
    }
}
