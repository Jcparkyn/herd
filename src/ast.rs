use std::fmt::Debug;

#[derive(PartialEq, Debug)]
pub struct Block {
    pub statements: Vec<Statement>,
}

#[derive(PartialEq, Debug)]
pub enum Statement {
    Declaration(String, Box<Expr>),
    Assignment(String, Box<Expr>),
    Expression(Box<Expr>),
    Print(Box<Expr>),
}

use std::fmt::Error;
use std::fmt::Formatter;

#[derive(PartialEq)]
pub enum Expr {
    Number(f64),
    Bool(bool),
    Nil,
    Op {
        op: Opcode,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Variable(String),
    Block(Block),
    If {
        condition: Box<Expr>,
        then_branch: Block,
        else_branch: Option<Block>,
    },
}

#[derive(PartialEq, Eq, Hash)]
pub enum Opcode {
    Mul,
    Div,
    Add,
    Sub,
    Gt,
    Lt,
    Eq,
    Neq,
}

impl Debug for Expr {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match self {
            Expr::Number(n) => n.fmt(f),
            Expr::Bool(b) => b.fmt(f),
            Expr::Op { op, lhs, rhs } => write!(f, "({:?} {:?} {:?})", lhs, op, rhs),
            Expr::Variable(v) => v.fmt(f),
            Expr::Block(b) => b.fmt(f),
            Expr::Nil => f.write_str("nil"),
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                f.write_str("if ")?;
                condition.fmt(f)?;
                f.write_str(" ")?;
                then_branch.fmt(f)?;
                if let Some(else_branch) = else_branch {
                    f.write_str(" else ")?;
                    else_branch.fmt(f)?;
                }
                Ok(())
            }
        }
    }
}

impl Debug for Opcode {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.write_str(match self {
            Opcode::Mul => "*",
            Opcode::Div => "/",
            Opcode::Add => "+",
            Opcode::Sub => "-",
            Opcode::Gt => ">",
            Opcode::Lt => "<",
            Opcode::Eq => "==",
            Opcode::Neq => "!=",
        })
    }
}
