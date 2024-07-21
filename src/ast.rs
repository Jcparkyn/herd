use std::fmt::Debug;

use std::fmt::Display;
use std::fmt::Error;
use std::fmt::Formatter;
use std::rc::Rc;

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum BuiltInFunction {
    Print,
    Not,
    Push,
    Len,
}

impl BuiltInFunction {
    pub fn name(&self) -> &'static str {
        match self {
            BuiltInFunction::Print => "print",
            BuiltInFunction::Not => "not",
            BuiltInFunction::Push => "push",
            BuiltInFunction::Len => "len",
        }
    }
}

impl Display for BuiltInFunction {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "{}", self.name())
    }
}

#[derive(PartialEq)]
pub struct Block {
    pub statements: Vec<Statement>,
    pub expression: Option<Box<Expr>>,
}

impl Debug for Block {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.debug_tuple("Block")
            .field(&self.statements)
            .field(&self.expression)
            .finish()
    }
}

#[derive(PartialEq, Debug)]
pub struct AssignmentTarget {
    pub var: String,
    pub path: Vec<Box<Expr>>,
}

#[derive(PartialEq, Debug)]
pub enum Statement {
    Declaration(String, Box<Expr>),
    Assignment(AssignmentTarget, Box<Expr>),
    Expression(Box<Expr>),
}

#[derive(PartialEq)]
pub enum Expr {
    Number(f64),
    Bool(bool),
    String(String),
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
    Call {
        callee: Box<Expr>,
        args: Vec<Box<Expr>>,
    },
    BuiltInFunction(BuiltInFunction),
    Lambda {
        params: Vec<String>,
        body: Rc<Block>,
    },
    Dict(Vec<(String, Box<Expr>)>),
    Array(Vec<Box<Expr>>),
    GetIndex(Box<Expr>, Box<Expr>),
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
    And,
    Or,
}

impl Debug for Expr {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match self {
            Expr::Number(n) => write!(f, "{}", n),
            Expr::Bool(b) => write!(f, "{}", b),
            Expr::String(s) => write!(f, "'{}'", s),
            Expr::BuiltInFunction(b) => write!(f, "{:?}", b),
            Expr::Op { op, lhs, rhs } => write!(f, "({:?} {:?} {:?})", lhs, op, rhs),
            Expr::Variable(v) => write!(f, "{:?}", v),
            Expr::Block(b) => b.fmt(f),
            Expr::Nil => f.write_str("nil"),
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                write!(f, "if ({:?}) then {:?}", condition, then_branch)?;
                if let Some(else_branch) = else_branch {
                    f.write_str(" else ")?;
                    else_branch.fmt(f)?;
                }
                Ok(())
            }
            Expr::Call { callee, args } => f.debug_tuple("Call").field(callee).field(args).finish(),
            Expr::Lambda { params, body } => {
                f.debug_tuple("Lambda").field(params).field(body).finish()
            }
            Expr::Dict(entries) => write!(f, "Dict{:?}", entries),
            Expr::Array(elements) => write!(f, "Array{:?}", elements),
            Expr::GetIndex(lhs, index) => write!(f, "GetIndex({:?}, {:?})", lhs, index),
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
            Opcode::And => "and",
            Opcode::Or => "or",
        })
    }
}
