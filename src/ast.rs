use std::fmt::Debug;

use std::fmt::Display;
use std::fmt::Error;
use std::fmt::Formatter;
use std::rc::Rc;

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum BuiltInFunction {
    Print,
    Not,
    Range,
    Push,
    Pop,
    Len,
    Sort,
    ShiftLeft,
    Floor,
}

impl BuiltInFunction {
    pub fn name(&self) -> &'static str {
        match self {
            BuiltInFunction::Print => "print",
            BuiltInFunction::Not => "not",
            BuiltInFunction::Range => "range",
            BuiltInFunction::Push => "push",
            BuiltInFunction::Pop => "pop",
            BuiltInFunction::Len => "len",
            BuiltInFunction::Sort => "sort",
            BuiltInFunction::ShiftLeft => "shiftLeft",
            BuiltInFunction::Floor => "floor",
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
    pub slot: u32,
    pub path: Vec<Box<Expr>>,
}

#[derive(PartialEq, Debug)]
pub enum Statement {
    Declaration(String, Box<Expr>),
    Assignment(AssignmentTarget, Box<Expr>),
    Expression(Box<Expr>),
    Return(Box<Expr>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct VarRef {
    pub name: String,
    pub slot: u32,
    /// True if this is guaranteed to be the final usage of this binding, so it can be dropped.
    /// This starts out true for all variables, and the analysis passes will clear it if there
    /// is a chance of this variable being used later.
    pub is_final: bool,
}

impl VarRef {
    pub fn new(name: String) -> VarRef {
        VarRef {
            name,
            slot: 0,
            is_final: true,
        }
    }
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
    Variable(VarRef),
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
        potential_captures: Vec<VarRef>,
    },
    Dict(Vec<(String, Box<Expr>)>),
    Array(Vec<Box<Expr>>),
    GetIndex(Box<Expr>, Box<Expr>),
    ForIn {
        iter: Box<Expr>,
        var: String,
        body: Block,
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
            Expr::Variable(v) => {
                write!(
                    f,
                    "{:?}[{}]{}",
                    v.name,
                    v.slot,
                    if (*v).is_final { "❌" } else { "" }
                )
            }
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
            Expr::Lambda {
                params,
                body,
                potential_captures,
            } => f
                .debug_tuple("Lambda")
                .field(params)
                .field(body)
                .field(potential_captures)
                .finish(),
            Expr::Dict(entries) => write!(f, "Dict{:?}", entries),
            Expr::Array(elements) => write!(f, "Array{:?}", elements),
            Expr::GetIndex(lhs, index) => write!(f, "GetIndex({:?}, {:?})", lhs, index),
            Expr::ForIn { iter, var, body } => {
                write!(f, "ForIn({var}, {iter:?}, {body:?})")
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
            Opcode::And => "and",
            Opcode::Or => "or",
        })
    }
}
