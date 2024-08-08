use std::fmt::Debug;

use std::fmt::Display;
use std::fmt::Error;
use std::fmt::Formatter;
use std::rc::Rc;

#[derive(PartialEq, Debug, Clone, Copy, Hash)]
pub enum BuiltInFunction {
    // general
    Print,
    Not,
    // numbers
    ShiftLeft,
    Floor,
    // arrays
    Range,
    Push,
    Pop,
    Len,
    Sort,
    Map,
    Filter,
    // dicts
    RemoveKey,
}

impl BuiltInFunction {
    pub const fn name(&self) -> &'static str {
        use BuiltInFunction::*;
        match self {
            Print => "print",
            Not => "not",
            Range => "range",
            Push => "push",
            Pop => "pop",
            Len => "len",
            Sort => "sort",
            Map => "map",
            Filter => "filter",
            ShiftLeft => "shiftLeft",
            Floor => "floor",
            RemoveKey => "removeKey",
        }
    }

    pub fn from_name(name: &str) -> Option<BuiltInFunction> {
        use BuiltInFunction::*;
        match name {
            "print" => Some(Print),
            "not" => Some(Not),
            "range" => Some(Range),
            "push" => Some(Push),
            "pop" => Some(Pop),
            "len" => Some(Len),
            "sort" => Some(Sort),
            "map" => Some(Map),
            "filter" => Some(Filter),
            "removeKey" => Some(RemoveKey),
            "shiftLeft" => Some(ShiftLeft),
            "floor" => Some(Floor),
            _ => None,
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
    pub var: VarRef,
    pub path: Vec<Box<Expr>>,
}

#[derive(PartialEq, Debug)]
pub enum Statement {
    Declaration(VarRef, Box<Expr>),
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

#[derive(PartialEq, Debug)]
pub struct LambdaExpr {
    pub params: Vec<String>,
    pub body: Rc<Block>,
    pub potential_captures: Vec<VarRef>,
    pub name: Option<String>,
}

impl LambdaExpr {
    pub fn new(params: Vec<String>, body: Rc<Block>) -> LambdaExpr {
        LambdaExpr {
            params,
            body,
            potential_captures: vec![],
            name: None,
        }
    }
}

#[derive(PartialEq)]
pub enum Expr {
    Number(f64),
    Bool(bool),
    String(Rc<String>),
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
    Lambda(LambdaExpr),
    Dict(Vec<(Box<Expr>, Box<Expr>)>),
    Array(Vec<Box<Expr>>),
    GetIndex(Box<Expr>, Box<Expr>),
    ForIn {
        iter: Box<Expr>,
        var: VarRef,
        body: Block,
    },
}

impl Expr {
    pub fn from_string(s: String) -> Expr {
        Expr::String(Rc::new(s))
    }
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
            Expr::Lambda(l) => f
                .debug_struct("Lambda")
                .field("name", &l.name)
                .field("params", &l.params)
                .field("body", &l.body)
                .field("captures", &l.potential_captures)
                .finish(),
            Expr::Dict(entries) => write!(f, "Dict{:?}", entries),
            Expr::Array(elements) => write!(f, "Array{:?}", elements),
            Expr::GetIndex(lhs, index) => write!(f, "GetIndex({:?}, {:?})", lhs, index),
            Expr::ForIn { iter, var, body } => {
                write!(f, "ForIn({var:?}, {iter:?}, {body:?})")
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
