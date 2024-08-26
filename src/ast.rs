use std::fmt::Debug;

use std::fmt::Display;
use std::fmt::Error;
use std::fmt::Formatter;
use std::iter;
use std::rc::Rc;

use crate::pos::Spanned;

#[derive(PartialEq, Debug, Clone, Copy, Hash)]
pub enum BuiltInFunction {
    // general
    Print,
    Not,
    // numbers
    ShiftLeft,
    XOR,
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
            XOR => "xor",
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
            "xor" => Some(XOR),
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
    pub statements: Vec<SpannedStatement>,
    pub expression: Option<Box<SpannedExpr>>,
}

impl Block {
    pub fn empty() -> Block {
        Block {
            statements: vec![],
            expression: None,
        }
    }
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
    pub path: Vec<Expr>,
}

#[derive(PartialEq, Debug)]
pub struct SpreadArrayPattern {
    pub before: Vec<MatchPattern>,
    pub spread: Box<MatchPattern>,
    pub after: Vec<MatchPattern>,
}

impl SpreadArrayPattern {
    pub fn all_parts_mut(&mut self) -> impl Iterator<Item = &mut MatchPattern> {
        self.before
            .iter_mut()
            .chain(iter::once(&mut *self.spread))
            .chain(self.after.iter_mut())
    }

    pub fn min_len(&self) -> usize {
        self.before.len() + self.after.len()
    }
}

#[derive(PartialEq, Debug)]
pub enum MatchPattern {
    SimpleArray(Vec<MatchPattern>),
    SpreadArray(SpreadArrayPattern),
    Declaration(VarRef),
    Assignment(AssignmentTarget),
    Discard,
    Constant(MatchConstant),
}

#[derive(PartialEq, Debug)]
pub enum MatchConstant {
    Number(f64),
    Bool(bool),
    String(String),
    Nil,
}

impl Display for MatchConstant {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match self {
            MatchConstant::Number(n) => write!(f, "{}", n),
            MatchConstant::Bool(b) => write!(f, "{}", b),
            MatchConstant::String(s) => write!(f, "'{}'", s),
            MatchConstant::Nil => write!(f, "nil"),
        }
    }
}

#[derive(PartialEq, Debug)]
pub enum Statement {
    PatternAssignment(MatchPattern, Box<Expr>),
    Expression(Box<Expr>),
    Return(Box<Expr>),
}

#[derive(PartialEq, Clone)]
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

impl Debug for VarRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?}[{}]{}",
            self.name,
            self.slot,
            if self.is_final { "❌" } else { "" }
        )
    }
}

#[derive(PartialEq, Debug)]
pub struct LambdaExpr {
    pub params: Rc<Vec<MatchPattern>>,
    pub body: Rc<SpannedExpr>,
    pub potential_captures: Vec<VarRef>,
    pub name: Option<String>,
}

impl LambdaExpr {
    pub fn new(params: Vec<MatchPattern>, body: Rc<SpannedExpr>) -> LambdaExpr {
        LambdaExpr {
            params: Rc::new(params),
            body,
            potential_captures: vec![],
            name: None,
        }
    }
}

#[derive(PartialEq, Debug)]
pub struct MatchExpr {
    pub condition: SpannedExpr,
    pub branches: Vec<(MatchPattern, SpannedExpr)>,
}

#[derive(PartialEq)]
pub enum Expr {
    Number(f64),
    Bool(bool),
    String(Rc<String>),
    Nil,
    Op {
        op: Opcode,
        lhs: Box<SpannedExpr>,
        rhs: Box<SpannedExpr>,
    },
    Variable(VarRef),
    Block(Block),
    If {
        condition: Box<SpannedExpr>,
        then_branch: Box<SpannedExpr>,
        else_branch: Option<Box<SpannedExpr>>,
    },
    Match(Box<MatchExpr>),
    Call {
        callee: Box<SpannedExpr>,
        args: Vec<SpannedExpr>,
    },
    BuiltInFunction(BuiltInFunction),
    Lambda(LambdaExpr),
    Dict(Vec<(SpannedExpr, SpannedExpr)>),
    Array(Vec<SpannedExpr>),
    GetIndex(Box<SpannedExpr>, Box<SpannedExpr>),
    ForIn {
        iter: Box<SpannedExpr>,
        var: VarRef,
        body: Block,
    },
    While {
        condition: Box<SpannedExpr>,
        body: Box<SpannedExpr>,
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
    Gte,
    Lt,
    Lte,
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
            Expr::Variable(v) => v.fmt(f),
            Expr::Block(b) => b.fmt(f),
            Expr::Nil => f.write_str("nil"),
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let mut s = f.debug_struct("If");
                s.field("condition", condition);
                s.field("then", then_branch);
                if let Some(else_branch) = else_branch {
                    s.field("else", else_branch);
                }
                s.finish()
            }
            Expr::Match(m) => m.fmt(f),
            Expr::Call { callee, args } => f.debug_tuple("Call").field(callee).field(args).finish(),
            Expr::Lambda(l) => l.fmt(f),
            Expr::Dict(entries) => {
                f.write_str("Dict")?;
                f.debug_list().entries(entries.iter()).finish()
            }
            Expr::Array(elements) => {
                f.write_str("Array")?;
                f.debug_list().entries(elements.iter()).finish()
            }
            Expr::GetIndex(lhs, index) => write!(f, "GetIndex({:?}, {:?})", lhs, index),
            Expr::ForIn { iter, var, body } => f
                .debug_tuple("ForIn")
                .field(iter)
                .field(var)
                .field(body)
                .finish(),
            Expr::While { condition, body } => {
                f.debug_tuple("While").field(condition).field(body).finish()
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
            Opcode::Gte => ">=",
            Opcode::Lt => "<",
            Opcode::Lte => "<=",
            Opcode::Eq => "==",
            Opcode::Neq => "!=",
            Opcode::And => "and",
            Opcode::Or => "or",
        })
    }
}

pub type SpannedExpr = Spanned<Expr>;

pub type SpannedStatement = Spanned<Statement>;
