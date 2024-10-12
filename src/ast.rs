use std::fmt::Debug;

use std::fmt::Display;
use std::fmt::Error;
use std::fmt::Formatter;
use std::iter;
use std::rc::Rc;

use strum::EnumString;
use strum::FromRepr;

use crate::pos::Spanned;

#[repr(u8)]
#[derive(PartialEq, Debug, Clone, Copy, Hash, FromRepr, EnumString, strum::Display)]
pub enum BuiltInFunction {
    // general
    #[strum(serialize = "print")]
    Print,
    #[strum(serialize = "not")]
    Not,
    // numbers
    #[strum(serialize = "shiftLeft")]
    ShiftLeft,
    #[strum(serialize = "xor")]
    XOR,
    #[strum(serialize = "floor")]
    Floor,
    // lists
    #[strum(serialize = "range")]
    Range,
    #[strum(serialize = "push")]
    Push,
    #[strum(serialize = "pop")]
    Pop,
    #[strum(serialize = "len")]
    Len,
    #[strum(serialize = "sort")]
    Sort,
    #[strum(serialize = "map")]
    Map,
    #[strum(serialize = "filter")]
    Filter,
    // dicts
    #[strum(serialize = "removeKey")]
    RemoveKey,
}

#[derive(PartialEq, Clone)]
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

#[derive(PartialEq, Debug, Clone)]
pub struct AssignmentTarget {
    pub var: VarRef,
    pub path: Vec<SpannedExpr>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct SpreadListPattern {
    pub before: Vec<MatchPattern>,
    pub spread: Box<MatchPattern>,
    pub after: Vec<MatchPattern>,
}

impl SpreadListPattern {
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

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum DeclarationType {
    Const,
    Mutable,
}

#[derive(PartialEq, Debug, Clone)]
pub enum MatchPattern {
    SimpleList(Vec<MatchPattern>),
    SpreadList(SpreadListPattern),
    Declaration(VarRef, DeclarationType),
    Assignment(AssignmentTarget),
    Discard,
    Constant(MatchConstant),
}

impl MatchPattern {
    pub const NIL: MatchPattern = MatchPattern::Constant(MatchConstant::Nil);
}

type SpannedPattern = Spanned<MatchPattern>;

#[derive(PartialEq, Debug, Clone)]
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
            MatchConstant::Nil => write!(f, "()"),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum Statement {
    PatternAssignment(SpannedPattern, SpannedExpr),
    Expression(SpannedExpr),
    Return(SpannedExpr),
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

#[derive(PartialEq, Debug, Clone)]
pub struct LambdaExpr {
    pub params: Rc<Vec<SpannedPattern>>,
    pub body: Rc<SpannedExpr>,
    pub potential_captures: Vec<VarRef>,
    pub name: Option<String>,
}

impl LambdaExpr {
    pub fn new(params: Vec<SpannedPattern>, body: Rc<SpannedExpr>) -> LambdaExpr {
        LambdaExpr {
            params: Rc::new(params),
            body,
            potential_captures: vec![],
            name: None,
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct MatchExpr {
    pub condition: SpannedExpr,
    pub branches: Vec<(SpannedPattern, SpannedExpr)>,
}

#[derive(PartialEq, Clone)]
pub enum Expr {
    Number(f64),
    Bool(bool),
    String(Rc<String>),
    Nil,
    Op {
        op: Opcode, // TODO spanned
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
    CallNative {
        callee: BuiltInFunction,
        args: Vec<SpannedExpr>,
    },
    Lambda(LambdaExpr),
    Dict(Vec<(SpannedExpr, SpannedExpr)>),
    List(Vec<SpannedExpr>),
    GetIndex(Box<SpannedExpr>, Box<SpannedExpr>),
    ForIn {
        iter: Box<SpannedExpr>,
        var: SpannedPattern,
        body: Box<SpannedExpr>,
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

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
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
            Expr::Op { op, lhs, rhs } => write!(f, "({:?} {:?} {:?})", lhs, op, rhs),
            Expr::Variable(v) => v.fmt(f),
            Expr::Block(b) => b.fmt(f),
            Expr::Nil => f.write_str("()"),
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
            Expr::CallNative { callee, args } => f
                .debug_tuple("CallNative")
                .field(callee)
                .field(args)
                .finish(),
            Expr::Lambda(l) => l.fmt(f),
            Expr::Dict(entries) => {
                f.write_str("Dict")?;
                f.debug_list().entries(entries.iter()).finish()
            }
            Expr::List(elements) => {
                f.write_str("List")?;
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
