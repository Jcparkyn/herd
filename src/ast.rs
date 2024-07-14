use std::fmt::Debug;
// pub struct Function {
//   params: Vec<String>,
//   body: Expr, // or something else?
// }

// pub enum ValueIndex {
//   ObjectIndex(String),
//   ArrayIndex(i64),
// }

#[derive(PartialEq)]
pub enum Statement {
  Assignment(String, Box<Expr>),
  Print(Box<Expr>),
}

use std::fmt::Error;
use std::fmt::Formatter;


#[derive(PartialEq)]
pub enum Expr {
    Number(f64),
    Bool(bool),
    Op(Box<Expr>, Opcode, Box<Expr>),
    Variable(String),
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
            Expr::Op(l, op, r) => write!(f, "({:?} {:?} {:?})", l, op, r),
            Expr::Variable(v) => v.fmt(f),
        }
    }
}

impl Debug for Opcode {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error>  {
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
