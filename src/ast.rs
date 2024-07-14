
pub struct Function {
  params: Vec<String>,
  body: Expr, // or something else?
}

pub enum ValueIndex {
  ObjectIndex(String),
  ArrayIndex(i64),
}

pub enum Statement {
  Assignment(Vec<ValueIndex>, Expr)
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub enum Expr {
    Number(i32),
    Op(Box<Expr>, Opcode, Box<Expr>),
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub enum Opcode {
    Mul,
    Div,
    Add,
    Sub,
}