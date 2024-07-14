use crate::ast::{Expr, Opcode};


pub struct Interpreter {

}

#[derive(PartialEq, Debug)]
pub enum Value {
  Number(f64),
}

impl Value {
  pub fn as_number(self) -> Result<f64, ()> {
    match self {
      Value::Number(n) => Ok(n),
    }
  }
}

impl Interpreter {
  pub fn new() -> Interpreter {
    Interpreter {}
  }

  pub fn eval(&self, expr: &Expr) -> Result<Value, ()> {
    match expr {
      Expr::Number(num) => Ok(Value::Number(*num)),
      Expr::Op(left_expr, op, right_expr) => {
        let l = self.eval(&left_expr)?;
        let r = self.eval(&right_expr)?;
        match op {
          Opcode::Add => Ok(Value::Number(l.as_number()? + r.as_number()?)),
          Opcode::Sub => Ok(Value::Number(l.as_number()? - r.as_number()?)),
          Opcode::Mul => Ok(Value::Number(l.as_number()? * r.as_number()?)),
          Opcode::Div => Ok(Value::Number(l.as_number()? / r.as_number()?)),
        }
      } 
    }
  }
}