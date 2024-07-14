use crate::ast::{Expr, Opcode, Statement};


pub struct Interpreter {

}

#[derive(PartialEq, Debug)]
pub enum Value {
  Number(f64),
  Bool(bool),
}

impl Value {
  pub fn as_number(self) -> Result<f64, ()> {
    match self {
      Value::Number(n) => Ok(n),
      _ => Err(()),
    }
  }

  // pub fn as_bool(self) -> Result<bool, ()> {
  //   match self {
  //     Value::Bool(b) => Ok(b),
  //     _ => Err(()),
  //   }
  // }
}

impl Interpreter {
  pub fn new() -> Interpreter {
    Interpreter {}
  }

  pub fn execute(&self, statement: &Statement) -> Result<(), ()> {
    match statement {
      // Statement::Let(name, expr) => {
      //   let value = self.eval(&expr)?;
      //   println!("{} = {:?}", name, value);
      // }
      Statement::Print(expr) => {
        let value = self.eval(&expr)?;
        println!("{:?}", value);
        Ok(())
      }
    }
  }

  pub fn eval(&self, expr: &Expr) -> Result<Value, ()> {
    use Value::*;
    match expr {
      Expr::Number(num) => Ok(Number(*num)),
      Expr::Bool(b) => Ok(Bool(*b)),
      Expr::Op(left_expr, op, right_expr) => {
        let l = self.eval(&left_expr)?;
        let r = self.eval(&right_expr)?;
        match op {
          Opcode::Add => Ok(Number(l.as_number()? + r.as_number()?)),
          Opcode::Sub => Ok(Number(l.as_number()? - r.as_number()?)),
          Opcode::Mul => Ok(Number(l.as_number()? * r.as_number()?)),
          Opcode::Div => Ok(Number(l.as_number()? / r.as_number()?)),
          Opcode::Gt => Ok(Bool(l.as_number()? > r.as_number()?)),
          Opcode::Lt => Ok(Bool(l.as_number()? < r.as_number()?)),
          Opcode::Eq => Ok(Bool(l == r)),
          Opcode::Neq => Ok(Bool(l != r)),
        }
      } 
    }
  }
}