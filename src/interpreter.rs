use std::{collections::{hash_map::Entry, HashMap}, error::Error, fmt::Display};

use crate::ast::{Expr, Opcode, Statement};


pub struct Interpreter {
  environment: Environment,
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Value {
  Number(f64),
  Bool(bool),
}

pub struct Environment {
  enclosing: Option<Box<Environment>>,
  values: HashMap<String, Value>,
}

#[derive(Debug, Clone)]
pub enum InterpreterError {
  VariableAlreadyDefined(String),
  VariableNotDefined(String),
  WrongType,
}

impl Display for InterpreterError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      VariableAlreadyDefined(name) => write!(f, "Variable {} is already defined", name),
      VariableNotDefined(name) => write!(f, "Variable {} is not defined", name),
      WrongType => write!(f, "Wrong type"),
    }
  }
}

// impl Error for InterpreterError {}

use InterpreterError::*;

impl Value {
  pub fn as_number(self) -> Result<f64, InterpreterError> {
    match self {
      Value::Number(n) => Ok(n),
      _ => Err(WrongType),
    }
  }
}

impl Environment {
  pub fn new() -> Environment {
    Environment {
      enclosing: None,
      values: HashMap::new(),
    }
  }

  pub fn declare(&mut self, name: String, value: Value) -> Result<(), InterpreterError> {
    match self.values.entry(name.clone()) {
      Entry::Occupied(_) => {
        Err(VariableAlreadyDefined(name))
      }
      Entry::Vacant(e) => {
        e.insert(value);
        Ok(())
      }
    }
  }

  pub fn assign(&mut self, name: String, value: Value) -> Result<(), InterpreterError> {
    if self.values.contains_key(&name) {
      self.values.insert(name, value);
      return Ok(());
    }
    if let Some(enclosing) = &mut self.enclosing {
      return enclosing.assign(name, value);
    }
    Err(VariableNotDefined(name))
  }

  pub fn get(&self, name: &str) -> Option<&Value> {
    if let Some(value) = self.values.get(name) {
      return Some(value);
    }
    if let Some(enclosing) = &self.enclosing {
      return enclosing.get(name);
    }
    return None;
  }
}

impl Interpreter {
  pub fn new() -> Interpreter {
    Interpreter { environment: Environment::new() }
  }

  pub fn execute(&mut self, statement: &Statement) -> Result<(), InterpreterError> {
    match statement {
      Statement::Print(expr) => {
        let value = self.eval(&expr)?;
        println!("{:?}", value);
        Ok(())
      }
      Statement::Declaration(name, expr) => {
        self.environment.declare(name.to_string(), self.eval(&expr)?)
      }
      Statement::Assignment(name, expr) => {
        self.environment.assign(name.to_string(), self.eval(&expr)?)?;
        Ok(())
      }
    }
  }

  pub fn eval(&self, expr: &Expr) -> Result<Value, InterpreterError> {
    use Value::*;
    match expr {
      Expr::Number(num) => Ok(Number(*num)),
      Expr::Bool(b) => Ok(Bool(*b)),
      Expr::Variable(name) => self.environment.get(name).copied().ok_or(VariableNotDefined(name.clone())),
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