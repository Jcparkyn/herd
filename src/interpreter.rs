use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::Display,
};

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
    // enclosing: Option<Box<Environment>>,
    // values: HashMap<String, Value>,
    scopes: Vec<HashMap<String, Value>>,
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

use InterpreterError::*;

impl Value {
    pub fn as_number(&self) -> Result<f64, InterpreterError> {
        match self {
            Value::Number(n) => Ok(*n),
            _ => Err(WrongType),
        }
    }
}

impl Environment {
    pub fn new() -> Environment {
        Environment {
            scopes: vec![HashMap::new()],
        }
    }

    pub fn declare(&mut self, name: String, value: Value) -> Result<(), InterpreterError> {
        match self.scopes.last_mut().unwrap().entry(name.clone()) {
            Entry::Occupied(_) => Err(VariableAlreadyDefined(name)),
            Entry::Vacant(e) => {
                e.insert(value);
                Ok(())
            }
        }
    }

    pub fn assign(&mut self, name: String, value: Value) -> Result<(), InterpreterError> {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(&name) {
                scope.insert(name, value);
                return Ok(());
            }
        }
        Err(VariableNotDefined(name))
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(name) {
                return Some(value);
            }
        }
        return None;
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }
}

impl Interpreter {
    pub fn new() -> Interpreter {
        Interpreter {
            environment: Environment::new(),
        }
    }

    pub fn execute(&mut self, statement: &Statement) -> Result<(), InterpreterError> {
        match statement {
            Statement::Print(expr) => {
                let value = self.eval(&expr)?;
                println!("{:?}", value);
                Ok(())
            }
            Statement::Declaration(name, expr) => {
                let value = self.eval(&expr)?;
                self.environment.declare(name.to_string(), value)
            }
            Statement::Assignment(name, expr) => {
                let value = self.eval(&expr)?;
                self.environment.assign(name.to_string(), value)?;
                Ok(())
            }
            Statement::Expression(expr) => {
                self.eval(&expr)?;
                Ok(())
            }
        }
    }

    pub fn eval(&mut self, expr: &Expr) -> Result<Value, InterpreterError> {
        use Value::*;
        match expr {
            Expr::Number(num) => Ok(Number(*num)),
            Expr::Bool(b) => Ok(Bool(*b)),
            Expr::Variable(name) => self
                .environment
                .get(name)
                .copied()
                .ok_or(VariableNotDefined(name.clone())),
            Expr::Op { op, lhs, rhs } => {
                let l = self.eval(&lhs)?;
                let r = self.eval(&rhs)?;
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
            Expr::Block(block) => {
                self.environment.push_scope();
                for stmt in block.statements.iter() {
                    self.execute(&stmt)?;
                }
                self.environment.pop_scope();
                Ok(Value::Number(-1.0)) // TODO
            }
        }
    }
}
