#![allow(dead_code)]

use std::{collections::HashMap, fmt::Display, rc::Rc};

use crate::{
    ast::{BuiltInFunction, MatchPattern, SpannedExpr},
    interpreter::InterpreterError,
    pos::Spanned,
};

use InterpreterError::*;

#[derive(PartialEq, Debug, Clone)]
pub enum Value {
    Number(f64),
    Bool(bool),
    String(Rc<String>),
    Builtin(BuiltInFunction),
    Lambda(Rc<LambdaFunction>),
    Dict(Rc<DictInstance>),
    Array(Rc<ArrayInstance>),
    Nil,
}

pub const NIL: Value = Value::Nil;

#[derive(Debug, Clone)]
pub enum Callable {
    Lambda(Rc<LambdaFunction>),
    Builtin(BuiltInFunction),
}

impl Display for Callable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Callable::Builtin(b) => write!(f, "{}", b.to_string()),
            Callable::Lambda(l) => write!(f, "{}", l),
        }
    }
}

impl Value {
    pub fn as_number(&self) -> Result<f64, InterpreterError> {
        match self {
            Value::Number(n) => Ok(*n),
            v => Err(WrongType {
                message: format!("Expected a number, found {v}"),
            }),
        }
    }

    pub fn is_number(&self) -> bool {
        matches!(self, Value::Number(_))
    }

    pub fn as_string(&self) -> Result<&str, InterpreterError> {
        match self {
            Value::String(s) => Ok(s),
            v => Err(WrongType {
                message: format!("Expected a string, found {v}"),
            }),
        }
    }

    pub fn is_string(&self) -> bool {
        matches!(self, Value::String(_))
    }

    pub fn to_dict(self) -> Result<Rc<DictInstance>, InterpreterError> {
        match self {
            Value::Dict(d) => Ok(d),
            v => Err(WrongType {
                message: format!("Expected a dict, found {v}"),
            }),
        }
    }

    pub fn to_array(self) -> Result<Rc<ArrayInstance>, InterpreterError> {
        match self {
            Value::Array(a) => Ok(a),
            v => Err(WrongType {
                message: format!("Expected an array, found {v}"),
            }),
        }
    }

    pub fn as_callable(&self) -> Result<Callable, InterpreterError> {
        match self {
            Value::Lambda(l) => Ok(Callable::Lambda(l.clone())),
            Value::Builtin(b) => Ok(Callable::Builtin(*b)),
            v => Err(WrongType {
                message: format!("Expected a callable, found {v}"),
            }),
        }
    }

    pub fn truthy(&self) -> bool {
        match self {
            Value::Number(n) => *n != 0.0,
            Value::Bool(b) => *b,
            Value::String(s) => !s.is_empty(),
            Value::Nil => false,
            Value::Builtin(_) => true,
            Value::Lambda { .. } => true,
            Value::Dict(_) => true,
            Value::Array(arr) => !arr.values.is_empty(),
        }
    }

    pub fn add(lhs: Value, rhs: Value) -> Result<Value, InterpreterError> {
        use Value::*;
        match (lhs, rhs) {
            (Number(n1), Number(n2)) => Ok(Number(n1 + n2)),
            (String(mut s1), String(s2)) => {
                let s1_mut = Rc::make_mut(&mut s1);
                s1_mut.push_str(s2.as_ref());
                return Ok(String(s1));
            }
            (x1, x2) => Err(WrongType {
                message: format!("Can't add {x1} to {x2}"),
            }),
        }
    }

    pub fn is_valid_dict_key(&self) -> bool {
        match self {
            Value::Dict(_) => false,
            Value::Lambda(_) => false,
            Value::Array(a) => a.as_ref().values.iter().all(Self::is_valid_dict_key),
            Value::Number(f) => !f.is_nan(),
            _ => true,
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Number(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", b),
            Value::String(s) => write!(f, "'{}'", s),
            Value::Builtin(b) => write!(f, "{}", b.to_string()),
            Value::Lambda(l) => write!(f, "{}", l),
            Value::Dict(d) => write!(f, "{}", d),
            Value::Array(a) => write!(f, "{}", a),
            Value::Nil => write!(f, "nil"),
        }
    }
}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Value::Number(n) => n.to_bits().hash(state),
            Value::Bool(b) => b.hash(state),
            Value::String(s) => s.hash(state),
            Value::Builtin(b) => b.hash(state),
            Value::Lambda(_) => panic!("Lambda functions cannot be used as keys inside dicts"),
            Value::Dict(_) => panic!("Dicts cannot be used as keys inside dicts"),
            Value::Array(a) => a.hash(state),
            Value::Nil => ().hash(state),
        }
    }
}

impl Eq for Value {}

#[derive(PartialEq, Debug)]
pub struct LambdaFunction {
    pub params: Rc<Vec<Spanned<MatchPattern>>>,
    pub body: Rc<SpannedExpr>,
    pub closure: Vec<Value>,
    pub self_name: Option<String>,
    pub recursive: bool,
}

impl Display for LambdaFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(self_name) = &self.self_name {
            write!(f, "<lambda: {}>", self_name)
        } else {
            write!(f, "<lambda>")
        }
    }
}

#[derive(PartialEq, Debug)]
pub struct DictInstance {
    pub values: HashMap<Value, Value>,
}

impl Clone for DictInstance {
    fn clone(&self) -> Self {
        #[cfg(debug_assertions)]
        println!("Cloning dict: {}", self);
        DictInstance {
            values: self.values.clone(),
        }
    }
}

impl Display for DictInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.values.is_empty() {
            return write!(f, "[:]");
        }
        let values: Vec<_> = self
            .values
            .iter()
            .map(|(name, v)| name.to_string() + ": " + &v.to_string())
            .collect();
        write!(f, "[{}]", values.join(", "))
    }
}

#[derive(PartialEq, Debug, Hash)]
pub struct ArrayInstance {
    pub values: Vec<Value>,
}

impl ArrayInstance {
    pub fn new(values: Vec<Value>) -> Self {
        ArrayInstance { values }
    }
}

impl Clone for ArrayInstance {
    fn clone(&self) -> Self {
        #[cfg(debug_assertions)]
        println!("Cloning array: {}", self);
        ArrayInstance {
            values: self.values.clone(),
        }
    }
}

impl Display for ArrayInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let values: Vec<_> = self.values.iter().map(|v| v.to_string()).collect();
        write!(f, "[{}]", values.join(", "))
    }
}
