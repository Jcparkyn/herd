use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::Display,
    rc::Rc,
};

use crate::ast::{Block, BuiltInFunction, Expr, Opcode, Statement};

pub struct Interpreter {
    environment: Environment,
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
    TooManyArguments,
    NotEnoughArguments,
    WrongType,
}

impl Display for InterpreterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VariableAlreadyDefined(name) => write!(f, "Variable {} is already defined", name),
            VariableNotDefined(name) => write!(f, "Variable {} is not defined", name),
            TooManyArguments => write!(f, "Too many arguments"),
            NotEnoughArguments => write!(f, "Not enough arguments"),
            WrongType => write!(f, "Wrong type"),
        }
    }
}

use InterpreterError::*;

#[derive(PartialEq, Debug, Clone)]
pub enum Value {
    Number(f64),
    Bool(bool),
    Builtin(BuiltInFunction),
    Lambda {
        params: Vec<String>,
        body: Rc<Block>,
    },
    Nil,
}

impl Value {
    pub fn as_number(&self) -> Result<f64, InterpreterError> {
        match self {
            Value::Number(n) => Ok(*n),
            _ => Err(WrongType),
        }
    }

    pub fn truthy(&self) -> bool {
        match self {
            Value::Number(n) => *n != 0.0,
            Value::Bool(b) => *b,
            Value::Nil => false,
            Value::Builtin(_) => true,
            Value::Lambda { .. } => true,
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

    pub fn execute_block(&mut self, block: &Block) -> Result<(), InterpreterError> {
        self.environment.push_scope();
        for stmt in block.statements.iter() {
            self.execute(stmt)?;
        }
        self.environment.pop_scope();
        Ok(())
    }

    pub fn eval(&mut self, expr: &Expr) -> Result<Value, InterpreterError> {
        use Value::*;
        match expr {
            Expr::Number(num) => Ok(Number(*num)),
            Expr::Bool(b) => Ok(Bool(*b)),
            Expr::Nil => Ok(Nil),
            Expr::Variable(name) => self
                .environment
                .get(name)
                .cloned()
                .ok_or(VariableNotDefined(name.clone())),
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond = self.eval(&condition)?;
                if cond.truthy() {
                    self.execute_block(&then_branch)?;
                    return Ok(Value::Nil); // TODO
                }
                if let Some(else_branch2) = else_branch {
                    self.execute_block(&else_branch2)?;
                    return Ok(Value::Nil);
                }
                Ok(Value::Nil)
            }
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
                    Opcode::And => Ok(Bool(l.truthy() && r.truthy())),
                    Opcode::Or => Ok(Bool(l.truthy() || r.truthy())),
                }
            }
            Expr::Block(block) => {
                self.environment.push_scope();
                for stmt in block.statements.iter() {
                    self.execute(&stmt)?;
                }
                self.environment.pop_scope();
                Ok(Value::Nil) // TODO
            }
            Expr::BuiltInFunction(f) => Ok(Builtin(f.clone())),
            Expr::Call { callee, args } => {
                let arg_values = args
                    .iter()
                    .map(|arg| self.eval(arg))
                    .collect::<Result<Vec<_>, _>>()?;

                match self.eval(&callee)? {
                    Value::Builtin(c) => self.call_builtin(c, arg_values),
                    Value::Lambda { params, body } => {
                        if params.len() != arg_values.len() {
                            return Err(TooManyArguments); // TODO
                        }
                        let mut lambda_interpreter = Interpreter::new();
                        for (name, value) in params.iter().zip(arg_values.iter()) {
                            lambda_interpreter
                                .environment
                                .declare(name.clone(), value.clone())?;
                        }
                        lambda_interpreter.execute_block(&body)?;
                        Ok(Value::Nil)
                    }
                    _ => Err(WrongType),
                }
            }
            Expr::Lambda { params, body } => Ok(Value::Lambda {
                params: params.to_vec(),
                body: body.clone(),
            }),
        }
    }

    fn call_builtin(
        &self,
        builtin: BuiltInFunction,
        args: Vec<Value>,
    ) -> Result<Value, InterpreterError> {
        match builtin {
            BuiltInFunction::Print => {
                match args.as_slice() {
                    [a] => println!("{:?}", a),
                    _ => println!("{:?}", args),
                }
                return Ok(Value::Nil);
            }
            BuiltInFunction::Not => match args.as_slice() {
                [] => Err(NotEnoughArguments),
                [a] => Ok(Value::Bool(!a.truthy())),
                _ => Err(TooManyArguments),
            },
        }
    }
}
