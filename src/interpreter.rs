use std::{
    cell::{Ref, RefCell, RefMut},
    collections::{hash_map::Entry, HashMap},
    fmt::{Debug, Display},
    rc::Rc,
};

use crate::ast::{Block, BuiltInFunction, Expr, Opcode, Statement};

pub struct Interpreter {
    environment: EnvironmentRef,
}

#[derive(Clone)]
pub struct EnvironmentRef {
    env_ref: Rc<RefCell<Environment>>,
    mutable: bool,
}

impl Debug for EnvironmentRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("EnvironmentRef")
            .field(&self.mutable)
            .finish()
    }
}

impl PartialEq for EnvironmentRef {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.env_ref, &other.env_ref) && self.mutable == other.mutable
    }
}

impl EnvironmentRef {
    pub fn new(environment: Environment, mutable: bool) -> EnvironmentRef {
        EnvironmentRef {
            env_ref: Rc::new(RefCell::new(environment)),
            mutable,
        }
    }

    pub fn new_child(&self, mutable_self: bool) -> EnvironmentRef {
        EnvironmentRef {
            env_ref: Rc::new(RefCell::new(Environment {
                enclosing: Some(self.clone()),
                values: HashMap::new(),
            })),
            mutable: mutable_self,
        }
    }

    pub fn get(&self) -> Ref<Environment> {
        (*self.env_ref).borrow()
    }

    pub fn get_mut(&mut self) -> RefMut<Environment> {
        (*self.env_ref).borrow_mut()
    }
}

pub struct Environment {
    enclosing: Option<EnvironmentRef>,
    values: HashMap<String, Value>,
    // root: bool,
    // scopes: Vec<HashMap<String, Value>>,
}

#[derive(Debug, Clone)]
pub enum InterpreterError {
    VariableAlreadyDefined(String),
    VariableNotDefined(String),
    VariableNotMutable(String),
    TooManyArguments,
    NotEnoughArguments,
    WrongType,
}

impl Display for InterpreterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VariableAlreadyDefined(name) => write!(f, "Variable {} is already defined", name),
            VariableNotDefined(name) => write!(f, "Variable {} is not defined", name),
            VariableNotMutable(name) => write!(
                f,
                "Variable {} was captured from another scope, so it can't be modified here",
                name
            ),
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
    String(String),
    Builtin(BuiltInFunction),
    Lambda {
        params: Vec<String>,
        body: Rc<Block>,
        closure: EnvironmentRef,
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
            Value::String(s) => !s.is_empty(),
            Value::Nil => false,
            Value::Builtin(_) => true,
            Value::Lambda { .. } => true,
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
            Entry::Occupied(_) => Err(VariableAlreadyDefined(name)),
            Entry::Vacant(e) => {
                e.insert(value);
                Ok(())
            }
        }
    }

    pub fn assign(&mut self, name: String, value: Value) -> Result<(), InterpreterError> {
        if let Some(v) = self.values.get_mut(&name) {
            *v = value;
            return Ok(());
        }
        if let Some(enclosing) = &mut self.enclosing {
            if !enclosing.mutable {
                return Err(VariableNotMutable(name));
            }
            let mut e = (*enclosing.env_ref).borrow_mut();
            return e.assign(name, value);
        }
        return Err(VariableNotDefined(name));
    }

    pub fn get(&self, name: &String) -> Option<Value> {
        if let Some(v) = self.values.get(name) {
            return Some(v.clone());
        }
        if let Some(enclosing) = &self.enclosing {
            let e = (*enclosing.env_ref).borrow();
            return e.get(name);
        }
        return None;
    }
}

impl Interpreter {
    pub fn new() -> Interpreter {
        Interpreter {
            environment: EnvironmentRef::new(Environment::new(), true),
        }
    }

    pub fn execute(&mut self, statement: &Statement) -> Result<(), InterpreterError> {
        match statement {
            Statement::Declaration(name, expr) => {
                let value = self.eval(&expr)?;
                self.environment.get_mut().declare(name.to_string(), value)
            }
            Statement::Assignment(name, expr) => {
                let value = self.eval(&expr)?;
                self.environment.get_mut().assign(name.to_string(), value)?;
                Ok(())
            }
            Statement::Expression(expr) => {
                self.eval(&expr)?;
                Ok(())
            }
        }
    }

    pub fn eval_block(&mut self, block: &Block) -> Result<Value, InterpreterError> {
        self.run_in_scope(|s| {
            for stmt in block.statements.iter() {
                s.execute(stmt)?;
            }
            if let Some(expr) = &block.expression {
                s.eval(expr)
            } else {
                Ok(Value::Nil)
            }
        })
    }

    pub fn eval(&mut self, expr: &Expr) -> Result<Value, InterpreterError> {
        use Value::*;
        match expr {
            Expr::Number(num) => Ok(Number(*num)),
            Expr::Bool(b) => Ok(Bool(*b)),
            Expr::String(s) => Ok(String(s.clone())),
            Expr::Nil => Ok(Nil),
            Expr::Variable(name) => self
                .environment
                .get()
                .get(name)
                .ok_or(VariableNotDefined(name.clone())),
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond = self.eval(&condition)?;
                if cond.truthy() {
                    return self.eval_block(&then_branch);
                }
                if let Some(else_branch2) = else_branch {
                    return self.eval_block(&else_branch2);
                }
                Ok(Value::Nil)
            }
            Expr::Op { op, lhs, rhs } => self.eval_binary_op(lhs, rhs, op),
            Expr::Block(block) => self.eval_block(block),
            Expr::BuiltInFunction(f) => Ok(Builtin(f.clone())),
            Expr::Call { callee, args } => {
                let arg_values = args
                    .iter()
                    .map(|arg| self.eval(arg))
                    .collect::<Result<Vec<_>, _>>()?;

                match self.eval(&callee)? {
                    Value::Builtin(c) => self.call_builtin(c, arg_values),
                    Value::Lambda {
                        params,
                        body,
                        closure,
                    } => self.call_lambda(params, arg_values, body, &closure),
                    _ => Err(WrongType),
                }
            }
            Expr::Lambda { params, body } => Ok(Value::Lambda {
                params: params.to_vec(),
                body: body.clone(),
                closure: self.environment.clone(),
            }),
        }
    }

    fn eval_binary_op(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        op: &Opcode,
    ) -> Result<Value, InterpreterError> {
        use Value::*;
        match op {
            Opcode::Add => match (self.eval(&lhs)?, self.eval(&rhs)?) {
                (Number(n1), Number(n2)) => Ok(Number(n1 + n2)),
                (String(s1), String(s2)) => Ok(String(s1 + &s2)),
                _ => Err(WrongType),
            },
            Opcode::Sub => Ok(Number(
                self.eval(&lhs)?.as_number()? - self.eval(&rhs)?.as_number()?,
            )),
            Opcode::Mul => Ok(Number(
                self.eval(&lhs)?.as_number()? * self.eval(&rhs)?.as_number()?,
            )),
            Opcode::Div => Ok(Number(
                self.eval(&lhs)?.as_number()? / self.eval(&rhs)?.as_number()?,
            )),
            Opcode::Gt => Ok(Bool(
                self.eval(&lhs)?.as_number()? > self.eval(&rhs)?.as_number()?,
            )),
            Opcode::Lt => Ok(Bool(
                self.eval(&lhs)?.as_number()? < self.eval(&rhs)?.as_number()?,
            )),
            Opcode::Eq => Ok(Bool(self.eval(&lhs)? == self.eval(&rhs)?)),
            Opcode::Neq => Ok(Bool(self.eval(&lhs)? != self.eval(&rhs)?)),
            Opcode::And => Ok(Bool(self.eval(&lhs)?.truthy() && self.eval(&rhs)?.truthy())),
            Opcode::Or => Ok(Bool(self.eval(&lhs)?.truthy() || self.eval(&rhs)?.truthy())),
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

    fn call_lambda(
        &mut self,
        params: Vec<String>,
        arg_values: Vec<Value>,
        body: Rc<Block>,
        closure: &EnvironmentRef,
    ) -> Result<Value, InterpreterError> {
        if params.len() != arg_values.len() {
            return Err(TooManyArguments); // TODO
        }
        let new_env = closure.new_child(false);
        let mut lambda_interpreter = Interpreter {
            environment: new_env,
        };
        for (name, value) in params.iter().zip(arg_values.iter()) {
            lambda_interpreter
                .environment
                .get_mut()
                .declare(name.clone(), value.clone())?;
        }
        lambda_interpreter.eval_block(&body)
    }

    fn run_in_scope<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let mut new_env = Environment::new();
        new_env.enclosing = Some(EnvironmentRef {
            env_ref: self.environment.env_ref.clone(),
            mutable: true,
        });
        // TODO: we shouldn't need to make a new interpreter for this
        let mut new_interpreter = Interpreter {
            environment: EnvironmentRef::new(new_env, true),
        };
        let result = f(&mut new_interpreter);
        return result;
    }
}
