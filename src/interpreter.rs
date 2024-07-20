use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    fmt::{Debug, Display},
    rc::Rc,
};

use crate::ast::{AssignmentTarget, Block, BuiltInFunction, Expr, Opcode, Statement};

pub struct Interpreter {
    environment: Environment,
}

pub struct Environment {
    scopes: Vec<HashMap<String, Value>>,
}

#[derive(Debug, Clone)]
pub enum InterpreterError {
    VariableAlreadyDefined(String),
    VariableNotDefined(String),
    FieldNotExists(String),
    TooManyArguments,
    NotEnoughArguments,
    WrongType,
}

impl Display for InterpreterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VariableAlreadyDefined(name) => write!(f, "Variable {} is already defined", name),
            VariableNotDefined(name) => write!(f, "Variable {} is not defined", name),
            FieldNotExists(name) => write!(f, "Field {} doesn't exist", name),
            TooManyArguments => write!(f, "Too many arguments"),
            NotEnoughArguments => write!(f, "Not enough arguments"),
            WrongType => write!(f, "Wrong type"),
        }
    }
}

use InterpreterError::*;

#[derive(PartialEq, Debug, Clone)]
pub struct LambdaFunction {
    params: Vec<String>,
    body: Rc<Block>,
    closure: HashMap<String, Value>,
    self_name: Option<String>,
}

#[derive(PartialEq, Debug)]
pub struct DictInstance {
    values: HashMap<String, Value>,
}

impl Clone for DictInstance {
    fn clone(&self) -> Self {
        println!("Cloning dict: {:?}", self);
        DictInstance {
            values: self.values.clone(),
        }
    }
}

impl Drop for DictInstance {
    fn drop(&mut self) {
        println!("Dropping dict: {:?}", self);
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum Value {
    Number(f64),
    Bool(bool),
    String(String),
    Builtin(BuiltInFunction),
    Lambda(Rc<LambdaFunction>),
    Dict(Rc<DictInstance>),
    Nil,
}

impl Value {
    pub fn as_number(&self) -> Result<f64, InterpreterError> {
        match self {
            Value::Number(n) => Ok(*n),
            _ => Err(WrongType),
        }
    }

    pub fn to_dict(self) -> Result<Rc<DictInstance>, InterpreterError> {
        match self {
            Value::Dict(d) => Ok(d),
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
            Value::Dict(_) => true,
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
            Value::Lambda(l) => write!(f, "<lambda: {}>", l.params.join(", ")),
            Value::Dict(d) => {
                if d.values.is_empty() {
                    return write!(f, "[:]");
                }
                let values: Vec<_> = d
                    .values
                    .iter()
                    .map(|(name, v)| name.clone() + ": " + &v.to_string())
                    .collect();
                write!(f, "[ {} ]", values.join(", "))
            }
            Value::Nil => write!(f, "nil"),
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

    fn assign_part(
        old: Value,
        rhs: Value,
        field: &String,
        path: &[String],
    ) -> Result<Value, InterpreterError> {
        match path {
            [] => {
                let mut dict = old.to_dict()?;
                let mut_dict = Rc::make_mut(&mut dict);
                mut_dict.values.insert(field.clone(), rhs);
                Ok(Value::Dict(dict))
            }
            [next_field, rest @ ..] => {
                let mut dict = old.to_dict()?;
                let mut_dict = Rc::make_mut(&mut dict);
                match mut_dict.values.entry(field.clone()) {
                    Entry::Occupied(mut entry) => {
                        let old_value = entry.insert(Value::Nil);
                        let new_value = Self::assign_part(old_value, rhs, next_field, rest)?;
                        entry.insert(new_value);
                        return Ok(Value::Dict(dict));
                    }
                    Entry::Vacant(_) => {
                        return Err(FieldNotExists(field.clone()));
                    }
                };
            }
        }
    }

    pub fn rebind(&mut self, name: &String, value: Value) -> Result<(), InterpreterError> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(v) = scope.get_mut(name) {
                *v = value;
                return Ok(());
            }
        }
        return Err(VariableNotDefined(name.clone()));
    }

    pub fn assign(
        &mut self,
        target: &AssignmentTarget,
        value: Value,
    ) -> Result<(), InterpreterError> {
        match &target.path[..] {
            [] => return self.rebind(&target.var, value),
            [field, rest @ ..] => {
                let current = match self.replace(&target.var, Value::Nil) {
                    Some(x) => x,
                    None => return Err(VariableNotDefined(target.var.clone())),
                };
                let new_value = Self::assign_part(current, value, field, rest)?;
                return self.rebind(&target.var, new_value);
            }
        }
    }

    pub fn get(&self, name: &String) -> Option<&Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v);
            }
        }
        return None;
    }

    pub fn replace(&mut self, name: &String, new_val: Value) -> Option<Value> {
        for scope in self.scopes.iter_mut().rev() {
            if let Entry::Occupied(mut v) = scope.entry(name.clone()) {
                return Some(v.insert(new_val));
            }
        }
        return None;
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
            Statement::Declaration(name, expr) => {
                let value = match **expr {
                    Expr::Lambda {
                        ref params,
                        ref body,
                    } => {
                        // Workaround to allow simple recursive functions without Rc cycles
                        let mut l = self.eval_lambda_definition(&body, &params);
                        l.self_name = Some(name.clone());
                        Value::Lambda(Rc::new(l))
                    }
                    ref e => self.eval(&e)?,
                };

                self.environment.declare(name.to_string(), value)?;

                Ok(())
            }
            Statement::Assignment(target, expr) => {
                let value = self.eval(&expr)?;
                self.environment.assign(target, value)?;
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
                    Value::Lambda(f) => self.call_lambda(f, arg_values),
                    _ => Err(WrongType),
                }
            }
            Expr::Lambda { params, body } => {
                let f = self.eval_lambda_definition(body, params);
                Ok(Value::Lambda(Rc::new(f)))
            }
            Expr::Dict(entries) => {
                let mut values = HashMap::new();
                for (k, v) in entries.iter() {
                    values.insert(k.clone(), self.eval(v)?);
                }
                Ok(Value::Dict(Rc::new(DictInstance { values })))
            }
        }
    }

    fn eval_lambda_definition(&mut self, body: &Rc<Block>, params: &Vec<String>) -> LambdaFunction {
        let mut potential_captures = HashSet::new();
        get_identifiers_in_block(body, &mut potential_captures);
        let mut captures = HashMap::new();
        for pc in potential_captures {
            if let Some(v) = self.environment.get(&pc) {
                captures.insert(pc, v.clone());
            }
        }
        LambdaFunction {
            params: params.to_vec(),
            body: body.clone(),
            closure: captures.into(),
            self_name: None,
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
                    [a] => println!("{}", a),
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
        function: Rc<LambdaFunction>,
        arg_values: Vec<Value>,
    ) -> Result<Value, InterpreterError> {
        if function.params.len() != arg_values.len() {
            return Err(TooManyArguments); // TODO
        }
        let mut lambda_interpreter = Interpreter::new();
        for (name, value) in function.closure.iter() {
            lambda_interpreter
                .environment
                .declare(name.clone(), value.clone())?;
        }
        if let Some(self_name) = &function.self_name {
            lambda_interpreter
                .environment
                .declare(self_name.clone(), Value::Lambda(function.clone()))?;
        }

        for (name, value) in function.params.iter().zip(arg_values.iter()) {
            lambda_interpreter
                .environment
                .declare(name.clone(), value.clone())?;
        }
        lambda_interpreter.eval_block(&function.body)
    }

    fn run_in_scope<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.environment.scopes.push(HashMap::new());
        let result = f(self);
        self.environment.scopes.pop();
        return result;
    }
}

// TODO Ideally this should only return captured variables (excluding ones that are assigned in the block)
fn get_identifiers_in_block(block: &Block, out: &mut HashSet<String>) {
    for stmt in &block.statements {
        match stmt {
            Statement::Declaration(name, _) => {
                out.insert(name.to_string());
            }
            Statement::Assignment(target, _) => {
                out.insert(target.path[0].clone());
            }
            Statement::Expression(expr) => {
                get_identifiers_in_expr(expr, out);
            }
        }
    }
    if let Some(expr) = &block.expression {
        get_identifiers_in_expr(expr, out);
    }
}

fn get_identifiers_in_expr(expr: &Expr, out: &mut HashSet<String>) {
    match expr {
        Expr::Number(_) => {}
        Expr::Bool(_) => {}
        Expr::String(_) => {}
        Expr::Nil => {}
        Expr::Variable(name) => {
            out.insert(name.to_string());
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            get_identifiers_in_expr(condition, out);
            get_identifiers_in_block(then_branch, out);
            if let Some(else_branch2) = else_branch {
                get_identifiers_in_block(&else_branch2, out);
            }
        }
        Expr::Op { op: _, lhs, rhs } => {
            get_identifiers_in_expr(lhs, out);
            get_identifiers_in_expr(rhs, out);
        }
        Expr::Block(block) => {
            get_identifiers_in_block(block, out);
        }
        Expr::BuiltInFunction(_) => {}
        Expr::Call { callee, args } => {
            get_identifiers_in_expr(callee, out);
            for arg in args {
                get_identifiers_in_expr(arg, out);
            }
        }
        Expr::Lambda { params: _, body } => {
            get_identifiers_in_block(body, out);
        }
        Expr::Dict(entries) => {
            for (_, v) in entries {
                get_identifiers_in_expr(v, out);
            }
        }
    }
}
