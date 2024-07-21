use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    fmt::{Debug, Display},
    rc::Rc,
};

use crate::ast::{Block, BuiltInFunction, Expr, Opcode, Statement};

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
    IndexOutOfRange { array_len: usize, accessed: usize },
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
            IndexOutOfRange {
                array_len,
                accessed,
            } => write!(
                f,
                "Cant access index {} of an array with {} elements",
                accessed, array_len
            ),
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
pub struct ArrayInstance {
    values: Vec<Value>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Value {
    Number(f64),
    Bool(bool),
    String(String),
    Builtin(BuiltInFunction),
    Lambda(Rc<LambdaFunction>),
    Dict(Rc<DictInstance>),
    Array(Rc<ArrayInstance>),
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

    pub fn to_array(self) -> Result<Rc<ArrayInstance>, InterpreterError> {
        match self {
            Value::Array(a) => Ok(a),
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
            Value::Array(arr) => !arr.values.is_empty(),
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
                write!(f, "[{}]", values.join(", "))
            }
            Value::Array(a) => {
                let values: Vec<_> = a.values.iter().map(|v| v.to_string()).collect();
                write!(f, "[{}]", values.join(", "))
            }
            Value::Nil => write!(f, "nil"),
        }
    }
}

fn try_into_int(value: f64) -> Result<usize, InterpreterError> {
    let int = value as usize;
    if int as f64 != value {
        return Err(WrongType);
    }
    Ok(int)
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
        index: &Value,
        path: &[Value],
    ) -> Result<Value, InterpreterError> {
        match index {
            Value::String(name) => Self::assign_dict_field(old.to_dict()?, rhs, name, path),
            Value::Number(idx) => {
                let idx_int = try_into_int(*idx)?;
                return Self::assign_array_index(old.to_array()?, rhs, idx_int, path);
            }
            _ => Err(WrongType),
        }
    }

    fn assign_dict_field(
        mut dict: Rc<DictInstance>,
        rhs: Value,
        field: &String,
        path: &[Value],
    ) -> Result<Value, InterpreterError> {
        let mut_dict = Rc::make_mut(&mut dict);
        match path {
            [] => {
                mut_dict.values.insert(field.clone(), rhs);
                Ok(Value::Dict(dict))
            }
            [next_field, rest @ ..] => {
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

    fn assign_array_index(
        mut array: Rc<ArrayInstance>,
        rhs: Value,
        index: usize,
        path: &[Value],
    ) -> Result<Value, InterpreterError> {
        let mut_array = Rc::make_mut(&mut array);
        match path {
            [] => {
                if index as usize >= mut_array.values.len() {
                    return Err(InterpreterError::IndexOutOfRange {
                        array_len: mut_array.values.len(),
                        accessed: index,
                    });
                }
                mut_array.values[index] = rhs;
                Ok(Value::Array(array))
            }
            _ => Err(InterpreterError::FieldNotExists(format!("{}", index))),
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
        var: &String,
        path: &[Value],
        value: Value,
    ) -> Result<(), InterpreterError> {
        match &path[..] {
            [] => return self.rebind(&var, value),
            [field, rest @ ..] => {
                let current = match self.replace(&var, Value::Nil) {
                    Some(x) => x,
                    None => return Err(VariableNotDefined(var.clone())),
                };
                let new_value = Self::assign_part(current, value, field, rest)?;
                return self.rebind(&var, new_value);
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

static BUILTIN_FUNCTIONS: phf::Map<&'static str, BuiltInFunction> = phf::phf_map! {
    "print" => BuiltInFunction::Print,
    "not" => BuiltInFunction::Not,
};

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
                let mut path_values = vec![];
                for index in &target.path {
                    path_values.push(self.eval(index)?);
                }
                self.environment.assign(&target.var, &path_values, value)?;
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
            Expr::Variable(name) => match BUILTIN_FUNCTIONS.get(&name) {
                Some(f) => Ok(Builtin(*f)),
                None => match self.environment.get(name) {
                    Some(v) => Ok(v.clone()),
                    None => Err(VariableNotDefined(name.clone())),
                },
            },
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
            Expr::Array(elements) => {
                let mut values = Vec::new();
                for e in elements.iter() {
                    values.push(self.eval(e)?);
                }
                Ok(Value::Array(Rc::new(ArrayInstance { values })))
            }
            Expr::GetIndex(lhs_expr, index_expr) => {
                let index = self.eval(&index_expr)?;
                let lhs = self.eval(&lhs_expr)?;
                match (lhs, index) {
                    (Value::Dict(d), Value::String(name)) => {
                        return Ok(d.values.get(&name).cloned().unwrap_or(Value::Nil));
                    }
                    (Value::Array(a), Value::Number(idx)) => {
                        let idx_int = try_into_int(idx)?;
                        return match a.values.get(idx_int) {
                            Some(v) => Ok(v.clone()),
                            None => Err(InterpreterError::IndexOutOfRange {
                                array_len: a.values.len(),
                                accessed: idx_int,
                            }),
                        };
                    }
                    _ => Err(WrongType),
                }
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
            Statement::Declaration(name, rhs) => {
                out.insert(name.to_string());
                get_identifiers_in_expr(rhs, out);
            }
            Statement::Assignment(target, rhs) => {
                out.insert(target.var.clone());
                get_identifiers_in_expr(rhs, out);
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
        Expr::Array(elements) => {
            for e in elements {
                get_identifiers_in_expr(e, out);
            }
        }
        Expr::GetIndex(lhs_expr, index_expr) => {
            get_identifiers_in_expr(lhs_expr, out);
            get_identifiers_in_expr(index_expr, out);
        }
    }
}
