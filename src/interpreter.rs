use std::{
    collections::{hash_map::Entry, HashMap},
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
    Return(Value), // Implemented as an error to simplifiy implementation.
    VariableAlreadyDefined(String),
    VariableNotDefined(String),
    FieldNotExists(String),
    IndexOutOfRange { array_len: usize, accessed: usize },
    WrongArgumentCount { expected: usize, supplied: usize },
    WrongType { message: String },
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
            WrongArgumentCount { expected, supplied } => {
                write!(
                    f,
                    "Wrong number of arguments for function. Expected {expected}, got {supplied}"
                )
            }
            WrongType { message } => f.write_str(&message),
            Return(val) => write!(f, "Returning {val}"),
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
    recursive: bool,
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

#[derive(PartialEq, Debug)]
pub struct ArrayInstance {
    values: Vec<Value>,
}

impl Clone for ArrayInstance {
    fn clone(&self) -> Self {
        println!("Cloning array: {:?}", self);
        ArrayInstance {
            values: self.values.clone(),
        }
    }
}

impl Drop for ArrayInstance {
    fn drop(&mut self) {
        // println!("Dropping array: {:?}", self);
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
    Array(Rc<ArrayInstance>),
    Nil,
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
        return Err(WrongType {
            message: format!("Expected a non-negative integer, found {value}"),
        });
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
            _ => Err(WrongType {
                message: format!("Expected a string or non-negative integer, found {index}"),
            }),
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
            if let Some(v) = scope.get_mut(name) {
                return Some(std::mem::replace(v, new_val));
            }
        }
        return None;
    }
}

static BUILTIN_FUNCTIONS: phf::Map<&'static str, BuiltInFunction> = phf::phf_map! {
    "print" => BuiltInFunction::Print,
    "not" => BuiltInFunction::Not,
    "range" => BuiltInFunction::Range,
    "len" => BuiltInFunction::Len,
    "push" => BuiltInFunction::Push,
    "pop" => BuiltInFunction::Pop,
    "sort" => BuiltInFunction::Sort,
    "shiftLeft" => BuiltInFunction::ShiftLeft,
    "floor" => BuiltInFunction::Floor,
};

fn destructure_args<const N: usize>(args: Vec<Value>) -> Result<[Value; N], InterpreterError> {
    match args.try_into() {
        Ok(arr) => Ok(arr),
        Err(vec) => Err(WrongArgumentCount {
            expected: N,
            supplied: vec.len(),
        }),
    }
}

impl Interpreter {
    pub fn new() -> Interpreter {
        Interpreter {
            environment: Environment::new(),
        }
    }

    pub fn list_globals(&self) -> impl Iterator<Item = &String> + '_ {
        self.environment.scopes.iter().flat_map(|s| s.keys())
    }

    pub fn execute(&mut self, statement: &Statement) -> Result<(), InterpreterError> {
        match statement {
            Statement::Declaration(name, expr) => {
                let value = match **expr {
                    Expr::Lambda {
                        ref params,
                        ref body,
                        ref potential_captures,
                    } => {
                        // Workaround to allow simple recursive functions without Rc cycles
                        let mut l =
                            self.eval_lambda_definition(&body, &params, &potential_captures);
                        l.self_name = Some(name.clone());
                        l.recursive = potential_captures.contains(name);
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
            Statement::Return(expr) => {
                let value = self.eval(&expr)?;
                Err(Return(value))
            }
        }
    }

    pub fn eval_block(&mut self, block: &Block) -> Result<Value, InterpreterError> {
        self.run_in_scope(|s| s.eval_block_without_scope(block))
    }

    pub fn eval_block_without_scope(&mut self, block: &Block) -> Result<Value, InterpreterError> {
        for stmt in block.statements.iter() {
            self.execute(stmt)?;
        }
        if let Some(expr) = &block.expression {
            self.eval(expr)
        } else {
            Ok(Value::Nil)
        }
    }

    pub fn eval(&mut self, expr: &Expr) -> Result<Value, InterpreterError> {
        use Value::*;
        match expr {
            Expr::Number(num) => Ok(Number(*num)),
            Expr::Bool(b) => Ok(Bool(*b)),
            Expr::String(s) => Ok(String(s.clone())),
            Expr::Nil => Ok(Nil),
            Expr::Variable {
                name,
                is_final,
                slot: _,
            } => match BUILTIN_FUNCTIONS.get(&name) {
                Some(f) => Ok(Builtin(*f)),
                None => {
                    if *is_final {
                        match self.environment.replace(name, Value::Nil) {
                            Some(v) => Ok(v),
                            None => Err(VariableNotDefined(name.clone())),
                        }
                    } else {
                        match self.environment.get(name) {
                            Some(v) => Ok(v.clone()),
                            None => Err(VariableNotDefined(name.clone())),
                        }
                    }
                }
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
                    v => Err(WrongType {
                        message: format!("Expected a function, found {v}"),
                    }),
                }
            }
            Expr::Lambda {
                params,
                body,
                potential_captures,
            } => {
                let f = self.eval_lambda_definition(body, params, potential_captures);
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
                    (Value::Dict(_), v) => Err(WrongType {
                        message: format!("Dicts can only be indexed with strings, found {v}"),
                    }),
                    (Value::Array(_), v) => Err(WrongType {
                        message: format!("Arrays can only be indexed with numbers, found {v}"),
                    }),
                    (v, _) => Err(WrongType {
                        message: format!("Can't index {v}"),
                    }),
                }
            }
            Expr::ForIn { iter, var, body } => {
                let iter_value = self.eval(&iter)?;
                match iter_value {
                    Value::Array(a) => {
                        for v in a.values.iter() {
                            self.run_in_scope(|s| {
                                s.environment.declare(var.clone(), v.clone())?;
                                s.eval_block(&body)?;
                                Ok(())
                            })?;
                        }
                        Ok(Value::Nil)
                    }
                    _ => Err(WrongType {
                        message: format!("Expected an array, found {iter_value}"),
                    }),
                }
            }
        }
    }

    fn eval_lambda_definition(
        &mut self,
        body: &Rc<Block>,
        params: &Vec<String>,
        potential_captures: &Vec<String>,
    ) -> LambdaFunction {
        let mut captures = HashMap::new();
        for pc in potential_captures {
            // TODO these should check liveness as well
            if let Some(v) = self.environment.get(&pc) {
                captures.insert(pc.clone(), v.clone());
            }
        }
        LambdaFunction {
            params: params.to_vec(),
            body: body.clone(),
            closure: captures.into(),
            self_name: None,
            recursive: false,
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
                (x1, x2) => Err(WrongType {
                    message: format!("Can't add {x1} to {x2}"),
                }),
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
                    _ => println!(
                        "{}",
                        args.iter()
                            .map(|v| v.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                }
                return Ok(Value::Nil);
            }
            BuiltInFunction::Not => {
                let [arg] = destructure_args(args)?;
                return Ok(Value::Bool(!arg.truthy()));
            }
            BuiltInFunction::Range => {
                let [start, stop] = destructure_args(args)?;
                let start_int = try_into_int(start.as_number()?)?;
                let stop_int = try_into_int(stop.as_number()?)?;
                let mut values = Vec::new();
                for i in start_int..stop_int {
                    values.push(Value::Number(i as f64));
                }
                return Ok(Value::Array(Rc::new(ArrayInstance { values })));
            }
            BuiltInFunction::Len => match destructure_args::<1>(args)? {
                [Value::Array(a)] => Ok(Value::Number(a.values.len() as f64)),
                [Value::Dict(d)] => Ok(Value::Number(d.values.len() as f64)),
                [v] => Err(WrongType {
                    message: format!("Expected an array or dict, found {v}"),
                }),
            },
            BuiltInFunction::Push => {
                let [array_val, new_value] = destructure_args(args)?;
                let mut array = array_val.to_array()?;
                let mut_array = Rc::make_mut(&mut array);
                mut_array.values.push(new_value);
                return Ok(Value::Array(array));
            }
            BuiltInFunction::Pop => {
                let [array_val] = destructure_args(args)?;
                let mut array = array_val.to_array()?;
                let mut_array = Rc::make_mut(&mut array);
                mut_array.values.pop();
                // TODO this should really return the value as well.
                return Ok(Value::Array(array));
            }
            BuiltInFunction::Sort => {
                let [array_val] = destructure_args(args)?;
                let mut array = array_val.to_array()?;
                match (*array).values.as_slice() {
                    [] => Ok(Value::Array(array)),
                    [Value::Number(_), rest @ ..] => {
                        if let Some(bad_val) = rest.iter().find(|v| !v.is_number()) {
                            return Err(WrongType {
                                message: format!(
                                    "Expected all values in array to be numbers, found {bad_val}"
                                ),
                            });
                        }
                        Rc::make_mut(&mut array).values.sort_by(|a, b| {
                            a.as_number().unwrap().total_cmp(&b.as_number().unwrap())
                        });
                        return Ok(Value::Array(array));
                    }
                    [Value::String(_), rest @ ..] => {
                        if let Some(bad_val) = rest.iter().find(|v| !v.is_string()) {
                            return Err(WrongType {
                                message: format!(
                                    "Expected all values in array to be strings, found {bad_val}"
                                ),
                            });
                        }
                        Rc::make_mut(&mut array)
                            .values
                            .sort_by(|a, b| a.as_string().unwrap().cmp(&b.as_string().unwrap()));
                        return Ok(Value::Array(array));
                    }
                    [first, _rest @ ..] => {
                        return Err(WrongType {
                            message: format!(
                                "Can only sort arrays of numbers or strings, found {first}"
                            ),
                        })
                    }
                }
            }
            BuiltInFunction::ShiftLeft => {
                let [val, shift_by] = destructure_args(args)?;
                let val_int = try_into_int(val.as_number()?)?;
                let shift_by_int = try_into_int(shift_by.as_number()?)?;
                return Ok(Value::Number((val_int << shift_by_int) as f64));
            }
            BuiltInFunction::Floor => {
                let [val] = destructure_args(args)?;
                let val_num = val.as_number()?;
                return Ok(Value::Number(val_num.floor()));
            }
        }
    }

    fn call_lambda(
        &mut self,
        function: Rc<LambdaFunction>,
        arg_values: Vec<Value>,
    ) -> Result<Value, InterpreterError> {
        if function.params.len() != arg_values.len() {
            return Err(WrongArgumentCount {
                expected: function.params.len(),
                supplied: arg_values.len(),
            });
        }
        let mut lambda_interpreter = Interpreter::new();
        for (name, value) in function.closure.iter() {
            lambda_interpreter
                .environment
                .declare(name.clone(), value.clone())?;
        }
        if let Some(self_name) = &function.self_name {
            // Optimization: don't add self unless we need to.
            if function.recursive {
                lambda_interpreter
                    .environment
                    .declare(self_name.clone(), Value::Lambda(function.clone()))?;
            }
        }

        for (name, value) in function.params.iter().zip(arg_values.into_iter()) {
            lambda_interpreter
                .environment
                .declare(name.clone(), value)?;
        }
        return match lambda_interpreter.eval_block(&function.body) {
            Ok(val) => Ok(val),
            Err(InterpreterError::Return(v)) => Ok(v),
            Err(e) => Err(e),
        };
    }

    fn run_in_scope<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.environment.scopes.push(HashMap::new());
        let result = f(self);
        self.environment.scopes.pop();
        return result;
    }
}
