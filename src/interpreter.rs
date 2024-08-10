use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    hash::Hash,
    rc::Rc,
    vec,
};

use crate::ast::{Block, BuiltInFunction, Expr, LambdaExpr, MatchPattern, Opcode, Statement};

pub struct Interpreter {
    environment: Environment,
}

pub struct Environment {
    slots: Vec<Value>,
    cur_frame: usize,
}

#[derive(Debug, Clone)]
pub enum InterpreterError {
    Return(Value), // Implemented as an error to simplify implementation.
    KeyNotExists(Value),
    IndexOutOfRange { array_len: usize, accessed: usize },
    WrongArgumentCount { expected: usize, supplied: usize },
    WrongType { message: String },
}

impl Display for InterpreterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeyNotExists(name) => write!(f, "Field {} doesn't exist", name),
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
    closure: Vec<Value>,
    self_name: Option<String>,
    recursive: bool,
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
    values: HashMap<Value, Value>,
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
    values: Vec<Value>,
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

enum Callable {
    Lambda(Rc<LambdaFunction>),
    Builtin(BuiltInFunction),
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

    fn as_callable(&self) -> Result<Callable, InterpreterError> {
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
            Value::Builtin(_) => false,
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
            slots: Vec::with_capacity(100),
            cur_frame: 0,
        }
    }

    fn assign_part(
        old: Value,
        rhs: Value,
        index: &Value,
        path: &[Value],
    ) -> Result<Value, InterpreterError> {
        match old {
            Value::Dict(dict) => Self::assign_dict_field(dict, rhs, index, path),
            Value::Array(array) => {
                Self::assign_array_index(array, rhs, try_into_int(index.as_number()?)?, path)
            }
            _ => Err(WrongType {
                message: format!(
                    "Can't assign to index {index}, because {old} is neither a dict nor an array."
                ),
            }),
        }
    }

    fn assign_dict_field(
        mut dict: Rc<DictInstance>,
        rhs: Value,
        key: &Value,
        path: &[Value],
    ) -> Result<Value, InterpreterError> {
        if !key.is_valid_dict_key() {
            return Err(WrongType {
                message: format!("Can't use {key} as a key in a dict. Valid keys are strings, numbers, booleans, and arrays.")
            });
        }
        let mut_dict = Rc::make_mut(&mut dict);
        match path {
            [] => {
                mut_dict.values.insert(key.clone(), rhs);
                Ok(Value::Dict(dict))
            }
            [next_field, rest @ ..] => {
                match mut_dict.values.get_mut(key) {
                    Some(entry) => {
                        let old_value = std::mem::replace(entry, Value::Nil);
                        let new_value = Self::assign_part(old_value, rhs, next_field, rest)?;
                        *entry = new_value;
                        return Ok(Value::Dict(dict));
                    }
                    None => {
                        return Err(KeyNotExists(key.clone()));
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
        if index >= array.values.len() {
            return Err(InterpreterError::IndexOutOfRange {
                array_len: array.values.len(),
                accessed: index,
            });
        }
        let mut_array = Rc::make_mut(&mut array);
        match path {
            [] => {
                mut_array.values[index] = rhs;
                Ok(Value::Array(array))
            }
            [next_field, rest @ ..] => {
                let old_value = std::mem::replace(&mut mut_array.values[index], Value::Nil);
                let new_value = Self::assign_part(old_value, rhs, next_field, rest)?;
                mut_array.values[index] = new_value;
                Ok(Value::Array(array))
            }
        }
    }

    pub fn assign(
        &mut self,
        slot: u32,
        path: &[Value],
        value: Value,
    ) -> Result<(), InterpreterError> {
        match &path[..] {
            [] => {
                self.replace(slot, value);
                Ok(())
            }
            [field, rest @ ..] => {
                let current = self.replace(slot, Value::Nil);
                let new_value = Self::assign_part(current, value, field, rest)?;
                self.replace(slot, new_value);
                Ok(())
            }
        }
    }

    pub fn set(&mut self, slot: u32, new_val: Value) {
        *self.slot(slot) = new_val;
    }

    pub fn get(&self, slot: u32) -> &Value {
        &self.slots[slot as usize + self.cur_frame]
    }

    pub fn replace(&mut self, slot: u32, new_val: Value) -> Value {
        std::mem::replace(self.slot(slot), new_val)
    }

    fn slot(&mut self, slot: u32) -> &mut Value {
        self.expand_slots(slot);
        &mut self.slots[slot as usize + self.cur_frame]
    }

    pub fn expand_slots(&mut self, new_slot: u32) {
        let new_len = (new_slot as usize + self.cur_frame) + 1;
        if new_len <= self.slots.len() {
            return;
        }
        for _ in 0..(new_len - self.slots.len()) {
            self.slots.push(Value::Nil);
        }
    }
}

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

    pub fn execute(&mut self, statement: &Statement) -> Result<(), InterpreterError> {
        match statement {
            Statement::PatternAssignment(pattern, expr) => {
                let value = self.eval(&expr)?;
                self.assign_pattern(pattern, value)?;
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
            Expr::Variable(v) => {
                if v.is_final {
                    Ok(self.environment.replace(v.slot, Value::Nil))
                } else {
                    Ok(self.environment.get(v.slot).clone())
                }
            }
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
                    Value::Lambda(f) => self.call_lambda(&f, arg_values),
                    v => Err(WrongType {
                        message: format!("Expected a function, found {v}"),
                    }),
                }
            }
            Expr::Lambda(l) => {
                let f = self.eval_lambda_definition(l);
                Ok(Value::Lambda(Rc::new(f)))
            }
            Expr::Dict(entries) => {
                let mut values = HashMap::new();
                for (k, v) in entries.iter() {
                    let key = self.eval(k)?;
                    if !key.is_valid_dict_key() {
                        return Err(WrongType {
                            message: format!("Can't use {key} as a key in a dict. Valid keys are strings, numbers, booleans, and arrays.")
                        });
                    }
                    let value = self.eval(v)?;
                    values.insert(key, value);
                }
                Ok(Value::Dict(Rc::new(DictInstance { values })))
            }
            Expr::Array(elements) => {
                let mut values = Vec::with_capacity(elements.len());
                for e in elements.iter() {
                    values.push(self.eval(e)?);
                }
                Ok(Value::Array(Rc::new(ArrayInstance::new(values))))
            }
            Expr::GetIndex(lhs_expr, index_expr) => {
                let index = self.eval(&index_expr)?;
                let lhs = self.eval(&lhs_expr)?;
                match lhs {
                    Value::Dict(d) => {
                        return Ok(d.values.get(&index).cloned().unwrap_or(Value::Nil));
                    }
                    Value::Array(a) => {
                        let idx_int =
                            index
                                .as_number()
                                .and_then(try_into_int)
                                .map_err(|_| WrongType {
                                    message: format!(
                                        "Arrays can only be indexed with integers, found {index}"
                                    ),
                                })?;
                        return match a.values.get(idx_int) {
                            Some(v) => Ok(v.clone()),
                            None => Err(InterpreterError::IndexOutOfRange {
                                array_len: a.values.len(),
                                accessed: idx_int,
                            }),
                        };
                    }
                    v => Err(WrongType {
                        message: format!("Can't index {v} (trying to get key {index})"),
                    }),
                }
            }
            Expr::ForIn { iter, var, body } => {
                let iter_value = self.eval(&iter)?;
                match iter_value {
                    Value::Array(a) => {
                        for v in a.values.iter() {
                            self.environment.set(var.slot, v.clone());
                            self.eval_block(body)?;
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

    fn eval_lambda_definition(&mut self, lambda: &LambdaExpr) -> LambdaFunction {
        let mut captures: Vec<Value> = vec![];
        for pc in &lambda.potential_captures {
            // TODO these should check liveness as well
            let v = self.environment.get(pc.slot);
            captures.push(v.clone());
        }
        LambdaFunction {
            params: lambda.params.to_vec(),
            body: lambda.body.clone(),
            closure: captures,
            self_name: lambda.name.clone(),
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
            Opcode::Add => Value::add(self.eval(lhs)?, self.eval(rhs)?),
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

    fn call(&mut self, callable: &Callable, args: Vec<Value>) -> Result<Value, InterpreterError> {
        match callable {
            Callable::Lambda(f) => self.call_lambda(f, args),
            Callable::Builtin(f) => self.call_builtin(*f, args),
        }
    }

    fn call_builtin(
        &mut self,
        builtin: BuiltInFunction,
        args: Vec<Value>,
    ) -> Result<Value, InterpreterError> {
        match builtin {
            BuiltInFunction::Print => {
                for arg in args {
                    match arg {
                        Value::String(s) => print!("{s}"),
                        v => print!("{v}"),
                    }
                }
                println!();
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
                return Ok(Value::Array(Rc::new(ArrayInstance::new(values))));
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
            BuiltInFunction::Map => {
                let [array_val, f_val] = destructure_args(args)?;
                let mut array = array_val.to_array()?;
                let f = f_val.as_callable()?;
                let mut_array = Rc::make_mut(&mut array);
                for v in mut_array.values.iter_mut() {
                    let v2 = std::mem::replace(v, Value::Nil);
                    *v = self.call(&f, vec![v2])?;
                }
                return Ok(Value::Array(array));
            }
            BuiltInFunction::Filter => {
                let [array_val, f_val] = destructure_args(args)?;
                let mut array = array_val.to_array()?;
                let f = f_val.as_callable()?;
                let mut_array = Rc::make_mut(&mut array);

                let mut err = None;
                mut_array
                    .values
                    .retain(|x| match self.call(&f, vec![x.clone()]) {
                        Ok(v) => v.truthy(),
                        Err(e) => {
                            err = Some(e);
                            false
                        }
                    });

                return match err {
                    Some(e) => Err(e),
                    None => Ok(Value::Array(array)),
                };
            }
            BuiltInFunction::RemoveKey => {
                let [dict_val, key] = destructure_args(args)?;
                let mut dict = dict_val.to_dict()?;
                let mut_dict = Rc::make_mut(&mut dict);
                mut_dict.values.remove(&key);
                return Ok(Value::Dict(dict));
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
        function: &Rc<LambdaFunction>,
        arg_values: Vec<Value>,
    ) -> Result<Value, InterpreterError> {
        if function.params.len() != arg_values.len() {
            return Err(WrongArgumentCount {
                expected: function.params.len(),
                supplied: arg_values.len(),
            });
        }
        let old_frame = self.environment.cur_frame;
        self.environment.cur_frame = self.environment.slots.len();
        for val in arg_values {
            self.environment.slots.push(val);
        }
        for val in &function.closure {
            self.environment.slots.push(val.clone());
        }
        if let Some(_) = &function.self_name {
            self.environment.slots.push(Value::Lambda(function.clone()));
            // // Optimization: don't add self unless we need to.
            // if function.recursive {
            // }
        }

        let result = match self.eval_block(&function.body) {
            Ok(val) => Ok(val),
            Err(InterpreterError::Return(v)) => Ok(v),
            Err(e) => Err(e),
        };
        self.environment.slots.truncate(self.environment.cur_frame);
        self.environment.cur_frame = old_frame;
        return result;
    }

    fn assign_pattern(
        &mut self,
        pattern: &MatchPattern,
        value: Value,
    ) -> Result<(), InterpreterError> {
        match pattern {
            MatchPattern::Declaration(var) => {
                self.environment.set(var.slot, value);
                Ok(())
            }
            MatchPattern::Assignment(target) => {
                let mut path_values = vec![];
                for index in &target.path {
                    path_values.push(self.eval(index)?);
                }
                self.environment
                    .assign(target.var.slot, &path_values, value)?;
                Ok(())
            }
            MatchPattern::Array(parts) => {
                match value {
                    Value::Array(arr) => {
                        // TODO
                        if arr.values.len() != parts.len() {
                            return Err(WrongType {
                                message: format!("Pattern matching error: expected array of length {}, found array of length {}", parts.len(), arr.values.len()),
                            });
                        }
                        match Rc::try_unwrap(arr) {
                            // If we're the only reference, move (replace) the elements to avoid clone.
                            Ok(a) => {
                                for (part, value) in parts.into_iter().zip(a.values.into_iter()) {
                                    self.assign_pattern(part, value)?;
                                }
                            }
                            // If the array is shared, just clone each item
                            Err(a) => {
                                for (part, value) in parts.iter().zip(a.values.iter()) {
                                    self.assign_pattern(part, value.clone())?;
                                }
                            }
                        }
                        Ok(())
                    }
                    _ => Err(WrongType {
                        message: format!("Expected array, found {value}"),
                    }),
                }
            }
        }
    }
}
