use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    hash::Hash,
    rc::Rc,
    vec,
};

use crate::ast::{
    AssignmentTarget, Block, BuiltInFunction, Expr, LambdaExpr, MatchConstant, MatchExpr,
    MatchPattern, Opcode, SpreadArrayPattern, Statement,
};

pub struct Interpreter {
    environment: Environment,
    // Temporary stack used for passing function arguments.
    // Using a shared vec so we don't need to re-allocate for each function call.
    arg_stack: Vec<Value>,
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
    PatternMatchFailed { message: String },
}

type IResult<T> = Result<T, InterpreterError>;

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
            PatternMatchFailed { message } => write!(f, "Unsuccessful pattern match: {}", message),
        }
    }
}

use InterpreterError::*;

#[derive(PartialEq, Debug, Clone)]
pub struct LambdaFunction {
    params: Vec<String>,
    body: Rc<Expr>,
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

const NIL: Value = Value::Nil;

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

impl Interpreter {
    pub fn new() -> Interpreter {
        Interpreter {
            environment: Environment::new(),
            arg_stack: Vec::with_capacity(32),
        }
    }

    pub fn execute(&mut self, statement: &Statement) -> Result<(), InterpreterError> {
        match statement {
            Statement::PatternAssignment(pattern, expr) => {
                let value = self.eval(&expr)?;
                self.match_pattern(pattern, value)?;
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
                    return self.eval(&then_branch);
                }
                if let Some(else_branch2) = else_branch {
                    return self.eval(&else_branch2);
                }
                Ok(Value::Nil)
            }
            Expr::Match(MatchExpr {
                condition,
                branches,
            }) => {
                let cond = self.eval(&condition)?;
                for (pattern, body) in branches {
                    if Interpreter::matches_pattern(pattern, &cond) {
                        self.match_pattern(pattern, cond)?;
                        return self.eval(body);
                    }
                }
                Err(PatternMatchFailed {
                    message: format!("No branches matched successfully"),
                })
            }
            Expr::Op { op, lhs, rhs } => self.eval_binary_op(lhs, rhs, op),
            Expr::Block(block) => self.eval_block(block),
            Expr::BuiltInFunction(f) => Ok(Builtin(f.clone())),
            Expr::Call { callee, args } => {
                let arg_stack_len_before = self.arg_stack.len();
                let arg_count = args.len();
                // self.arg_stack.clear();
                for arg in args.iter() {
                    match self.eval(arg) {
                        Ok(v) => self.arg_stack.push(v),
                        Err(e) => {
                            self.arg_stack.truncate(arg_stack_len_before);
                            return Err(e);
                        }
                    };
                }

                let result = match self.eval(&callee)? {
                    Value::Builtin(c) => self.call_builtin(c, arg_count),
                    Value::Lambda(f) => self.call_lambda(&f, arg_count),
                    v => Err(WrongType {
                        message: format!("Expected a function, found {v}"),
                    }),
                };
                self.arg_stack.truncate(arg_stack_len_before);
                return result;
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

    fn call(&mut self, callable: &Callable, arg_count: usize) -> Result<Value, InterpreterError> {
        match callable {
            Callable::Lambda(f) => self.call_lambda(f, arg_count),
            Callable::Builtin(f) => self.call_builtin(*f, arg_count),
        }
    }

    fn call_internal<const N: usize>(
        &mut self,
        callable: &Callable,
        args: [Value; N],
    ) -> IResult<Value> {
        let arg_stack_len_before = self.arg_stack.len();
        self.arg_stack.extend(args);
        let result = self.call(callable, N);
        self.arg_stack.truncate(arg_stack_len_before);
        result
    }

    fn call_builtin(
        &mut self,
        builtin: BuiltInFunction,
        arg_count: usize,
    ) -> Result<Value, InterpreterError> {
        match builtin {
            BuiltInFunction::Print => {
                for arg in Self::pop_args(&mut self.arg_stack, arg_count) {
                    match arg {
                        Value::String(s) => print!("{s}"),
                        v => print!("{v}"),
                    }
                }
                println!();
                return Ok(Value::Nil);
            }
            BuiltInFunction::Not => {
                let [arg] = self.destructure_args(arg_count)?;
                return Ok(Value::Bool(!arg.truthy()));
            }
            BuiltInFunction::Range => {
                let [start, stop] = self.destructure_args(arg_count)?;
                let start_int = try_into_int(start.as_number()?)?;
                let stop_int = try_into_int(stop.as_number()?)?;
                let mut values = Vec::new();
                for i in start_int..stop_int {
                    values.push(Value::Number(i as f64));
                }
                return Ok(Value::Array(Rc::new(ArrayInstance::new(values))));
            }
            BuiltInFunction::Len => match self.destructure_args(arg_count)? {
                [Value::Array(a)] => Ok(Value::Number(a.values.len() as f64)),
                [Value::Dict(d)] => Ok(Value::Number(d.values.len() as f64)),
                [v] => Err(WrongType {
                    message: format!("Expected an array or dict, found {v}"),
                }),
            },
            BuiltInFunction::Push => {
                let [array_val, new_value] = self.destructure_args(arg_count)?;
                let mut array = array_val.to_array()?;
                let mut_array = Rc::make_mut(&mut array);
                mut_array.values.push(new_value);
                return Ok(Value::Array(array));
            }
            BuiltInFunction::Pop => {
                let [array_val] = self.destructure_args(arg_count)?;
                let mut array = array_val.to_array()?;
                let mut_array = Rc::make_mut(&mut array);
                mut_array.values.pop();
                // TODO this should really return the value as well.
                return Ok(Value::Array(array));
            }
            BuiltInFunction::Sort => {
                let [array_val] = self.destructure_args(arg_count)?;
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
                let [array_val, f_val] = self.destructure_args(arg_count)?;
                let mut array = array_val.to_array()?;
                let f = f_val.as_callable()?;
                let mut_array = Rc::make_mut(&mut array);
                for v in mut_array.values.iter_mut() {
                    let v2 = std::mem::replace(v, Value::Nil);
                    *v = self.call_internal(&f, [v2])?;
                }
                return Ok(Value::Array(array));
            }
            BuiltInFunction::Filter => {
                let [array_val, f_val] = self.destructure_args(arg_count)?;
                let mut array = array_val.to_array()?;
                let f = f_val.as_callable()?;
                let mut_array = Rc::make_mut(&mut array);

                let mut err = None;
                mut_array.values.retain(|x| -> bool {
                    match self.call_internal(&f, [x.clone()]) {
                        Ok(v) => v.truthy(),
                        Err(e) => {
                            err = Some(e);
                            false
                        }
                    }
                });

                return match err {
                    Some(e) => Err(e),
                    None => Ok(Value::Array(array)),
                };
            }
            BuiltInFunction::RemoveKey => {
                let [dict_val, key] = self.destructure_args(arg_count)?;
                let mut dict = dict_val.to_dict()?;
                let mut_dict = Rc::make_mut(&mut dict);
                mut_dict.values.remove(&key);
                return Ok(Value::Dict(dict));
            }
            BuiltInFunction::ShiftLeft => {
                let [val, shift_by] = self.destructure_args(arg_count)?;
                let val_int = try_into_int(val.as_number()?)?;
                let shift_by_int = try_into_int(shift_by.as_number()?)?;
                return Ok(Value::Number((val_int << shift_by_int) as f64));
            }
            BuiltInFunction::Floor => {
                let [val] = self.destructure_args(arg_count)?;
                let val_num = val.as_number()?;
                return Ok(Value::Number(val_num.floor()));
            }
        }
    }

    fn call_lambda(
        &mut self,
        function: &Rc<LambdaFunction>,
        arg_count: usize,
    ) -> Result<Value, InterpreterError> {
        if function.params.len() != arg_count {
            return Err(WrongArgumentCount {
                expected: function.params.len(),
                supplied: arg_count,
            });
        }
        let old_frame = self.environment.cur_frame;
        self.environment.cur_frame = self.environment.slots.len();
        for val in Self::pop_args(&mut self.arg_stack, arg_count) {
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

        let result = match self.eval(&function.body) {
            Ok(val) => Ok(val),
            Err(InterpreterError::Return(v)) => Ok(v),
            Err(e) => Err(e),
        };
        self.environment.slots.truncate(self.environment.cur_frame);
        self.environment.cur_frame = old_frame;
        return result;
    }

    fn destructure_args<const N: usize>(
        &mut self,
        arg_count: usize,
    ) -> Result<[Value; N], InterpreterError> {
        if arg_count != N {
            return Err(InterpreterError::WrongArgumentCount {
                expected: N,
                supplied: self.arg_stack.len(),
            });
        }
        let mut result = [NIL; N];
        for (i, arg) in Self::pop_args(&mut self.arg_stack, N).enumerate() {
            result[i] = arg;
        }
        Ok(result)
    }

    fn pop_args(arg_stack: &mut Vec<Value>, arg_count: usize) -> vec::Drain<Value> {
        assert!(arg_stack.len() >= arg_count);
        // self.arg_stack.truncate(self.arg_stack.len() - arg_count);
        arg_stack.drain(arg_stack.len() - arg_count..)
    }

    fn match_slice(&mut self, parts: &[MatchPattern], values: &mut [Value]) -> IResult<()> {
        if parts.len() != values.len() {
            return Err(PatternMatchFailed {
                message: format!(
                    "Expected an array with length={}, but actual array had length={}",
                    parts.len(),
                    values.len()
                ),
            });
        }
        for (part, value) in parts.iter().zip(values) {
            self.match_pattern(part, std::mem::replace(value, Value::Nil))?;
        }
        return Ok(());
    }

    fn match_array_spread(&mut self, pattern: &MatchPattern, values: &mut [Value]) -> IResult<()> {
        fn to_value_array(values: &mut [Value]) -> Value {
            let mut vec = Vec::with_capacity(values.len());
            for value in values {
                vec.push(std::mem::replace(value, Value::Nil));
            }
            Value::Array(Rc::new(ArrayInstance::new(vec)))
        }
        match pattern {
            MatchPattern::Discard => Ok(()),
            MatchPattern::Declaration(var) => {
                self.environment.set(var.slot, to_value_array(values));
                Ok(())
            }
            MatchPattern::Assignment(target) => {
                self.assign(target, to_value_array(values))?;
                Ok(())
            }
            MatchPattern::SimpleArray(parts) => self.match_slice(parts, values),
            MatchPattern::SpreadArray(pattern) => self.match_spread_array(pattern, values),
            MatchPattern::Constant(c) => Err(InterpreterError::PatternMatchFailed {
                message: format!("Can't use a constant ({c}) as a spread parameter (..)"),
            }),
        }
    }

    fn match_spread_array(
        &mut self,
        pattern: &SpreadArrayPattern,
        values: &mut [Value],
    ) -> IResult<()> {
        let (values_before, rest) = values.split_at_mut(pattern.before.len());
        let (values_spread, values_after) = rest.split_at_mut(rest.len() - pattern.after.len());
        self.match_slice(&pattern.before, values_before)?;
        self.match_array_spread(&pattern.spread, values_spread)?;
        self.match_slice(&pattern.after, values_after)?;
        return Ok(());
    }

    fn match_pattern(&mut self, pattern: &MatchPattern, value: Value) -> IResult<()> {
        match pattern {
            MatchPattern::Discard => Ok(()),
            MatchPattern::Declaration(var) => {
                self.environment.set(var.slot, value);
                Ok(())
            }
            MatchPattern::Assignment(target) => {
                self.assign(target, value)?;
                Ok(())
            }
            MatchPattern::SimpleArray(parts) => match value {
                Value::Array(mut a) => {
                    let mut_values = &mut Rc::make_mut(&mut a).values;
                    self.match_slice(parts, mut_values)
                }
                _ => Err(PatternMatchFailed {
                    message: format!("Expected an array, found {value}"),
                }),
            },
            MatchPattern::SpreadArray(pattern) => match value {
                Value::Array(mut a) => {
                    if a.values.len() < pattern.min_len() {
                        return Err(PatternMatchFailed {
                            message: format!(
                                "Expected an array with length >= {}, but actual array had length = {}",
                                pattern.min_len(),
                                a.values.len()
                            ),
                        });
                    }
                    let mut_values = &mut Rc::make_mut(&mut a).values;
                    self.match_spread_array(pattern, mut_values)
                }
                _ => Err(PatternMatchFailed {
                    message: format!("Expected an array, found {value}"),
                }),
            },
            MatchPattern::Constant(c) => {
                if Self::matches_constant(c, &value) {
                    Ok(())
                } else {
                    Err(PatternMatchFailed {
                        message: format!("Expected constant {c}, found {value}"),
                    })
                }
            }
        }
    }

    fn matches_pattern(pattern: &MatchPattern, value: &Value) -> bool {
        fn matches_slice(parts: &[MatchPattern], values: &[Value]) -> bool {
            if parts.len() != values.len() {
                return false;
            }
            let mut zip = parts.iter().zip(values);
            zip.all(|(p, v)| Interpreter::matches_pattern(&p, &v))
        }

        fn matches_array_spread(pattern: &MatchPattern, values: &[Value]) -> bool {
            match pattern {
                MatchPattern::Discard => true,
                MatchPattern::Declaration(_) => true,
                MatchPattern::Assignment(_) => true,
                MatchPattern::SimpleArray(parts) => matches_slice(parts, values),
                MatchPattern::SpreadArray(pattern) => matches_spread_array(pattern, values),
                MatchPattern::Constant(_) => false,
            }
        }

        fn matches_spread_array(pattern: &SpreadArrayPattern, values: &[Value]) -> bool {
            if values.len() < pattern.min_len() {
                return false;
            }
            let (values_before, rest) = values.split_at(pattern.before.len());
            let (values_spread, values_after) = rest.split_at(rest.len() - pattern.after.len());
            return matches_slice(&pattern.before, values_before)
                && matches_array_spread(&pattern.spread, values_spread)
                && matches_slice(&pattern.after, values_after);
        }

        match pattern {
            MatchPattern::Discard => true,
            MatchPattern::Declaration(_) => true,
            MatchPattern::Assignment(_) => true,
            MatchPattern::SimpleArray(parts) => match value {
                Value::Array(a) => matches_slice(parts, &a.values),
                _ => false,
            },
            MatchPattern::SpreadArray(pattern) => match value {
                Value::Array(a) => matches_spread_array(pattern, &a.values),
                _ => false,
            },
            MatchPattern::Constant(c) => Self::matches_constant(c, value),
        }
    }

    fn matches_constant(constant: &MatchConstant, value: &Value) -> bool {
        match (constant, value) {
            (MatchConstant::Number(n), Value::Number(m)) => n == m,
            (MatchConstant::String(s), Value::String(m)) => s == m.as_ref(),
            (MatchConstant::Bool(b), Value::Bool(m)) => b == m,
            (MatchConstant::Nil, Value::Nil) => true,
            _ => false,
        }
    }

    fn assign(&mut self, target: &AssignmentTarget, value: Value) -> Result<(), InterpreterError> {
        let mut path_values = vec![];
        for index in &target.path {
            path_values.push(self.eval(index)?);
        }
        self.environment
            .assign(target.var.slot, &path_values, value)?;
        Ok(())
    }
}
