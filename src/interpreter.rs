use std::{collections::HashMap, fmt::Debug, rc::Rc, vec};

use crate::{
    ast::{
        AssignmentTarget, Block, BuiltInFunction, Expr, LambdaExpr, MatchConstant, MatchPattern,
        Opcode, SpannedExpr, SpannedStatement, SpreadListPattern, Statement,
    },
    pos::{Span, Spanned},
    value64::{DictInstance, LambdaFunction, ListInstance, Value64 as Value},
};

pub const NIL: Value = Value::NIL;

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
    IndexOutOfRange {
        list_len: usize,
        accessed: usize,
    },
    WrongArgumentCount {
        expected: usize,
        supplied: usize,
    },
    WrongType {
        message: String,
    },
    PatternMatchFailed {
        message: String,
    },
    FunctionCallFailed {
        function: Rc<LambdaFunction>,
        inner: Box<Spanned<InterpreterError>>,
    },
}

type IResult<T> = Result<T, InterpreterError>;
type SpannedResult<T> = Result<T, Spanned<InterpreterError>>;

trait SpannableResult<T> {
    fn with_span(self, span: &Span) -> SpannedResult<T>
    where
        Self: Sized;
}

impl<T> SpannableResult<T> for IResult<T> {
    fn with_span(self, span: &Span) -> SpannedResult<T> {
        self.map_err(|e| Spanned::new(*span, e))
    }
}

impl InterpreterError {
    fn with_span(self, span: &Span) -> Spanned<InterpreterError> {
        Spanned::new(*span, self)
    }
}

use InterpreterError::*;

fn expect_into_type<T>(
    value: Value,
    converter: impl FnOnce(Value) -> Result<T, Value>,
    expected: &str,
) -> IResult<T> {
    match converter(value) {
        Ok(v) => Ok(v),
        Err(v) => Err(WrongType {
            message: format!("Expected {}, found {}", expected, &v),
        }),
    }
}

fn expect_f64(value: &Value) -> IResult<f64> {
    match value.as_f64() {
        Some(v) => Ok(v),
        None => Err(WrongType {
            message: format!("Expected a number, found {}", &value),
        }),
    }
}

fn expect_usize(value: &Value) -> IResult<usize> {
    let result = value.as_f64().and_then(|v| {
        let int = v as usize;
        if int as f64 != v {
            None
        } else {
            Some(int)
        }
    });
    match result {
        Some(v) => Ok(v),
        None => Err(WrongType {
            message: format!("Expected a non-negative integer, found {}", &value),
        }),
    }
}

fn expect_i64(value: &Value) -> IResult<i64> {
    let result = value.as_f64().and_then(|v| {
        let int = v as i64;
        if int as f64 != v {
            None
        } else {
            Some(int)
        }
    });
    match result {
        Some(v) => Ok(v),
        None => Err(WrongType {
            message: format!("Expected an integer, found {}", &value),
        }),
    }
}

fn expect_list(value: Value) -> IResult<Rc<ListInstance>> {
    match value.try_into_list() {
        Ok(v) => Ok(v),
        Err(v) => Err(WrongType {
            message: format!("Expected a list, found {}", &v),
        }),
    }
}

fn expect_into_dict(value: Value) -> IResult<Rc<DictInstance>> {
    expect_into_type(value, Value::try_into_dict, "a dict")
}

fn expect_into_lambda(value: Value) -> IResult<Rc<LambdaFunction>> {
    expect_into_type(value, Value::try_into_lambda, "a lambda")
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
        if old.is_dict() {
            let dict = old.try_into_dict().unwrap();
            Self::assign_dict_field(dict, rhs, index, path)
        } else if old.is_list() {
            let list = old.try_into_list().unwrap();
            Self::assign_list_index(list, rhs, expect_usize(index)?, path)
        } else {
            Err(WrongType {
                message: format!(
                    "Can't assign to index {index}, because {old} is neither a dict nor a list."
                ),
            })
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
                message: format!("Can't use {key} as a key in a dict. Valid keys are strings, numbers, booleans, and lists.")
            });
        }
        let mut_dict = Rc::make_mut(&mut dict);
        match path {
            [] => {
                mut_dict.values.insert(key.clone(), rhs);
                Ok(Value::from_dict(dict))
            }
            [next_field, rest @ ..] => {
                match mut_dict.values.get_mut(key) {
                    Some(entry) => {
                        let old_value = std::mem::replace(entry, NIL);
                        let new_value = Self::assign_part(old_value, rhs, next_field, rest)?;
                        *entry = new_value;
                        return Ok(Value::from_dict(dict));
                    }
                    None => {
                        return Err(KeyNotExists(key.clone()));
                    }
                };
            }
        }
    }

    fn assign_list_index(
        mut list: Rc<ListInstance>,
        rhs: Value,
        index: usize,
        path: &[Value],
    ) -> Result<Value, InterpreterError> {
        if index >= list.values.len() {
            return Err(InterpreterError::IndexOutOfRange {
                list_len: list.values.len(),
                accessed: index,
            });
        }
        let mut_list = Rc::make_mut(&mut list);
        match path {
            [] => {
                mut_list.values[index] = rhs;
                Ok(Value::from_list(list))
            }
            [next_field, rest @ ..] => {
                let old_value = std::mem::replace(&mut mut_list.values[index], NIL);
                let new_value = Self::assign_part(old_value, rhs, next_field, rest)?;
                mut_list.values[index] = new_value;
                Ok(Value::from_list(list))
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
                let current = self.replace(slot, NIL);
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
            self.slots.push(NIL);
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

    pub fn execute(&mut self, statement: &SpannedStatement) -> SpannedResult<()> {
        let span = &statement.span;
        match &statement.value {
            Statement::PatternAssignment(pattern, expr) => {
                let value = self.eval(&expr)?;
                self.match_pattern(&pattern.value, value)
                    .with_span(&pattern.span)?;
                Ok(())
            }
            Statement::Expression(expr) => {
                self.eval(&expr)?;
                Ok(())
            }
            Statement::Return(expr) => {
                let value = self.eval(&expr)?;
                Err(Spanned::new(*span, Return(value)))
            }
        }
    }

    pub fn eval_block(&mut self, block: &Block) -> SpannedResult<Value> {
        for stmt in block.statements.iter() {
            self.execute(&stmt)?;
        }
        if let Some(expr) = &block.expression {
            self.eval(&expr)
        } else {
            Ok(NIL)
        }
    }

    pub fn eval(&mut self, expr: &SpannedExpr) -> SpannedResult<Value> {
        match &expr.value {
            Expr::Number(num) => Ok(Value::from_f64(*num)),
            Expr::Bool(b) => Ok(Value::from_bool(*b)),
            Expr::String(s) => Ok(Value::from_string(s.clone())),
            Expr::Nil => Ok(NIL),
            Expr::Variable(v) => {
                if v.is_final {
                    Ok(self.environment.replace(v.slot, NIL))
                } else {
                    Ok(self.environment.get(v.slot).clone())
                }
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond = self.eval(condition)?;
                if cond.truthy() {
                    return self.eval(then_branch);
                }
                if let Some(else_branch2) = else_branch {
                    return self.eval(else_branch2);
                }
                Ok(NIL)
            }
            Expr::Match(m) => {
                let cond = self.eval(&m.condition)?;
                for (pattern, body) in &m.branches {
                    if Interpreter::matches_pattern(&pattern.value, &cond) {
                        self.match_pattern(&pattern.value, cond).unwrap();
                        return self.eval(&body);
                    }
                }
                Err(PatternMatchFailed {
                    message: format!("No branches matched successfully"),
                }
                .with_span(&expr.span))
            }
            Expr::Op { op, lhs, rhs } => self.eval_binary_op(lhs, rhs, op),
            Expr::Block(block) => self.eval_block(block),
            Expr::Call { callee, args } => self.run_with_args(args, |s| {
                let callee_val = s.eval(callee)?;
                let lambda = expect_into_lambda(callee_val).with_span(&callee.span)?;
                s.call_lambda(&lambda, args.len()).with_span(&expr.span)
            }),
            Expr::CallNative { callee, args } => self.run_with_args(args, |s| {
                s.call_builtin(*callee, args.len()).with_span(&expr.span)
            }),
            Expr::Lambda(l) => {
                let f = self.eval_lambda_definition(l);
                Ok(Value::from_lambda(Rc::new(f)))
            }
            Expr::Dict(entries) => {
                let mut values = HashMap::new();
                for (k, v) in entries.iter() {
                    let key = self.eval(k)?;
                    if !key.is_valid_dict_key() {
                        return Err(WrongType {
                            message: format!("Can't use {key} as a key in a dict. Valid keys are strings, numbers, booleans, and lists.")
                        }.with_span(&k.span));
                    }
                    let value = self.eval(v)?;
                    values.insert(key, value);
                }
                Ok(Value::from_dict(Rc::new(DictInstance { values })))
            }
            Expr::List(elements) => {
                let mut values = Vec::with_capacity(elements.len());
                for e in elements.iter() {
                    values.push(self.eval(e)?);
                }
                Ok(Value::from_list(Rc::new(ListInstance::new(values))))
            }
            Expr::GetIndex(lhs_expr, index_expr) => {
                let index = self.eval(index_expr)?;
                let lhs = self.eval(lhs_expr)?;
                if lhs.is_dict() {
                    let d = lhs.try_into_dict().unwrap();
                    return Ok(d.values.get(&index).cloned().unwrap_or(NIL));
                } else if lhs.is_list() {
                    let a = lhs.try_into_list().unwrap();
                    let idx_int = expect_usize(&index).map_err(|_| {
                        WrongType {
                            message: format!(
                                "Lists can only be indexed with integers, found {index}"
                            ),
                        }
                        .with_span(&index_expr.span)
                    })?;
                    return match a.values.get(idx_int) {
                        Some(v) => Ok(v.clone()),
                        None => Err(InterpreterError::IndexOutOfRange {
                            list_len: a.values.len(),
                            accessed: idx_int,
                        }
                        .with_span(&index_expr.span)),
                    };
                } else {
                    return Err(WrongType {
                        message: format!("Can't index {lhs} (trying to get key {index})"),
                    }
                    .with_span(&lhs_expr.span));
                }
            }
            Expr::ForIn { iter, var, body } => {
                let iter_value = self.eval(iter)?;
                match iter_value.try_into_list() {
                    Ok(a) => {
                        for v in a.values.iter() {
                            self.match_pattern(&var.value, v.clone())
                                .with_span(&var.span)?;
                            self.eval(body)?;
                        }
                        Ok(NIL)
                    }
                    Err(v) => {
                        return Err(WrongType {
                            message: format!("Expected a list, found {v}"),
                        }
                        .with_span(&iter.span))
                    }
                }
            }
            Expr::While { condition, body } => {
                while self.eval(condition)?.truthy() {
                    self.eval(body)?;
                }
                return Ok(NIL);
            }
            Expr::Import { .. } => todo!("Imports not supported in tree walking mode"),
        }
    }

    fn run_with_args(
        &mut self,
        args: &[SpannedExpr],
        task: impl FnOnce(&mut Self) -> SpannedResult<Value>,
    ) -> SpannedResult<Value> {
        let arg_stack_len_before = self.arg_stack.len();
        for arg in args.iter() {
            match self.eval(arg) {
                Ok(v) => self.arg_stack.push(v),
                Err(e) => {
                    self.arg_stack.truncate(arg_stack_len_before);
                    return Err(e);
                }
            };
        }
        let result = task(self);
        self.arg_stack.truncate(arg_stack_len_before);
        result
    }

    fn eval_lambda_definition(&mut self, lambda: &LambdaExpr) -> LambdaFunction {
        let mut captures: Vec<Value> = vec![];
        for pc in &lambda.potential_captures {
            // TODO these should check liveness as well
            let v = self.environment.get(pc.slot);
            captures.push(v.clone());
        }
        LambdaFunction {
            params: lambda.params.clone(),
            param_count: lambda.params.len(),
            body: lambda.body.clone(),
            closure: captures,
            self_name: lambda.name.clone(),
            recursive: false,
            func_ptr: None,
        }
    }

    fn add(lhs: Value, rhs: Value) -> IResult<Value> {
        fn get_err(lhs: &Value, rhs: &Value) -> IResult<Value> {
            Err(WrongType {
                message: format!("Can't add {lhs} to {rhs}"),
            })
        }
        if let Some(n1) = lhs.as_f64() {
            if let Some(n2) = rhs.as_f64() {
                return Ok(Value::from_f64(n1 + n2));
            }
        } else if lhs.is_string() {
            if let Some(s2) = rhs.as_string() {
                let mut lhs_str = lhs.try_into_string().unwrap();
                Rc::make_mut(&mut lhs_str).push_str(s2.as_ref());
                return Ok(Value::from_string(lhs_str));
            }
        }
        get_err(&lhs, &rhs)
    }

    fn eval_binary_op(
        &mut self,
        lhs: &SpannedExpr,
        rhs: &SpannedExpr,
        op: &Opcode,
    ) -> SpannedResult<Value> {
        let mut try_num = |expr: &SpannedExpr| -> SpannedResult<f64> {
            let value = self.eval(expr)?;
            expect_f64(&value).with_span(&expr.span)
        };
        match op {
            Opcode::Add => Self::add(self.eval(lhs)?, self.eval(rhs)?).with_span(&lhs.span),
            Opcode::Sub => Ok(Value::from_f64(try_num(lhs)? - try_num(rhs)?)),
            Opcode::Mul => Ok(Value::from_f64(try_num(lhs)? * try_num(rhs)?)),
            Opcode::Div => Ok(Value::from_f64(try_num(lhs)? / try_num(rhs)?)),
            Opcode::Gt => Ok(Value::from_bool(try_num(lhs)? > try_num(rhs)?)),
            Opcode::Gte => Ok(Value::from_bool(try_num(lhs)? >= try_num(rhs)?)),
            Opcode::Lt => Ok(Value::from_bool(try_num(lhs)? < try_num(rhs)?)),
            Opcode::Lte => Ok(Value::from_bool(try_num(lhs)? <= try_num(rhs)?)),
            Opcode::Eq => Ok(Value::from_bool(self.eval(&lhs)? == self.eval(&rhs)?)),
            Opcode::Neq => Ok(Value::from_bool(self.eval(&lhs)? != self.eval(&rhs)?)),
            Opcode::And => Ok(Value::from_bool(
                self.eval(&lhs)?.truthy() && self.eval(&rhs)?.truthy(),
            )),
            Opcode::Or => Ok(Value::from_bool(
                self.eval(&lhs)?.truthy() || self.eval(&rhs)?.truthy(),
            )),
        }
    }

    fn call_internal<const N: usize>(
        &mut self,
        func: &Rc<LambdaFunction>,
        args: [Value; N],
    ) -> IResult<Value> {
        let arg_stack_len_before = self.arg_stack.len();
        self.arg_stack.extend(args);
        let result = self.call_lambda(func, N);
        self.arg_stack.truncate(arg_stack_len_before);
        result
    }

    fn call_builtin(&mut self, builtin: BuiltInFunction, arg_count: usize) -> IResult<Value> {
        match builtin {
            BuiltInFunction::Print => {
                for arg in Self::pop_args(&mut self.arg_stack, arg_count) {
                    match arg.try_into_string() {
                        Ok(s) => print!("{s}"),
                        Err(v) => print!("{v}"),
                    }
                }
                println!();
                return Ok(NIL);
            }
            BuiltInFunction::Not => {
                let [arg] = self.destructure_args(arg_count)?;
                return Ok(Value::from_bool(!arg.truthy()));
            }
            BuiltInFunction::Range => {
                let [start, stop] = self.destructure_args(arg_count)?;
                let start_int = expect_i64(&start)?;
                let stop_int = expect_i64(&stop)?;
                let mut values = Vec::new();
                for i in start_int..stop_int {
                    values.push(Value::from_f64(i as f64));
                }
                return Ok(Value::from_list(Rc::new(ListInstance::new(values))));
            }
            BuiltInFunction::Len => {
                let [arg] = self.destructure_args(arg_count)?;
                if let Some(a) = arg.as_list() {
                    Ok(Value::from_f64(a.values.len() as f64))
                } else if let Some(d) = arg.as_dict() {
                    Ok(Value::from_f64(d.values.len() as f64))
                } else {
                    Err(WrongType {
                        message: format!("Expected a list or dict, found {arg}"),
                    })
                }
            }
            BuiltInFunction::Push => {
                let [list_val, new_value] = self.destructure_args(arg_count)?;
                let mut list = expect_list(list_val)?;
                let mut_list = Rc::make_mut(&mut list);
                mut_list.values.push(new_value);
                return Ok(Value::from_list(list));
            }
            BuiltInFunction::Pop => {
                let [list_val] = self.destructure_args(arg_count)?;
                let mut list = expect_list(list_val)?;
                let mut_list = Rc::make_mut(&mut list);
                mut_list.values.pop();
                // TODO this should really return the value as well.
                return Ok(Value::from_list(list));
            }
            BuiltInFunction::Sort => {
                let [list_val] = self.destructure_args(arg_count)?;
                let mut list = expect_list(list_val)?;
                match (*list).values.as_slice() {
                    [] => Ok(Value::from_list(list)),
                    [first, rest @ ..] => {
                        if first.is_f64() {
                            if let Some(bad_val) = rest.iter().find(|v| !v.is_f64()) {
                                return Err(WrongType {
                                    message: format!(
                                    "Expected all values in list to be numbers, found {bad_val}"
                                ),
                                });
                            }
                            Rc::make_mut(&mut list).values.sort_by(|a, b| {
                                a.as_f64().unwrap().total_cmp(&b.as_f64().unwrap())
                            });
                            return Ok(Value::from_list(list));
                        } else if first.is_string() {
                            if let Some(bad_val) = rest.iter().find(|v| !v.is_string()) {
                                return Err(WrongType {
                                    message: format!(
                                    "Expected all values in list to be strings, found {bad_val}"
                                ),
                                });
                            }
                            Rc::make_mut(&mut list).values.sort_by(|a, b| {
                                a.as_string().unwrap().cmp(&b.as_string().unwrap())
                            });
                            return Ok(Value::from_list(list));
                        }
                        return Err(WrongType {
                            message: format!(
                                "Can only sort lists of numbers or strings, found {first}"
                            ),
                        });
                    }
                }
            }
            BuiltInFunction::Map => {
                let [list_val, f_val] = self.destructure_args(arg_count)?;
                let mut list = expect_list(list_val)?;
                let f = expect_into_lambda(f_val)?;
                let mut_list = Rc::make_mut(&mut list);
                for v in mut_list.values.iter_mut() {
                    let v2 = std::mem::replace(v, NIL);
                    *v = self.call_internal(&f, [v2])?;
                }
                return Ok(Value::from_list(list));
            }
            BuiltInFunction::Filter => {
                let [list_val, f_val] = self.destructure_args(arg_count)?;
                let mut list = expect_list(list_val)?;
                let f = expect_into_lambda(f_val)?;
                let mut_list = Rc::make_mut(&mut list);

                let mut err = None;
                mut_list.values.retain(|x| -> bool {
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
                    None => Ok(Value::from_list(list)),
                };
            }
            BuiltInFunction::RemoveKey => {
                let [dict_val, key] = self.destructure_args(arg_count)?;
                let mut dict = expect_into_dict(dict_val)?;
                let mut_dict = Rc::make_mut(&mut dict);
                mut_dict.values.remove(&key);
                return Ok(Value::from_dict(dict));
            }
            BuiltInFunction::ShiftLeft => {
                let [val, shift_by] = self.destructure_args(arg_count)?;
                let val_int = expect_i64(&val)?;
                let shift_by_int = expect_i64(&shift_by)?;
                return Ok(Value::from_f64((val_int << shift_by_int) as f64));
            }
            BuiltInFunction::BitwiseXOR => {
                let [lhs, rhs] = self.destructure_args(arg_count)?;
                let lhs_int = expect_i64(&lhs)?;
                let rhs_int = expect_i64(&rhs)?;
                return Ok(Value::from_f64((lhs_int ^ rhs_int) as f64));
            }
            BuiltInFunction::Floor => {
                let [val] = self.destructure_args(arg_count)?;
                let val_num = expect_f64(&val)?;
                return Ok(Value::from_f64(val_num.floor()));
            }
            _ => todo!(),
        }
    }

    fn call_lambda(&mut self, function: &Rc<LambdaFunction>, arg_count: usize) -> IResult<Value> {
        if function.params.len() != arg_count {
            let inner_err = function.body.span.wrap(WrongArgumentCount {
                expected: function.params.len(),
                supplied: arg_count,
            });
            return Err(FunctionCallFailed {
                function: function.clone(),
                inner: Box::new(inner_err),
            });
        }
        let old_frame = self.environment.cur_frame;
        self.environment.cur_frame = self.environment.slots.len();
        for param_idx in (0..arg_count).rev() {
            let arg = self.arg_stack.pop().unwrap();
            let pattern = &function.params[param_idx];
            self.match_pattern(&pattern.value, arg)
                .with_span(&pattern.span)
                .map_err(|e| FunctionCallFailed {
                    function: function.clone(),
                    inner: Box::new(e),
                })?;
        }
        for val in &function.closure {
            self.environment.slots.push(val.clone());
        }
        if let Some(_) = &function.self_name {
            self.environment
                .slots
                .push(Value::from_lambda(function.clone()));
        }

        let result = match self.eval(&function.body) {
            Err(Spanned {
                value: InterpreterError::Return(v),
                span: _,
            }) => Ok(v),
            Ok(v) => Ok(v),
            Err(e) => Err(FunctionCallFailed {
                function: function.clone(),
                inner: Box::new(e),
            }),
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

    fn match_slice_replace(&mut self, parts: &[MatchPattern], values: &mut [Value]) -> IResult<()> {
        assert_slice_len(parts, values)?;
        for (part, value) in parts.iter().zip(values) {
            self.match_pattern(part, std::mem::replace(value, NIL))?;
        }
        return Ok(());
    }

    fn match_slice_clone(&mut self, parts: &[MatchPattern], values: &[Value]) -> IResult<()> {
        assert_slice_len(parts, values)?;
        for (part, value) in parts.iter().zip(values) {
            // TODO: We shouldn't need to clone the whole value (parts might not be used)
            self.match_pattern(part, value.clone())?;
        }
        return Ok(());
    }

    fn match_list_spread(&mut self, pattern: &MatchPattern, values: &mut [Value]) -> IResult<()> {
        fn to_value_list(values: &mut [Value]) -> Value {
            let mut vec = Vec::with_capacity(values.len());
            for value in values {
                vec.push(std::mem::replace(value, NIL));
            }
            Value::from_list(Rc::new(ListInstance::new(vec)))
        }
        match pattern {
            MatchPattern::Discard => Ok(()),
            MatchPattern::Declaration(var, _) => {
                self.environment.set(var.slot, to_value_list(values));
                Ok(())
            }
            MatchPattern::Assignment(target) => {
                self.assign(target, to_value_list(values))?;
                Ok(())
            }
            MatchPattern::SimpleList(parts) => self.match_slice_replace(parts, values),
            MatchPattern::SpreadList(pattern) => self.match_spread_list(pattern, values),
            MatchPattern::Constant(c) => Err(InterpreterError::PatternMatchFailed {
                message: format!("Can't use a constant ({c}) as a spread parameter (..)"),
            }),
            MatchPattern::Dict(_dict_pattern) => todo!(),
        }
    }

    fn match_spread_list(
        &mut self,
        pattern: &SpreadListPattern,
        values: &mut [Value],
    ) -> IResult<()> {
        let (values_before, rest) = values.split_at_mut(pattern.before.len());
        let (values_spread, values_after) = rest.split_at_mut(rest.len() - pattern.after.len());
        self.match_slice_replace(&pattern.before, values_before)?;
        self.match_list_spread(&pattern.spread, values_spread)?;
        self.match_slice_replace(&pattern.after, values_after)?;
        return Ok(());
    }

    fn match_pattern(&mut self, pattern: &MatchPattern, value: Value) -> IResult<()> {
        match pattern {
            MatchPattern::Discard => Ok(()),
            MatchPattern::Declaration(var, _) => {
                self.environment.set(var.slot, value);
                Ok(())
            }
            MatchPattern::Assignment(target) => {
                self.assign(target, value)?;
                Ok(())
            }
            MatchPattern::SimpleList(parts) => match value.try_into_list() {
                Ok(a) => match Rc::try_unwrap(a) {
                    Ok(mut arr) => self.match_slice_replace(&parts, &mut arr.values),
                    Err(rc) => self.match_slice_clone(&parts, &rc.values),
                },
                Err(v) => Err(PatternMatchFailed {
                    message: format!("Expected a list, found {v}"),
                }),
            },
            MatchPattern::SpreadList(pattern) => match value.try_into_list() {
                Ok(mut a) => {
                    if a.values.len() < pattern.min_len() {
                        return Err(PatternMatchFailed {
                            message: format!(
                                "Expected a list with length >= {}, but actual list had length = {}",
                                pattern.min_len(),
                                a.values.len()
                            ),
                        });
                    }
                    let mut_values = &mut Rc::make_mut(&mut a).values;
                    self.match_spread_list(pattern, mut_values)
                }
                Err(v) => Err(PatternMatchFailed {
                    message: format!("Expected a list, found {v}"),
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
            MatchPattern::Dict(_dict_pattern) => todo!(),
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

        fn matches_list_spread(pattern: &MatchPattern, values: &[Value]) -> bool {
            match pattern {
                MatchPattern::Discard => true,
                MatchPattern::Declaration(_, _) => true,
                MatchPattern::Assignment(_) => true,
                MatchPattern::SimpleList(parts) => matches_slice(parts, values),
                MatchPattern::SpreadList(pattern) => matches_spread_list(pattern, values),
                MatchPattern::Constant(_) => false,
                MatchPattern::Dict(_dict_pattern) => todo!(),
            }
        }

        fn matches_spread_list(pattern: &SpreadListPattern, values: &[Value]) -> bool {
            if values.len() < pattern.min_len() {
                return false;
            }
            let (values_before, rest) = values.split_at(pattern.before.len());
            let (values_spread, values_after) = rest.split_at(rest.len() - pattern.after.len());
            return matches_slice(&pattern.before, values_before)
                && matches_list_spread(&pattern.spread, values_spread)
                && matches_slice(&pattern.after, values_after);
        }

        match pattern {
            MatchPattern::Discard => true,
            MatchPattern::Declaration(_, _) => true,
            MatchPattern::Assignment(_) => true,
            MatchPattern::SimpleList(parts) => match value.as_list() {
                Some(a) => matches_slice(parts, &a.values),
                _ => false,
            },
            MatchPattern::SpreadList(pattern) => match value.as_list() {
                Some(a) => matches_spread_list(pattern, &a.values),
                _ => false,
            },
            MatchPattern::Constant(c) => Self::matches_constant(c, value),
            MatchPattern::Dict(_dict_pattern) => todo!(),
        }
    }

    fn matches_constant(constant: &MatchConstant, value: &Value) -> bool {
        match constant {
            MatchConstant::Number(n) => value.as_f64() == Some(*n), // Use NaN equality?
            MatchConstant::String(s) => value.as_string() == Some(s),
            MatchConstant::Bool(b) => value.as_bool() == Some(*b),
            MatchConstant::Nil => value.is_nil(),
        }
    }

    fn assign(&mut self, target: &AssignmentTarget, value: Value) -> Result<(), InterpreterError> {
        let mut path_values = vec![];
        for index in &target.path {
            let result = self.eval(index);
            // Ignore the span, we'll use the one from the whole pattern for simplicity
            let value = result.map_err(|s| s.value)?;
            path_values.push(value);
        }
        self.environment
            .assign(target.var.slot, &path_values, value)?;
        Ok(())
    }
}

fn assert_slice_len(parts: &[MatchPattern], values: &[Value]) -> IResult<()> {
    if parts.len() != values.len() {
        return Err(PatternMatchFailed {
            message: format!(
                "Expected a list with length={}, but actual list had length={}",
                parts.len(),
                values.len()
            ),
        });
    }
    Ok(())
}
