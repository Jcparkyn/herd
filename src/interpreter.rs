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
    Return(Value), // Implemented as an error to simplifiy implementation.
    VariableAlreadyDefined(String),
    VariableNotDefined(String),
    FieldNotExists(String),
    IndexOutOfRange { array_len: usize, accessed: usize },
    WrongArgumentCount { expected: usize, supplied: usize },
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
            WrongArgumentCount { expected, supplied } => {
                write!(
                    f,
                    "Wrong number of arguments for function. Expected {expected}, got {supplied}"
                )
            }
            WrongType => write!(f, "Wrong type"),
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
            _ => Err(WrongType),
        }
    }

    pub fn is_number(&self) -> bool {
        matches!(self, Value::Number(_))
    }

    pub fn as_string(&self) -> Result<&str, InterpreterError> {
        match self {
            Value::String(s) => Ok(s),
            _ => Err(WrongType),
        }
    }

    pub fn is_string(&self) -> bool {
        matches!(self, Value::String(_))
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
    "range" => BuiltInFunction::Range,
    "len" => BuiltInFunction::Len,
    "push" => BuiltInFunction::Push,
    "pop" => BuiltInFunction::Pop,
    "sort" => BuiltInFunction::Sort,
    "shiftLeft" => BuiltInFunction::ShiftLeft,
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
            Expr::Variable { name, is_final } => match BUILTIN_FUNCTIONS.get(&name) {
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
                    _ => Err(WrongType),
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
                    _ => Err(WrongType),
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
                    _ => Err(WrongType),
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
                [_] => Err(WrongType),
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
                        if rest.iter().any(|v| !v.is_number()) {
                            return Err(WrongType);
                        }
                        Rc::make_mut(&mut array).values.sort_by(|a, b| {
                            a.as_number().unwrap().total_cmp(&b.as_number().unwrap())
                        });
                        return Ok(Value::Array(array));
                    }
                    [Value::String(_), rest @ ..] => {
                        if rest.iter().any(|v| !v.is_string()) {
                            return Err(WrongType);
                        }
                        Rc::make_mut(&mut array)
                            .values
                            .sort_by(|a, b| a.as_string().unwrap().cmp(&b.as_string().unwrap()));
                        return Ok(Value::Array(array));
                    }
                    _ => return Err(WrongType),
                }
            }
            BuiltInFunction::ShiftLeft => {
                let [val, shift_by] = destructure_args(args)?;
                let val_int = try_into_int(val.as_number()?)?;
                let shift_by_int = try_into_int(shift_by.as_number()?)?;
                return Ok(Value::Number((val_int << shift_by_int) as f64));
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
            lambda_interpreter
                .environment
                .declare(self_name.clone(), Value::Lambda(function.clone()))?;
        }

        for (name, value) in function.params.iter().zip(arg_values.iter()) {
            lambda_interpreter
                .environment
                .declare(name.clone(), value.clone())?;
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

pub fn analyze_statements(stmts: &mut [Statement], deps: &mut HashSet<String>) {
    for stmt in stmts.iter_mut().rev() {
        analyze_statement(stmt, deps);
    }
}

pub fn analyze_statement(stmt: &mut Statement, deps: &mut HashSet<String>) {
    match stmt {
        Statement::Declaration(name, rhs) => {
            deps.remove(name);
            analyze_expr(rhs, deps);
        }
        Statement::Assignment(target, rhs) => {
            if target.path.is_empty() {
                deps.remove(&target.var);
            }
            analyze_expr(rhs, deps);
        }
        Statement::Expression(expr) => {
            analyze_expr(expr, deps);
        }
        Statement::Return(expr) => analyze_expr(expr, deps),
    }
}

fn analyze_block(block: &mut Block, deps: &mut HashSet<String>) {
    // deps: Variables that may be depended on at the current execution point.
    // Code is processed in reverse order, starting with the final expression.
    if let Some(expr) = &mut block.expression {
        analyze_expr(expr, deps);
    }
    analyze_statements(&mut block.statements, deps);
}

fn analyze_expr(expr: &mut Expr, deps: &mut HashSet<String>) {
    // deps: Variables that may be depended on at the current execution point.
    // Code is processed in reverse order.
    match expr {
        Expr::Number(_) => {}
        Expr::Bool(_) => {}
        Expr::String(_) => {}
        Expr::Nil => {}
        Expr::Variable { name, is_final } => {
            // If is_final was already cleared, don't set it.
            // This is required for loops, which are analyzed twice.
            *is_final = *is_final && deps.insert(name.to_string());
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let mut deps_else = deps.clone(); // need to clone even if else is empty, in case deps got removed in if branch.
            if let Some(else_branch) = else_branch {
                analyze_block(else_branch, &mut deps_else);
            }
            analyze_block(then_branch, deps);
            // Deps at start of expression are union of both blocks (because we don't know which branch will be taken).
            for dep in deps_else {
                deps.insert(dep);
            }
            analyze_expr(condition, deps);
        }
        Expr::Op { op: _, lhs, rhs } => {
            analyze_expr(rhs, deps);
            analyze_expr(lhs, deps);
        }
        Expr::Block(block) => {
            analyze_block(block, deps);
        }
        Expr::BuiltInFunction(_) => {}
        Expr::Call { callee, args } => {
            for arg in args.iter_mut().rev() {
                analyze_expr(arg, deps);
            }
            analyze_expr(callee, deps);
        }
        Expr::Lambda {
            body,
            potential_captures,
            params,
        } => {
            // TODO assuming no shared references to body at this point.
            let mut_body = Rc::get_mut(body).unwrap();
            // analyze lambda body, in a separate scope.
            let mut lambda_deps = HashSet::new();
            analyze_block(mut_body, &mut lambda_deps);
            for p in params {
                lambda_deps.remove(p);
            }
            for dep in lambda_deps {
                (*potential_captures).push(dep);
            }
        }
        Expr::Dict(entries) => {
            for (_, v) in entries.iter_mut().rev() {
                analyze_expr(v, deps);
            }
        }
        Expr::Array(elements) => {
            for e in elements.iter_mut().rev() {
                analyze_expr(e, deps);
            }
        }
        Expr::GetIndex(lhs_expr, index_expr) => {
            analyze_expr(lhs_expr, deps);
            analyze_expr(index_expr, deps);
        }
        Expr::ForIn { var, iter, body } => {
            // TODO (I think) this doesn't account for the fact that variables declared in the loop (including the loop variable)
            // are cleared each loop. We could be a bit more aggressive here.
            let mut deps_last_loop = deps.clone();
            analyze_block(body, &mut deps_last_loop);
            let mut deps_other_loops = deps_last_loop.clone();
            deps_other_loops.remove(var); // var can't persist between loops.
            analyze_block(body, &mut deps_other_loops);
            // final dependency set is union of 0 loops, 1 loop, and >1 loops.
            for dep in deps_last_loop {
                deps.insert(dep);
            }
            for dep in deps_other_loops {
                deps.insert(dep);
            }
            analyze_expr(iter, deps);
            deps.remove(var);
        }
    }
}
