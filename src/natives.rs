use std::{
    collections::HashMap,
    fs,
    io::{self, BufRead, Write},
    mem::ManuallyDrop,
    ops::Deref,
    panic::AssertUnwindSafe,
    time::SystemTime,
};

use cranelift::prelude::{AbiParam, Signature, Type, types};
use rand::Rng;
use rayon::prelude::*;
use regex::Regex;
use strum::{EnumIter, IntoEnumIterator};

use crate::{
    dict::DictInstance,
    error::HerdError,
    jit::VmContext,
    rc::Rc,
    value64::{LambdaFunction, ListInstance, Value64, rc_mutate, rc_new},
};

#[derive(Clone, Copy, Debug)]
pub struct NativeFuncDef {
    pub name: &'static str,
    pub func_ptr: *const u8,
    pub make_sig: fn(&mut Signature) -> (),
    pub needs_vm: bool,
    pub fallible: bool,
}

impl NativeFuncDef {
    pub fn with_vm(mut self) -> Self {
        self.needs_vm = true;
        self
    }

    pub fn fallible(mut self) -> Self {
        self.fallible = true;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, EnumIter)]
pub enum NativeFuncId {
    ListNew,
    ListLenU64,
    ListGetU64,
    ListBorrowU64,
    DictNew,
    DictInsert,
    DictLookup,
    Clone,
    Drop,
    ValBorrowIndex,
    ValReplaceIndex,
    ValSetIndex,
    ValEq,
    ValEqU8,
    ValTruthy,
    ValConcat,
    ValGetLambdaDetails,
    ConstructLambda,
    ImportModule,
    Print,
    AllocError,
    StringTemplate,
}

trait AbiType {
    const TYPE: Type;
}

impl AbiType for () {
    // HACK: use as sentinel value for no return
    const TYPE: Type = types::INVALID;
}

impl AbiType for Value64 {
    const TYPE: Type = types::F64;
}

impl Default for Value64 {
    fn default() -> Self {
        Value64::NIL
    }
}

impl AbiType for Value64Ref {
    const TYPE: Type = types::F64;
}

impl AbiType for i64 {
    const TYPE: Type = types::I64;
}

impl AbiType for u64 {
    const TYPE: Type = types::I64;
}

impl AbiType for u8 {
    const TYPE: Type = types::I8;
}

impl<T> AbiType for *const T {
    const TYPE: Type = types::I64;
}

impl<T> AbiType for *mut T {
    const TYPE: Type = types::I64;
}

impl AbiType for &VmContext {
    const TYPE: Type = types::I64;
}

macro_rules! func_name {
    ($func:expr) => {
        concat!("NATIVE:", stringify!($func))
    };
}

macro_rules! get_def {
    (0, $func:expr) => {
        get_def0(func_name!($func), $func)
    };
    (1, $func:expr) => {
        get_def1(func_name!($func), $func)
    };
    (2, $func:expr) => {
        get_def2(func_name!($func), $func)
    };
    (3, $func:expr) => {
        get_def3(func_name!($func), $func)
    };
    (4, $func:expr) => {
        get_def4(func_name!($func), $func)
    };
    (5, $func:expr) => {
        get_def5(func_name!($func), $func)
    };
}

fn get_native_func_def(func: NativeFuncId) -> NativeFuncDef {
    match func {
        NativeFuncId::ListNew => get_def!(2, list_new),
        NativeFuncId::ListLenU64 => get_def!(1, list_len_u64), // assert in JIT
        NativeFuncId::ListGetU64 => get_def!(2, list_get_u64), // assert in JIT (TODO: for loops)
        NativeFuncId::ListBorrowU64 => get_def!(2, list_borrow_u64), // assert in JIT
        NativeFuncId::DictNew => get_def!(1, dict_new),
        NativeFuncId::DictInsert => get_def!(3, dict_insert), // assert in JIT
        NativeFuncId::DictLookup => get_def!(3, dict_lookup),
        NativeFuncId::Clone => get_def!(1, clone),
        NativeFuncId::Drop => get_def!(1, val_drop),
        NativeFuncId::ValBorrowIndex => get_def!(3, val_borrow_index).fallible(),
        NativeFuncId::ValReplaceIndex => get_def!(4, val_take_index).fallible(),
        NativeFuncId::ValSetIndex => get_def!(4, val_set_index).fallible(),
        NativeFuncId::ValEq => get_def!(2, val_eq),
        NativeFuncId::ValEqU8 => get_def!(2, val_eq_u8),
        NativeFuncId::ValTruthy => get_def!(1, val_truthy_u8),
        NativeFuncId::ValConcat => get_def!(2, val_concat), // TODO
        NativeFuncId::ValGetLambdaDetails => get_def!(4, get_lambda_details).fallible(),
        NativeFuncId::ConstructLambda => get_def!(5, construct_lambda),
        NativeFuncId::ImportModule => get_def!(3, import_module).fallible().with_vm(),
        NativeFuncId::Print => get_def!(1, print),
        NativeFuncId::AllocError => get_def!(4, alloc_herd_error),
        NativeFuncId::StringTemplate => get_def!(3, string_template),
    }
}

pub fn get_natives() -> HashMap<NativeFuncId, NativeFuncDef> {
    let mut map = HashMap::new();
    for func in NativeFuncId::iter() {
        map.insert(func, get_native_func_def(func));
    }
    return map;
}

pub fn get_builtins() -> HashMap<&'static str, NativeFuncDef> {
    let mut map = HashMap::new();
    map.insert("print", get_def!(1, print));
    map.insert("readline", get_def!(1, readln).fallible());
    map.insert("not", get_def!(1, val_not));
    map.insert("shiftLeft", get_def!(3, val_shift_left).fallible());
    map.insert("range", get_def!(3, range).fallible());
    map.insert("push", get_def!(3, list_push).fallible());
    map.insert("pop", get_def!(2, list_pop).fallible());
    map.insert("len", get_def!(2, len).fallible());
    map.insert("sort", get_def!(2, list_sort).fallible());
    map.insert("listSlice", get_def!(4, list_slice).fallible());
    map.insert("removeKey", get_def!(3, dict_remove_key).fallible());
    map.insert("dictKeys", get_def!(2, dict_keys).fallible());
    map.insert("dictEntries", get_def!(2, dict_entries).fallible());
    map.insert("randomInt", get_def!(3, random_int).fallible());
    map.insert("randomFloat", get_def!(3, random_float).fallible());
    map.insert("floatPow", get_def!(3, float_pow).fallible());
    map.insert("assertTruthy", get_def!(2, assert_truthy).fallible());
    map.insert("regexFind", get_def!(3, regex_find).fallible());
    map.insert("regexReplace", get_def!(4, regex_replace).fallible());
    map.insert("toString", get_def!(1, val_to_string));
    map.insert("stringToChars", get_def!(2, string_to_chars).fallible());
    map.insert("stringToUpper", get_def!(2, string_upper).fallible());
    map.insert("stringToLower", get_def!(2, string_lower).fallible());
    map.insert("stringSplit", get_def!(3, string_split).fallible());
    map.insert(
        "parallelMap",
        get_def!(4, parallel_map).fallible().with_vm(),
    );
    map.insert("epochTime", get_def!(1, epoch_time).fallible());
    map.insert("parseFloat", get_def!(2, parse_float).fallible());
    map.insert("programArgs", get_def!(1, program_args).with_vm());
    map.insert("fileToString", get_def!(2, file_to_string).fallible());
    map.insert("fileLines", get_def!(2, file_lines).fallible());

    return map;
}

macro_rules! generate_get_def {
    ($fname:ident $(, $param:ident)*) => {
        #[allow(dead_code)]
        fn $fname<$($param: AbiType,)* TRet: AbiType>(
            name: &'static str,
            func: extern "C" fn($($param),*) -> TRet,
        ) -> NativeFuncDef {
            NativeFuncDef {
                name,
                func_ptr: func as *const u8,
                make_sig: |sig| {
                    $(
                        sig.params.push(AbiParam::new($param::TYPE));
                    )*
                    if TRet::TYPE != types::INVALID {
                        sig.returns.push(AbiParam::new(TRet::TYPE));
                    }
                },
                needs_vm: false,
                fallible: false,
            }
        }
    };
}

generate_get_def!(get_def0);
generate_get_def!(get_def1, T1);
generate_get_def!(get_def2, T1, T2);
generate_get_def!(get_def3, T1, T2, T3);
generate_get_def!(get_def4, T1, T2, T3, T4);
generate_get_def!(get_def5, T1, T2, T3, T4, T5);

fn handle_c_result<F, TReturn>(error_out: ErrorOut, f: F) -> TReturn
where
    TReturn: Default,
    F: FnOnce() -> Result<TReturn, HerdError>,
{
    unsafe {
        *error_out = std::ptr::null();
    }
    f().unwrap_or_else(|err| {
        if !error_out.is_null() {
            unsafe {
                *error_out = Box::into_raw(Box::new(err));
            }
        }
        TReturn::default()
    })
}

// Not using &Value64, so that the ABI for these functions still takes
// regular f64 values.
#[repr(transparent)]
pub struct Value64Ref(ManuallyDrop<Value64>);

impl Value64Ref {
    const NIL: Self = Self(ManuallyDrop::new(Value64::NIL));

    pub fn from_ref(val: &Value64) -> Self {
        Self(ManuallyDrop::new(unsafe { std::ptr::read(val) }))
    }

    pub fn as_ref(&self) -> &Value64 {
        &self.0
    }
}

impl Deref for Value64Ref {
    type Target = Value64;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl std::fmt::Display for Value64Ref {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&*self.0, f)
    }
}

impl Default for Value64Ref {
    fn default() -> Self {
        Self::NIL
    }
}

fn guard_f64(val: &Value64) -> Result<f64, HerdError> {
    val.as_f64()
        .ok_or_else(|| HerdError::native_code(format!("Expected a number, got {}", val)))
}

fn guard_i64(val: &Value64) -> Result<i64, HerdError> {
    guard_f64(val).map(|f| f as i64)
}

fn guard_into_list(val: Value64) -> Result<Rc<ListInstance>, HerdError> {
    val.try_into_list()
        .map_err(|v| HerdError::native_code(format!("Expected a list, got {}", v)))
}

fn guard_into_dict(val: Value64) -> Result<Rc<DictInstance>, HerdError> {
    val.try_into_dict()
        .map_err(|v| HerdError::native_code(format!("Expected a dict, got {}", v)))
}

fn guard_str(val: &Value64) -> Result<&str, HerdError> {
    val.as_str()
        .ok_or_else(|| HerdError::native_code(format!("Expected a string, got {}", val)))
}

fn guard_lambda(val: &Value64) -> Result<&LambdaFunction, HerdError> {
    val.as_lambda()
        .ok_or_else(|| HerdError::native_code(format!("Expected a lambda, got {}", val)))
}

fn parse_list_index_f64(val: f64, len: usize) -> Option<usize> {
    let i = val.abs() as usize;
    if val < 0.0 {
        if i > len { None } else { Some(len - i) }
    } else {
        if i >= len { None } else { Some(i) }
    }
}

fn guard_list_index(val: &Value64, len: usize) -> Result<usize, HerdError> {
    let f = match val.as_f64() {
        Some(f) => f,
        None => {
            return Err(HerdError::native_code(format!(
                "List index should be an integer, got {}",
                *val
            )));
        }
    };
    match parse_list_index_f64(f, len) {
        Some(i) => Ok(i),
        None => Err(HerdError::native_code(format!(
            "List index out of range, got {} but length is {}",
            f, len
        ))),
    }
}

type Out<T> = *mut T;
type ErrorOut = *mut *const HerdError;

pub extern "C" fn list_new(len: u64, items: *const Value64) -> Value64 {
    let items_slice = unsafe { std::slice::from_raw_parts(items, len as usize) };
    let mut items = Vec::with_capacity(len as usize);
    for item in items_slice {
        // values in items array are all owned, so we don't need to clone them
        items.push(unsafe { item.cheeky_copy() });
    }
    Value64::from_list(rc_new(ListInstance::new(items)))
}

pub extern "C" fn list_push(error_out: ErrorOut, list: Value64, val: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let mut list = guard_into_list(list)?;
        rc_mutate(&mut list, |l| l.values.push(val));
        Ok(Value64::from_list(list))
    })
}

pub extern "C" fn list_pop(error_out: ErrorOut, list: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let mut list = guard_into_list(list)?;
        rc_mutate(&mut list, |l| {
            l.values.pop();
        });
        Ok(Value64::from_list(list))
    })
}

pub extern "C" fn list_len_u64(list: Value64Ref) -> u64 {
    list.as_list().unwrap().values.len() as u64
}

pub extern "C" fn list_get_u64(list: Value64Ref, index: u64) -> Value64 {
    let list2 = list.as_list().unwrap();
    list2.values[index as usize].clone()
}

pub extern "C" fn list_borrow_u64(list: Value64Ref, index: u64) -> Value64Ref {
    let list2 = list.as_list().unwrap();
    Value64Ref::from_ref(&list2.values[index as usize])
}

pub extern "C" fn list_sort(error_out: ErrorOut, list_val: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let mut list = guard_into_list(list_val)?;
        rc_mutate(&mut list, |l| {
            l.values.sort_by(|a, b| a.display_cmp(b));
        });
        Ok(Value64::from_list(list))
    })
}

fn get_slice_index(index: i64, len: usize) -> usize {
    if index < 0 {
        let abs_index = (-index) as usize;
        if abs_index > len { 0 } else { len - abs_index }
    } else {
        let index_usize = index as usize;
        if index_usize > len { len } else { index_usize }
    }
}

pub extern "C" fn list_slice(
    error_out: ErrorOut,
    list_val: Value64,
    start_index: Value64,
    stop_index: Value64,
) -> Value64 {
    handle_c_result(error_out, || {
        let list = guard_into_list(list_val)?;
        let len = list.len();
        let start = if start_index.is_nil() {
            0
        } else {
            get_slice_index(guard_i64(&start_index)?, len)
        };
        let stop = if stop_index.is_nil() {
            len
        } else {
            get_slice_index(guard_i64(&stop_index)?, len)
        };
        if start >= stop {
            return Err(HerdError::native_code(
                "Start index must be less than stop index".to_string(),
            ));
        }
        let sliced_values = list.values[start..stop].to_vec();
        Ok(Value64::from_list(rc_new(ListInstance::new(sliced_values))))
    })
}

pub extern "C" fn dict_new(capacity: u64) -> Value64 {
    Value64::from_dict(rc_new(DictInstance::with_capacity(capacity as usize)))
}

pub extern "C" fn dict_insert(dict: Value64, key: Value64, val: Value64) -> Value64 {
    let mut dict = dict.try_into_dict().unwrap();
    rc_mutate(&mut dict, |d| {
        d.insert(key, val);
    });
    Value64::from_dict(dict)
}

pub extern "C" fn dict_lookup(dict: Value64Ref, key: Value64Ref, found: Out<u8>) -> Value64 {
    // TODO: take a string directly for faster lookup
    let dict = dict.as_dict().unwrap();
    match dict.get(&key) {
        Some(v) => {
            unsafe {
                *found = 1;
            }
            v.clone()
        }
        None => {
            unsafe {
                *found = 0;
            }
            Value64::NIL
        }
    }
}

pub extern "C" fn dict_remove_key(error_out: ErrorOut, dict: Value64, key: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let mut dict = guard_into_dict(dict)?;
        rc_mutate(&mut dict, |d| {
            d.remove(&key);
        });
        Ok(Value64::from_dict(dict))
    })
}

pub extern "C" fn dict_keys(error_out: ErrorOut, dict: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let dict = guard_into_dict(dict)?;
        let keys: Vec<Value64> = dict.keys().cloned().collect();
        Ok(Value64::from_list(rc_new(ListInstance::new(keys))))
    })
}

pub extern "C" fn dict_entries(error_out: ErrorOut, dict: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let dict = guard_into_dict(dict)?;
        let entries: Vec<Value64> = dict
            .iter()
            .map(|(k, v)| {
                let entry = vec![k.clone(), v.clone()];
                Value64::from_list(rc_new(ListInstance::new(entry)))
            })
            .collect();
        Ok(Value64::from_list(rc_new(ListInstance::new(entries))))
    })
}

pub extern "C" fn range(error_out: ErrorOut, start: Value64, stop: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let start_int = guard_i64(&start)?;
        let stop_int = guard_i64(&stop)?;
        let mut values = Vec::new();
        for i in start_int..stop_int {
            values.push(Value64::from_f64(i as f64));
        }
        Ok(Value64::from_list(rc_new(ListInstance::new(values))))
    })
}

pub extern "C" fn len(error_out: ErrorOut, val: Value64) -> Value64 {
    handle_c_result(error_out, || {
        if val.is_list() {
            let list = val.try_into_list().unwrap();
            Ok(Value64::from_f64(list.len() as f64))
        } else if val.is_dict() {
            let dict = val.try_into_dict().unwrap();
            Ok(Value64::from_f64(dict.len() as f64))
        } else if val.is_string() {
            let s = guard_str(&val)?;
            Ok(Value64::from_f64(s.len() as f64))
        } else {
            Err(HerdError::native_code(format!(
                "Expected list, dict, or string, got {}",
                val
            )))
        }
    })
}

pub extern "C" fn clone(val: Value64Ref) -> Value64 {
    val.clone()
}

pub extern "C" fn val_drop(val: Value64) {
    std::mem::drop(val);
}

// Replaces the element at an index with NIL, and returns the old element via element_out.
// The return value is the new list/dict.
pub extern "C" fn val_take_index(
    error_out: ErrorOut,
    val: Value64,
    index: Value64Ref,
    element_out: *mut Value64,
) -> Value64 {
    unsafe { *element_out = Value64::NIL };
    handle_c_result(error_out, || {
        if val.is_list() {
            let mut list = val.try_into_list().unwrap();
            let index_int = guard_list_index(index.as_ref(), list.len())?;
            let element = rc_mutate(&mut list, |l| {
                std::mem::replace(&mut l.values[index_int], Value64::NIL)
            });
            unsafe { *element_out = element };
            Ok(Value64::from_list(list))
        } else if val.is_dict() {
            let mut dict = val.try_into_dict().unwrap();
            let element = rc_mutate(&mut dict, |d| {
                if let Some(value) = d.get_mut(&index) {
                    std::mem::replace(value, Value64::NIL)
                } else {
                    Value64::NIL
                }
            });
            unsafe { *element_out = element };
            Ok(Value64::from_dict(dict))
        } else {
            Err(HerdError::native_code(format!(
                "Expected list or dict, was {}",
                &val
            )))
        }
    })
}

pub extern "C" fn val_borrow_index(
    error_out: ErrorOut,
    val: Value64Ref,
    index: Value64Ref,
) -> Value64Ref {
    handle_c_result(error_out, || {
        if let Some(list) = val.as_list() {
            let index_int = guard_list_index(index.as_ref(), list.len())?;
            Ok(Value64Ref::from_ref(&list.values[index_int]))
        } else if let Some(dict) = val.as_dict() {
            Ok(Value64Ref::from_ref(
                dict.get(&index).unwrap_or(&Value64::NIL),
            ))
        } else {
            Err(HerdError::native_code(format!(
                "Expected list or dict, was {}",
                *val
            )))
        }
    })
}

pub extern "C" fn val_set_index(
    error_out: ErrorOut,
    val: Value64,
    index: Value64,
    new_val: Value64,
) -> Value64 {
    handle_c_result(error_out, || {
        if val.is_list() {
            let mut list = val.try_into_list().unwrap();
            let index_int = guard_list_index(&index, list.len())?;
            rc_mutate(&mut list, |l| {
                l.values[index_int] = new_val;
            });
            Ok(Value64::from_list(list))
        } else if val.is_dict() {
            let mut dict = val.try_into_dict().unwrap();
            rc_mutate(&mut dict, |d| {
                d.insert(index, new_val);
            });
            Ok(Value64::from_dict(dict))
        } else {
            Err(HerdError::native_code(format!(
                "Expected list or dict, was {}",
                &val
            )))
        }
    })
}

pub extern "C" fn val_eq(val1: Value64Ref, val2: Value64Ref) -> Value64 {
    Value64::from_bool(*val1 == *val2)
}

pub extern "C" fn val_eq_u8(val1: Value64Ref, val2: Value64Ref) -> u8 {
    (*val1 == *val2) as u8
}

pub extern "C" fn val_truthy_u8(val: Value64Ref) -> u8 {
    val.truthy() as u8
}

pub extern "C" fn val_concat(val1: Value64, val2: Value64) -> Value64 {
    if val1.is_string() {
        if !val2.is_string() {
            panic!("Expected string, was {}", val2)
        }
        let mut str1 = val1.try_into_string().unwrap();
        let str2 = val2.as_str().unwrap();
        rc_mutate(&mut str1, |s| s.push_str(&str2));
        Value64::from_string(str1)
    } else if val1.is_list() {
        if !val2.is_list() {
            panic!("Expected list, was {}", val2)
        }
        let mut list1 = val1.try_into_list().unwrap();
        let list2 = val2.try_into_list().unwrap();
        rc_mutate(&mut list1, |l| {
            l.values.extend(list2.values.iter().cloned())
        });
        Value64::from_list(list1)
    } else {
        panic!("Expected string or list, was {}", val1)
    }
}

pub extern "C" fn float_pow(error_out: ErrorOut, base: Value64, exponent: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let base_float = guard_f64(&base)?;
        let exp_float = guard_f64(&exponent)?;
        Ok(Value64::from_f64(base_float.powf(exp_float)))
    })
}

pub extern "C" fn assert_truthy(error_out: ErrorOut, val: Value64) -> Value64 {
    handle_c_result(error_out, || {
        if !val.truthy() {
            Err(HerdError::native_code(format!(
                "Assertion failed. Expected truthy value, was {}",
                val
            )))
        } else {
            Ok(Value64::NIL)
        }
    })
}

pub extern "C" fn get_lambda_details(
    error_out: ErrorOut,
    val: Value64Ref,
    param_count: u64,
    closure_out: Out<*const Value64>,
) -> *const u8 {
    handle_c_result(error_out, || {
        let Some(lambda) = val.as_lambda() else {
            return Err(HerdError::native_code(format!(
                "Tried to call something that isn't a function: {}",
                val
            )));
        };
        if lambda.param_count != param_count as usize {
            return Err(HerdError::native_code(format!(
                "Wrong number of arguments passed to function {}. Expected {}, got {}",
                val, lambda.param_count, param_count
            )));
        } else {
            unsafe {
                // We can only return one value, so passing this via pointer
                *closure_out = lambda.closure.as_ptr();
            }
            return Ok(lambda.func_ptr.unwrap());
        }
    })
}

pub extern "C" fn construct_lambda(
    param_count: u64,
    name: Value64Ref, // string or ()
    func_ptr: *const u8,
    capture_count: u64,
    captures: *const Value64,
) -> Value64 {
    let closure_slice = unsafe { std::slice::from_raw_parts(captures, capture_count as usize) };
    let lambda = LambdaFunction {
        param_count: param_count as usize,
        closure: closure_slice.to_vec(),
        self_name: name.as_str().map(|x| x.to_string()),
        func_ptr: Some(func_ptr),
    };
    Value64::from_lambda(rc_new(lambda))
}

pub extern "C" fn import_module(error_out: ErrorOut, vm: &VmContext, name: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let path = name.as_str().ok_or_else(|| {
            HerdError::native_code(format!("Module name must be a string, got {}", name))
        })?;
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| vm.execute_file(&path, true)));
        match result {
            Ok(Ok(Ok(result))) => Ok(result),
            Ok(Ok(Err(err))) => Err(err.wrap_native(format!("Error importing module {}", name))),
            Ok(Err(err)) => Err(HerdError::native_code(format!(
                "Error importing module {}: {:?}",
                name, err
            ))),
            Err(err) => Err(HerdError::native_code(format!(
                "Panic occurred while importing module {}: {:?}",
                name, err
            ))),
        }
    })
}

pub extern "C" fn val_shift_left(error_out: ErrorOut, val: Value64, by: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let a = guard_i64(&val)?;
        let b = guard_i64(&by)?;
        let result = Value64::from_f64((a << b) as f64);
        Ok(result)
    })
}

pub extern "C" fn val_not(val: Value64) -> Value64 {
    Value64::from_bool(!val.truthy())
}

pub extern "C" fn random_int(error_out: ErrorOut, min: Value64, max: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let min_int = guard_i64(&min)?;
        let max_int = guard_i64(&max)?;
        if min_int >= max_int {
            return Err(HerdError::native_code("min should be < max in randomInt"));
        }
        let result = rand::rng().random_range(min_int..max_int);
        Ok(Value64::from_f64(result as f64))
    })
}

pub extern "C" fn random_float(error_out: ErrorOut, min: Value64, max: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let min_float = guard_f64(&min)?;
        let max_float = guard_f64(&max)?;
        if min_float >= max_float {
            return Err(HerdError::native_code("min should be < max in randomFloat"));
        }
        let result = rand::rng().random_range(min_float..=max_float);
        Ok(Value64::from_f64(result))
    })
}

pub extern "C" fn regex_find(error_out: ErrorOut, text: Value64, regex: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let regex_str = guard_str(&regex)?;
        let text_str = guard_str(&text)?;
        let regex = Regex::new(regex_str).map_err(|e| HerdError::native_code(e.to_string()))?;
        let result = regex.find(text_str);
        match result {
            Some(m) => {
                let start = Value64::from_usize(m.start());
                let end = Value64::from_usize(m.end());
                Ok(Value64::from_list(rc_new(ListInstance::new(vec![
                    start, end,
                ]))))
            }
            None => Ok(Value64::NIL),
        }
    })
}

pub extern "C" fn regex_replace(
    error_out: ErrorOut,
    text: Value64,
    regex: Value64,
    replacement: Value64,
) -> Value64 {
    handle_c_result(error_out, || {
        let regex_str = guard_str(&regex)?;
        let text_str = guard_str(&text)?;
        let replacement_str = guard_str(&replacement)?;
        let regex = Regex::new(regex_str).map_err(|e| HerdError::native_code(e.to_string()))?;
        let result = regex.replace_all(text_str, replacement_str);
        Ok(Value64::from_string(rc_new(result.to_string())))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn parallel_map(
    error_out: ErrorOut,
    vm: &VmContext,
    list: Value64,
    func: Value64,
) -> Value64 {
    handle_c_result(error_out, || {
        let list = guard_into_list(list)?;
        let _ = guard_lambda(&func)?;

        let result = list
            .values
            .par_iter()
            .map(|v| vm.run_lambda(&func, &[v.clone()]))
            .collect::<Result<Vec<_>, HerdError>>();

        let result_mapped = result.map_err(|e| HerdError {
            message: String::new(),
            pos: None,
            inner: Some(Box::new(e)),
            file_id: None,
        });

        Ok(Value64::from_list(rc_new(ListInstance::new(
            result_mapped?,
        ))))
    })
}

pub extern "C" fn epoch_time(error_out: ErrorOut) -> Value64 {
    handle_c_result(error_out, || {
        let duration_since_epoch = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|e| HerdError::native_code(e.to_string()))?;
        Ok(Value64::from_f64(duration_since_epoch.as_secs_f64()))
    })
}

pub extern "C" fn parse_float(error_out: ErrorOut, val: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let s = guard_str(&val)?;
        match s.parse::<f64>() {
            Ok(f) => Ok(Value64::from_f64(f)),
            Err(_) => Ok(Value64::NIL),
        }
    })
}

pub extern "C" fn val_to_string(val: Value64) -> Value64 {
    if val.is_string() {
        return val;
    }
    Value64::from_string(rc_new(format!("{}", val)))
}

pub extern "C" fn string_to_chars(error_out: ErrorOut, val: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let s = guard_str(&val)?;
        let chars: Vec<Value64> = s
            .chars()
            .map(|c| Value64::from_string(rc_new(c.to_string())))
            .collect();
        Ok(Value64::from_list(rc_new(ListInstance::new(chars))))
    })
}

pub extern "C" fn string_lower(error_out: ErrorOut, val: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let s = guard_str(&val)?;
        Ok(Value64::from_string(rc_new(s.to_lowercase())))
    })
}

pub extern "C" fn string_upper(error_out: ErrorOut, val: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let s = guard_str(&val)?;
        Ok(Value64::from_string(rc_new(s.to_uppercase())))
    })
}

pub extern "C" fn string_split(error_out: ErrorOut, val: Value64, delimiter: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let s = guard_str(&val)?;
        let delim_str = guard_str(&delimiter)?;
        let parts: Vec<Value64> = s
            .split(delim_str)
            .map(|part| Value64::from_string(rc_new(part.to_string())))
            .collect();
        Ok(Value64::from_list(rc_new(ListInstance::new(parts))))
    })
}

pub extern "C" fn program_args(vm: &VmContext) -> Value64 {
    let args = vm.program_args.clone();
    let mut result = Vec::new();
    for arg in args {
        result.push(Value64::from_string(rc_new(arg)));
    }
    Value64::from_list(rc_new(ListInstance::new(result)))
}

pub extern "C" fn file_to_string(error_out: ErrorOut, path: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let path_str = path.as_str().ok_or_else(|| {
            HerdError::native_code(format!("Expected path to be a string, found {}", path))
        })?;
        let content =
            fs::read_to_string(path_str).map_err(|e| HerdError::native_code(e.to_string()))?;
        Ok(Value64::from_string(rc_new(content)))
    })
}

pub extern "C" fn file_lines(error_out: ErrorOut, path: Value64) -> Value64 {
    handle_c_result(error_out, || {
        let path_str = guard_str(&path)?;
        let file =
            std::fs::File::open(path_str).map_err(|e| HerdError::native_code(e.to_string()))?;
        let reader = std::io::BufReader::new(file);
        let mut lines = Vec::new();
        for line in reader.lines() {
            match line {
                Ok(l) => lines.push(Value64::from_string(rc_new(l))),
                Err(e) => {
                    return Err(HerdError::native_code(format!(
                        "Error reading line from file {}: {}",
                        path_str, e
                    )));
                }
            }
        }
        Ok(Value64::from_list(rc_new(ListInstance::new(lines))))
    })
}

// TODO: Variadic functions
pub extern "C" fn print(val: Value64) {
    match val.try_into_string() {
        Ok(s) => print!("{s}"),
        Err(v) => print!("{v}"),
    }
    io::stdout().flush().unwrap();
}

pub extern "C" fn readln(error_out: ErrorOut) -> Value64 {
    handle_c_result(error_out, || {
        let mut input = String::new();
        match std::io::stdin().read_line(&mut input) {
            Ok(_) => Ok(Value64::from_string(rc_new(input.trim_end().to_string()))),
            Err(e) => Err(HerdError::native_code(format!("Error reading line: {}", e))),
        }
    })
}

pub extern "C" fn alloc_herd_error(
    msg: Value64,
    inner: *mut HerdError,
    pos: u64,
    file_id: u64,
) -> *mut HerdError {
    let inner_box = if inner.is_null() {
        None
    } else {
        unsafe { Some(Box::from_raw(inner)) }
    };
    let error = HerdError {
        message: msg.as_str().unwrap_or("").to_string(),
        pos: Some(pos as usize),
        inner: inner_box,
        file_id: Some(file_id as usize),
    };
    Box::into_raw(Box::new(error))
}

pub extern "C" fn string_template(
    template: Value64,
    parts_len: u64,
    parts: *const Value64,
) -> Value64 {
    let template = template.as_str().unwrap();
    let parts_slice = unsafe { std::slice::from_raw_parts(parts, parts_len as usize) };
    let mut result = String::new();
    let mut part_index = 0;
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '{' && chars.peek() == Some(&'}') {
            chars.next(); // consume '}'
            if part_index < parts_slice.len() {
                let part = &parts_slice[part_index];
                match part.as_str() {
                    Some(s) => {
                        result.push_str("'");
                        result.push_str(s);
                        result.push_str("'");
                    }
                    None => result.push_str(&part.to_string()),
                }
                part_index += 1;
            } else {
                println!("ERROR: Not enough parts provided for string template");
                return Value64::ERROR;
            }
        } else {
            result.push(c);
        }
    }
    Value64::from_string(rc_new(result))
}
