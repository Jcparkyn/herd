use std::{
    collections::HashMap, mem::ManuallyDrop, ops::Deref, panic::AssertUnwindSafe, path::PathBuf,
    ptr::null, time::SystemTime,
};

use cranelift::prelude::{types, AbiParam, Signature, Type};
use rand::Rng;
use rayon::prelude::*;
use regex::Regex;
use strum::{EnumIter, IntoEnumIterator};

use crate::{
    analysis::Analyzer,
    dict::DictInstance,
    jit::VmContext,
    lang::ProgramParser,
    rc::Rc,
    stdlib::load_stdlib_module,
    value64::{Boxable, LambdaFunction, ListInstance, Value64},
};

#[cfg(debug_assertions)]
use crate::value64::RC_TRACKER;

#[derive(Clone, Copy, Debug)]
pub struct NativeFuncDef {
    pub name: &'static str,
    pub func_ptr: *const u8,
    pub make_sig: fn(&mut Signature) -> (),
    pub needs_vm: bool,
}

impl NativeFuncDef {
    pub fn with_vm(mut self) -> Self {
        self.needs_vm = true;
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
        NativeFuncId::ValBorrowIndex => get_def!(2, val_borrow_index), // TODO
        NativeFuncId::ValReplaceIndex => get_def!(3, val_take_index),  // TODO
        NativeFuncId::ValSetIndex => get_def!(3, val_set_index),       // TODO
        NativeFuncId::ValEq => get_def!(2, val_eq),
        NativeFuncId::ValEqU8 => get_def!(2, val_eq_u8),
        NativeFuncId::ValTruthy => get_def!(1, val_truthy_u8),
        NativeFuncId::ValConcat => get_def!(2, val_concat), // TODO
        NativeFuncId::ValGetLambdaDetails => get_def!(3, get_lambda_details), // check result in JIT
        NativeFuncId::ConstructLambda => get_def!(4, construct_lambda),
        NativeFuncId::ImportModule => get_def!(2, import_module), // check result in JIT
        NativeFuncId::Print => get_def!(1, print),
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
    map.insert("not", get_def!(1, val_not));
    map.insert("shiftLeft", get_def!(2, val_shift_left));
    map.insert("range", get_def!(2, range));
    map.insert("push", get_def!(2, list_push));
    map.insert("pop", get_def!(1, list_pop));
    map.insert("len", get_def!(1, len));
    map.insert("sort", get_def!(1, list_sort));
    map.insert("removeKey", get_def!(2, dict_remove_key));
    map.insert("dictKeys", get_def!(1, dict_keys));
    map.insert("dictEntries", get_def!(1, dict_entries));
    map.insert("randomInt", get_def!(2, random_int));
    map.insert("randomFloat", get_def!(2, random_float));
    map.insert("floatPow", get_def!(2, float_pow));
    map.insert("assertTruthy", get_def!(1, assert_truthy));
    map.insert("regexFind", get_def!(2, regex_find));
    map.insert("regexReplace", get_def!(3, regex_replace));
    map.insert("parallelMap", get_def!(3, parallel_map).with_vm());
    map.insert("epochTime", get_def!(0, epoch_time));
    map.insert("parseFloat", get_def!(1, parse_float));
    map.insert("programArgs", get_def!(1, program_args).with_vm());

    return map;
}

macro_rules! generate_get_def {
    ($fname:ident $(, $param:ident)*) => {
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
            }
        }
    };
}

generate_get_def!(get_def0);
generate_get_def!(get_def1, T1);
generate_get_def!(get_def2, T1, T2);
generate_get_def!(get_def3, T1, T2, T3);
generate_get_def!(get_def4, T1, T2, T3, T4);

// Not using &Value64, so that the ABI for these functions still takes
// regular f64 values.
#[repr(transparent)]
pub struct Value64Ref(ManuallyDrop<Value64>);

impl Value64Ref {
    pub fn from_ref(val: &Value64) -> Self {
        Self(ManuallyDrop::new(unsafe { std::ptr::read(val) }))
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

fn rc_new<T: Boxable>(val: T) -> Rc<T> {
    let rc = Rc::new(val);
    #[cfg(debug_assertions)]
    RC_TRACKER.with(|tracker| {
        tracker.borrow_mut().track(&rc);
    });
    rc
}

fn rc_mutate<T: Boxable + Clone, F: FnOnce(&mut T) -> TRet, TRet>(
    rc: &mut Rc<T>,
    action: F,
) -> TRet {
    let mut_val = Rc::make_mut(rc);
    let ret = action(mut_val);
    #[cfg(debug_assertions)]
    RC_TRACKER.with(|tracker| {
        tracker.borrow_mut().track(rc);
    });
    ret
}

macro_rules! guard_f64 {
    ($val:expr) => {
        match $val.as_f64() {
            Some(f) => f,
            None => {
                println!("ERROR: Expected f64, got {}", $val);
                return Value64::ERROR;
            }
        }
    };
}

macro_rules! guard_usize {
    ($val:expr) => {
        // TODO assert int
        guard_f64!($val) as usize
    };
}

macro_rules! guard_i64 {
    ($val:expr) => {
        // TODO assert int
        guard_f64!($val) as i64
    };
}

macro_rules! guard_list {
    ($val:expr) => {
        match $val.as_list() {
            Some(l) => l,
            None => {
                println!("ERROR: Expected a list, got {}", $val);
                return Value64::ERROR;
            }
        }
    };
}

macro_rules! guard_into_list {
    ($val:expr) => {
        match $val.try_into_list() {
            Ok(l) => l,
            Err(v) => {
                println!("ERROR: Expected a list, got {}", v);
                return Value64::ERROR;
            }
        }
    };
}

macro_rules! guard_into_dict {
    ($val:expr) => {
        match $val.try_into_dict() {
            Ok(d) => d,
            Err(v) => {
                println!("ERROR: Expected a dict, got {}", v);
                return Value64::ERROR;
            }
        }
    };
}

macro_rules! guard_string {
    ($val:expr) => {
        match $val.as_string() {
            Some(l) => l,
            None => {
                println!("ERROR: Expected a string, got {}", $val);
                return Value64::ERROR;
            }
        }
    };
}

macro_rules! guard_lambda {
    ($val:expr) => {
        match $val.as_lambda() {
            Some(l) => l,
            None => {
                println!("ERROR: Expected a lambda, got {}", $val);
                return Value64::ERROR;
            }
        }
    };
}

type Out<T> = *mut T;

pub extern "C" fn list_new(len: u64, items: *const Value64) -> Value64 {
    let items_slice = unsafe { std::slice::from_raw_parts(items, len as usize) };
    let mut items = Vec::with_capacity(len as usize);
    for item in items_slice {
        // values in items array are all owned, so we don't need to clone them
        items.push(unsafe { item.cheeky_copy() });
    }
    Value64::from_list(rc_new(ListInstance::new(items)))
}

pub extern "C" fn list_push(list: Value64, val: Value64) -> Value64 {
    let mut list = guard_into_list!(list);
    rc_mutate(&mut list, |l| l.values.push(val));
    Value64::from_list(list)
}

pub extern "C" fn list_pop(list: Value64) -> Value64 {
    let mut list = guard_into_list!(list);
    rc_mutate(&mut list, |l| {
        l.values.pop();
    });
    Value64::from_list(list)
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

pub extern "C" fn list_sort(list_val: Value64) -> Value64 {
    let mut list = guard_into_list!(list_val);
    rc_mutate(&mut list, |l| {
        l.values.sort_by(|a, b| a.display_cmp(b));
    });
    Value64::from_list(list)
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

pub extern "C" fn dict_remove_key(dict: Value64, key: Value64) -> Value64 {
    let mut dict = guard_into_dict!(dict);
    rc_mutate(&mut dict, |d| {
        d.remove(&key);
    });
    Value64::from_dict(dict)
}

pub extern "C" fn dict_keys(dict: Value64) -> Value64 {
    let dict = guard_into_dict!(dict);
    let keys: Vec<Value64> = dict.keys().cloned().collect();
    Value64::from_list(rc_new(ListInstance::new(keys)))
}

pub extern "C" fn dict_entries(dict: Value64) -> Value64 {
    let dict = guard_into_dict!(dict);
    let entries: Vec<Value64> = dict
        .iter()
        .map(|(k, v)| {
            let entry = vec![k.clone(), v.clone()];
            Value64::from_list(rc_new(ListInstance::new(entry)))
        })
        .collect();
    Value64::from_list(rc_new(ListInstance::new(entries)))
}

pub extern "C" fn range(start: Value64, stop: Value64) -> Value64 {
    let start_int = guard_i64!(start);
    let stop_int = guard_i64!(stop);
    let mut values = Vec::new();
    for i in start_int..stop_int {
        values.push(Value64::from_f64(i as f64));
    }
    return Value64::from_list(rc_new(ListInstance::new(values)));
}

pub extern "C" fn len(list: Value64) -> Value64 {
    let list2 = guard_list!(list);
    Value64::from_f64(list2.values.len() as f64)
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
    val: Value64,
    index: Value64Ref,
    element_out: *mut Value64,
) -> Value64 {
    if val.is_list() {
        let index_int = guard_usize!(index);
        let mut list = val.try_into_list().unwrap();
        if index_int >= list.values.len() {
            println!("ERROR: Out of range");
            return Value64::ERROR;
        }
        let element = rc_mutate(&mut list, |l| {
            std::mem::replace(&mut l.values[index_int], Value64::from_f64(69.0))
        });
        unsafe { *element_out = element };
        Value64::from_list(list)
    } else if val.is_dict() {
        let mut dict = val.try_into_dict().unwrap();
        let element = rc_mutate(&mut dict, |d| {
            if let Some(value) = d.get_mut(&index) {
                std::mem::replace(value, Value64::from_f64(70.0))
            } else {
                Value64::NIL
            }
        });
        unsafe { *element_out = element };
        Value64::from_dict(dict)
    } else {
        println!("Expected list or dict, was {}", val);
        Value64::ERROR
    }
}

pub extern "C" fn val_borrow_index(val: Value64Ref, index: Value64Ref) -> Value64Ref {
    if let Some(list) = val.as_list() {
        match index.as_f64() {
            Some(i) => {
                if i < 0.0 || i >= list.values.len() as f64 {
                    println!("ERROR: Index out of range");
                    return Value64Ref::from_ref(&Value64::ERROR);
                }
                let index_int = i as usize;
                return Value64Ref::from_ref(&list.values[index_int]);
            }
            _ => {
                println!("ERROR: Out of range");
                return Value64Ref::from_ref(&Value64::ERROR);
            }
        }
    } else if let Some(dict) = val.as_dict() {
        Value64Ref::from_ref(dict.get(&index).unwrap_or(&Value64::NIL))
    } else {
        panic!("Expected list or dict, was {}", *val)
    }
}

pub extern "C" fn val_set_index(val: Value64, index: Value64, new_val: Value64) -> Value64 {
    if val.is_list() {
        let index_int = guard_usize!(index);
        let mut list = val.try_into_list().unwrap();
        rc_mutate(&mut list, |l| {
            l.values[index_int] = new_val;
        });
        Value64::from_list(list)
    } else if val.is_dict() {
        let mut dict = val.try_into_dict().unwrap();
        rc_mutate(&mut dict, |d| {
            d.insert(index, new_val);
        });
        Value64::from_dict(dict)
    } else {
        panic!("Expected list or dict, was {}", val)
    }
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
        let str2 = val2.as_string().unwrap();
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

pub extern "C" fn float_pow(base: Value64, exponent: Value64) -> Value64 {
    let base_float = guard_f64!(base);
    let exp_float = guard_f64!(exponent);
    Value64::from_f64(base_float.powf(exp_float))
}

pub extern "C" fn assert_truthy(val: Value64) -> Value64 {
    if !val.truthy() {
        println!(
            "ERROR: Assertion failed. Expected truthy value, was {}",
            val
        );
        return Value64::ERROR;
    }
    return Value64::NIL;
}

pub extern "C" fn get_lambda_details(
    val: Value64Ref,
    param_count: u64,
    closure_out: Out<*const Value64>,
) -> *const u8 {
    let lambda = match val.as_lambda() {
        Some(l) => l,
        None => {
            println!("Not a lambda");
            return null();
        }
    };
    if lambda.param_count != param_count as usize {
        println!("Wrong number of parameters");
        return null();
    } else {
        unsafe {
            // We can only return one value, so passing this via pointer
            *closure_out = lambda.closure.as_ptr();
        }
        return lambda.func_ptr.unwrap();
    }
}

pub extern "C" fn construct_lambda(
    param_count: u64,
    func_ptr: *const u8,
    capture_count: u64,
    captures: *const Value64,
) -> Value64 {
    let closure_slice = unsafe { std::slice::from_raw_parts(captures, capture_count as usize) };
    let lambda = LambdaFunction {
        param_count: param_count as usize,
        closure: closure_slice.to_vec(),
        self_name: Some("TEMP lambda".to_string()),
        recursive: false, // TODO
        func_ptr: Some(func_ptr),
    };
    Value64::from_lambda(rc_new(lambda))
}

pub extern "C" fn import_module(vm: &VmContext, name: Value64) -> Value64 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| import_module_panic(vm, &name)));
    match result {
        Ok(Ok(result)) => result,
        err => {
            println!("Error while importing {}: {:?}", name, err);
            Value64::ERROR
        }
    }
}

fn import_module_panic(vmc: &VmContext, name: &Value64) -> Result<Value64, String> {
    let name = name.as_string().unwrap();
    let path = name.as_str();

    let mut vm = vmc.jit.try_lock().unwrap();
    if let Some(maybe_module) = vm.modules.get(path) {
        if let Some(ref module_result) = *maybe_module {
            return Ok(module_result.clone());
        } else {
            return Err("Import cycle detected!".to_string());
        }
    }

    vm.modules.insert(path.to_string(), None);

    // Compile the module
    let is_stdlib = path.starts_with("@");
    let program = if is_stdlib {
        load_stdlib_module(path)
    } else {
        &vm.module_loader.load(&path).map_err(|e| e.to_string())?
    };
    let parser = ProgramParser::new();
    let mut program_ast = parser.parse(&program).map_err(|e| e.to_string())?;
    if !is_stdlib {
        let prelude_ast = parser.parse(include_str!("../src/prelude.herd")).unwrap();
        program_ast.splice(0..0, prelude_ast);
    }
    let mut analyzer = Analyzer::new();
    analyzer
        .analyze_statements(&mut program_ast)
        .map_err(|e| format!("{:?}", e))?;

    let main_func = vm
        .compile_program_as_function(&program_ast, &PathBuf::from(path))
        .map_err(|e| e.to_string())?;
    drop(vm);

    let result = unsafe { vmc.run_func(main_func, vec![]) };
    let mut vm = vmc.jit.try_lock().unwrap();
    vm.modules.insert(path.to_string(), Some(result.clone()));
    return Ok(result);
}

pub extern "C" fn val_shift_left(val: Value64, by: Value64) -> Value64 {
    let a = guard_i64!(val);
    let b = guard_i64!(by);
    let result = Value64::from_f64((a << b) as f64);
    result
}

pub extern "C" fn val_xor(val1: Value64, val2: Value64) -> Value64 {
    let a = guard_usize!(val1);
    let b = guard_usize!(val2);
    Value64::from_f64((a ^ b) as f64)
}

pub extern "C" fn val_not(val: Value64) -> Value64 {
    Value64::from_bool(!val.truthy())
}

pub extern "C" fn random_int(min: Value64, max: Value64) -> Value64 {
    let min_int = guard_i64!(min);
    let max_int = guard_i64!(max);
    let result = rand::thread_rng().gen_range(min_int..=max_int);
    Value64::from_f64(result as f64)
}

pub extern "C" fn random_float(min: Value64, max: Value64) -> Value64 {
    let min_float = guard_f64!(min);
    let max_float = guard_f64!(max);
    let result = rand::thread_rng().gen_range(min_float..=max_float);
    Value64::from_f64(result)
}

pub extern "C" fn regex_find(text: Value64, regex: Value64) -> Value64 {
    let regex_str = guard_string!(regex);
    let text_str = guard_string!(text);
    let regex = Regex::new(regex_str).unwrap();
    let result = regex.find(text_str);
    match result {
        Some(m) => {
            let start = Value64::from_usize(m.start());
            let end = Value64::from_usize(m.end());
            Value64::from_list(rc_new(ListInstance::new(vec![start, end])))
        }
        None => Value64::NIL,
    }
}

pub extern "C" fn regex_replace(text: Value64, regex: Value64, replacement: Value64) -> Value64 {
    let regex_str = guard_string!(regex);
    let text_str = guard_string!(text);
    let replacement_str = guard_string!(replacement);
    let regex = Regex::new(regex_str).unwrap();
    let result = regex.replace_all(text_str, replacement_str);
    Value64::from_string(rc_new(result.to_string()))
}

pub extern "C" fn parallel_map(vm: &VmContext, list: Value64, func: Value64) -> Value64 {
    let list = guard_list!(list);
    let _ = guard_lambda!(func);

    let result: Vec<_> = list
        .values
        .par_iter()
        .map(|v| vm.run_lambda(&func, &[v.clone()]))
        .collect();

    Value64::from_list(rc_new(ListInstance::new(result)))
}

pub extern "C" fn epoch_time() -> Value64 {
    // time::Instant::
    // let start = std::time::Instant::now();
    // let elapsed = start.elapsed();
    // std::time::Instant::
    let duration_since_epoch = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap();
    Value64::from_f64(duration_since_epoch.as_secs_f64())
}

pub extern "C" fn parse_float(val: Value64) -> Value64 {
    let s = guard_string!(val);
    match s.parse::<f64>() {
        Ok(f) => Value64::from_f64(f),
        Err(_) => Value64::NIL,
    }
}

pub extern "C" fn program_args(vm: &VmContext) -> Value64 {
    let args = vm.program_args.clone();
    let mut result = Vec::new();
    for arg in args {
        result.push(Value64::from_string(rc_new(arg)));
    }
    Value64::from_list(rc_new(ListInstance::new(result)))
}

// TODO: Variadic functions
pub extern "C" fn print(val: Value64) {
    match val.try_into_string() {
        Ok(s) => print!("{s}"),
        Err(v) => print!("{v}"),
    }
}
