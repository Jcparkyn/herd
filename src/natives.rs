use std::{
    collections::HashMap, mem::ManuallyDrop, ops::Deref, panic::AssertUnwindSafe, path::PathBuf,
    ptr::null, rc::Rc,
};

use cranelift::prelude::{types, AbiParam, Signature, Type};
use strum::{EnumIter, IntoEnumIterator};

use crate::{
    analysis::Analyzer,
    jit::VmContext,
    lang::ProgramParser,
    stdlib::load_stdlib_module,
    value64::{Boxable, DictInstance, LambdaFunction, ListInstance, Value64},
};

#[cfg(debug_assertions)]
use crate::value64::RC_TRACKER;

#[derive(Clone, Copy, Debug)]
pub struct NativeFuncDef {
    pub name: &'static str,
    pub func_ptr: *const u8,
    pub make_sig: fn(&mut Signature) -> (),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, EnumIter)]
pub enum NativeFuncId {
    ListNew,
    ListLenU64,
    ListGetU64,
    ListBorrowU64,
    DictNew,
    DictInsert,
    Clone,
    Drop,
    ValGetIndex,
    ValSetIndex,
    ValEq,
    ValEqU8,
    ValTruthy,
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

impl AbiType for f64 {
    const TYPE: Type = types::F64;
}

impl<T> AbiType for *const T {
    const TYPE: Type = types::I64;
}

impl<T> AbiType for *mut T {
    const TYPE: Type = types::I64;
}

impl AbiType for &mut VmContext {
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
        NativeFuncId::ListLenU64 => get_def!(1, list_len_u64),
        NativeFuncId::ListGetU64 => get_def!(2, list_get_u64),
        NativeFuncId::ListBorrowU64 => get_def!(2, list_borrow_u64),
        NativeFuncId::DictNew => get_def!(1, dict_new),
        NativeFuncId::DictInsert => get_def!(3, dict_insert),
        NativeFuncId::Clone => get_def!(1, clone),
        NativeFuncId::Drop => get_def!(1, drop),
        NativeFuncId::ValGetIndex => get_def!(2, val_get_index),
        NativeFuncId::ValSetIndex => get_def!(3, val_set_index),
        NativeFuncId::ValEq => get_def!(2, val_eq),
        NativeFuncId::ValEqU8 => get_def!(2, val_eq_u8),
        NativeFuncId::ValTruthy => get_def!(1, val_truthy_u8),
        NativeFuncId::ValGetLambdaDetails => get_def!(3, get_lambda_details),
        NativeFuncId::ConstructLambda => get_def!(4, construct_lambda),
        NativeFuncId::ImportModule => get_def!(2, import_module),
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
    // map.insert("sort", get_def!(1, sort));
    map.insert("removeKey", get_def!(2, dict_remove_key));

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
            }
        }
    };
}

// generate_get_def!(get_def0);
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

fn rc_new<T: Boxable>(val: T) -> Rc<T> {
    let rc = Rc::new(val);
    #[cfg(debug_assertions)]
    RC_TRACKER.with(|tracker| {
        tracker.borrow_mut().track(&rc);
    });
    rc
}

fn rc_mutate<T: Boxable + Clone, F: FnOnce(&mut T)>(rc: &mut Rc<T>, action: F) {
    let mut_val = Rc::make_mut(rc);
    action(mut_val);
    #[cfg(debug_assertions)]
    RC_TRACKER.with(|tracker| {
        tracker.borrow_mut().track(rc);
    });
}

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
    let mut list = list.try_into_list().unwrap();
    rc_mutate(&mut list, |l| l.values.push(val));
    Value64::from_list(list)
}

pub extern "C" fn list_pop(list: Value64) -> Value64 {
    let mut list = list.try_into_list().unwrap();
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

pub extern "C" fn dict_new(capacity: u64) -> Value64 {
    Value64::from_dict(rc_new(DictInstance {
        values: HashMap::with_capacity(capacity as usize),
    }))
}

pub extern "C" fn dict_insert(dict: Value64, key: Value64, val: Value64) -> Value64 {
    let mut dict = dict.try_into_dict().unwrap();
    rc_mutate(&mut dict, |d| {
        d.values.insert(key, val);
    });
    Value64::from_dict(dict)
}

pub extern "C" fn dict_remove_key(dict: Value64, key: Value64) -> Value64 {
    let mut dict = dict.try_into_dict().unwrap();
    rc_mutate(&mut dict, |d| {
        d.values.remove(&key);
    });
    Value64::from_dict(dict)
}

pub extern "C" fn range(start: Value64, stop: Value64) -> Value64 {
    let start_int = start.as_f64().unwrap() as i64;
    let stop_int = stop.as_f64().unwrap() as i64;
    let mut values = Vec::new();
    for i in start_int..stop_int {
        values.push(Value64::from_f64(i as f64));
    }
    return Value64::from_list(rc_new(ListInstance::new(values)));
}

pub extern "C" fn len(list: Value64) -> Value64 {
    let list2 = list.as_list().unwrap();
    Value64::from_f64(list2.values.len() as f64)
}

pub extern "C" fn clone(val: Value64Ref) -> Value64 {
    val.clone()
}

pub extern "C" fn drop(val: Value64) {
    std::mem::drop(val);
}

pub extern "C" fn val_get_index(val: Value64Ref, index: Value64Ref) -> Value64 {
    if let Some(list) = val.as_list() {
        list.values
            .get(index.as_f64().unwrap() as usize)
            .unwrap() // TODO
            .clone()
    } else if let Some(dict) = val.as_dict() {
        dict.values.get(&index).cloned().unwrap_or(Value64::NIL)
    } else {
        panic!("Expected list or dict, was {}", *val)
    }
}

pub extern "C" fn val_set_index(val: Value64, index: Value64, new_val: Value64) -> Value64 {
    if val.is_list() {
        let mut list = val.try_into_list().unwrap();
        rc_mutate(&mut list, |l| {
            l.values[index.as_f64().unwrap() as usize] = new_val;
        });
        Value64::from_list(list)
    } else if val.is_dict() {
        let mut dict = val.try_into_dict().unwrap();
        rc_mutate(&mut dict, |d| {
            d.values.insert(index, new_val);
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

pub extern "C" fn get_lambda_details(
    val: Value64Ref,
    param_count: u64,
    closure_out: *mut *const Value64,
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

pub extern "C" fn import_module(vm: &mut VmContext, name: Value64) -> Value64 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| import_module_panic(vm, &name)));
    match result {
        Ok(Ok(result)) => result,
        err => {
            println!("Error while importing {}: {:?}", name, err);
            Value64::ERROR
        }
    }
}

fn import_module_panic(vm: &mut VmContext, name: &Value64) -> Result<Value64, String> {
    let name = name.as_string().unwrap();
    let path = name.as_str();

    if let Some(maybe_module) = vm.modules.get(path) {
        if let Some(module_result) = maybe_module {
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
        let prelude_ast = parser.parse(include_str!("../src/prelude.bovine")).unwrap();
        program_ast.splice(0..0, prelude_ast);
    }
    let mut analyzer = Analyzer::new();
    analyzer
        .analyze_statements(&mut program_ast)
        .map_err(|e| format!("{:?}", e))?;

    let main_func = vm
        .compile_program_as_function(&program_ast, &PathBuf::from(path))
        .map_err(|e| e.to_string())?;

    let result = unsafe { vm.run_func(main_func, Value64::NIL) };
    vm.modules.insert(path.to_string(), Some(result.clone()));
    return Ok(result);
}

pub extern "C" fn val_shift_left(val: Value64, by: Value64) -> Value64 {
    let a = val.as_f64().unwrap() as u64;
    let b = by.as_f64().unwrap() as u8;
    let result = Value64::from_f64((a << b) as f64);
    result
}

pub extern "C" fn val_xor(val1: Value64, val2: Value64) -> Value64 {
    let a = val1.as_f64().unwrap() as u64;
    let b = val2.as_f64().unwrap() as u64;
    Value64::from_f64((a ^ b) as f64)
}

pub extern "C" fn val_not(val: Value64) -> Value64 {
    Value64::from_bool(!val.truthy())
}

// TODO: Variadic functions
pub extern "C" fn print(val: Value64) {
    match val.try_into_string() {
        Ok(s) => print!("{s}"),
        Err(v) => print!("{v}"),
    }
}
