use std::{collections::HashMap, mem::ManuallyDrop, ops::Deref, path::PathBuf, ptr::null, rc::Rc};

use crate::{
    analysis::Analyzer,
    ast,
    jit::VmContext,
    lang::ProgramParser,
    pos::{Span, Spanned},
    value64::{Boxable, DictInstance, LambdaFunction, ListInstance, Value64},
};

#[cfg(debug_assertions)]
use crate::value64::RC_TRACKER;

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
    #[cfg(debug_assertions)]
    let will_clone = Rc::strong_count(rc) > 1;
    let mut_val = Rc::make_mut(rc);
    action(mut_val);
    #[cfg(debug_assertions)]
    if will_clone {
        RC_TRACKER.with(|tracker| {
            tracker.borrow_mut().track(rc);
        });
    }
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

pub extern "C" fn public_list_push(list: Value64, val: Value64) -> Value64 {
    let mut list = list.try_into_list().unwrap();
    rc_mutate(&mut list, |l| l.values.push(val));
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

pub extern "C" fn public_range(start: Value64, stop: Value64) -> Value64 {
    let start_int = start.as_f64().unwrap() as i64;
    let stop_int = stop.as_f64().unwrap() as i64;
    let mut values = Vec::new();
    for i in start_int..stop_int {
        values.push(Value64::from_f64(i as f64));
    }
    return Value64::from_list(rc_new(ListInstance::new(values)));
}

pub extern "C" fn public_len(list: Value64) -> Value64 {
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

pub extern "C" fn val_truthy(val: Value64Ref) -> i8 {
    val.truthy() as i8
}

pub extern "C" fn val_get_lambda_details(
    val: Value64Ref,
    param_count: u32,
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
    param_count: usize,
    func_ptr: *const u8,
    capture_count: usize,
    captures: *const Value64,
) -> Value64 {
    let closure_slice = unsafe { std::slice::from_raw_parts(captures, capture_count) };
    let lambda = LambdaFunction {
        params: Rc::new(vec![]), // Not used in JIT
        param_count,
        body: Rc::new(Spanned::new(Span::new(0, 0), ast::Expr::Nil)), // Not used in JIT
        closure: closure_slice.to_vec(),
        self_name: Some("TEMP lambda".to_string()),
        recursive: false, // TODO
        func_ptr: Some(func_ptr),
    };
    Value64::from_lambda(rc_new(lambda))
}

pub extern "C" fn import_module(vm: &mut VmContext, name: Value64) -> Value64 {
    let name = name.try_into_string().unwrap();
    let path = PathBuf::from(name.as_str());

    if let Some(maybe_module) = vm.modules.get(&path) {
        if let Some(module_result) = maybe_module {
            return module_result.clone();
        } else {
            panic!("Import cycle detected!");
        }
    }
    vm.modules.insert(path.clone(), None);
    // Compile the module
    let program = vm.module_loader.load(&path).unwrap();
    let parser = ProgramParser::new();
    let prelude_ast = parser.parse(include_str!("../src/prelude.bovine")).unwrap();
    let mut program_ast = parser.parse(&program).unwrap();
    program_ast.splice(0..0, prelude_ast);
    let mut analyzer = Analyzer::new();
    analyzer.analyze_statements(&mut program_ast).unwrap();

    let main_func = vm.compile_program_as_function(&program_ast, &path).unwrap();

    let result = unsafe { vm.run_func(main_func, Value64::NIL) };
    vm.modules.insert(path, Some(result.clone()));
    return result;
}

pub extern "C" fn public_val_shift_left(val: Value64, by: Value64) -> Value64 {
    let a = val.as_f64().unwrap() as u64;
    let b = by.as_f64().unwrap() as u8;
    let result = Value64::from_f64((a << b) as f64);
    result
}

pub extern "C" fn public_val_xor(val1: Value64, val2: Value64) -> Value64 {
    let a = val1.as_f64().unwrap() as u64;
    let b = val2.as_f64().unwrap() as u64;
    Value64::from_f64((a ^ b) as f64)
}

pub extern "C" fn public_val_not(val: Value64) -> Value64 {
    Value64::from_bool(!val.truthy())
}

// TODO: Variadic functions
pub extern "C" fn public_print(val: Value64) {
    match val.try_into_string() {
        Ok(s) => print!("{s}"),
        Err(v) => print!("{v}"),
    }
}
