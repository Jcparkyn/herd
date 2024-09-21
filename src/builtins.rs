use std::{mem::forget, rc::Rc};

use crate::value64::{ListInstance, Value64};

pub extern "C" fn list_new() -> Value64 {
    Value64::from_list(Rc::new(ListInstance::new(vec![])))
}

pub extern "C" fn list_push(list: Value64, val: Value64) -> Value64 {
    let mut list = list.try_into_list().unwrap();
    let mut_list = Rc::make_mut(&mut list);
    mut_list.values.push(val);
    Value64::from_list(list)
}

pub extern "C" fn list_len_u64(list: Value64) -> u64 {
    let list2 = list.as_list().unwrap();
    let result = list2.values.len() as u64;
    forget(list);
    result
}

pub extern "C" fn list_get_u64(list: Value64, index: u64) -> Value64 {
    let list2 = list.as_list().unwrap();
    let result = list2.values[index as usize].clone();
    forget(list);
    result
}

pub extern "C" fn range(start: Value64, stop: Value64) -> Value64 {
    let start_int = start.as_f64().unwrap() as i64;
    let stop_int = stop.as_f64().unwrap() as i64;
    let mut values = Vec::new();
    for i in start_int..stop_int {
        values.push(Value64::from_f64(i as f64));
    }
    return Value64::from_list(Rc::new(ListInstance::new(values)));
}

pub extern "C" fn len(list: Value64) -> Value64 {
    let list2 = list.as_list().unwrap();
    let result = Value64::from_f64(list2.values.len() as f64);
    forget(list);
    result
}

pub extern "C" fn clone(val: Value64) -> Value64 {
    forget(val.clone());
    val
}

pub extern "C" fn val_get_index(val: Value64, index: Value64) -> Value64 {
    let list = match val.as_list() {
        Some(l) => l,
        None => panic!("Expected list, was {}", val),
    };
    let result = list.values[index.as_f64().unwrap() as usize].clone();
    forget(val);
    result
}

pub extern "C" fn val_eq(val1: Value64, val2: Value64) -> Value64 {
    let result = Value64::from_bool(val1 == val2);
    forget(val1);
    forget(val2);
    result
}

pub extern "C" fn val_truthy(val: Value64) -> i8 {
    let result = if val.truthy() { 1 } else { 0 };
    forget(val);
    result
}

pub extern "C" fn val_shift_left(val: Value64, by: Value64) -> Value64 {
    let a = val.as_f64().unwrap() as u64;
    let b = by.as_f64().unwrap() as u8;
    let result = Value64::from_f64((a << b) as f64);
    forget(val);
    result
}

pub extern "C" fn val_xor(val1: Value64, val2: Value64) -> Value64 {
    let a = val1.as_f64().unwrap() as u64;
    let b = val2.as_f64().unwrap() as u64;
    let result = Value64::from_f64((a ^ b) as f64);
    forget(val1);
    forget(val2);
    result
}

pub extern "C" fn val_not(val: Value64) -> Value64 {
    let result = Value64::from_bool(!val.truthy());
    forget(val);
    result
}

// TODO: Variadic functions
pub extern "C" fn print(val: Value64) {
    println!("{}", val);
}
