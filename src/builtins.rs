use std::{collections::HashMap, mem::ManuallyDrop, ops::Deref, rc::Rc};

use crate::value64::{DictInstance, ListInstance, Value64};

// Not using &Value64, so that the ABI for these functions still takes
// regular f64 values.
#[repr(transparent)]
pub struct Value64Ref(ManuallyDrop<Value64>);

impl Deref for Value64Ref {
    type Target = Value64;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

pub extern "C" fn list_new(capacity: u64) -> Value64 {
    Value64::from_list(Rc::new(ListInstance::new(Vec::with_capacity(
        capacity as usize,
    ))))
}

pub extern "C" fn list_push(list: Value64, val: Value64) -> Value64 {
    let mut list = list.try_into_list().unwrap();
    let mut_list = Rc::make_mut(&mut list);
    mut_list.values.push(val);
    Value64::from_list(list)
}

pub extern "C" fn list_len_u64(list: Value64Ref) -> u64 {
    list.as_list().unwrap().values.len() as u64
}

pub extern "C" fn list_get_u64(list: Value64Ref, index: u64) -> Value64 {
    let list2 = list.as_list().unwrap();
    list2.values[index as usize].clone()
}

pub extern "C" fn dict_new(capacity: u64) -> Value64 {
    Value64::from_dict(Rc::new(DictInstance {
        values: HashMap::with_capacity(capacity as usize),
    }))
}

pub extern "C" fn dict_insert(dict: Value64, key: Value64, val: Value64) -> Value64 {
    let mut dict = dict.try_into_dict().unwrap();
    let mut_dict = Rc::make_mut(&mut dict);
    mut_dict.values.insert(key, val);
    Value64::from_dict(dict)
}

pub extern "C" fn public_range(start: Value64, stop: Value64) -> Value64 {
    let start_int = start.as_f64().unwrap() as i64;
    let stop_int = stop.as_f64().unwrap() as i64;
    let mut values = Vec::new();
    for i in start_int..stop_int {
        values.push(Value64::from_f64(i as f64));
    }
    return Value64::from_list(Rc::new(ListInstance::new(values)));
}

pub extern "C" fn public_len(list: Value64) -> Value64 {
    let list2 = list.as_list().unwrap();
    Value64::from_f64(list2.values.len() as f64)
}

pub extern "C" fn clone(val: Value64Ref) -> Value64 {
    val.clone()
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

pub extern "C" fn val_eq(val1: Value64Ref, val2: Value64Ref) -> Value64 {
    Value64::from_bool(*val1 == *val2)
}

pub extern "C" fn val_truthy(val: Value64Ref) -> i8 {
    val.truthy() as i8
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
