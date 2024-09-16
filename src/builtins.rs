use std::rc::Rc;

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
    let list = list.try_into_list().unwrap();
    list.values.len() as u64
}

pub extern "C" fn list_get_u64(list: Value64, index: u64) -> Value64 {
    let list = list.try_into_list().unwrap();
    list.values[index as usize].clone()
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
