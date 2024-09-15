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
