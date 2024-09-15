use std::rc::Rc;

use crate::value64::{ListInstance, Value64};

pub unsafe extern "C" fn build_list_1(val: f64) -> f64 {
    let val64 = Value64::from_f64_unsafe(val);
    let result = Value64::from_list(Rc::new(ListInstance::new(vec![val64])));
    result.into_f64_unsafe()
}
