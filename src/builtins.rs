use std::rc::Rc;

use crate::value64::{ListInstance, Value64};

pub unsafe extern "C" fn build_list_1(val: Value64) -> Value64 {
    Value64::from_list(Rc::new(ListInstance::new(vec![val])))
}
