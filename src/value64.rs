#![allow(dead_code)]

use std::{
    fmt::{Debug, Display},
    rc::Rc,
};

use crate::{
    ast::{BuiltInFunction, MatchPattern, SpannedExpr},
    pos::Spanned,
    value::{ArrayInstance, DictInstance, Value as FatValue},
};

#[derive(PartialEq, Debug, Clone, Copy)]
enum PointerTag {
    String,
    Dict,
    Array,
    Lambda,
}

const fn pointer_mask(tag: PointerTag) -> u64 {
    ((tag as u64) << 48) | 0xFFFC_0000_0000_0000
}

const fn extract_ptr<T>(value: u64) -> *const T {
    assert!((value & 0xFFFC000000000000) == 0xFFFC000000000000);
    (value & 0x0000FFFFFFFFFFFF) as usize as *const T
}

const fn try_get_ptr_tag(value: u64) -> Option<PointerTag> {
    if is_ptr(value) {
        return None;
    }
    return match (value >> 48) & 0b11 {
        0 => Some(PointerTag::String),
        1 => Some(PointerTag::Dict),
        2 => Some(PointerTag::Array),
        3 => Some(PointerTag::Lambda),
        _ => unreachable!(),
    };
}

const fn is_ptr(value: u64) -> bool {
    (value & 0xFFFC000000000000) == 0xFFFC000000000000
}

const fn is_ptr_type(value: u64, tag: PointerTag) -> bool {
    (value & NANISH_MASK) == pointer_mask(tag)
}

const fn is_safe_addr(ptr: u64) -> bool {
    (ptr & TAG_MASK) == 0
}

const QNAN: u64 = 0x7FF8000000000000;
const NANISH: u64 = 0x7FFC000000000000;
const NANISH_MASK: u64 = 0xFFFF000000000000;
// const POINTER_MASK: u64 = 0x0000FFFFFFFFFFFF;
const DICT_MASK: u64 = pointer_mask(PointerTag::Dict);
const ARRAY_MASK: u64 = pointer_mask(PointerTag::Array);
const STRING_MASK: u64 = pointer_mask(PointerTag::String);
const BOOL_MASK: u64 = 0x7FFE000000000002;
const TAG_MASK: u64 = 0xFFFF000000000000;
const REF_TAG: u64 = 0x7FFA000000000000;

const TRUE_VALUE: u64 = 0x7FFE000000000003;
const FALSE_VALUE: u64 = 0x7FFE000000000002;
const NIL_VALUE: u64 = 0x7FFC000000000000;

pub struct Value64 {
    bits: u64,
}

impl Value64 {
    const NIL: Value64 = Value64::from_bits(NIL_VALUE);

    const fn from_bits(bits: u64) -> Self {
        Value64 { bits }
    }

    pub const fn is_ptr(&self) -> bool {
        (self.bits & 0xFFFC000000000000) == 0xFFFC000000000000
    }

    fn from_ptr<T>(ptr: *const T, tag: PointerTag) -> Self {
        let addr = ptr as usize as u64;
        assert!(is_safe_addr(addr));
        Self::from_bits(addr | pointer_mask(tag))
    }

    unsafe fn into_rc<T>(self, tag: PointerTag) -> Option<Rc<T>> {
        if is_ptr_type(self.bits, tag) {
            let ptr = extract_ptr::<T>(self.bits);
            unsafe { Some(Rc::from_raw(ptr)) }
        } else {
            None
        }
    }

    // fn into_fat_value(self) -> FatValue {
    //     match try_get_ptr_tag(self.bits) {
    //         Some(PointerTag::String) => FatValue::String(self.try_into_string().unwrap()),
    //         Some(PointerTag::Dict) => FatValue::Dict(self.try_into_dict().unwrap()),
    //         Some(PointerTag::Array) => FatValue::Array(self.try_into_array().unwrap()),
    //         None => {
    //             if self.bits == TRUE_VALUE {
    //                 FatValue::Bool(true)
    //             } else if self.bits == FALSE_VALUE {
    //                 FatValue::Bool(false)
    //             } else if self.bits == NIL_VALUE {
    //                 FatValue::Nil
    //             } else {
    //                 FatValue::Number(self.as_f64().unwrap())
    //             }
    //         }
    //     }
    // }

    pub fn truthy(&self) -> bool {
        // TODO
        match try_get_ptr_tag(self.bits) {
            Some(PointerTag::String) => true,
            Some(PointerTag::Dict) => true,
            Some(PointerTag::Array) => true,
            Some(PointerTag::Lambda) => true,
            None => {
                if self.bits == TRUE_VALUE {
                    true
                } else if self.bits == FALSE_VALUE {
                    false
                } else if self.bits == NIL_VALUE {
                    false
                } else {
                    self.as_f64().unwrap() != 0.0
                }
            }
        }
    }

    // FLOATS

    pub fn from_f64(value: f64) -> Self {
        assert!(!value.is_nan() || value.to_bits() == QNAN);
        Value64::from_bits(value.to_bits())
    }

    pub const fn is_f64(&self) -> bool {
        (self.bits & NANISH) != NANISH
    }

    pub fn as_f64(&self) -> Option<f64> {
        if self.is_f64() {
            Some(f64::from_bits(self.bits))
        } else {
            None
        }
    }

    // BOOLS

    pub const fn from_bool(value: bool) -> Self {
        if value {
            Value64::from_bits(TRUE_VALUE)
        } else {
            Value64::from_bits(FALSE_VALUE)
        }
    }

    pub fn is_bool(&self) -> bool {
        self.bits == TRUE_VALUE || self.bits == FALSE_VALUE
    }

    pub fn as_bool(&self) -> Option<bool> {
        if self.bits == TRUE_VALUE {
            Some(true)
        } else if self.bits == FALSE_VALUE {
            Some(false)
        } else {
            None
        }
    }

    // STRINGS

    pub fn from_string(value: Rc<String>) -> Self {
        let ptr = Rc::into_raw(value);
        Self::from_ptr(ptr, PointerTag::String)
    }

    pub const fn is_string(&self) -> bool {
        is_ptr_type(self.bits, PointerTag::String)
    }

    pub fn try_into_string(self) -> Option<Rc<String>> {
        unsafe { self.into_rc(PointerTag::String) }
    }

    // DICTS

    pub fn from_dict(value: Rc<DictInstance>) -> Self {
        let ptr = Rc::into_raw(value);
        Self::from_ptr(ptr, PointerTag::Dict)
    }

    pub const fn is_dict(&self) -> bool {
        is_ptr_type(self.bits, PointerTag::Dict)
    }

    pub fn try_into_dict(self) -> Option<Rc<DictInstance>> {
        unsafe { self.into_rc(PointerTag::Dict) }
    }

    // ARRAYS

    pub fn from_array(value: Rc<ArrayInstance>) -> Self {
        let ptr = Rc::into_raw(value);
        Self::from_ptr(ptr, PointerTag::Array)
    }

    pub const fn is_array(&self) -> bool {
        is_ptr_type(self.bits, PointerTag::Array)
    }

    pub fn try_into_array(self) -> Option<Rc<ArrayInstance>> {
        unsafe { self.into_rc(PointerTag::Array) }
    }

    // LAMBDAS

    pub fn from_lambda(value: Rc<LambdaFunction>) -> Self {
        let ptr = Rc::into_raw(value);
        Self::from_ptr(ptr, PointerTag::Lambda)
    }
}

impl PartialEq for Value64 {
    fn eq(&self, other: &Self) -> bool {
        // TODO value equality, etc
        self.bits == other.bits
    }
}

impl Debug for Value64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Value64").field("bits", &self.bits).finish()
    }
}

#[derive(Debug, Clone)]
pub enum Callable {
    Lambda(Rc<LambdaFunction>),
    Builtin(BuiltInFunction),
}

#[derive(PartialEq, Debug)]
pub struct LambdaFunction {
    pub params: Rc<Vec<Spanned<MatchPattern>>>,
    pub body: Rc<SpannedExpr>,
    pub closure: Vec<Value64>,
    pub self_name: Option<String>,
    pub recursive: bool,
}

impl Display for LambdaFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(self_name) = &self.self_name {
            write!(f, "<lambda: {}>", self_name)
        } else {
            write!(f, "<lambda>")
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn f64_round_trip() {
        let inputs = [0.0, -1.0, 1.0, f64::INFINITY, f64::NEG_INFINITY];
        for input in inputs {
            let val = Value64::from_f64(input);
            assert_eq!(val.as_f64(), Some(input));
            assert!(!val.is_ptr());
        }
    }

    #[test]
    fn nan_round_trip() {
        let val = Value64::from_f64(f64::NAN);
        assert!(val.as_f64().unwrap().is_nan());
        assert!(!val.is_ptr());
    }

    #[test]
    fn bool_round_trip() {
        let t = Value64::from_bool(true);
        let f = Value64::from_bool(false);
        assert_eq!(t.as_bool(), Some(true));
        assert_eq!(f.as_bool(), Some(false));
        assert!(!t.is_ptr());
        assert!(!f.is_ptr());
    }

    #[test]
    fn dict_round_trip() {
        let mut dict_values = HashMap::new();
        dict_values.insert(
            FatValue::String(Rc::new("a".to_string())),
            FatValue::Number(1.0),
        );
        let dict = Rc::new(DictInstance {
            values: dict_values,
        });
        let val = Value64::from_dict(dict.clone());
        assert!(val.is_dict());
        assert!(val.is_ptr());
        let dict2 = val.try_into_dict();
        assert_eq!(dict2, Some(dict));
    }

    #[test]
    fn array_round_trip() {
        let arr = Rc::new(ArrayInstance::new(vec![
            FatValue::Nil,
            FatValue::Bool(true),
            FatValue::Number(1.0),
        ]));
        let val = Value64::from_array(arr.clone());
        assert!(val.is_array());
        assert!(val.is_ptr());
        let arr2 = val.try_into_array();
        assert_eq!(arr2, Some(arr));
    }

    #[test]
    fn string_round_trip() {
        let s = Rc::new("test".to_string());
        let val = Value64::from_string(s.clone());
        assert!(val.is_string());
        assert!(val.is_ptr());
        let s2 = val.try_into_string();
        assert_eq!(s2, Some(s));
    }
}
