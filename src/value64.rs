#![allow(dead_code)]

use std::rc::Rc;

use crate::value::ArrayInstance;

#[derive(PartialEq, Debug, Clone, Copy)]
enum PointerTag {
    String,
    Lambda,
    Dict,
    Array,
}

const fn pointer_mask(tag: PointerTag) -> u64 {
    ((tag as u64) << 48) | 0xFFFC_0000_0000_0000
}

const fn extract_ptr<T>(value: u64) -> *const T {
    assert!((value & 0xFFFC000000000000) == 0xFFFC000000000000);
    (value & 0x0000FFFFFFFFFFFF) as usize as *const T
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

const TRUE_VALUE: u64 = BOOL_MASK | 3;
const FALSE_VALUE: u64 = BOOL_MASK | 2;

// #[derive(PartialEq, Debug)]
// pub enum RefValue {
//     String(String),
//     Lambda(LambdaFunction),
//     Dict(DictInstance),
//     Array(ArrayInstance),
//     // Nil,
// }

// pub enum InlineValue {
//     Nil,
//     Builtin(BuiltInFunction),
//     Bool(bool),
// }

pub struct Value64 {
    bits: u64,
}

impl Value64 {
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

    // ARRAYS

    pub fn from_array(value: Rc<ArrayInstance>) -> Self {
        let ptr = Rc::into_raw(value);
        Self::from_ptr(ptr, PointerTag::Array)
    }

    pub const fn is_array(&self) -> bool {
        (self.bits & NANISH_MASK) == ARRAY_MASK
    }

    pub fn into_array(self) -> Option<Rc<ArrayInstance>> {
        if self.is_array() {
            let ptr = extract_ptr::<ArrayInstance>(self.bits);
            unsafe { Some(Rc::from_raw(ptr)) }
        } else {
            None
        }
    }
}

const fn is_safe_addr(ptr: u64) -> bool {
    (ptr & TAG_MASK) == 0
}

#[cfg(test)]
mod tests {
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
    fn array_round_trip() {
        let arr = Rc::new(ArrayInstance::new(vec![]));
        let val = Value64::from_array(arr.clone());
        assert!(val.is_array());
        assert!(val.is_ptr());
        let arr2 = val.into_array();
        assert_eq!(arr2, Some(arr));
    }
}
