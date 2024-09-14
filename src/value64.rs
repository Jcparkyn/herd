use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    rc::Rc,
};

use crate::{
    ast::{BuiltInFunction, MatchPattern, SpannedExpr},
    pos::Spanned,
};

/*
Pointer layout:
    - Sign bit: 1
    - 11 bit exponent: all 1s
    - 2 bits: 11
    - 2 bits: tag
    - Last 48 bits: pointer
Inline layout:
    - Sign bit: 0
    - 11 bit exponent: all 1s
    - 2 bits: 11
    - 2 bits: tag (00=builtin, 10=bool/nil)
    - Last 48 bits: value
*/

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum PointerTag {
    String,
    Dict,
    Array,
    Lambda,
}

const fn pointer_mask(tag: PointerTag) -> u64 {
    ((tag as u64) << 48) | 0xFFFC_0000_0000_0000
}

const fn extract_ptr<T: Boxable>(value: u64) -> *const T {
    assert!((value & 0xFFFC000000000000) == 0xFFFC000000000000);
    (value & 0x0000FFFFFFFFFFFF) as usize as *const T
}

const fn try_get_ptr_tag(value: u64) -> Option<PointerTag> {
    if !is_ptr(value) {
        return None;
    }
    // probably a hot path, should optimize
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
const TAG_MASK: u64 = 0xFFFF000000000000;

const TRUE_VALUE: u64 = 0x7FFE000000000003;
const FALSE_VALUE: u64 = 0x7FFE000000000002;
const NIL_VALUE: u64 = 0x7FFE000000000000;
const BUILTIN_MASK: u64 = 0x7FFC000000000000;

pub trait Boxable {
    const TAG: PointerTag;
}

impl Boxable for String {
    const TAG: PointerTag = PointerTag::String;
}

impl Boxable for ArrayInstance {
    const TAG: PointerTag = PointerTag::Array;
}

impl Boxable for DictInstance {
    const TAG: PointerTag = PointerTag::Dict;
}

impl Boxable for LambdaFunction {
    const TAG: PointerTag = PointerTag::Lambda;
}

pub struct Value64 {
    bits: u64,
}

impl Value64 {
    pub const NIL: Value64 = Value64::from_bits(NIL_VALUE);

    const fn from_bits(bits: u64) -> Self {
        Value64 { bits }
    }

    #[allow(dead_code)]
    pub const fn is_ptr(&self) -> bool {
        (self.bits & 0xFFFC000000000000) == 0xFFFC000000000000
    }

    fn from_ptr<T>(ptr: *const T, tag: PointerTag) -> Self {
        let addr = ptr as usize as u64;
        assert!(is_safe_addr(addr));
        Self::from_bits(addr | pointer_mask(tag))
    }

    fn from_rc<T: Boxable>(rc: Rc<T>) -> Self {
        let ptr = Rc::into_raw(rc);
        Self::from_ptr(ptr, T::TAG)
    }

    fn as_ref<'a, T: Boxable>(&self) -> Option<&'a T> {
        if is_ptr_type(self.bits, T::TAG) {
            let ptr = extract_ptr::<T>(self.bits);
            unsafe { Some(&*ptr) }
        } else {
            None
        }
    }

    pub fn is_nil(&self) -> bool {
        self.bits == NIL_VALUE
    }

    fn into_rc<T: Boxable>(self) -> Result<Rc<T>, Self> {
        if is_ptr_type(self.bits, T::TAG) {
            let ptr = extract_ptr::<T>(self.bits);
            std::mem::forget(self); // Don't run destructor for self, we've "moved" it
            unsafe { Ok(Rc::from_raw(ptr)) }
        } else {
            Err(self)
        }
    }

    pub fn truthy(&self) -> bool {
        match try_get_ptr_tag(self.bits) {
            Some(PointerTag::String) => !self.as_string().unwrap().is_empty(),
            Some(PointerTag::Dict) => true,
            Some(PointerTag::Array) => !self.as_array().unwrap().values.is_empty(),
            Some(PointerTag::Lambda) => true,
            None => {
                if let Some(float) = self.as_f64() {
                    !float.is_nan() && float != 0.0
                } else if self.bits == TRUE_VALUE {
                    true
                } else if self.bits == FALSE_VALUE {
                    false
                } else if self.bits == NIL_VALUE {
                    false
                } else {
                    true
                }
            }
        }
    }

    pub fn is_valid_dict_key(&self) -> bool {
        if self.is_dict() || self.is_lambda() {
            return false;
        }
        if let Some(x) = self.as_f64() {
            return !x.is_nan();
        }
        if let Some(a) = self.as_array() {
            return a.values.iter().all(Self::is_valid_dict_key);
        }
        return true;
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

    #[allow(dead_code)]
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
        Self::from_rc(value)
    }

    pub const fn is_string(&self) -> bool {
        is_ptr_type(self.bits, PointerTag::String)
    }

    pub fn try_into_string(self) -> Result<Rc<String>, Self> {
        self.into_rc()
    }

    pub fn as_string(&self) -> Option<&String> {
        self.as_ref()
    }

    // DICTS

    pub fn from_dict(value: Rc<DictInstance>) -> Self {
        Self::from_rc(value)
    }

    pub const fn is_dict(&self) -> bool {
        is_ptr_type(self.bits, PointerTag::Dict)
    }

    pub fn try_into_dict(self) -> Result<Rc<DictInstance>, Self> {
        self.into_rc()
    }

    pub fn as_dict(&self) -> Option<&DictInstance> {
        self.as_ref()
    }

    // ARRAYS

    pub fn from_array(value: Rc<ArrayInstance>) -> Self {
        Self::from_rc(value)
    }

    pub const fn is_array(&self) -> bool {
        is_ptr_type(self.bits, PointerTag::Array)
    }

    pub fn try_into_array(self) -> Result<Rc<ArrayInstance>, Self> {
        self.into_rc()
    }

    pub fn as_array(&self) -> Option<&ArrayInstance> {
        self.as_ref()
    }

    // LAMBDAS

    pub fn from_lambda(value: Rc<LambdaFunction>) -> Self {
        Self::from_rc(value)
    }

    pub const fn is_lambda(&self) -> bool {
        is_ptr_type(self.bits, PointerTag::Lambda)
    }

    pub fn try_into_lambda(self) -> Result<Rc<LambdaFunction>, Self> {
        self.into_rc()
    }

    pub fn as_lambda(&self) -> Option<&LambdaFunction> {
        self.as_ref()
    }

    // BUILTINS:

    pub fn from_builtin(value: BuiltInFunction) -> Self {
        Value64::from_bits((value as u64) | BUILTIN_MASK)
    }

    pub fn is_builtin(&self) -> bool {
        (self.bits & NANISH_MASK) == BUILTIN_MASK
    }

    pub fn as_builtin(&self) -> Option<BuiltInFunction> {
        if self.is_builtin() {
            Some(BuiltInFunction::from_repr(self.bits as u8).unwrap())
        } else {
            None
        }
    }

    pub fn try_into_callable(self) -> Result<Callable, Self> {
        if let Some(b) = self.as_builtin() {
            Ok(Callable::Builtin(b))
        } else {
            self.try_into_lambda().map(Callable::Lambda)
        }
    }
}

impl PartialEq for Value64 {
    fn eq(&self, other: &Self) -> bool {
        match try_get_ptr_tag(self.bits) {
            Some(PointerTag::String) => match other.as_string() {
                Some(b) => self.as_string().unwrap() == b,
                None => false,
            },
            Some(PointerTag::Array) => match other.as_array() {
                Some(b) => self.as_array().unwrap() == b,
                None => false,
            },
            Some(PointerTag::Dict) => match other.as_dict() {
                Some(b) => self.as_dict().unwrap() == b,
                None => false,
            },
            Some(PointerTag::Lambda) => match other.as_lambda() {
                Some(b) => self.as_lambda().unwrap() == b,
                None => false,
            },
            None => {
                if let (Some(a), Some(b)) = (self.as_f64(), other.as_f64()) {
                    a == b
                } else {
                    self.bits == other.bits
                }
            }
        }
    }
}

impl Eq for Value64 {}

impl Debug for Value64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Value64").field("bits", &self.bits).finish()
    }
}

impl std::hash::Hash for Value64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match try_get_ptr_tag(self.bits) {
            Some(PointerTag::String) => self.as_string().unwrap().hash(state),
            Some(PointerTag::Array) => self.as_array().unwrap().hash(state),
            Some(PointerTag::Dict) => panic!("Dicts cannot be used as keys inside dicts"),
            Some(PointerTag::Lambda) => {
                panic!("Lambda functions cannot be used as keys inside dicts")
            }
            None => self.bits.hash(state),
        }
    }
}

impl Clone for Value64 {
    fn clone(&self) -> Self {
        match try_get_ptr_tag(self.bits) {
            Some(PointerTag::String) => {
                let ptr = extract_ptr::<String>(self.bits);
                unsafe { Rc::increment_strong_count(ptr) };
            }
            Some(PointerTag::Array) => {
                let ptr = extract_ptr::<ArrayInstance>(self.bits);
                unsafe { Rc::increment_strong_count(ptr) };
            }
            Some(PointerTag::Dict) => {
                let ptr = extract_ptr::<DictInstance>(self.bits);
                unsafe { Rc::increment_strong_count(ptr) };
            }
            Some(PointerTag::Lambda) => {
                let ptr = extract_ptr::<LambdaFunction>(self.bits);
                unsafe { Rc::increment_strong_count(ptr) };
            }
            None => {}
        }
        Value64::from_bits(self.bits)
    }
}

impl Drop for Value64 {
    fn drop(&mut self) {
        match try_get_ptr_tag(self.bits) {
            Some(PointerTag::String) => {
                let ptr = extract_ptr::<String>(self.bits);
                unsafe { Rc::decrement_strong_count(ptr) };
            }
            Some(PointerTag::Array) => {
                let ptr = extract_ptr::<ArrayInstance>(self.bits);
                unsafe { Rc::decrement_strong_count(ptr) };
            }
            Some(PointerTag::Dict) => {
                let ptr = extract_ptr::<DictInstance>(self.bits);
                unsafe { Rc::decrement_strong_count(ptr) };
            }
            Some(PointerTag::Lambda) => {
                let ptr = extract_ptr::<LambdaFunction>(self.bits);
                unsafe { Rc::decrement_strong_count(ptr) };
            }
            None => {}
        }
    }
}

// impl Display for Value {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match self {
//             Value::Number(n) => write!(f, "{}", n),
//             Value::Bool(b) => write!(f, "{}", b),
//             Value::String(s) => write!(f, "'{}'", s),
//             Value::Builtin(b) => write!(f, "{}", b.to_string()),
//             Value::Lambda(l) => write!(f, "{}", l),
//             Value::Dict(d) => write!(f, "{}", d),
//             Value::Array(a) => write!(f, "{}", a),
//             Value::Nil => write!(f, "nil"),
//         }
//     }
// }

impl Display for Value64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match try_get_ptr_tag(self.bits) {
            Some(PointerTag::String) => write!(f, "'{}'", self.as_string().unwrap()),
            Some(PointerTag::Array) => write!(f, "{}", self.as_array().unwrap()),
            Some(PointerTag::Dict) => write!(f, "{}", self.as_dict().unwrap()),
            Some(PointerTag::Lambda) => write!(f, "{}", self.as_lambda().unwrap()),
            None => {
                if let Some(float) = self.as_f64() {
                    write!(f, "{}", float)
                } else if self.bits == TRUE_VALUE {
                    write!(f, "true")
                } else if self.bits == FALSE_VALUE {
                    write!(f, "false")
                } else if self.bits == NIL_VALUE {
                    write!(f, "nil")
                } else if let Some(b) = self.as_builtin() {
                    write!(f, "{}", b)
                } else {
                    write!(f, "<unknown value: {}>", self.bits)
                }
            }
        }
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

#[derive(PartialEq, Debug)]
pub struct DictInstance {
    pub values: HashMap<Value64, Value64>,
}

impl Clone for DictInstance {
    fn clone(&self) -> Self {
        #[cfg(debug_assertions)]
        println!("Cloning dict: {}", self);
        DictInstance {
            values: self.values.clone(),
        }
    }
}

impl Display for DictInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.values.is_empty() {
            return write!(f, "[:]");
        }
        let mut values = vec![];
        for (key, value) in &self.values {
            if let Some(s) = key.as_string() {
                values.push(format!("{}: {}", s, value));
            } else {
                values.push(format!("[{}]: {}", key, value));
            }
        }
        write!(f, "[{}]", values.join(", "))
    }
}

#[derive(PartialEq, Debug, Hash)]
pub struct ArrayInstance {
    pub values: Vec<Value64>,
}

impl ArrayInstance {
    pub fn new(values: Vec<Value64>) -> Self {
        ArrayInstance { values }
    }
}

impl Clone for ArrayInstance {
    fn clone(&self) -> Self {
        #[cfg(debug_assertions)]
        println!("Cloning array: {}", self);
        ArrayInstance {
            values: self.values.clone(),
        }
    }
}

impl Display for ArrayInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let values: Vec<_> = self.values.iter().map(|v| v.to_string()).collect();
        write!(f, "[{}]", values.join(", "))
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
    fn f64_eq() {
        assert_eq!(Value64::from_f64(1.0), Value64::from_f64(1.0));
        assert_ne!(Value64::from_f64(1.0), Value64::from_f64(2.0));
        assert_ne!(Value64::from_f64(1.0), Value64::from_f64(f64::NAN));
        assert_ne!(Value64::from_f64(f64::NAN), Value64::from_f64(f64::NAN));
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
        for val in [t, f] {
            assert!(val.is_bool());
            assert!(!val.is_ptr());
            assert!(!val.is_builtin());
            assert!(!val.is_nil());
        }
    }

    #[test]
    fn bool_eq() {
        assert_eq!(Value64::from_bool(true), Value64::from_bool(true));
        assert_eq!(Value64::from_bool(false), Value64::from_bool(false));
        assert_ne!(Value64::from_bool(false), Value64::NIL);
    }

    #[test]
    fn dict_round_trip() {
        let mut dict_values = HashMap::new();
        dict_values.insert(
            Value64::from_string(Rc::new("a".to_string())),
            Value64::from_f64(1.0),
        );
        let dict = Rc::new(DictInstance {
            values: dict_values,
        });
        let val = Value64::from_dict(dict.clone());
        assert!(val.is_dict());
        assert!(val.is_ptr());
        let dict2 = val.try_into_dict();
        assert_eq!(dict2, Ok(dict));
    }

    #[test]
    fn dict_eq() {
        let mut dict_values = HashMap::new();
        dict_values.insert(
            Value64::from_string(Rc::new("a".to_string())),
            Value64::from_f64(1.0),
        );
        let dict1 = Rc::new(DictInstance {
            values: dict_values.clone(),
        });
        let dict2 = Rc::new(DictInstance {
            values: dict_values,
        });
        let dict3 = Rc::new(DictInstance {
            values: HashMap::new(),
        });
        assert_eq!(Value64::from_dict(dict1.clone()), Value64::from_dict(dict2));
        assert_ne!(Value64::from_dict(dict1), Value64::from_dict(dict3));
    }

    #[test]
    fn array_round_trip() {
        let arr = Rc::new(ArrayInstance::new(vec![
            Value64::NIL,
            Value64::from_bool(true),
            Value64::from_f64(1.0),
        ]));
        let val = Value64::from_array(arr.clone());
        assert!(val.is_array());
        assert!(val.is_ptr());
        let arr2 = val.try_into_array();
        assert_eq!(arr2, Ok(arr));
    }

    #[test]
    fn string_round_trip() {
        let s = Rc::new("test".to_string());
        let val = Value64::from_string(s.clone());
        assert!(val.is_string());
        assert!(val.is_ptr());
        let s2 = val.try_into_string();
        assert_eq!(s2, Ok(s));
    }

    #[test]
    fn builtin_round_trip() {
        let builtin = BuiltInFunction::Map;
        let val = Value64::from_builtin(builtin);
        assert!(val.is_builtin());
        assert!(!val.is_ptr());
        assert!(!val.is_bool());
        assert!(!val.is_nil());
        assert_eq!(val.as_builtin(), Some(BuiltInFunction::Map));
    }

    #[test]
    fn ref_counting() {
        let s = Rc::new("test".to_string());
        let val = Value64::from_string(s.clone());
        let val2 = val.clone();
        assert_eq!(Rc::strong_count(&s), 3);
        let s2 = val2.try_into_string();
        drop(val);
        assert_eq!(Rc::strong_count(&s), 2);
        drop(s2);
        assert_eq!(Rc::strong_count(&s), 1);
    }
}
