use std::{collections::HashMap, fmt::Display};

use crate::{rc::Rc, Value64};

const MAP_THRESHOLD: usize = 16;

#[derive(PartialEq, Debug, Clone)]
pub struct DictShape(pub Vec<String>);

#[derive(PartialEq, Debug, Clone)]
pub struct VecDict {
    shape: Rc<Vec<String>>,
    values: Vec<Value64>,
}

impl VecDict {
    fn key_index(&self, key: &str) -> Option<usize> {
        self.shape.iter().position(|k| key == k)
    }
}

#[derive(PartialEq, Debug)]
pub enum DictInstance {
    Vec(VecDict),
    Map(HashMap<Value64, Value64>),
}

pub enum DictEntries<'a> {
    Map(std::collections::hash_map::Iter<'a, Value64, Value64>),
    Vec(&'a VecDict, usize),
    // Vec(std::slice::Iter<'a, (Rc<String>, Value64)>),
}

impl<'a> Iterator for DictEntries<'a> {
    type Item = (Value64, &'a Value64);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DictEntries::Map(iter) => iter.next().map(|(k, v)| (k.clone(), v)),
            // DictEntries::Vec(iter) => iter
            //     .next()
            //     .map(|(k, v)| (Value64::from_string((*k).clone()), v)),
            DictEntries::Vec(vec, index) => {
                if *index >= vec.values.len() {
                    None
                } else {
                    let key = Rc::new(vec.shape[*index].clone());
                    let value = &vec.values[*index as usize];
                    *index += 1;
                    Some((Value64::from_string(key), value))
                }
            }
        }
    }
}

fn val_eq_str(val: &Value64, s: &str) -> bool {
    match val.as_string() {
        Some(v) => v == s,
        None => false,
    }
}

impl DictInstance {
    pub fn new() -> Self {
        DictInstance::Map(HashMap::new())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        if capacity < MAP_THRESHOLD {
            // DictInstance::Vec(
            //     Rc::new(DictShape(Vec::with_capacity(capacity))),
            //     Vec::with_capacity(capacity),
            // )
            DictInstance::Vec(VecDict {
                shape: Rc::new(Vec::with_capacity(capacity)),
                values: Vec::with_capacity(capacity),
            })
        } else {
            DictInstance::Map(HashMap::with_capacity(capacity))
        }
    }

    pub fn from_hashmap(values: HashMap<Value64, Value64>) -> Self {
        DictInstance::Map(values)
    }

    pub fn insert(&mut self, key: Value64, value: Value64) {
        match self {
            DictInstance::Map(map) => {
                map.insert(key, value);
            }
            DictInstance::Vec(v) => {
                if v.values.len() < MAP_THRESHOLD {
                    match key.try_into_string() {
                        Ok(key) => match v.key_index(&key) {
                            Some(index) => v.values[index] = value,
                            None => {
                                Rc::make_mut(&mut v.shape).push(key.to_string());
                                v.values.push(value);
                            }
                        },
                        Err(k) => {
                            self.mapify();
                            self.insert(k, value);
                        }
                    }
                } else {
                    self.mapify();
                    self.insert(key, value);
                }
            }
        }
    }

    pub fn get(&self, key: &Value64) -> Option<&Value64> {
        match self {
            DictInstance::Map(map) => map.get(key),
            DictInstance::Vec(vec) => {
                let index = vec.key_index(key.as_string()?);
                Some(&vec.values[index?])
            }
        }
    }

    pub fn get_mut(&mut self, key: &Value64) -> Option<&mut Value64> {
        match self {
            DictInstance::Map(map) => map.get_mut(key),
            DictInstance::Vec(vec) => {
                let index = vec.key_index(key.as_string()?);
                Some(&mut vec.values[index?])
            }
        }
    }

    pub fn remove(&mut self, key: &Value64) -> Option<Value64> {
        match self {
            DictInstance::Map(map) => map.remove(key),
            DictInstance::Vec(vec) => {
                if let Some(index) = vec.key_index(key.as_string()?) {
                    Rc::make_mut(&mut vec.shape).remove(index);
                    Some(vec.values.remove(index))
                } else {
                    None
                }
            }
        }
    }

    pub fn iter(&self) -> DictEntries {
        match self {
            DictInstance::Map(map) => DictEntries::Map(map.iter()),
            DictInstance::Vec(vec) => {
                // let cloned: Vec<_> = vec
                //     .shape
                //     .iter()
                //     .zip(vec.values.iter())
                //     .map(|(k, v)| (Rc::new(k.clone()), v.clone()))
                //     .collect();
                // DictEntries::Vec(cloned.iter())
                DictEntries::Vec(&vec, 0)
            }
        }
    }

    pub fn keys(&self) -> impl Iterator<Item = Value64> + '_ {
        self.iter().map(|(k, _)| k)
    }

    fn mapify(&mut self) {
        match self {
            DictInstance::Vec(vec) => {
                let mut map = HashMap::with_capacity(vec.values.len() * 2); // some spare capacity since we're currently expanding it.
                for (i, key) in vec.shape.iter().enumerate() {
                    // TODO: Move key instead of clone()
                    let value = std::mem::replace(&mut vec.values[i], Value64::NIL);
                    map.insert(Value64::from_string(Rc::new(key.clone())), value.clone());
                }
                *self = DictInstance::Map(map);
            }
            _ => {}
        }
    }
}

fn vec_to_map(vec: Vec<(Rc<String>, Value64)>) -> HashMap<Value64, Value64> {
    let mut map = HashMap::with_capacity(vec.len());
    for (k, v) in vec {
        map.insert(Value64::from_string(k), v);
    }
    map
}

impl Clone for DictInstance {
    fn clone(&self) -> Self {
        #[cfg(debug_assertions)]
        println!("Cloning dict: {}", self);
        match self {
            DictInstance::Map(map) => DictInstance::Map(map.clone()),
            DictInstance::Vec(vec) => DictInstance::Vec(vec.clone()),
        }
    }
}

impl Display for DictInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut entries: Vec<_> = self.iter().collect();
        entries.sort_unstable_by(|a, b| a.0.display_cmp(&b.0));
        let mut values = vec![];
        for (key, value) in entries {
            if let Some(s) = key.as_string() {
                values.push(format!("{}: {}", s, value));
            } else {
                values.push(format!("[{}]: {}", key, value));
            }
        }
        write!(f, "{{{}}}", values.join(", "))
    }
}
