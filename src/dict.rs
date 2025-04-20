use std::{collections::HashMap, fmt::Display};

use crate::{rc::Rc, Value64};

const MAP_THRESHOLD: usize = 16;

#[derive(PartialEq, Debug)]
pub enum DictInstance {
    Vec(Vec<(Rc<String>, Value64)>),
    Map(HashMap<Value64, Value64>),
}

pub enum DictEntries<'a> {
    Map(std::collections::hash_map::Iter<'a, Value64, Value64>),
    Vec(std::slice::Iter<'a, (Rc<String>, Value64)>),
}

impl<'a> Iterator for DictEntries<'a> {
    type Item = (Value64, &'a Value64);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DictEntries::Map(iter) => iter.next().map(|(k, v)| (k.clone(), v)),
            DictEntries::Vec(iter) => iter
                .next()
                .map(|(k, v)| (Value64::from_string((*k).clone()), v)),
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
        DictInstance::Vec(Vec::new())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        if capacity < MAP_THRESHOLD {
            DictInstance::Vec(Vec::with_capacity(capacity))
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
            DictInstance::Vec(vec) => {
                if vec.len() < MAP_THRESHOLD {
                    match key.try_into_string() {
                        Ok(key) => match vec.iter_mut().find(|(k, _)| key == *k) {
                            Some((_, v)) => *v = value,
                            None => vec.push((key, value)),
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
            DictInstance::Vec(vec) => vec.iter().find(|(k, _)| val_eq_str(key, k)).map(|(_, v)| v),
        }
    }

    pub fn get_mut(&mut self, key: &Value64) -> Option<&mut Value64> {
        match self {
            DictInstance::Map(map) => map.get_mut(key),
            DictInstance::Vec(vec) => vec
                .iter_mut()
                .find(|(k, _)| val_eq_str(key, k))
                .map(|(_, v)| v),
        }
    }

    pub fn remove(&mut self, key: &Value64) -> Option<Value64> {
        match self {
            DictInstance::Map(map) => map.remove(key),
            DictInstance::Vec(vec) => {
                if let Some(index) = vec.iter().position(|(k, _)| val_eq_str(key, k)) {
                    Some(vec.remove(index).1)
                } else {
                    None
                }
            }
        }
    }

    pub fn iter(&self) -> DictEntries {
        match self {
            DictInstance::Map(map) => DictEntries::Map(map.iter()),
            DictInstance::Vec(vec) => DictEntries::Vec(vec.iter()),
        }
    }

    pub fn keys(&self) -> impl Iterator<Item = Value64> + '_ {
        self.iter().map(|(k, _)| k)
    }

    fn mapify(&mut self) {
        match self {
            DictInstance::Vec(vec) => {
                *self = DictInstance::Map(vec_to_map(std::mem::take(vec)));
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
