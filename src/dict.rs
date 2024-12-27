use std::{collections::HashMap, fmt::Display};

use crate::Value64;

const MAP_THRESHOLD: usize = 16;

#[derive(PartialEq, Debug)]
pub enum DictInstance {
    Vec(Vec<(Value64, Value64)>),
    Map(HashMap<Value64, Value64>),
}

pub enum DictEntries<'a> {
    Map(std::collections::hash_map::Iter<'a, Value64, Value64>),
    Vec(std::slice::Iter<'a, (Value64, Value64)>),
}

impl<'a> Iterator for DictEntries<'a> {
    type Item = (&'a Value64, &'a Value64);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DictEntries::Map(iter) => iter.next(),
            DictEntries::Vec(iter) => iter.next().map(|(k, v)| (k, v)),
        }
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
                    match vec.iter_mut().find(|(k, _)| k == &key) {
                        Some((_, v)) => *v = value,
                        None => vec.push((key, value)),
                    }
                } else {
                    let mut new_map = HashMap::with_capacity(vec.len() * 2);
                    for (k, v) in vec.drain(..) {
                        new_map.insert(k, v);
                    }
                    new_map.insert(key, value);
                    *self = DictInstance::Map(new_map);
                }
            }
        }
    }

    pub fn get(&self, key: &Value64) -> Option<&Value64> {
        match self {
            DictInstance::Map(map) => map.get(key),
            DictInstance::Vec(vec) => vec.iter().find(|(k, _)| k == key).map(|(_, v)| v),
        }
    }

    pub fn get_mut(&mut self, key: &Value64) -> Option<&mut Value64> {
        match self {
            DictInstance::Map(map) => map.get_mut(key),
            DictInstance::Vec(vec) => vec.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
        }
    }

    pub fn remove(&mut self, key: &Value64) -> Option<Value64> {
        match self {
            DictInstance::Map(map) => map.remove(key),
            DictInstance::Vec(vec) => {
                if let Some(index) = vec.iter().position(|(k, _)| k == key) {
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

    pub fn keys(&self) -> impl Iterator<Item = &Value64> {
        self.iter().map(|(k, _)| k)
    }
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
        entries.sort_unstable_by(|a, b| a.0.display_cmp(b.0));
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
