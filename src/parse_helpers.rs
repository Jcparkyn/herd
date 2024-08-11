use crate::ast::{MatchPattern, SpreadArrayPattern};

pub fn parse_string_literal(s: &str) -> String {
    s[1..s.len() - 1].to_string()
}

pub fn process_array_match(parts: Vec<(MatchPattern, bool)>) -> Result<MatchPattern, &'static str> {
    let mut spread_idx = None;
    for (i, (_, is_spread)) in parts.iter().enumerate() {
        if *is_spread {
            if spread_idx.is_some() {
                return Err("Array patterns can only contain one spread variable (..)");
            }
            spread_idx = Some(i);
        }
    }
    let mut parts_vec: Vec<MatchPattern> = parts.into_iter().map(|p| p.0).collect();
    match spread_idx {
        Some(idx) => {
            let after = parts_vec.split_off(idx + 1);
            let spread = Box::new(parts_vec.pop().unwrap());
            let before = parts_vec;
            Ok(MatchPattern::SpreadArray(SpreadArrayPattern {
                before,
                spread,
                after,
            }))
        }
        None => Ok(MatchPattern::SimpleArray(parts_vec)),
    }
}
