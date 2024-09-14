use std::rc::Rc;

use crate::{
    ast::{
        Block, DeclarationType, Expr, LambdaExpr, MatchPattern, SpannedExpr, SpreadListPattern,
        Statement, VarRef,
    },
    pos::Spanned,
};

pub fn parse_string_literal(s: &str) -> String {
    s[1..s.len() - 1].to_string()
}

pub fn process_list_match(parts: Vec<(MatchPattern, bool)>) -> Result<MatchPattern, &'static str> {
    let mut spread_idx = None;
    for (i, (_, is_spread)) in parts.iter().enumerate() {
        if *is_spread {
            if spread_idx.is_some() {
                return Err("List patterns can only contain one spread variable (..)");
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
            Ok(MatchPattern::SpreadList(SpreadListPattern {
                before,
                spread,
                after,
            }))
        }
        None => Ok(MatchPattern::SimpleList(parts_vec)),
    }
}

pub fn process_declaration(
    name: Spanned<String>,
    decl_type: DeclarationType,
    mut rhs: SpannedExpr,
) -> Statement {
    if let Expr::Lambda(lambda) = &mut rhs.value {
        lambda.name = Some(name.value.clone());
    }
    let pattern = MatchPattern::Declaration(VarRef::new(name.value), decl_type);
    return Statement::PatternAssignment(Spanned::new(name.span, pattern), rhs);
}

pub fn make_implicit_lambda(body: Spanned<Block>) -> Expr {
    let param = MatchPattern::Declaration(VarRef::new("_".to_string()), DeclarationType::Mutable);
    Expr::Lambda(LambdaExpr::new(
        vec![Spanned::new(body.span, param)],
        Rc::new(body.map(Expr::Block)),
    ))
}
