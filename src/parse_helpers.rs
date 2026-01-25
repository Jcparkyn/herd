use crate::{
    ast::{
        AssignmentTarget, Block, DeclarationType, Expr, LambdaExpr, MatchPattern, SpannedExpr,
        SpreadListPattern, Statement, VarRef,
    },
    pos::Spanned,
    rc::Rc,
};

pub fn parse_string_literal(s: &str) -> String {
    let mut result = String::new();
    let mut chars = s[1..s.len() - 1].chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('\\') => result.push('\\'),
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('"') => result.push('"'),
                Some('\'') => result.push('\''),
                Some('e') => result.push('\x1B'),
                Some('x') => {
                    let mut num_val = 0;
                    while let Some(next_digit) = chars.peek().and_then(|c| c.to_digit(16)) {
                        num_val = num_val * 16 + next_digit;
                        chars.next();
                    }
                    if let Some(ch) = std::char::from_u32(num_val) {
                        result.push(ch);
                    } else {
                        panic!("Invalid character code in string escape");
                    }
                }
                _ => panic!("Unexpected string escape"),
            }
        } else {
            result.push(ch);
        }
    }
    result
}

pub fn process_list_match(
    parts: Vec<(Spanned<MatchPattern>, bool)>,
) -> Result<MatchPattern, &'static str> {
    let mut spread_idx = None;
    for (i, (_, is_spread)) in parts.iter().enumerate() {
        if *is_spread {
            if spread_idx.is_some() {
                return Err("List patterns can only contain one spread variable (..)");
            }
            spread_idx = Some(i);
        }
    }
    let mut parts_vec: Vec<Spanned<MatchPattern>> = parts.into_iter().map(|p| p.0).collect();
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

pub fn process_lambda_params(
    params: Vec<(Spanned<MatchPattern>, bool)>,
) -> Result<(Vec<Spanned<MatchPattern>>, bool), &'static str> {
    let mut has_spread = false;
    for (i, (_, is_spread)) in params.iter().enumerate() {
        if *is_spread {
            if has_spread {
                return Err("Lambda functions can only contain one spread parameter (..)");
            }
            if i != params.len() - 1 {
                return Err("Spread parameter (..) must be the last parameter");
            }
            has_spread = true;
        }
    }
    let params_vec: Vec<Spanned<MatchPattern>> = params.into_iter().map(|p| p.0).collect();
    Ok((params_vec, has_spread))
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
        false,
        Rc::new(body.map(Expr::Block)),
    ))
}

pub fn assignment_target_to_index_expr(target: &Spanned<AssignmentTarget>) -> SpannedExpr {
    let init = Spanned::new(target.span, Expr::Variable(target.value.var.clone()));
    target.value.path.iter().cloned().fold(init, |e, index| {
        Spanned::new(e.span, Expr::GetIndex(Box::new(e), Box::new(index)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_string_literal_test() {
        assert_eq!(parse_string_literal("\'hello\'"), "hello");
        assert_eq!(parse_string_literal("\"hello\\nworld\""), "hello\nworld");
        assert_eq!(parse_string_literal("\"hello\\\"world\""), "hello\"world");
        assert_eq!(parse_string_literal("\"hello\\'world\""), "hello'world");
        assert_eq!(parse_string_literal("\"hello\\\\world\""), "hello\\world");
        assert_eq!(parse_string_literal("\"hello\\tworld\""), "hello\tworld");
    }
}
