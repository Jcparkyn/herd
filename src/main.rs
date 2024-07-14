use ast::Expr;
use lalrpop_util::lalrpop_mod;
use std::{fmt::Debug, io::{stdin, stdout, Write}};

mod ast;

lalrpop_mod!(
    #[allow(clippy::ptr_arg)]
    #[rustfmt::skip]
    pub lang
);

fn print_expr(expr: Expr) -> String {
    // expr.fmt(f)
    format!("{:?}", expr)
    // match expr {
    //     Expr::Number(x) => "2".to_string(),
    //     Expr::Op(left, op, right) => print_expr(*left) + &print_expr(*right),
    // }
}

// fn print

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let parser = lang::ExprParser::new();
    loop {
        print!("> ");
        stdout().flush()?;
        let mut buffer = String::new();
        stdin().read_line(&mut buffer)?;
        let ast_result = parser.parse(&buffer);
        match ast_result {
            Ok(ast) => println!("ast: {}", print_expr(*ast)),
            Err(err) => println!("{}", err),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::lang;
    use rstest::rstest;

    #[rstest(
    exp,
    expected,
    case("1 + 2", 3),
    case("2 * 2", 4),
    case("10 / 4", 2),
    case("3 - 9", -6),
    case("-5 * -7", 35),
    case("100 * 0", 0),
    case("8 * 2 + 3", 19),
    case("6 * (2 + 3)", 30),
    case("-100 - (-1000)", 900),
    case("150 / (20 - 5) * 3", 30),
    case("150 / ((20 - 5) * 5)", 2),
    case("10 * 15 / ((20 - 5) * 5)", 2),
    case("(100 + 50) / ((20 - 5) * 5)", 2),
    case("100 + 150 / ((20 - 5) * 5)", 102),
    )]
    fn when_expression_evaluated_then_correct_value_returned(exp: &str, expected: i32) {
        assert!(lang::ExprParser::new().parse(exp).is_ok());
    }
}

