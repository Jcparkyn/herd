use interpreter::Interpreter;
use lalrpop_util::{lalrpop_mod, ParseError};
use std::io::{stdin, stdout, Write};

mod ast;
mod interpreter;
mod parse_helpers;

lalrpop_mod!(
    #[allow(clippy::ptr_arg)]
    #[rustfmt::skip]
    pub lang
);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let parser = lang::StatementParser::new();
    let mut interpreter = Interpreter::new();
    loop {
        print!("> ");
        stdout().flush()?;
        let mut buffer = String::new();
        loop {
            stdin().read_line(&mut buffer)?;
            let ast_result = parser.parse(&buffer);
            match ast_result {
                Err(ParseError::UnrecognizedEof { .. }) => {}
                Err(err) => {
                    println!("Error while parsing: {}", err);
                    break;
                }
                Ok(statement) => {
                    println!("ast: {:?}", statement);
                    match interpreter.execute(&statement) {
                        Ok(()) => {}
                        Err(err) => println!("Error while evaluating: {}", err),
                    }
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{interpreter, lang};
    use rstest::rstest;

    #[rstest(
    exp,
    expected,
    case("1 + 2", 3),
    case("2 * 2", 4),
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
        use crate::interpreter::Value;

        let expr = lang::ExprParser::new().parse(exp).unwrap();
        assert_eq!(
            interpreter::Interpreter::new().eval(&expr).unwrap(),
            Value::Number(expected as f64)
        );
    }
}
