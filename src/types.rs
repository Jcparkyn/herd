#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Value {
    Number(f64),
    Bool(bool),
    Builtin(BuiltInFunction),
    Nil,
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum BuiltInFunction {
    Print,
    Not,
}
