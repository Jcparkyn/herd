use crate::pos::Pos;

#[derive(Debug)]
pub struct HerdError {
    pub message: String,
    pub pos: Option<Pos>,
    pub inner: Option<Box<HerdError>>,
}

impl HerdError {
    pub fn new(message: impl Into<String>) -> Self {
        HerdError {
            message: message.into(),
            pos: None,
            inner: None,
        }
    }
}
