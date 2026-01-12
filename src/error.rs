use crate::pos::Pos;

#[derive(Debug, Clone)]
pub struct HerdError {
    pub message: String,
    pub inner: Option<Box<HerdError>>,
    pub pos: Option<Pos>,
    pub file_id: Option<usize>,
}

impl HerdError {
    pub fn new(message: impl Into<String>) -> Self {
        HerdError {
            message: message.into(),
            pos: None,
            inner: None,
            file_id: Some(0),
        }
    }

    pub fn wrap(self, message: impl Into<String>) -> Self {
        HerdError {
            message: message.into(),
            pos: None,
            file_id: self.file_id,
            inner: Some(Box::new(self)),
        }
    }
}
