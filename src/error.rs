use crate::pos::Pos;

#[derive(Debug, Clone)]
pub struct HerdError {
    pub message: String,
    pub inner: Option<Box<HerdError>>,
    pub pos: Option<Pos>,
    pub file_id: Option<usize>,
    pub func_name: Option<String>,
}

impl HerdError {
    pub fn native_code(message: impl Into<String>) -> Self {
        HerdError {
            message: message.into(),
            pos: None,
            inner: None,
            file_id: None,
            func_name: None,
        }
    }

    pub fn wrap_native(self, message: impl Into<String>) -> Self {
        HerdError {
            message: message.into(),
            pos: None,
            file_id: None,
            inner: Some(Box::new(self)),
            func_name: None,
        }
    }
}
