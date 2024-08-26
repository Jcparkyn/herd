use std::fmt;

pub type Pos = usize;

/// A span between two locations in a source file
#[derive(Copy, Clone, Default, Eq, Debug, PartialEq)]
pub struct Span {
    pub start: Pos,
    pub end: Pos,
}

impl Span {
    pub fn new(start: Pos, end: Pos) -> Self {
        Span { start, end }
    }
}

#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct Spanned<T> {
    pub span: Span,
    pub value: T,
}

impl<T> Spanned<T> {
    pub fn new(span: Span, value: T) -> Self {
        Spanned { span, value }
    }
}

impl<T: fmt::Debug> fmt::Debug for Spanned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@[{},{}] ", self.span.start, self.span.end)?;
        self.value.fmt(f)
    }
}
