use std::fmt;

use crate::pos::Pos;

// Copied from https://github.com/gluon-lang/gluon/blob/d7ce3e81c1fcfdf25cdd6d1abde2b6e376b4bf50/base/src/source.rs

/// A location in a source file
#[derive(Copy, Clone, Default, Eq, PartialEq, Debug, Hash, Ord, PartialOrd)]
pub struct Location {
    pub line: usize,
    pub column: usize,
    pub absolute: Pos,
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}", self.line + 1, self.column + 1)
    }
}

/// Type which provides a bidirectional mapping between byte offsets and line and column locations
/// for some source file
#[derive(Clone, Debug)]
pub struct Lines {
    starting_bytes: Vec<Pos>,
    end: usize,
}

impl Lines {
    /// Creates a mapping for `src`
    pub fn new<I>(src: I) -> Lines
    where
        I: IntoIterator<Item = u8>,
    {
        use std::iter;

        let mut len = 0;
        let starting_bytes = {
            let input_indices = src
                .into_iter()
                .inspect(|_| len += 1)
                .enumerate()
                .filter(|&(_, b)| b == b'\n')
                .map(|(i, _)| i + 1); // index of first char in the line

            iter::once(0).chain(input_indices).collect()
        };
        Lines {
            starting_bytes,
            end: len,
        }
    }

    /// Returns the byte offset of the start of `line_number`
    pub fn line(&self, line_number: usize) -> Option<usize> {
        let line_number = line_number;
        self.starting_bytes.get(line_number).cloned()
    }

    #[allow(dead_code)]
    pub fn offset(&self, line: usize, column: usize) -> Option<Pos> {
        self.line(line).and_then(|mut offset| {
            offset += column;
            if offset >= self.end {
                None
            } else {
                Some(offset)
            }
        })
    }

    /// Returns the line and column location of `byte`
    pub fn location(&self, byte: Pos) -> Option<Location> {
        if byte <= self.end {
            let line_index = self.line_number_at_byte(byte);

            self.line(line_index).map(|line_byte| Location {
                line: line_index,
                column: byte - line_byte,
                absolute: byte,
            })
        } else {
            None
        }
    }

    /// Returns which line `byte` points to
    pub fn line_number_at_byte(&self, byte: Pos) -> usize {
        let num_lines = self.starting_bytes.len();

        (0..num_lines)
            .filter(|&i| self.starting_bytes[i] > byte)
            .map(|i| i - 1)
            .next()
            .unwrap_or(num_lines - 1)
    }
}
