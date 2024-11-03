pub mod analysis;
pub mod ast;
pub mod builtins;
pub mod interpreter;
pub mod jit;
pub mod lines;
pub mod parse_helpers;
pub mod pos;
mod stdlib;
pub mod value64;

use lalrpop_util::lalrpop_mod;
pub use value64::Value64;

lalrpop_mod!(
    #[allow(clippy::ptr_arg)]
    #[rustfmt::skip]
    pub lang
);
