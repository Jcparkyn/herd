pub mod analysis;
pub mod ast;
mod dict;
pub mod error;
pub mod jit;
pub mod lines;
pub mod natives;
pub mod parse_helpers;
pub mod pos;
pub mod prelude;
pub mod rc;
mod stdlib;
pub mod value64;

use lalrpop_util::lalrpop_mod;
pub use value64::Value64;

lalrpop_mod!(
    #[allow(clippy::ptr_arg)]
    #[rustfmt::skip]
    pub lang
);
