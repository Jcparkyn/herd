use codegen::ir::{self, StackSlot};
use core::panic;
use cranelift::{codegen::ir::BlockArg, prelude::*};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, FuncId, FuncOrDataId, Linkage, Module, ModuleError};
use std::{
    collections::HashMap,
    hash::Hash,
    mem::{self, forget, size_of},
    ops::Deref,
    path::{Path, PathBuf},
    sync::Mutex,
};
use strum::Display;
use types::I64;

use crate::{
    Value64,
    ast::{
        self, Expr, LambdaExpr, MatchConstant, MatchExpr, MatchPattern, Opcode, SpannedExpr,
        SpannedStatement, Statement, VarRef,
    },
    error::HerdError,
    natives::{self, NativeFuncDef, NativeFuncId},
    pos::{Span, Spanned},
    rc::Rc,
    value64::{self, PointerTag},
};

type FuncExpr = LambdaExpr;

const VAL64: ir::Type = ir::types::F64;
const PTR: ir::Type = ir::types::I64;

#[derive(Debug, Display)]
pub enum JITError {
    Module(ModuleError),
}

pub type JITResult<T> = Result<T, JITError>;

pub trait ModuleLoader: Send + Sync {
    fn load(&self, path: &str) -> std::io::Result<String>;
}

pub struct DefaultModuleLoader {
    pub base_path: PathBuf,
}

impl ModuleLoader for DefaultModuleLoader {
    fn load(&self, path: &str) -> std::io::Result<String> {
        let path = self.base_path.join(path);
        std::fs::read_to_string(path)
    }
}

/// The basic JIT class.
pub struct JIT {
    /// The function builder context, which is reused across multiple
    /// FunctionBuilder instances.
    builder_context: FunctionBuilderContext,

    /// The main Cranelift context, which holds the state for codegen. Cranelift
    /// separates this from `Module` to allow for parallel compilation, with a
    /// context per thread, though this isn't in the simple demo here.
    ctx: codegen::Context,

    /// The data description, which is to data objects what `ctx` is to functions.
    #[allow(dead_code)]
    data_description: DataDescription,

    /// The module, with the jit backend, which manages the JIT'd
    /// functions.
    module: JITModule,

    /// Interned string constants, which are used for string literals.
    string_constants: HashMap<String, Value64>,

    /// Return values from herd modules (files). Files currently being evaluated
    /// are stored as `None`.
    pub modules: HashMap<String, Option<Value64>>,

    pub module_loader: Box<dyn ModuleLoader>,
    builtins_map: HashMap<&'static str, NativeFunc>,
    natives_map: HashMap<NativeFuncId, NativeFunc>,
}

pub struct VmContext {
    pub jit: Mutex<JIT>,
    pub program_args: Vec<String>,
}

impl VmContext {
    pub fn new(jit: JIT, program_args: Vec<String>) -> Self {
        Self {
            jit: Mutex::new(jit),
            program_args,
        }
    }

    pub unsafe fn run_func(
        &self,
        func_id: FuncId,
        inputs: Vec<Value64>,
    ) -> Result<Value64, HerdError> {
        let func_ptr = self
            .jit
            .lock()
            .unwrap()
            .module
            .get_finalized_function(func_id);
        // Cast the raw pointer to a typed function pointer. This is unsafe, because
        // this is the critical point where you have to trust that the generated code
        // is safe to be called.
        let code_fn = unsafe {
            mem::transmute::<
                _,
                extern "C" fn(&Self, *const Value64, *mut *const HerdError) -> Value64,
            >(func_ptr)
        };
        let inputs_ptr = inputs.as_ptr();
        // Make code_fn "consume" the inputs
        forget(inputs);

        let mut error_ptr: *const HerdError = std::ptr::null();
        // And now we can call it!
        let result = code_fn(self, inputs_ptr, &mut error_ptr);

        if !error_ptr.is_null() {
            let error_box = unsafe { Box::from_raw(error_ptr as *mut HerdError) };
            return Err(*error_box);
        }
        Ok(result)
    }

    pub fn run_lambda(
        &self,
        lambda_val: &Value64,
        params: &[Value64],
    ) -> Result<Value64, HerdError> {
        let lambda = lambda_val.as_lambda().unwrap();
        assert!(lambda.param_count == params.len());
        let func_ptr = lambda.func_ptr.unwrap();

        let mut error_ptr: *const HerdError = std::ptr::null();
        let result = match params {
            [x1] => unsafe {
                let code_fn = mem::transmute::<
                    _,
                    extern "C" fn(
                        &Self,
                        closure: *const Value64,
                        error_out: *mut *const HerdError,
                        x1: Value64,
                        recursion: Value64,
                    ) -> Value64,
                >(func_ptr);
                code_fn(
                    self,
                    lambda.closure.as_ptr(),
                    &mut error_ptr,
                    x1.clone(),
                    lambda_val.cheeky_copy(),
                )
            },
            _ => {
                panic!("Argument count not supported: {}", params.len())
            }
        };
        if !error_ptr.is_null() {
            let error_box = unsafe { Box::from_raw(error_ptr as *mut HerdError) };
            return Err(*error_box);
        }
        Ok(result)
    }
}

/// An owned Value64
type OValue = Value;

/// A borrowed Value64
type BValue = Value;

#[derive(Clone, Copy)]
struct MValue {
    value: Value,
    owned: bool,
}

impl MValue {
    pub fn owned(value: Value) -> Self {
        Self { value, owned: true }
    }

    pub fn borrowed(value: Value) -> Self {
        Self {
            value,
            owned: false,
        }
    }

    pub fn into_owned(&self, translator: &mut FunctionTranslator) -> OValue {
        if self.owned {
            return self.value;
        } else {
            translator.clone_val64(self.value)
        }
    }

    pub fn borrow(&self) -> Value {
        self.value
    }
}

trait AsBValue: Copy {
    fn as_bvalue(&self) -> BValue;
}

impl AsBValue for MValue {
    fn as_bvalue(&self) -> BValue {
        self.borrow()
    }
}

impl AsBValue for Value {
    fn as_bvalue(&self) -> BValue {
        *self
    }
}

trait ValueExt {
    fn assert_owned(self) -> MValue;
    fn assert_borrowed(self) -> MValue;
}

impl ValueExt for Value {
    fn assert_owned(self) -> MValue {
        MValue::owned(self)
    }

    fn assert_borrowed(self) -> MValue {
        MValue::borrowed(self)
    }
}

#[derive(Debug, Clone)]
struct NativeFunc {
    func: FuncId,
    sig: Signature,
    needs_vm: bool,
    fallible: bool,
}

impl NativeFunc {
    fn expected_arg_count(&self) -> usize {
        self.sig.params.len() - (self.needs_vm as usize) - (self.fallible as usize)
    }
}

fn build_native_funcs<TKey>(
    module: &mut JITModule,
    funcs: HashMap<TKey, NativeFuncDef>,
) -> HashMap<TKey, NativeFunc>
where
    TKey: Eq + Hash,
{
    let mut result = HashMap::new();
    for (key, def) in funcs {
        let mut sig = module.make_signature();
        (def.make_sig)(&mut sig);
        let func = module
            .declare_function(&def.name, Linkage::Import, &sig)
            .expect("problem declaring function");
        result.insert(
            key,
            NativeFunc {
                func,
                sig,
                needs_vm: def.needs_vm,
                fallible: def.fallible,
            },
        );
    }
    return result;
}

impl JIT {
    pub fn new(module_loader: Box<dyn ModuleLoader>) -> Self {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();
        flag_builder.set("preserve_frame_pointers", "true").unwrap();
        flag_builder.set("unwind_info", "true").unwrap();
        let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
            panic!("host machine is not supported: {}", msg);
        });
        let flags = settings::Flags::new(flag_builder);
        let isa = isa_builder.finish(flags).unwrap();
        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let builtins_map = natives::get_builtins();
        let natives_map = natives::get_natives();

        for (_, func) in &builtins_map {
            builder.symbol(func.name, func.func_ptr);
        }
        for (_, func) in &natives_map {
            builder.symbol(func.name, func.func_ptr);
        }

        let mut module = JITModule::new(builder);

        let builtins_map = build_native_funcs(&mut module, builtins_map);

        let natives_map = build_native_funcs(&mut module, natives_map);

        Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            data_description: DataDescription::new(),
            module,
            string_constants: HashMap::new(),
            modules: HashMap::new(),
            module_loader,
            builtins_map,
            natives_map,
        }
    }

    pub fn compile_program_as_function(
        &mut self,
        program: &[SpannedStatement],
        src_path: &Path,
    ) -> JITResult<FuncId> {
        let body = Spanned::with_zero_span(Expr::Block(ast::Block {
            statements: program.to_vec(),
            expression: None,
        }));
        let func_id = self.compile_main_func(&body, src_path, &[], false)?;
        Ok(func_id)
    }

    pub fn compile_repl_as_function(
        &mut self,
        program: &[SpannedStatement],
        src_path: &Path,
        globals: &[String],
    ) -> JITResult<FuncId> {
        let body = Spanned::with_zero_span(Expr::Block(ast::Block {
            statements: program.to_vec(),
            expression: None,
        }));
        let func_id = self.compile_main_func(&body, src_path, globals, true)?;
        Ok(func_id)
    }

    fn compile_main_func(
        &mut self,
        body: &SpannedExpr,
        src_path: &Path,
        implicit_vars: &[String],
        repl_mode: bool,
    ) -> JITResult<FuncId> {
        // Then, translate the AST nodes into Cranelift IR.
        self.translate_main_func(body, src_path, implicit_vars, repl_mode)?;

        let id = if repl_mode {
            // Don't use a name in JIT mode, to avoid collisions
            self.module
                .declare_anonymous_function(&self.ctx.func.signature)
                .map_err(JITError::Module)?
        } else {
            let func_name = format!("MAIN:{}", src_path.to_str().unwrap());

            self.module
                .declare_function(&func_name, Linkage::Export, &self.ctx.func.signature)
                .map_err(JITError::Module)?
        };

        // Define the function to jit. This finishes compilation, although
        // there may be outstanding relocations to perform. Currently, jit
        // cannot finish relocations until all functions to be called are
        // defined. For this toy demo for now, we'll just finalize the
        // function below.
        self.module
            .define_function(id, &mut self.ctx)
            .map_err(JITError::Module)?;

        // Now that compilation is finished, we can clear out the context state.
        self.module.clear_context(&mut self.ctx);

        // Finalize the functions which we just defined, which resolves any
        // outstanding relocations (patching in addresses, now that they're
        // available).
        self.module
            .finalize_definitions()
            .map_err(JITError::Module)?;

        Ok(id)
    }

    pub fn get_func_id(&self, name: &str) -> Option<FuncId> {
        match self.module.get_name(name) {
            Some(FuncOrDataId::Func(f)) => Some(f),
            _ => None,
        }
    }

    fn translate_main_func(
        &mut self,
        body: &SpannedExpr,
        src_path: &Path,
        implicit_vars: &[String],
        repl_mode: bool,
    ) -> JITResult<()> {
        self.ctx.func.collect_debug_info();
        self.ctx.func.signature.params.push(AbiParam::new(PTR)); // VmContext ptr
        self.ctx.func.signature.params.push(AbiParam::new(PTR)); // args (array of VAL64)
        self.ctx.func.signature.params.push(AbiParam::new(PTR)); // error return pointer

        self.ctx.func.signature.returns.push(AbiParam::new(VAL64));

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        // Walk the AST and declare all implicitly-declared variables.
        let variables = {
            let implicit_vars_ptr = builder.block_params(entry_block)[1]; // Value64 array
            let mut implicit_var_vals = vec![];
            for (i, var) in implicit_vars.iter().enumerate() {
                let val = builder.ins().load(
                    VAL64,
                    MemFlags::new(),
                    implicit_vars_ptr,
                    (i * size_of::<Value64>()) as i32,
                );
                implicit_var_vals.push((var.as_str(), val));
            }

            let mut variable_builder = VariableBuilder::new(&mut builder);
            for (var, val) in implicit_var_vals {
                variable_builder.create_variable(var, val, true);
            }
            variable_builder.declare_variables_in_expr(body);
            variable_builder.variables
        };

        builder.seal_all_blocks();

        let return_block = builder.create_block();
        // Now translate the statements of the function body.
        let mut trans = FunctionTranslator {
            src_path,
            builder,
            variables,
            module: &mut self.module,
            string_constants: &mut self.string_constants,
            return_block,
            entry_block,
            builtins_map: &self.builtins_map,
            natives_map: &self.natives_map,
        };
        let return_value = trans.translate_expr(body);

        // Jump to the return block.
        trans.translate_return_ok(return_value.borrow());

        trans.translate_return_block(repl_mode);
        // Tell the builder we're done with this function.
        trans.builder.finalize();
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FuncVar {
    var: Variable,
    owned: bool,
}

struct VariableBuilder<'a, 'b> {
    builder: &'a mut FunctionBuilder<'b>,
    variables: HashMap<String, FuncVar>,
}

impl<'a, 'b> VariableBuilder<'a, 'b> {
    fn new(builder: &'a mut FunctionBuilder<'b>) -> Self {
        Self {
            builder,
            variables: HashMap::new(),
        }
    }

    fn declare_variables_in_func(&mut self, func: &FuncExpr, entry_block: Block) {
        let closure_ptr = self.builder.block_params(entry_block)[1];
        for (i, var) in func.potential_captures.iter().enumerate() {
            let val = self.builder.ins().load(
                VAL64,
                MemFlags::new(),
                closure_ptr,
                (i * size_of::<Value64>()) as i32,
            );
            self.create_variable(&var.name, val, false);
        }
        for pattern in &*func.params {
            self.declare_variables_in_pattern(pattern);
            // Actual match is done in function body, so we can reuse the logic from FunctionTranslator.
        }
        // Reference to self function for recursion
        let self_val = self
            .builder
            .block_params(entry_block)
            .last()
            .copied()
            .unwrap();
        if let Some(name) = &func.name {
            self.create_variable(name, self_val, false);
        } else {
            // HACK: We rely on variables to do cleanup of this value, so add it here even if unused.
            self.create_variable("<UNUSED SELF>", self_val, false);
        }

        self.declare_variables_in_expr(&func.body);
    }

    fn declare_variables_in_expr(&mut self, expr: &SpannedExpr) {
        match &expr.value {
            Expr::Block(b) => {
                for stmt in &b.statements {
                    self.declare_variables_in_stmt(stmt);
                }
                if let Some(ref expr) = b.expression {
                    self.declare_variables_in_expr(expr);
                }
            }
            Expr::Variable(_) => {}
            Expr::Bool(_) => {}
            Expr::Number(_) => {}
            Expr::String(_) => {}
            Expr::Nil => {}
            Expr::Op { op: _, lhs, rhs } => {
                self.declare_variables_in_expr(&lhs);
                self.declare_variables_in_expr(&rhs);
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.declare_variables_in_expr(condition);
                self.declare_variables_in_expr(then_branch);
                if let Some(else_branch) = else_branch {
                    self.declare_variables_in_expr(else_branch);
                }
            }
            Expr::ForIn { iter, var, body } => {
                self.declare_variables_in_expr(iter);
                self.declare_variables_in_pattern(var);
                self.declare_variables_in_expr(body);
            }
            Expr::While { condition, body } => {
                self.declare_variables_in_expr(condition);
                self.declare_variables_in_expr(body);
            }
            Expr::Call { callee, args } => {
                for arg in args {
                    self.declare_variables_in_expr(arg);
                }
                self.declare_variables_in_expr(callee);
            }
            Expr::CallBuiltin { callee: _, args } => {
                for arg in args {
                    self.declare_variables_in_expr(arg);
                }
            }
            Expr::List(l) => {
                for item in l {
                    self.declare_variables_in_expr(item);
                }
            }
            Expr::Dict(d) => {
                for (key, value) in d {
                    self.declare_variables_in_expr(key);
                    self.declare_variables_in_expr(value);
                }
            }
            Expr::GetIndex(val, index) => {
                // TODO: Should this order be reversed?
                self.declare_variables_in_expr(index);
                self.declare_variables_in_expr(val);
            }
            Expr::Lambda(_) => {
                // The lambda body is analyzed in a separate scope later on.
            }
            Expr::Match(m) => self.declare_variables_in_match(m),
            Expr::Import { .. } => {}
        }
    }

    fn declare_variables_in_stmt(&mut self, stmt: &SpannedStatement) {
        match &stmt.value {
            Statement::Expression(e) => self.declare_variables_in_expr(e),
            Statement::Return(e) => self.declare_variables_in_expr(e),
            Statement::PatternAssignment(pattern, rhs) => {
                self.declare_variables_in_expr(rhs);
                self.declare_variables_in_pattern(pattern);
            }
        }
    }

    fn declare_variables_in_match(&mut self, match_expr: &MatchExpr) {
        for (pattern, body) in &match_expr.branches {
            self.declare_variables_in_pattern(pattern);
            self.declare_variables_in_expr(body);
        }
    }

    fn declare_variables_in_pattern(&mut self, pattern: &Spanned<MatchPattern>) {
        match &pattern.value {
            MatchPattern::Declaration(var, _) => {
                let nil = self
                    .builder
                    .ins()
                    .f64const(f64::from_bits(value64::NIL_VALUE));
                self.create_variable(&var.name, nil, true);
            }
            MatchPattern::Discard => {}
            MatchPattern::Constant(_) => {}
            MatchPattern::Assignment(target) => {
                for index in &target.path {
                    self.declare_variables_in_expr(&index);
                }
            }
            MatchPattern::SimpleList(parts) => {
                for part in parts {
                    self.declare_variables_in_pattern(part);
                }
            }
            MatchPattern::SpreadList(spread) => {
                for part in &spread.before {
                    self.declare_variables_in_pattern(part);
                }
                self.declare_variables_in_pattern(&spread.spread);
                for part in &spread.after {
                    self.declare_variables_in_pattern(part);
                }
            }
            MatchPattern::Dict(dict) => {
                for (_key, pattern) in &dict.entries {
                    self.declare_variables_in_pattern(pattern);
                }
            }
        }
    }

    fn create_variable(&mut self, name: &str, default_value: Value, owned: bool) {
        if !self.variables.contains_key(name) {
            let var = self.builder.declare_var(VAL64);
            // let var = Variable::new(self.index);
            self.variables.insert(name.into(), FuncVar { var, owned });
            self.define_variable(var, default_value, name);
        }
    }

    fn define_variable(&mut self, var: Variable, val: Value, name: &str) {
        self.builder.try_def_var(var, val).unwrap_or_else(|error| {
            panic!(
                "Error defining variable {}: {}. Assigning value of type {}.",
                name,
                error,
                self.builder.func.dfg.value_type(val)
            );
        })
    }
}

struct FunctionTranslator<'a> {
    src_path: &'a Path,
    builder: FunctionBuilder<'a>,
    variables: HashMap<String, FuncVar>,
    module: &'a mut JITModule,
    string_constants: &'a mut HashMap<String, Value64>,
    entry_block: Block,
    return_block: Block,
    builtins_map: &'a HashMap<&'static str, NativeFunc>,
    natives_map: &'a HashMap<NativeFuncId, NativeFunc>,
}

impl<'a> FunctionTranslator<'a> {
    fn translate_function_entry(&mut self, lambda: &LambdaExpr, entry_block: Block) {
        for (i, pattern) in lambda.params.iter().enumerate() {
            let value = self.builder.block_params(entry_block)[i + 3];
            self.translate_match_pattern(pattern, value.assert_owned());
        }
    }

    /// When you write out instructions in Cranelift, you get back `Value`s. You
    /// can then use these references in other instructions.
    fn translate_expr(&mut self, expr: &SpannedExpr) -> MValue {
        self.set_src_span(&expr.span);
        match &expr.value {
            Expr::Variable(var) => self.use_var(var),
            Expr::Block(b) => {
                for stmt in &b.statements {
                    self.translate_stmt(stmt);
                }
                if let Some(ref expr) = b.expression {
                    return self.translate_expr(expr);
                } else {
                    return MValue::owned(self.const_nil());
                }
            }
            Expr::Bool(b) => MValue::owned(self.const_bool(*b)),
            Expr::Number(f) => self.builder.ins().f64const(*f).assert_owned(),
            Expr::String(s) => self
                .string_literal_borrow(s.deref().clone())
                .assert_borrowed(),
            Expr::Nil => self.const_nil().assert_owned(),
            Expr::Op { op, lhs, rhs } => self.translate_op(*op, lhs, rhs),
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => self.translate_if_else(condition, then_branch, else_branch),
            Expr::ForIn { iter, var, body } => self.translate_for_in(iter, var, body),
            Expr::While { condition, body } => self.translate_while_loop(condition, body),
            Expr::Call { callee, args } => self.translate_indirect_call(callee, args),
            Expr::CallBuiltin { callee, args } => self.translate_builtin_call(callee, args),
            Expr::List(l) => {
                let slot = self.create_stack_slot(VAL64, l.len());
                for (i, item) in l.iter().enumerate() {
                    let val = self.translate_expr(item).into_owned(self);
                    self.builder.ins().stack_store(val, slot, 8 * i as i32);
                }
                let len_value = self.builder.ins().iconst(types::I64, l.len() as i64);
                let items_ptr = self.builder.ins().stack_addr(PTR, slot, 0);
                self.call_native(NativeFuncId::ListNew, &[len_value, items_ptr])[0].assert_owned()
            }
            Expr::GetIndex(val, index) => {
                let index = self.translate_expr(index);
                let val = self.translate_expr(val);
                let result = self.call_native(
                    NativeFuncId::ValBorrowIndex,
                    &[val.borrow(), index.borrow()],
                )[0];
                if val.owned {
                    // result is a borrowed reference, so if we drop val first then result becomes invalid.
                    let result_owned = self.clone_val64_sometimes(result.as_bvalue());
                    self.drop_val64(index);
                    self.drop_val64(val);
                    result_owned.assert_owned()
                } else {
                    // if val is borrowed, we can safely return the borrowed result.
                    self.drop_val64(index);
                    result.assert_borrowed()
                }
            }
            Expr::Dict(d) => {
                let len_value = self.builder.ins().iconst(types::I64, d.len() as i64);
                let mut dict = self.call_native(NativeFuncId::DictNew, &[len_value])[0];
                for (key, value) in d {
                    let key = self.translate_expr(key).into_owned(self);
                    let value = self.translate_expr(value).into_owned(self);
                    dict = self.call_native(NativeFuncId::DictInsert, &[dict, key, value])[0];
                }
                dict.assert_owned()
            }
            Expr::Lambda(l) => self.translate_lambda_definition(l).assert_owned(),
            Expr::Match(m) => self.translate_match_expr(m),
            Expr::Import { path } => self.translate_import(path, &expr.span),
        }
    }

    fn translate_return_ok(&mut self, val: Value) {
        let null_ptr = self.builder.ins().iconst(PTR, 0);
        self.builder.ins().jump(
            self.return_block,
            &[BlockArg::Value(val), BlockArg::Value(null_ptr)],
        );
    }

    fn translate_return_err(&mut self, err_ptr: Value) {
        let nil = self.const_nil();
        self.builder.ins().jump(
            self.return_block,
            &[BlockArg::Value(nil), BlockArg::Value(err_ptr)],
        );
    }

    fn translate_return_block(&mut self, repl_mode: bool) {
        self.builder.switch_to_block(self.return_block);
        self.builder.seal_block(self.return_block);
        self.builder.append_block_param(self.return_block, VAL64);
        self.builder.append_block_param(self.return_block, PTR); // error pointer

        let return_value = self.builder.block_params(self.return_block)[0];
        // TODO: Separate block for error return
        let error_ptr = self.builder.block_params(self.return_block)[1]; // pointer to error or null
        let error_return_ptr = self.get_error_return_ptr(); // return area pointer to store error pointer
        self.builder
            .ins()
            .store(MemFlags::new(), error_ptr, error_return_ptr, 0);

        if repl_mode {
            let variables = self
                .variables
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect::<Vec<_>>();
            let capacity = self
                .builder
                .ins()
                .iconst(types::I64, variables.len() as i64);
            let mut return_dict = self.call_native(NativeFuncId::DictNew, &[capacity])[0];
            for (name, var) in variables {
                let val = self.builder.use_var(var.var);
                let key = self.string_literal_owned(name);
                return_dict =
                    self.call_native(NativeFuncId::DictInsert, &[return_dict, key, val])[0];
            }
            let return_key = self.string_literal_owned("<returnval>".to_string());
            return_dict = self.call_native(
                NativeFuncId::DictInsert,
                &[return_dict, return_key, return_value],
            )[0];
            self.builder.ins().return_(&[return_dict]);
        } else {
            let variables = self.variables.values().cloned().collect::<Vec<_>>();
            for var in variables {
                if var.owned {
                    let val = self.builder.use_var(var.var);
                    self.drop_val64(val.assert_owned());
                }
            }
            self.builder.ins().return_(&[return_value]);
        }
    }

    fn translate_indirect_call(&mut self, callee: &SpannedExpr, args: &Vec<SpannedExpr>) -> MValue {
        let callee_val = self.translate_expr(callee);

        let (func_ptr, closure_ptr) = self.get_lambda_details(args, callee_val.borrow());

        // Null pointer means callee wasn't a lambda
        self.assert(func_ptr, &callee.span, |s| {
            s.string_template(
                "Tried to call something that isn't a function (or wrong parameter count): {}",
                &[callee_val.borrow()],
            )
        });
        let mut sig = self.module.make_signature();
        Self::build_function_signature(&mut sig, args.len());
        let sig_ref = self.builder.import_signature(sig);

        let mut arg_values = Vec::new();
        arg_values.push(self.get_vm_ptr());
        arg_values.push(closure_ptr);
        let err_slot = self.create_stack_slot(I64, 1);
        let err_ptr = self.builder.ins().stack_addr(I64, err_slot, 0);
        arg_values.push(err_ptr);
        for arg in args {
            arg_values.push(self.translate_expr(arg).into_owned(self))
        }
        // Add the function itself as a parameter, to support recursion.
        arg_values.push(callee_val.borrow());
        let call = self
            .builder
            .ins()
            .call_indirect(sig_ref, func_ptr, &arg_values);

        self.drop_val64(callee_val);
        let call_result = self.builder.inst_results(call)[0];
        let err_val = self.builder.ins().stack_load(I64, err_slot, 0);
        self.assert_func_ret_ok(err_val, &callee.span);

        return call_result.assert_owned();
    }

    fn get_vm_ptr(&mut self) -> Value {
        self.builder.block_params(self.entry_block)[0]
    }

    fn get_error_return_ptr(&mut self) -> Value {
        self.builder.block_params(self.entry_block)[2]
    }

    fn build_function_signature(sig: &mut Signature, param_count: usize) {
        // VM context
        sig.params
            .push(AbiParam::special(PTR, ir::ArgumentPurpose::VMContext));
        // Closure pointer
        sig.params.push(AbiParam::new(I64));
        // Error return pointer
        sig.params.push(AbiParam::new(PTR));
        // User params
        for _ in 0..param_count {
            sig.params.push(AbiParam::new(VAL64));
        }
        // Self for recursion
        sig.params.push(AbiParam::new(VAL64));
        // Return value
        sig.returns.push(AbiParam::new(VAL64));
    }

    fn get_lambda_details(
        &mut self,
        args: &Vec<Spanned<Expr>>,
        callee_val: BValue,
    ) -> (Value, Value) {
        let closure_ptr_slot = self.create_stack_slot(PTR, 1);
        let closure_ptr_ptr_val = self.builder.ins().stack_addr(I64, closure_ptr_slot, 0);
        let arg_count_val = self.builder.ins().iconst(types::I64, args.len() as i64);
        let func_ptr = self.call_native(
            NativeFuncId::ValGetLambdaDetails,
            &[callee_val, arg_count_val, closure_ptr_ptr_val],
        )[0];
        let closure_ptr = self.builder.ins().stack_load(I64, closure_ptr_slot, 0);
        (func_ptr, closure_ptr)
    }

    fn translate_builtin_call(&mut self, callee: &str, args: &Vec<SpannedExpr>) -> MValue {
        fn bitwise_op(
            s: &mut FunctionTranslator,
            args: &[SpannedExpr],
            op: impl FnOnce(&mut FunctionTranslator, Value, Value) -> Value,
        ) -> MValue {
            assert!(args.len() == 2);
            let lhs = s.translate_expr(&args[0]);
            let rhs = s.translate_expr(&args[1]);
            s.guard_f64(lhs, &args[0].span);
            s.guard_f64(rhs, &args[1].span);
            let lhs2 = s.builder.ins().fcvt_to_sint_sat(types::I64, lhs.borrow());
            let rhs2 = s.builder.ins().fcvt_to_sint_sat(types::I64, rhs.borrow());
            let result = op(s, lhs2, rhs2);
            s.builder
                .ins()
                .fcvt_from_sint(types::F64, result)
                .assert_owned()
        }
        match callee {
            "bitwiseAnd" => bitwise_op(self, args, |s, lhs, rhs| s.builder.ins().band(lhs, rhs)),
            "bitwiseOr" => bitwise_op(self, args, |s, lhs, rhs| s.builder.ins().bor(lhs, rhs)),
            "bitwiseXor" => bitwise_op(self, args, |s, lhs, rhs| s.builder.ins().bxor(lhs, rhs)),
            "floor" => {
                assert!(args.len() == 1);
                let val = self.translate_expr(&args[0]);
                self.guard_f64(val, &args[0].span);
                self.builder.ins().floor(val.borrow()).assert_owned()
            }
            _ => {
                if let Some(func) = self.builtins_map.get(callee) {
                    return self.call_native_eval(&func, args).assert_owned();
                }
                panic!("ERROR: Native function not implemented ({})\n", callee);
            }
        }
    }

    /// Wraps [Self::call_native] and evaluates arguments
    fn call_native_eval(&mut self, func: &NativeFunc, args: &Vec<SpannedExpr>) -> Value {
        let mut arg_values = Vec::new();
        for arg in args {
            arg_values.push(self.translate_expr(arg).into_owned(self))
        }

        match &self.call_native_func_fallible(func, &arg_values)[..] {
            [val] => *val,
            [] => self.const_nil(),
            _ => panic!("Built-in functions should only return zero or one value"),
        }
    }

    fn translate_stmt(&mut self, stmt: &SpannedStatement) {
        match &stmt.value {
            Statement::Expression(e) => {
                self.translate_expr(e);
            }
            Statement::Return(e) => {
                let return_value = self.translate_expr(e).into_owned(self);
                self.translate_return_ok(return_value);

                // Create a new block for after the return instruction, so that other instructions
                // can still be added after it. Normally, Cranelift rejects instructions after
                // return, but we need them to keep the code simple.
                let after_return_block = self.builder.create_block();
                self.builder.switch_to_block(after_return_block);
                self.builder.seal_block(after_return_block);
            }
            Statement::PatternAssignment(pattern, rhs) => {
                self.translate_pattern_assignment(rhs, pattern);
            }
        }
    }

    fn translate_pattern_assignment(
        &mut self,
        rhs: &Spanned<Expr>,
        pattern: &Spanned<MatchPattern>,
    ) {
        let rhs_value = self.translate_expr(rhs);
        self.translate_match_pattern(&pattern, rhs_value);
    }

    fn translate_match_pattern(&mut self, pattern: &Spanned<MatchPattern>, value: MValue) {
        self.set_src_span(&pattern.span);
        match &pattern.value {
            MatchPattern::Discard => {
                self.drop_val64(value);
            }
            MatchPattern::Declaration(var_ref, _) => {
                // TODO add replace_var method?
                let old_value = self.use_var(var_ref).borrow().assert_owned();
                self.drop_val64(old_value);
                self.def_var(var_ref, value);
            }
            MatchPattern::Assignment(target) => {
                let owned_value = value.into_owned(self);
                self.translate_assignment(target, owned_value);
            }
            MatchPattern::Constant(c) => {
                let matches = self.translate_matches_constant(c, value.borrow());
                self.assert(matches, &pattern.span, |s| {
                    let template =
                        format!("Pattern match failed: Expected constant {}, found {{}}", c);
                    s.string_template(template, &[value.borrow()])
                });
                self.drop_val64(value);
            }
            MatchPattern::SimpleList(parts) => {
                let is_list = self.is_ptr_type(value, PointerTag::List);
                self.assert(is_list, &pattern.span, |s| {
                    s.string_template(
                        "Pattern match failed: Expected list, found {}",
                        &[value.borrow()],
                    )
                });
                let value_len = self.call_native(NativeFuncId::ListLenU64, &[value.borrow()])[0];
                let len_eq =
                    self.builder
                        .ins()
                        .icmp_imm(IntCC::Equal, value_len, parts.len() as i64);
                self.assert(len_eq, &pattern.span, |s| {
                    let template = format!(
                        "Pattern match failed: Expected list of length {}, found {{}}",
                        parts.len()
                    );
                    s.string_template(template, &[value.borrow()])
                });
                for (i, part) in parts.iter().enumerate() {
                    let ival = self.builder.ins().iconst(I64, i as i64);
                    let element =
                        self.call_native(NativeFuncId::ListGetU64, &[value.borrow(), ival])[0];
                    self.translate_match_pattern(part, element.assert_owned());
                }
                self.drop_val64(value);
            }
            MatchPattern::SpreadList(_) => {
                todo!("Pattern matching with ... isn't supported here yet")
            }
            MatchPattern::Dict(dict) => {
                for (key, pattern) in &dict.entries {
                    if pattern.value == MatchPattern::Discard {
                        continue;
                    }
                    // TODO: assert dict type
                    let keyval = self.string_literal_borrow(key.clone());
                    let (element, found) = self.dict_lookup(value, keyval);
                    self.assert(found, &pattern.span, |s| {
                        s.string_literal_owned(format!(
                            "Pattern match failed: Dict key {key} was not found\n",
                        ))
                    });
                    self.translate_match_pattern(pattern, element.assert_owned());
                }
                self.drop_val64(value);
            }
        }
    }

    fn translate_matches_pattern(&mut self, pattern: &MatchPattern, value: BValue) -> Value {
        // value is borrowed
        let true_val = self.builder.ins().iconst(types::I8, 1);
        let false_val = self.builder.ins().iconst(types::I8, 0);
        match pattern {
            MatchPattern::Discard => true_val,
            MatchPattern::Declaration(_, _) => true_val,
            MatchPattern::Assignment(_) => true_val,
            MatchPattern::Constant(c) => self.translate_matches_constant(c, value),
            MatchPattern::SimpleList(parts) => {
                let block0 = self.builder.create_block();
                let block1 = self.builder.create_block();
                let merge_block = self.builder.create_block();
                self.builder.append_block_param(merge_block, types::I8);

                // Return false if not a list
                let is_list = self.is_ptr_type(value, PointerTag::List);
                self.builder.ins().brif(
                    is_list,
                    block0,
                    &[],
                    merge_block,
                    &[BlockArg::Value(false_val)],
                );
                self.builder.switch_to_block(block0);
                self.builder.seal_block(block0);

                // Return false if length doesn't match
                let value_len = self.call_native(NativeFuncId::ListLenU64, &[value])[0];
                let len_eq =
                    self.builder
                        .ins()
                        .icmp_imm(IntCC::Equal, value_len, parts.len() as i64);

                self.builder.ins().brif(
                    len_eq,
                    block1,
                    &[],
                    merge_block,
                    &[BlockArg::Value(false_val)],
                );
                self.builder.switch_to_block(block1);
                self.builder.seal_block(block1);
                let mut matches_all = self.builder.ins().iconst(types::I8, 1);
                for (i, part) in parts.iter().enumerate() {
                    if pattern.always_matches() {
                        // skip simple patterns that will always be true (e.g. _)
                        continue;
                    }
                    let ival = self.builder.ins().iconst(I64, i as i64);
                    let element = self.call_native(NativeFuncId::ListBorrowU64, &[value, ival])[0];
                    let matches = self.translate_matches_pattern(&part.value, element);
                    // TODO short-circuit
                    matches_all = self.builder.ins().band(matches_all, matches);
                }
                self.builder
                    .ins()
                    .jump(merge_block, &[BlockArg::Value(matches_all)]);

                self.builder.switch_to_block(merge_block);
                self.builder.seal_block(merge_block);
                let phi = self.builder.block_params(merge_block)[0];
                phi
            }
            MatchPattern::SpreadList(_) => {
                todo!("Pattern matching with ... isn't supported here yet")
            }
            MatchPattern::Dict(dict_pattern) => {
                let block0 = self.builder.create_block();
                // let block1 = self.builder.create_block();
                let merge_block = self.builder.create_block();
                self.builder.append_block_param(merge_block, types::I8);

                // Return false if not a dict
                let is_dict = self.is_ptr_type(value, PointerTag::Dict);
                self.builder.ins().brif(
                    is_dict,
                    block0,
                    &[],
                    merge_block,
                    &[BlockArg::Value(false_val)],
                );
                self.builder.switch_to_block(block0);
                self.builder.seal_block(block0);

                let mut matches_all = self.builder.ins().iconst(types::I8, 1);
                for (key, pattern) in &dict_pattern.entries {
                    if pattern.value.always_matches() {
                        // skip simple patterns that will always be true (e.g. _)
                        continue;
                    }
                    let keyval = self.string_literal_borrow(key.clone());

                    let (element, found) = self.dict_lookup(value, keyval);
                    let matches = self.translate_matches_pattern(&pattern.value, element);
                    self.drop_val64(element.assert_owned());
                    // TODO short-circuit
                    matches_all = self.builder.ins().band(matches_all, found);
                    matches_all = self.builder.ins().band(matches_all, matches);
                }
                self.builder
                    .ins()
                    .jump(merge_block, &[BlockArg::Value(matches_all)]);

                self.builder.switch_to_block(merge_block);
                self.builder.seal_block(merge_block);
                let phi = self.builder.block_params(merge_block)[0];
                phi
            }
        }
    }

    fn translate_matches_constant(&mut self, c: &MatchConstant, value: BValue) -> Value {
        match c {
            MatchConstant::Nil => self.is_nil(value),
            MatchConstant::Bool(b) => {
                let expected_bits = if *b {
                    value64::TRUE_VALUE
                } else {
                    value64::FALSE_VALUE
                };
                self.cmp_bits_imm(IntCC::Equal, value, expected_bits)
            }
            MatchConstant::Number(f) => self.cmp_bits_imm(IntCC::Equal, value, (*f).to_bits()),
            MatchConstant::String(s) => {
                let comp_val = self.string_literal_borrow(s.clone());
                self.call_native(NativeFuncId::ValEqU8, &[comp_val, value])[0]
            }
        }
    }

    /// Returns (element: OValue, found: u8)
    fn dict_lookup(&mut self, dict: impl AsBValue, key: impl AsBValue) -> (Value, Value) {
        let found_slot = self.create_stack_slot(types::I8, 1);
        let found_ptr = self.builder.ins().stack_addr(types::I64, found_slot, 0);
        let element = self.call_native(
            NativeFuncId::DictLookup,
            &[dict.as_bvalue(), key.as_bvalue(), found_ptr],
        )[0];
        let found = self.builder.ins().stack_load(types::I8, found_slot, 0);
        (element, found)
    }

    fn translate_assignment(&mut self, target: &ast::AssignmentTarget, rhs: OValue) {
        let mut path_values = vec![];
        for index in &target.path {
            let value = self.translate_expr(index).into_owned(self);
            path_values.push(value);
        }
        let variable = self
            .variables
            .get(&target.var.name)
            .copied()
            .expect("variable not defined");
        match &path_values[..] {
            [] => {
                let old_val = self.use_var(&target.var);
                self.drop_val64(old_val);
                self.def_var(&target.var, rhs.assert_owned());
            }
            [index, rest @ ..] => {
                // "move" the value to update it, then write it back.
                // We don't need to explicitly replace the value with NIL because we overwrite it later.
                let old_val = self.builder.use_var(variable.var);
                let new_val = self.assign_part(old_val, *index, rest, rhs);
                self.builder.def_var(variable.var, new_val);
            }
        };
    }

    fn assign_part(
        &mut self,
        val: OValue,
        index: OValue,
        rest_path: &[OValue],
        rhs: OValue,
    ) -> Value {
        match rest_path {
            [] => {
                return self.call_native(NativeFuncId::ValSetIndex, &[val, index, rhs])[0];
            }
            [next_index, rest @ ..] => {
                let old_part_slot = self.create_stack_slot(VAL64, 1);
                // For some reason if we don't store something in the slot, we occasionally get
                // STATUS_ACCESS_VIOLATION when writing to this pointer from Rust?
                self.builder.ins().stack_store(rhs, old_part_slot, 0);
                let old_part_ptr = self.builder.ins().stack_addr(PTR, old_part_slot, 0);
                let val = self.call_native_fallible(
                    NativeFuncId::ValReplaceIndex,
                    &[val, index, old_part_ptr],
                )[0];
                let old_part = self.builder.ins().stack_load(VAL64, old_part_slot, 0);
                let new_part = self.assign_part(old_part, *next_index, rest, rhs);
                return self.call_native(NativeFuncId::ValSetIndex, &[val, index, new_part])[0];
            }
        }
    }

    fn translate_cmp(&mut self, cmp: FloatCC, lhs: &SpannedExpr, rhs: &SpannedExpr) -> MValue {
        let lhs = self.translate_expr(lhs);
        let rhs = self.translate_expr(rhs);
        let cmp_val = self.builder.ins().fcmp(cmp, lhs.borrow(), rhs.borrow());
        self.bool_to_val64(cmp_val).assert_owned()
    }

    fn translate_if_else(
        &mut self,
        condition: &SpannedExpr,
        then_body: &SpannedExpr,
        else_body: &Option<Box<SpannedExpr>>,
    ) -> MValue {
        let condition_value = self.translate_expr(condition);
        let condition_u8 = self.is_truthy(condition_value);
        self.drop_val64(condition_value);

        let then_block = self.builder.create_block();
        let else_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        // If-else constructs in the toy language have a return value.
        // In traditional SSA form, this would produce a PHI between
        // the then and else bodies. Cranelift uses block parameters,
        // so set up a parameter in the merge block, and we'll pass
        // the return values to it from the branches.
        self.builder.append_block_param(merge_block, VAL64);

        // Test the if condition and conditionally branch.
        self.builder
            .ins()
            .brif(condition_u8, then_block, &[], else_block, &[]);

        self.builder.switch_to_block(then_block);
        self.builder.seal_block(then_block);
        let then_return = self.translate_expr(then_body).into_owned(self);

        // Jump to the merge block, passing it the block return value.
        self.builder
            .ins()
            .jump(merge_block, &[BlockArg::Value(then_return)]);

        self.builder.switch_to_block(else_block);
        self.builder.seal_block(else_block);
        let mut else_return = self.const_nil();
        if let Some(expr) = else_body {
            else_return = self.translate_expr(expr).into_owned(self);
        }

        // Jump to the merge block, passing it the block return value.
        self.builder
            .ins()
            .jump(merge_block, &[BlockArg::Value(else_return)]);

        // Switch to the merge block for subsequent statements.
        self.builder.switch_to_block(merge_block);

        // We've now seen all the predecessors of the merge block.
        self.builder.seal_block(merge_block);

        // Read the value of the if-else by reading the merge block
        // parameter.
        let phi = self.builder.block_params(merge_block)[0];

        phi.assert_owned()
    }

    fn translate_for_in(
        &mut self,
        iter: &SpannedExpr,
        var: &Spanned<MatchPattern>,
        body: &SpannedExpr,
    ) -> MValue {
        let header_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let exit_block = self.builder.create_block();

        let iter_value = self.translate_expr(iter);
        let len_value = self.call_native(NativeFuncId::ListLenU64, &[iter_value.borrow()])[0];
        let initial_index = self.builder.ins().iconst(I64, 0);
        self.builder
            .ins()
            .jump(header_block, &[BlockArg::Value(initial_index)]);

        // HEADER BLOCK
        self.set_src_span(&iter.span);
        self.builder.append_block_param(header_block, I64); // Use block param for list index
        self.builder.switch_to_block(header_block);

        let current_index = self.builder.block_params(header_block)[0];
        let should_continue =
            self.builder
                .ins()
                .icmp(IntCC::UnsignedLessThan, current_index, len_value);
        self.builder.ins().brif(
            should_continue,
            body_block,
            &[BlockArg::Value(current_index)],
            exit_block,
            &[],
        );

        // BODY BLOCK
        self.set_src_span(&body.span);
        self.builder.append_block_param(body_block, I64);
        self.builder.switch_to_block(body_block);
        self.builder.seal_block(body_block);

        let current_index = self.builder.block_params(header_block)[0];
        let current_item = self.call_native(
            NativeFuncId::ListGetU64,
            &[iter_value.borrow(), current_index],
        )[0]
        .assert_owned();
        self.translate_match_pattern(&var, current_item);
        self.translate_expr(body);
        let next_index = self.builder.ins().iadd_imm(current_index, 1);
        self.builder
            .ins()
            .jump(header_block, &[BlockArg::Value(next_index)]);

        // EXIT BLOCK
        self.builder.switch_to_block(exit_block);
        self.drop_val64(iter_value);

        // We've reached the bottom of the loop, so there will be no
        // more backedges to the header to exits to the bottom.
        self.builder.seal_block(header_block);
        self.builder.seal_block(exit_block);

        self.const_nil().assert_owned()
    }

    fn translate_while_loop(&mut self, condition: &SpannedExpr, body: &SpannedExpr) -> MValue {
        let header_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let exit_block = self.builder.create_block();

        self.builder.ins().jump(header_block, &[]);
        self.builder.switch_to_block(header_block);

        let condition_value = self.translate_expr(condition);
        let condition_bool = self.is_truthy(condition_value);
        self.builder
            .ins()
            .brif(condition_bool, body_block, &[], exit_block, &[]);

        self.builder.switch_to_block(body_block);
        self.builder.seal_block(body_block);

        self.translate_expr(body);
        self.builder.ins().jump(header_block, &[]);

        self.builder.switch_to_block(exit_block);

        // We've reached the bottom of the loop, so there will be no
        // more backedges to the header to exits to the bottom.
        self.builder.seal_block(header_block);
        self.builder.seal_block(exit_block);

        self.const_nil().assert_owned()
    }

    fn translate_op(&mut self, op: Opcode, lhs: &SpannedExpr, rhs: &SpannedExpr) -> MValue {
        fn eval_f64(s: &mut FunctionTranslator, expr: &SpannedExpr) -> Value {
            let result = s.translate_expr(expr).borrow();
            s.guard_f64(result, &expr.span);
            result
        }
        match op {
            Opcode::Add => {
                let lhs = eval_f64(self, lhs);
                let rhs = eval_f64(self, rhs);
                self.builder.ins().fadd(lhs, rhs).assert_owned()
            }
            Opcode::Sub => {
                let lhs = eval_f64(self, lhs);
                let rhs = eval_f64(self, rhs);
                self.builder.ins().fsub(lhs, rhs).assert_owned()
            }
            Opcode::Mul => {
                let lhs = eval_f64(self, lhs);
                let rhs = eval_f64(self, rhs);
                self.builder.ins().fmul(lhs, rhs).assert_owned()
            }
            Opcode::Div => {
                let lhs = eval_f64(self, lhs);
                let rhs = eval_f64(self, rhs);
                self.builder.ins().fdiv(lhs, rhs).assert_owned()
            }
            Opcode::Eq => {
                let lhs_val = self.translate_expr(lhs);
                let rhs_val = self.translate_expr(rhs);
                let result =
                    self.call_native(NativeFuncId::ValEq, &[lhs_val.borrow(), rhs_val.borrow()])[0];
                self.drop_val64(lhs_val);
                self.drop_val64(rhs_val);
                result.assert_owned()
            }
            Opcode::Neq => {
                let lhs_val = self.translate_expr(lhs);
                let rhs_val = self.translate_expr(rhs);
                let eq_result = self
                    .call_native(NativeFuncId::ValEqU8, &[lhs_val.borrow(), rhs_val.borrow()])[0];
                self.drop_val64(lhs_val);
                self.drop_val64(rhs_val);
                let neq_result = self.builder.ins().bxor_imm(eq_result, 1);
                self.bool_to_val64(neq_result).assert_owned()
            }
            Opcode::Lt => self.translate_cmp(FloatCC::LessThan, lhs, rhs),
            Opcode::Lte => self.translate_cmp(FloatCC::LessThanOrEqual, lhs, rhs),
            Opcode::Gt => self.translate_cmp(FloatCC::GreaterThan, lhs, rhs),
            Opcode::Gte => self.translate_cmp(FloatCC::GreaterThanOrEqual, lhs, rhs),
            Opcode::And => {
                let lhs_val = self.translate_expr(lhs);
                let lhs_truthy = self.is_truthy(lhs_val);
                self.drop_val64(lhs_val);
                let false_u8 = self.builder.ins().iconst(types::I8, 0);
                let bool_result = self.eval_if(
                    lhs_truthy,
                    // If LHS truthy, evaluate RHS
                    |s| {
                        let rhs_val = s.translate_expr(rhs);
                        let rhs_truthy = s.is_truthy(rhs_val);
                        s.drop_val64(rhs_val);
                        rhs_truthy
                    },
                    false_u8,
                );
                self.bool_to_val64(bool_result).assert_owned()
            }
            Opcode::Or => {
                let lhs_val = self.translate_expr(lhs);
                let lhs_truthy = self.is_truthy(lhs_val);
                let lhs_not_truthy = self.builder.ins().bxor_imm(lhs_truthy, 1);
                self.drop_val64(lhs_val);
                let true_u8 = self.builder.ins().iconst(types::I8, 1);
                let bool_result = self.eval_if(
                    lhs_not_truthy,
                    // If LHS falsy, evaluate RHS
                    |s| {
                        let rhs_val = s.translate_expr(rhs);
                        let rhs_truthy = s.is_truthy(rhs_val);
                        s.drop_val64(rhs_val);
                        rhs_truthy
                    },
                    true_u8,
                );
                self.bool_to_val64(bool_result).assert_owned()
            }
            Opcode::Concat => {
                let lhs = self.translate_expr(lhs).into_owned(self);
                let rhs = self.translate_expr(rhs).into_owned(self);
                self.call_native(NativeFuncId::ValConcat, &[lhs, rhs])[0].assert_owned()
            }
        }
    }

    fn translate_lambda_definition(&mut self, lambda: &LambdaExpr) -> OValue {
        let mut ctx = self.module.make_context();
        ctx.func.collect_debug_info();

        Self::build_function_signature(&mut ctx.func.signature, lambda.params.len());

        let mut builder_context = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_context);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let variables = {
            let mut variable_builder = VariableBuilder::new(&mut builder);
            variable_builder.declare_variables_in_func(lambda, entry_block);
            variable_builder.variables
        };

        let return_block = builder.create_block();
        let mut trans = FunctionTranslator {
            src_path: &self.src_path,
            builder,
            variables,
            module: &mut self.module,
            string_constants: &mut self.string_constants,
            return_block,
            entry_block,
            builtins_map: &self.builtins_map,
            natives_map: &self.natives_map,
        };
        trans.translate_function_entry(lambda, entry_block);

        let return_value = trans.translate_expr(&lambda.body).into_owned(&mut trans);

        // let null_ptr = self.builder.ins().iconst(PTR, 8);
        // trans.builder.ins().jump(
        //     trans.return_block,
        //     &[BlockArg::Value(return_value), BlockArg::Value(null_ptr)],
        // );
        trans.translate_return_ok(return_value);

        trans.translate_return_block(false);
        // println!("func {:?}: {:?}", &lambda.name, &trans.builder.func);
        trans.builder.finalize();

        // Using a name here (instead of declare_anonymous_function) for profiling info on Linux.
        let func_name = format!(
            "USER:{}:{}:{}",
            lambda.name.as_deref().unwrap_or("lambda"),
            self.src_path.to_str().unwrap(),
            lambda.body.span.start,
        );
        let func_id = self
            .module
            .declare_function(&func_name, Linkage::Export, &ctx.func.signature)
            .expect("Error declaring function");

        self.module
            .define_function(func_id, &mut ctx)
            .expect("Error defining function");
        self.module.clear_context(&mut ctx);

        self.module.finalize_definitions().unwrap();

        let func_ptr = self.module.get_finalized_function(func_id);
        let param_count_val = self.builder.ins().iconst(I64, lambda.params.len() as i64);
        let func_ptr_val = self.builder.ins().iconst(PTR, func_ptr as i64);
        let capture_count_val = self
            .builder
            .ins()
            .iconst(I64, lambda.potential_captures.len() as i64);
        let closure_slot = self.create_stack_slot(VAL64, lambda.potential_captures.len());
        for (i, var_ref) in lambda.potential_captures.iter().enumerate() {
            let var = *self.variables.get(&var_ref.name).unwrap();
            let var_val = self.builder.use_var(var.var);
            self.builder
                .ins()
                .stack_store(var_val, closure_slot, (i * 8) as i32);
        }
        let closure_ptr = self.builder.ins().stack_addr(PTR, closure_slot, 0);
        let name_val = if let Some(name) = &lambda.name {
            self.string_literal_borrow(name.clone())
        } else {
            self.const_nil()
        };
        // let name_val = self.const_nil();
        let lambda_val = self.call_native(
            NativeFuncId::ConstructLambda,
            &[
                param_count_val,
                name_val,
                func_ptr_val,
                capture_count_val,
                closure_ptr,
            ],
        )[0];
        return lambda_val;
    }

    fn translate_match_expr(&mut self, match_expr: &MatchExpr) -> MValue {
        let merge_block = self.builder.create_block();
        self.builder.append_block_param(merge_block, VAL64);

        let mut branch_check_blocks = vec![];
        let mut branch_body_blocks = vec![];
        for _ in &match_expr.branches {
            let check_block = self.builder.create_block();
            let body_block = self.builder.create_block();
            branch_check_blocks.push(check_block);
            branch_body_blocks.push(body_block);
        }

        let subject = self.translate_expr(&match_expr.condition);
        self.builder.ins().jump(branch_check_blocks[0], &[]);

        for (i, (pattern, body)) in match_expr.branches.iter().enumerate() {
            let is_last_branch = i == match_expr.branches.len() - 1;
            self.builder.switch_to_block(branch_check_blocks[i]);
            self.set_src_span(&pattern.span);
            let matches = self.translate_matches_pattern(&pattern.value, subject.borrow());
            if is_last_branch {
                self.assert(matches, &match_expr.condition.span, |s| {
                    s.string_template(
                        "No branches matched in match expression. Value: {}",
                        &[subject.borrow()],
                    )
                });
                self.builder.ins().jump(branch_body_blocks[i], &[]);
            } else {
                self.builder.ins().brif(
                    matches,
                    branch_body_blocks[i],
                    &[],
                    branch_check_blocks[i + 1],
                    &[],
                );
            };

            // Branch body
            self.builder.switch_to_block(branch_body_blocks[i]);
            self.translate_match_pattern(pattern, subject);
            let body_value = self.translate_expr(body).into_owned(self);
            self.builder
                .ins()
                .jump(merge_block, &[BlockArg::Value(body_value)]);
        }

        // Seal all branch blocks
        for block in branch_check_blocks {
            self.builder.seal_block(block);
        }
        for block in branch_body_blocks {
            self.builder.seal_block(block);
        }

        // Get value from merge block
        self.builder.switch_to_block(merge_block);
        self.builder.seal_block(merge_block);
        let phi = self.builder.block_params(merge_block)[0];
        phi.assert_owned()
    }

    fn call_native_func(&mut self, func: &NativeFunc, args: &[Value]) -> &[Value] {
        let local_callee = self
            .module
            .declare_func_in_func(func.func, self.builder.func);
        let call = self.builder.ins().call(local_callee, args);
        self.builder.inst_results(call)
    }

    fn call_native(&mut self, func: NativeFuncId, args: &[Value]) -> &[Value] {
        let func_def = self.natives_map.get(&func).unwrap();
        self.call_native_func(func_def, args)
    }

    fn call_native_func_fallible(&mut self, func: &NativeFunc, user_args: &[Value]) -> Vec<Value> {
        let expected_arg_count = func.expected_arg_count();
        assert_eq!(
            expected_arg_count,
            user_args.len(),
            "ERROR: Native function called with wrong number of arguments\n"
        );

        let mut arg_values = Vec::new();
        let err_slot = self.create_stack_slot(I64, 1);
        if func.fallible {
            let err_ptr = self.builder.ins().stack_addr(I64, err_slot, 0);
            arg_values.push(err_ptr);
        }
        if func.needs_vm {
            arg_values.push(self.get_vm_ptr());
        }
        arg_values.extend_from_slice(user_args);
        let result: Vec<_> = self.call_native_func(func, &arg_values).to_vec();
        if func.fallible {
            let err_val = self.builder.ins().stack_load(I64, err_slot, 0);
            self.assert_func_ret_ok(err_val, &Span::default()); // TODO span
        }
        return result;
    }

    fn call_native_fallible(&mut self, func: NativeFuncId, args: &[Value]) -> Vec<Value> {
        let func_def = self.natives_map.get(&func).unwrap();
        self.call_native_func_fallible(func_def, args)
    }

    fn create_stack_slot(&mut self, ty: Type, count: usize) -> StackSlot {
        self.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            ty.bytes() * (count as u32),
            8, // align by 2^8 = 64
        ))
    }

    /// Asserts that the given condition is non-zero, otherwise builds and raises an error using the given build_msg closure.
    fn assert(
        &mut self,
        assertion: Value,
        span: &Span,
        build_msg: impl FnOnce(&mut Self) -> Value,
    ) {
        let before_block = self.builder.current_block().unwrap();
        let fail_block = self.builder.create_block();
        let ok_block = self.builder.create_block();
        self.builder.set_cold_block(fail_block);
        self.builder
            .ins()
            .brif(assertion, ok_block, &[], fail_block, &[]);

        self.builder.switch_to_block(fail_block);
        self.builder.seal_block(fail_block);
        let msg = build_msg(self);
        let null_ptr = self.builder.ins().iconst(PTR, 0);
        let err_ptr = self.alloc_error(msg, null_ptr, span);
        self.translate_return_err(err_ptr);

        self.builder.switch_to_block(ok_block);
        self.builder.seal_block(ok_block);
        self.builder.insert_block_after(ok_block, before_block);
    }

    fn string_template(&mut self, template: impl Into<String>, parts: &[Value]) -> Value {
        let template_val = self.string_literal_borrow(template.into());
        let parts_count = self.builder.ins().iconst(I64, parts.len() as i64);
        let parts_slot = self.create_stack_slot(VAL64, parts.len());
        for (i, part) in parts.iter().enumerate() {
            self.builder
                .ins()
                .stack_store(*part, parts_slot, (i * 8) as i32);
        }
        let parts_ptr = self.builder.ins().stack_addr(PTR, parts_slot, 0);
        self.call_native(
            NativeFuncId::StringTemplate,
            &[template_val, parts_count, parts_ptr],
        )[0]
    }

    fn assert_func_ret_ok(&mut self, err_val: Value, span: &Span) {
        let is_ok = self.builder.ins().icmp_imm(IntCC::Equal, err_val, 0);

        let before_block = self.builder.current_block().unwrap();
        let fail_block = self.builder.create_block();
        let ok_block = self.builder.create_block();
        self.builder.set_cold_block(fail_block);
        self.builder
            .ins()
            .brif(is_ok, ok_block, &[], fail_block, &[]);

        self.builder.switch_to_block(fail_block);
        self.builder.seal_block(fail_block);

        self.print_string_constant(format!(
            "TEMP ERROR (at {}): Function call failed\n",
            span.start
        ));
        let msg = self.string_literal_owned(String::new());
        let err_ptr = self.alloc_error(msg, err_val, span);
        self.translate_return_err(err_ptr);

        self.builder.switch_to_block(ok_block);
        self.builder.seal_block(ok_block);
        self.builder.insert_block_after(ok_block, before_block);
    }

    fn alloc_error(&mut self, msg: Value, inner: Value, span: &Span) -> Value {
        let pos = self.builder.ins().iconst(I64, span.start as i64);
        self.call_native(NativeFuncId::AllocError, &[msg, inner, pos])[0]
    }

    fn do_if(&mut self, cond: Value, then: impl FnOnce(&mut Self)) {
        let then_block = self.builder.create_block();
        let after_block = self.builder.create_block();
        self.builder
            .ins()
            .brif(cond, then_block, &[], after_block, &[]);

        self.builder.switch_to_block(then_block);
        self.builder.seal_block(then_block);
        then(self);
        self.builder.ins().jump(after_block, &[]);

        self.builder.switch_to_block(after_block);
        self.builder.seal_block(after_block);
    }

    fn eval_if(
        &mut self,
        cond: Value,
        then: impl FnOnce(&mut Self) -> Value,
        fallback: Value,
    ) -> Value {
        let then_block = self.builder.create_block();
        let after_block = self.builder.create_block();
        self.builder.append_block_param(after_block, types::I8); // TODO: Can we make this generic?
        self.builder.ins().brif(
            cond,
            then_block,
            &[],
            after_block,
            &[BlockArg::Value(fallback)],
        );

        self.builder.switch_to_block(then_block);
        self.builder.seal_block(then_block);
        let then_val = then(self);
        self.builder
            .ins()
            .jump(after_block, &[BlockArg::Value(then_val)]);

        self.builder.switch_to_block(after_block);
        self.builder.seal_block(after_block);
        let phi = self.builder.block_params(after_block)[0];
        phi
    }

    fn use_var(&mut self, var_ref: &VarRef) -> MValue {
        let var = *self
            .variables
            .get(&var_ref.name)
            .unwrap_or_else(|| panic!("variable {} not defined", &var_ref.name));
        let val = self.builder.use_var(var.var);
        if !var.owned {
            return val.assert_borrowed();
        }
        if var_ref.is_final {
            // Replace the variable instead of copying, because we know this won't be accessed again.
            let nil = self.const_nil();
            self.builder.def_var(var.var, nil);
            val.assert_owned()
        } else {
            val.assert_borrowed()
        }
    }

    fn def_var(&mut self, var_ref: &VarRef, val: MValue) {
        let var = *self
            .variables
            .get(&var_ref.name)
            .unwrap_or_else(|| panic!("variable {} not defined", &var_ref.name));
        assert!(var.owned, "can't assign values to borrowed variables");
        let val_owned = val.into_owned(self);
        self.builder.def_var(var.var, val_owned); // TODO ?
    }

    fn const_nil(&mut self) -> Value {
        self.const_value64_bits(value64::NIL_VALUE)
    }

    fn const_bool(&mut self, b: bool) -> Value {
        self.const_value64_bits(if b {
            value64::TRUE_VALUE
        } else {
            value64::FALSE_VALUE
        })
    }

    fn const_value64_bits(&mut self, bits: u64) -> Value {
        self.builder.ins().f64const(f64::from_bits(bits))
    }

    fn clone_val64(&mut self, val: Value) -> OValue {
        self.call_native(NativeFuncId::Clone, &[val])[0]
    }

    /// Same as [Self::clone_val64] but only does a function call if the value is a reference type.
    /// This slows things down when used everywhere, should only be used in places where reference types are unlikely.
    #[allow(dead_code)]
    fn clone_val64_sometimes(&mut self, val: BValue) -> OValue {
        let is_ptr = self.is_ptr_val(val);
        let clone_block = self.builder.create_block();
        // Short-circuit if it's not a reference type
        let after_block = self.builder.create_block();

        self.builder
            .ins()
            .brif(is_ptr, clone_block, &[], after_block, &[]);
        self.builder.switch_to_block(clone_block);
        self.builder.seal_block(clone_block);
        self.call_native(NativeFuncId::Clone, &[val]);
        self.builder.ins().jump(after_block, &[]);

        self.builder.switch_to_block(after_block);
        self.builder.seal_block(after_block);
        // TODO should we use phi values to pass the actual return value?
        val
    }

    fn drop_val64(&mut self, val: MValue) {
        if !val.owned {
            return;
        }
        let is_ptr = self.is_ptr_val(val.borrow());
        // Short-circuit if it's not a reference type
        self.do_if(is_ptr, |s| {
            s.call_native(NativeFuncId::Drop, &[val.borrow()]);
        });
    }

    fn set_src_span(&mut self, span: &Span) {
        self.builder
            .set_srcloc(ir::SourceLoc::new(span.start as u32));
    }

    fn is_truthy(&mut self, val: impl AsBValue) -> Value {
        let not_bool_block = self.builder.create_block();
        let merge_block = self.builder.create_block();
        self.builder.append_block_param(merge_block, types::I8);

        let val_bits = self.val_bits(val);
        let is_t = self
            .builder
            .ins()
            .icmp_imm(IntCC::Equal, val_bits, value64::TRUE_VALUE as i64);
        let is_f = self
            .builder
            .ins()
            .icmp_imm(IntCC::Equal, val_bits, value64::FALSE_VALUE as i64);
        let is_bool = self.builder.ins().bor(is_t, is_f);
        self.builder.ins().brif(
            is_bool,
            merge_block,
            &[BlockArg::Value(is_t)],
            not_bool_block,
            &[],
        );

        self.builder.switch_to_block(not_bool_block);
        self.builder.seal_block(not_bool_block);
        let truthy = self.call_native(NativeFuncId::ValTruthy, &[val.as_bvalue()])[0];
        self.builder
            .ins()
            .jump(merge_block, &[BlockArg::Value(truthy)]);

        self.builder.switch_to_block(merge_block);
        self.builder.seal_block(merge_block);
        self.builder.block_params(merge_block)[0]
    }

    fn is_nil(&mut self, val: impl AsBValue) -> Value {
        self.cmp_bits_imm(IntCC::Equal, val.as_bvalue(), value64::NIL_VALUE)
    }

    fn val_bits(&mut self, val: impl AsBValue) -> Value {
        let val = val.as_bvalue();
        assert_eq!(VAL64, self.builder.func.dfg.value_type(val));
        self.builder.ins().bitcast(I64, MemFlags::new(), val)
    }

    fn cmp_bits_imm(&mut self, cond: IntCC, lhs: impl AsBValue, rhs: u64) -> Value {
        let lhs_bits = self.val_bits(lhs);
        self.builder.ins().icmp_imm(cond, lhs_bits, rhs as i64)
    }

    fn bool_to_val64(&mut self, b: Value) -> Value {
        let t = self.const_bool(true);
        let f = self.const_bool(false);
        self.builder.ins().select(b, t, f)
    }

    fn is_ptr_val(&mut self, val: impl AsBValue) -> Value {
        let val_int = self.val_bits(val);
        let and = self
            .builder
            .ins()
            .band_imm(val_int, 0xFFFC000000000000u64 as i64);
        self.builder
            .ins()
            .icmp_imm(IntCC::Equal, and, 0xFFFC000000000000u64 as i64)
    }

    fn is_ptr_type(&mut self, val: impl AsBValue, tag: PointerTag) -> Value {
        let mask = value64::NANISH_MASK as i64;
        let val_int = self.val_bits(val);
        let and = self.builder.ins().band_imm(val_int, mask);
        self.builder
            .ins()
            .icmp_imm(IntCC::Equal, and, value64::pointer_mask(tag) as i64)
    }

    fn guard_f64(&mut self, val: impl AsBValue, span: &Span) {
        let val_int = self.val_bits(val);
        let nanish = value64::NANISH as i64;
        let and = self.builder.ins().band_imm(val_int, nanish);
        let is_f64 = self.builder.ins().icmp_imm(IntCC::NotEqual, and, nanish);
        self.assert(is_f64, span, |s| {
            s.string_template("Expected an f64, found {}", &[val.as_bvalue()])
        });
    }

    fn string_literal_borrow(&mut self, string: String) -> BValue {
        let string_val64 = self
            .string_constants
            .entry(string)
            .or_insert_with_key(|key| Value64::from_string(Rc::new(key.to_string())));
        let bits = string_val64.bits();
        self.const_value64_bits(bits)
    }

    fn string_literal_owned(&mut self, string: impl Into<String>) -> BValue {
        let string_val = self.string_literal_borrow(string.into());
        self.clone_val64(string_val)
    }

    fn print_string_constant(&mut self, string: String) {
        let string_val64 = self.string_literal_owned(string);
        self.call_native(NativeFuncId::Print, &[string_val64]);
    }

    fn translate_import(&mut self, path: &str, span: &Span) -> MValue {
        let vm_ptr = self.get_vm_ptr();
        let path_val = self.string_literal_owned(path.to_string());
        let result = self.call_native(NativeFuncId::ImportModule, &[vm_ptr, path_val])[0];
        let is_ok = self.cmp_bits_imm(IntCC::NotEqual, result, value64::ERROR_VALUE);

        self.assert(is_ok, span, |s| {
            let msg_path = s.string_literal_borrow(path.to_string());
            s.string_template("Import failed for path: {}", &[msg_path])
        });
        result.assert_owned()
    }
}
