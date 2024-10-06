#![allow(dead_code)]

use codegen::ir::{self};
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, FuncId, FuncOrDataId, Linkage, Module, ModuleError};
use std::{
    collections::HashMap,
    mem::{self, size_of},
    ops::Deref,
    rc::Rc,
};
use strum::Display;
use types::I64;

use crate::{
    ast::{
        self, BuiltInFunction, Expr, LambdaExpr, MatchPattern, Opcode, SpannedExpr,
        SpannedStatement, Statement,
    },
    builtins,
    pos::{Span, Spanned},
    value64::{self},
    Value64,
};

type FuncExpr = LambdaExpr;

const VAL64: ir::Type = ir::types::F64;
const PTR: ir::Type = ir::types::I64;

#[derive(Debug, Display)]
pub enum JITError {
    Module(ModuleError),
}

pub type JITResult<T> = Result<T, JITError>;

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
    data_description: DataDescription,

    /// The module, with the jit backend, which manages the JIT'd
    /// functions.
    module: JITModule,

    natives: NativeMethods,

    /// Interned string constants, which are used for string literals.
    string_constants: HashMap<String, Value64>,
}

#[derive(Debug, Clone)]
struct NativeMethod {
    func: FuncId,
    sig: Signature,
}

#[derive(Debug, Clone)]
struct NativeMethods {
    list_new: NativeMethod,
    list_push: NativeMethod,
    list_len_u64: NativeMethod,
    list_get_u64: NativeMethod,
    range: NativeMethod,
    clone: NativeMethod,
    print: NativeMethod,
    len: NativeMethod,
    val_get_index: NativeMethod,
    val_eq: NativeMethod,
    val_truthy: NativeMethod,
    val_shift_left: NativeMethod,
    val_xor: NativeMethod,
    val_not: NativeMethod,
    dict_new: NativeMethod,
    dict_insert: NativeMethod,
    val_set_index: NativeMethod,
    val_get_lambda_details: NativeMethod,
    construct_lambda: NativeMethod,
}

fn make_sig(module: &mut JITModule, params: &[ir::Type], returns: &[ir::Type]) -> ir::Signature {
    let mut sig = module.make_signature();
    for param in params {
        sig.params.push(AbiParam::new(*param));
    }
    for ret in returns {
        sig.returns.push(AbiParam::new(*ret));
    }
    sig
}

fn get_native_methods<'a, 'b>(module: &'b mut JITModule) -> NativeMethods {
    fn make_method(
        module: &mut JITModule,
        name: &str,
        params: &[ir::Type],
        returns: &[ir::Type],
    ) -> NativeMethod {
        let sig = make_sig(module, params, returns);
        let func = module
            .declare_function(name, Linkage::Import, &sig)
            .expect("problem declaring function");
        NativeMethod { func, sig }
    }

    let native_methods = NativeMethods {
        list_new: make_method(module, "NATIVE:list_new", &[I64], &[VAL64]),
        list_push: make_method(module, "NATIVE:list_push", &[VAL64, VAL64], &[VAL64]),
        list_len_u64: make_method(module, "NATIVE:list_len_u64", &[VAL64], &[I64]),
        list_get_u64: make_method(module, "NATIVE:list_get_u64", &[VAL64, I64], &[VAL64]),
        dict_new: make_method(module, "NATIVE:dict_new", &[I64], &[VAL64]),
        dict_insert: make_method(
            module,
            "NATIVE:dict_insert",
            &[VAL64, VAL64, VAL64],
            &[VAL64],
        ),
        val_get_index: make_method(module, "NATIVE:val_get_index", &[VAL64, VAL64], &[VAL64]),
        val_set_index: make_method(
            module,
            "NATIVE:val_set_index",
            &[VAL64, VAL64, VAL64],
            &[VAL64],
        ),
        val_eq: make_method(module, "NATIVE:val_eq", &[VAL64, VAL64], &[VAL64]),
        val_truthy: make_method(module, "NATIVE:val_truthy", &[VAL64], &[types::I8]),
        val_get_lambda_details: make_method(
            module,
            "NATIVE:val_get_lambda_details",
            &[VAL64, types::I32, types::I64],
            &[I64],
        ),
        construct_lambda: make_method(
            module,
            "NATIVE:construct_lambda",
            &[I64, PTR, I64, PTR],
            &[VAL64],
        ),
        val_shift_left: make_method(module, "NATIVE:val_shift_left", &[VAL64, VAL64], &[VAL64]),
        val_xor: make_method(module, "NATIVE:val_xor", &[VAL64, VAL64], &[VAL64]),
        range: make_method(module, "NATIVE:range", &[VAL64, VAL64], &[VAL64]),
        len: make_method(module, "NATIVE:len", &[VAL64], &[VAL64]),
        clone: make_method(module, "NATIVE:clone", &[VAL64], &[VAL64]),
        print: make_method(module, "NATIVE:print", &[VAL64], &[]),
        val_not: make_method(module, "NATIVE:val_not", &[VAL64], &[VAL64]),
    };
    native_methods
}

impl JIT {
    pub fn new() -> Self {
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
        builder.symbol("NATIVE:list_new", builtins::list_new as *const u8);
        builder.symbol("NATIVE:list_push", builtins::list_push as *const u8);
        builder.symbol("NATIVE:list_len_u64", builtins::list_len_u64 as *const u8);
        builder.symbol("NATIVE:list_get_u64", builtins::list_get_u64 as *const u8);
        builder.symbol("NATIVE:dict_new", builtins::dict_new as *const u8);
        builder.symbol("NATIVE:dict_insert", builtins::dict_insert as *const u8);
        builder.symbol("NATIVE:val_get_index", builtins::val_get_index as *const u8);
        builder.symbol("NATIVE:val_set_index", builtins::val_set_index as *const u8);
        builder.symbol("NATIVE:val_eq", builtins::val_eq as *const u8);
        builder.symbol("NATIVE:val_truthy", builtins::val_truthy as *const u8);
        builder.symbol(
            "NATIVE:val_get_lambda_details",
            builtins::val_get_lambda_details as *const u8,
        );
        builder.symbol(
            "NATIVE:construct_lambda",
            builtins::construct_lambda as *const u8,
        );
        builder.symbol(
            "NATIVE:val_shift_left",
            builtins::public_val_shift_left as *const u8,
        );
        builder.symbol("NATIVE:val_xor", builtins::public_val_xor as *const u8);
        builder.symbol("NATIVE:range", builtins::public_range as *const u8);
        builder.symbol("NATIVE:len", builtins::public_len as *const u8);
        builder.symbol("NATIVE:clone", builtins::clone as *const u8);
        builder.symbol("NATIVE:print", builtins::public_print as *const u8);
        builder.symbol("NATIVE:val_not", builtins::public_val_not as *const u8);

        let mut module = JITModule::new(builder);

        let natives = get_native_methods(&mut module);

        Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            data_description: DataDescription::new(),
            module,
            natives,
            string_constants: HashMap::new(),
        }
    }

    pub fn compile_program_as_function(
        &mut self,
        program: &[SpannedStatement],
    ) -> JITResult<FuncId> {
        let body = Spanned::with_zero_span(Expr::Block(ast::Block {
            statements: program.to_vec(),
            expression: None,
        }));
        let func_id = self.compile_main_func(&body)?;
        Ok(func_id)
    }

    pub unsafe fn run_func(&mut self, func_id: FuncId, input: Value64) -> Value64 {
        let func_ptr = self.module.get_finalized_function(func_id);
        // Cast the raw pointer to a typed function pointer. This is unsafe, because
        // this is the critical point where you have to trust that the generated code
        // is safe to be called.
        let code_fn = mem::transmute::<_, extern "C" fn(Value64) -> Value64>(func_ptr);
        // And now we can call it!
        code_fn(input)
    }

    fn compile_main_func(&mut self, body: &SpannedExpr) -> JITResult<FuncId> {
        // Then, translate the AST nodes into Cranelift IR.
        self.translate_main_func(body)?;

        // Next, declare the function to jit. Functions must be declared
        // before they can be called, or defined.
        let id = self
            .module
            .declare_function("MAIN", Linkage::Export, &self.ctx.func.signature)
            .map_err(JITError::Module)?;

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

    fn translate_main_func(&mut self, body: &SpannedExpr) -> JITResult<()> {
        self.ctx.func.collect_debug_info();
        self.ctx.func.signature.params.push(AbiParam::new(VAL64)); // program args
        self.ctx.func.signature.returns.push(AbiParam::new(VAL64));

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        // Walk the AST and declare all implicitly-declared variables.
        let variables = {
            let args_val = builder.block_params(entry_block)[0];
            let mut variable_builder = VariableBuilder::new(&mut builder);
            let args_var = variable_builder.declare_variable("args");
            variable_builder.declare_variables_in_expr(body);
            variable_builder.define_variable(args_var, args_val, "args");
            variable_builder.variables
        };

        builder.seal_all_blocks();

        // Now translate the statements of the function body.
        let mut trans = FunctionTranslator {
            builder,
            variables,
            module: &mut self.module,
            natives: &self.natives,
            string_constants: &mut self.string_constants,
        };
        let return_value = trans.translate_expr(body);

        // Emit the return instruction.
        trans.builder.ins().return_(&[return_value]);

        // Tell the builder we're done with this function.
        trans.builder.finalize();
        Ok(())
    }
}

struct VariableBuilder<'a, 'b> {
    builder: &'a mut FunctionBuilder<'b>,
    variables: HashMap<String, Variable>,
    index: usize,
}

impl<'a, 'b> VariableBuilder<'a, 'b> {
    fn new(builder: &'a mut FunctionBuilder<'b>) -> Self {
        Self {
            builder,
            variables: HashMap::new(),
            index: 0,
        }
    }

    fn declare_variables_in_func(&mut self, func: &FuncExpr, entry_block: Block) {
        let closure_ptr = self.builder.block_params(entry_block)[0];
        for (i, var_ref) in func.potential_captures.iter().enumerate() {
            let var = self.declare_variable(&var_ref.name);
            let val = self.builder.ins().load(
                VAL64,
                MemFlags::new(),
                closure_ptr,
                (i * size_of::<Value64>()) as i32,
            );
            self.define_variable(var, val, &var_ref.name);
        }
        for (i, pattern) in func.params.iter().enumerate() {
            let name = match pattern.value {
                MatchPattern::Declaration(ref var, _) => &var.name,
                MatchPattern::Discard => continue,
                MatchPattern::Constant(_) => continue,
                _ => todo!("Pattern matching in functions isn't supported"),
            };
            let val = self.builder.block_params(entry_block)[i + 1]; // actual params start at 1, first is closure
            let var = self.declare_variable(name);
            self.define_variable(var, val, &name);
        }
        if let Some(name) = &func.name {
            let val = self
                .builder
                .block_params(entry_block)
                .last()
                .copied()
                .unwrap();
            let var = self.declare_variable(name);
            self.define_variable(var, val, &name);
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
            // Expr::BuiltInFunction(_) => {}
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
                if let Some(ref else_branch) = else_branch {
                    self.declare_variables_in_expr(else_branch);
                }
            }
            Expr::ForIn { iter, var, body } => {
                self.declare_variables_in_expr(iter);
                let (var_ref, _) = var.expect_declaration();
                self.declare_variable(&var_ref.name);
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
            Expr::CallNative { callee: _, args } => {
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
                // for capture in l.potential_captures {
                //     self.declare_variable(&capture.name);
                // }
            }
            Expr::Match(m) => todo!("Match expressions not supported: {:?}", m),
        }
    }

    fn declare_variables_in_stmt(&mut self, stmt: &SpannedStatement) {
        match &stmt.value {
            Statement::Expression(e) => self.declare_variables_in_expr(e),
            Statement::Return(e) => self.declare_variables_in_expr(e),
            Statement::PatternAssignment(pattern, rhs) => {
                self.declare_variables_in_expr(rhs);
                let (var_ref, _) = pattern.expect_declaration();
                self.declare_variable(&var_ref.name);
            }
        }
    }

    fn declare_variables_in_pattern(&mut self, pattern: &MatchPattern) {
        match pattern {
            MatchPattern::Declaration(var, _) => {
                self.declare_variable(&var.name);
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
        }
    }

    fn declare_variable(&mut self, name: &str) -> Variable {
        let var = Variable::new(self.index);
        if !self.variables.contains_key(name) {
            self.variables.insert(name.into(), var);
            self.builder.declare_var(var, VAL64);
            self.index += 1;
        }
        var
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
    builder: FunctionBuilder<'a>,
    variables: HashMap<String, Variable>,
    module: &'a mut JITModule,
    natives: &'a NativeMethods,
    string_constants: &'a mut HashMap<String, Value64>,
}

impl<'a> FunctionTranslator<'a> {
    /// When you write out instructions in Cranelift, you get back `Value`s. You
    /// can then use these references in other instructions.
    fn translate_expr(&mut self, expr: &SpannedExpr) -> Value {
        self.set_src_span(&expr.span);
        match &expr.value {
            Expr::Variable(ref var) => {
                let variable = self
                    .variables
                    .get(&var.name)
                    .unwrap_or_else(|| panic!("variable {} not defined", var.name));
                self.builder.use_var(*variable)
            }
            Expr::Block(b) => {
                for stmt in &b.statements {
                    self.translate_stmt(stmt);
                }
                if let Some(ref expr) = b.expression {
                    return self.translate_expr(expr);
                } else {
                    return self.builder.ins().f64const(0.0);
                }
            }
            Expr::Bool(b) => self.const_bool(*b),
            Expr::Number(f) => self.builder.ins().f64const(*f),
            Expr::String(s) => self.string_literal(s.deref().clone()),
            Expr::Nil => self.const_nil(),
            Expr::Op { op, lhs, rhs } => self.translate_op(*op, lhs, rhs),
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => self.translate_if_else(condition, then_branch, else_branch),
            Expr::ForIn { iter, var, body } => self.translate_for_in(iter, var, body),
            Expr::While { condition, body } => self.translate_while_loop(condition, body),
            Expr::Call { callee, args } => self.translate_indirect_call(callee, args),
            Expr::CallNative { callee, args } => self.translate_native_call(*callee, args),
            Expr::List(l) => {
                let len_value = self.builder.ins().iconst(types::I64, l.len() as i64);
                let mut list = self.call_native(&self.natives.list_new, &[len_value])[0];
                for item in l {
                    let val = self.translate_expr(item);
                    list = self.call_native(&self.natives.list_push, &[list, val])[0];
                }
                list
            }
            Expr::GetIndex(val, index) => {
                let index = self.translate_expr(index);
                let val = self.translate_expr(val);
                self.call_native(&self.natives.val_get_index, &[val, index])[0]
            }
            Expr::Dict(d) => {
                let len_value = self.builder.ins().iconst(types::I64, d.len() as i64);
                let mut dict = self.call_native(&self.natives.dict_new, &[len_value])[0];
                for (key, value) in d {
                    let key = self.translate_expr(key);
                    let value = self.translate_expr(value);
                    dict = self.call_native(&self.natives.dict_insert, &[dict, key, value])[0];
                }
                dict
            }
            Expr::Lambda(l) => self.translate_lambda_definition(l),
            Expr::Match(_) => todo!(),
        }
    }

    fn translate_call(&mut self, callee: &SpannedExpr, args: &Vec<SpannedExpr>) -> Value {
        let mut sig = self.module.make_signature();

        for _arg in args {
            sig.params.push(AbiParam::new(VAL64));
        }

        sig.returns.push(AbiParam::new(VAL64));

        let name = match &callee.value {
            Expr::Variable(var) => var.name.clone(),
            _ => todo!("Only calls directly to functions are supported"),
        };

        let callee = self
            .module
            .declare_function(&name, Linkage::Import, &sig)
            .expect("problem declaring function");
        let local_callee = self.module.declare_func_in_func(callee, self.builder.func);

        let mut arg_values = Vec::new();
        for arg in args {
            let value = self.translate_expr(arg);
            arg_values.push(self.clone_val64(value))
        }
        let call = self.builder.ins().call(local_callee, &arg_values);
        self.builder.inst_results(call)[0]
    }

    fn translate_indirect_call(&mut self, callee: &SpannedExpr, args: &Vec<SpannedExpr>) -> Value {
        let callee_val = self.translate_expr(callee);

        let (func_ptr, closure_ptr) = self.get_lambda_details(args, callee_val);

        // Null pointer means callee wasn't a lambda
        self.builder.ins().trapz(func_ptr, TrapCode::User(3));

        let mut sig = self.module.make_signature();

        sig.params.push(AbiParam::new(I64)); // closure pointer

        for _arg in args {
            sig.params.push(AbiParam::new(VAL64));
        }

        sig.params.push(AbiParam::new(VAL64));

        sig.returns.push(AbiParam::new(VAL64));
        let sig_ref = self.builder.import_signature(sig);

        let mut arg_values = Vec::new();
        arg_values.push(closure_ptr);
        for arg in args {
            let value = self.translate_expr(arg);
            arg_values.push(self.clone_val64(value))
        }
        arg_values.push(self.clone_val64(callee_val));
        let call = self
            .builder
            .ins()
            .call_indirect(sig_ref, func_ptr, &arg_values);

        return self.builder.inst_results(call)[0];
    }

    fn get_lambda_details(
        &mut self,
        args: &Vec<Spanned<Expr>>,
        callee_val: Value,
    ) -> (Value, Value) {
        let closure_ptr_slot = self.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            8,
            0,
        ));
        let closure_ptr_ptr_val = self.builder.ins().stack_addr(I64, closure_ptr_slot, 0);
        let arg_count_val = self.builder.ins().iconst(types::I32, args.len() as i64);
        let func_ptr = self.call_native(
            &self.natives.val_get_lambda_details,
            &[callee_val, arg_count_val, closure_ptr_ptr_val],
        )[0];
        let closure_ptr = self.builder.ins().stack_load(I64, closure_ptr_slot, 0);
        (func_ptr, closure_ptr)
    }

    fn translate_native_call(&mut self, callee: BuiltInFunction, args: &Vec<SpannedExpr>) -> Value {
        let mut sig = self.module.make_signature();

        for _arg in args {
            sig.params.push(AbiParam::new(VAL64));
        }

        sig.returns.push(AbiParam::new(VAL64));

        let method = match get_native_method_for_builtin(&self.natives, callee) {
            Some(m) => m,
            None => {
                // FIXME: using trapz because trap makes it impossible to return Value
                let zero = self.builder.ins().iconst(types::I8, 0);
                self.builder.ins().trapz(zero, TrapCode::User(1));
                return self.const_nil();
            }
        };

        let mut arg_values = Vec::new();
        for arg in args {
            let value = self.translate_expr(arg);
            arg_values.push(self.clone_val64(value))
        }
        match self.call_native(method, &arg_values) {
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
                let return_value = self.translate_expr(e);
                self.builder.ins().return_(&[return_value]);

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
        match &pattern.value {
            MatchPattern::Discard => {}
            MatchPattern::Declaration(var_ref, _) => {
                let variable = self
                    .variables
                    .get(&var_ref.name)
                    .expect("variable not defined");
                self.builder.def_var(*variable, rhs_value);
            }
            MatchPattern::Assignment(target) => {
                self.translate_assignment(target, rhs_value);
            }
            _ => todo!("Pattern matching isn't supported here yet"),
        }
    }

    fn translate_assignment(&mut self, target: &ast::AssignmentTarget, rhs: Value) {
        let mut path_values = vec![];
        for index in &target.path {
            let value = self.translate_expr(index);
            path_values.push(value);
        }
        let variable = self
            .variables
            .get(&target.var.name)
            .copied()
            .expect("variable not defined");
        let new_val = match &path_values[..] {
            [] => rhs,
            [index, rest @ ..] => {
                let old_val = self.builder.use_var(variable);
                let old_val = self.clone_val64(old_val);
                self.assign_part(old_val, *index, rest, rhs)
            }
        };
        self.builder.def_var(variable, new_val);
    }

    fn assign_part(&mut self, val: Value, index: Value, rest_path: &[Value], rhs: Value) -> Value {
        match rest_path {
            [] => {
                return self.call_native(&self.natives.val_set_index, &[val, index, rhs])[0];
            }
            [next_index, rest @ ..] => {
                let old_val = self.call_native(&self.natives.val_get_index, &[val, index])[0];
                let new_val = self.assign_part(old_val, *next_index, rest, rhs);
                return self.call_native(&self.natives.val_set_index, &[val, index, new_val])[0];
            }
        }
    }

    fn translate_cmp(&mut self, cmp: FloatCC, lhs: &SpannedExpr, rhs: &SpannedExpr) -> Value {
        let lhs = self.translate_expr(lhs);
        let rhs = self.translate_expr(rhs);
        let cmp_val = self.builder.ins().fcmp(cmp, lhs, rhs);
        self.bool_to_val64(cmp_val)
    }

    fn translate_if_else(
        &mut self,
        condition: &SpannedExpr,
        then_body: &SpannedExpr,
        else_body: &Option<Box<SpannedExpr>>,
    ) -> Value {
        let condition_value = self.translate_expr(condition);
        let condition_value = self.call_native(&self.natives.val_truthy, &[condition_value])[0];

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
            .brif(condition_value, then_block, &[], else_block, &[]);

        self.builder.switch_to_block(then_block);
        self.builder.seal_block(then_block);
        let then_return = self.translate_expr(then_body);

        // Jump to the merge block, passing it the block return value.
        self.builder.ins().jump(merge_block, &[then_return]);

        self.builder.switch_to_block(else_block);
        self.builder.seal_block(else_block);
        let mut else_return = self.const_nil();
        if let Some(expr) = else_body {
            else_return = self.translate_expr(expr);
        }

        // Jump to the merge block, passing it the block return value.
        self.builder.ins().jump(merge_block, &[else_return]);

        // Switch to the merge block for subsequent statements.
        self.builder.switch_to_block(merge_block);

        // We've now seen all the predecessors of the merge block.
        self.builder.seal_block(merge_block);

        // Read the value of the if-else by reading the merge block
        // parameter.
        let phi = self.builder.block_params(merge_block)[0];

        phi
    }

    fn translate_for_in(
        &mut self,
        iter: &SpannedExpr,
        var: &Spanned<MatchPattern>,
        body: &SpannedExpr,
    ) -> Value {
        let header_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let exit_block = self.builder.create_block();
        let (var_ref, _) = var.expect_declaration();

        let iter_value = self.translate_expr(iter);
        let len_value = self.call_native(&self.natives.list_len_u64, &[iter_value])[0];
        let initial_index = self.builder.ins().iconst(I64, 0);
        self.builder.ins().jump(header_block, &[initial_index]);

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
            &[current_index],
            exit_block,
            &[],
        );

        // BODY BLOCK
        self.set_src_span(&body.span);
        self.builder.append_block_param(body_block, I64);
        self.builder.switch_to_block(body_block);
        self.builder.seal_block(body_block);

        let current_index = self.builder.block_params(header_block)[0];
        let current_item =
            self.call_native(&self.natives.list_get_u64, &[iter_value, current_index])[0];
        self.clone_val64(current_item);
        let variable = self
            .variables
            .get(&var_ref.name)
            .expect("variable not defined");
        self.builder.def_var(*variable, current_item);
        self.translate_expr(body);
        let next_index = self.builder.ins().iadd_imm(current_index, 1);
        self.builder.ins().jump(header_block, &[next_index]);

        // EXIT BLOCK
        self.builder.switch_to_block(exit_block);

        // We've reached the bottom of the loop, so there will be no
        // more backedges to the header to exits to the bottom.
        self.builder.seal_block(header_block);
        self.builder.seal_block(exit_block);

        self.const_nil()
    }

    fn translate_while_loop(&mut self, condition: &SpannedExpr, body: &SpannedExpr) -> Value {
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

        self.const_nil()
    }

    fn translate_op(&mut self, op: Opcode, lhs: &SpannedExpr, rhs: &SpannedExpr) -> Value {
        match op {
            Opcode::Add => {
                let lhs = self.translate_expr(lhs);
                self.guard_f64(lhs);
                let rhs = self.translate_expr(rhs);
                self.guard_f64(rhs);
                self.builder.ins().fadd(lhs, rhs)
            }
            Opcode::Sub => {
                let lhs = self.translate_expr(lhs);
                self.guard_f64(lhs);
                let rhs = self.translate_expr(rhs);
                self.guard_f64(rhs);
                self.builder.ins().fsub(lhs, rhs)
            }
            Opcode::Mul => {
                let lhs = self.translate_expr(lhs);
                self.guard_f64(lhs);
                let rhs = self.translate_expr(rhs);
                self.guard_f64(rhs);
                self.builder.ins().fmul(lhs, rhs)
            }
            Opcode::Div => {
                let lhs = self.translate_expr(lhs);
                self.guard_f64(lhs);
                let rhs = self.translate_expr(rhs);
                self.guard_f64(rhs);
                self.builder.ins().fdiv(lhs, rhs)
            }
            Opcode::Eq => {
                let lhs_val = self.translate_expr(lhs);
                let rhs_val = self.translate_expr(rhs);
                self.call_native(&self.natives.val_eq, &[lhs_val, rhs_val])[0]
            }
            Opcode::Neq => self.translate_cmp(FloatCC::NotEqual, lhs, rhs),
            Opcode::Lt => self.translate_cmp(FloatCC::LessThan, lhs, rhs),
            Opcode::Lte => self.translate_cmp(FloatCC::LessThanOrEqual, lhs, rhs),
            Opcode::Gt => self.translate_cmp(FloatCC::GreaterThan, lhs, rhs),
            Opcode::Gte => self.translate_cmp(FloatCC::GreaterThanOrEqual, lhs, rhs),
            Opcode::And => {
                // TODO: conditional evaluation
                let lhs = self.translate_expr(lhs);
                let rhs = self.translate_expr(rhs);
                let lhs_truthy = self.is_truthy(lhs);
                let rhs_truthy = self.is_truthy(rhs);
                let bool_val = self.builder.ins().band(lhs_truthy, rhs_truthy);
                self.bool_to_val64(bool_val)
            }
            Opcode::Or => {
                // TODO: conditional evaluation
                let lhs = self.translate_expr(lhs);
                let rhs = self.translate_expr(rhs);
                let lhs_truthy = self.is_truthy(lhs);
                let rhs_truthy = self.is_truthy(rhs);
                let bool_val = self.builder.ins().bor(lhs_truthy, rhs_truthy);
                self.bool_to_val64(bool_val)
            }
        }
    }

    fn translate_lambda_definition(&mut self, lambda: &LambdaExpr) -> Value {
        let mut ctx = self.module.make_context();
        ctx.func.collect_debug_info();
        ctx.func.signature.params.push(AbiParam::new(PTR)); // closure
        for _ in 0..lambda.params.len() {
            ctx.func.signature.params.push(AbiParam::new(VAL64));
        }
        ctx.func.signature.params.push(AbiParam::new(VAL64)); // self for recursion
        ctx.func.signature.returns.push(AbiParam::new(VAL64)); // return value

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

        builder.seal_all_blocks(); // ??

        let mut trans = FunctionTranslator {
            builder,
            variables,
            module: &mut self.module,
            natives: &self.natives,
            string_constants: &mut self.string_constants,
        };
        let return_value = trans.translate_expr(&lambda.body);

        trans.builder.ins().return_(&[return_value]);

        trans.builder.finalize();

        let func_id = self
            .module
            .declare_anonymous_function(&ctx.func.signature)
            .unwrap();

        self.module.define_function(func_id, &mut ctx).unwrap();
        self.module.clear_context(&mut ctx);

        self.module.finalize_definitions().unwrap();

        let func_ptr = self.module.get_finalized_function(func_id);
        let param_count_val = self.builder.ins().iconst(I64, lambda.params.len() as i64);
        let func_ptr_val = self.builder.ins().iconst(PTR, func_ptr as i64);
        let capture_count_val = self
            .builder
            .ins()
            .iconst(I64, lambda.potential_captures.len() as i64);
        let closure_slot = self.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (8 * lambda.potential_captures.len()) as u32,
            0,
        ));
        for (i, var_ref) in lambda.potential_captures.iter().enumerate() {
            let var = self.variables.get(&var_ref.name).unwrap();
            let var_val = self.builder.use_var(*var);
            self.builder
                .ins()
                .stack_store(var_val, closure_slot, (i * 8) as i32);
        }
        let closure_ptr = self.builder.ins().stack_addr(PTR, closure_slot, 0);
        let lambda_val = self.call_native(
            &self.natives.construct_lambda,
            &[
                param_count_val,
                func_ptr_val,
                capture_count_val,
                closure_ptr,
            ],
        )[0];
        return lambda_val;
    }

    fn call_native(&mut self, method: &NativeMethod, args: &[Value]) -> &[Value] {
        let local_callee = self
            .module
            .declare_func_in_func(method.func, self.builder.func);
        let call = self.builder.ins().call(local_callee, args);
        self.builder.inst_results(call)
    }

    fn const_nil(&mut self) -> Value {
        self.builder
            .ins()
            .f64const(f64::from_bits(value64::NIL_VALUE))
    }

    fn const_bool(&mut self, b: bool) -> Value {
        self.builder.ins().f64const(if b {
            f64::from_bits(value64::TRUE_VALUE)
        } else {
            f64::from_bits(value64::FALSE_VALUE)
        })
    }

    fn clone_val64(&mut self, val: Value) -> Value {
        self.call_native(&self.natives.clone, &[val])[0]
    }

    fn set_src_span(&mut self, span: &Span) {
        self.builder
            .set_srcloc(ir::SourceLoc::new(span.start as u32));
    }

    fn is_truthy(&mut self, val: Value) -> Value {
        self.call_native(&self.natives.val_truthy, &[val])[0]
    }

    fn bool_to_val64(&mut self, b: Value) -> Value {
        let t = self.const_bool(true);
        let f = self.const_bool(false);
        self.builder.ins().select(b, t, f)
    }

    fn guard_f64(&mut self, val: Value) {
        let val_int = self.builder.ins().bitcast(I64, MemFlags::new(), val);
        let nanish = self.builder.ins().iconst(I64, value64::NANISH as i64);
        let and = self.builder.ins().band(val_int, nanish);
        let is_f64 = self.builder.ins().icmp(IntCC::NotEqual, and, nanish);
        self.builder.ins().trapz(is_f64, TrapCode::User(2));
    }

    fn string_literal(&mut self, string: String) -> Value {
        let string_val64 = self
            .string_constants
            .entry(string)
            .or_insert_with_key(|key| Value64::from_string(Rc::new(key.clone())));
        let string_val = self.builder.ins().f64const(string_val64.bits_f64());
        self.clone_val64(string_val)
    }
}

fn get_native_method_for_builtin(
    methods: &NativeMethods,
    func: BuiltInFunction,
) -> Option<&NativeMethod> {
    match func {
        BuiltInFunction::Range => Some(&methods.range),
        BuiltInFunction::Print => Some(&methods.print),
        BuiltInFunction::Len => Some(&methods.len),
        BuiltInFunction::ShiftLeft => Some(&methods.val_shift_left),
        BuiltInFunction::XOR => Some(&methods.val_xor),
        BuiltInFunction::Not => Some(&methods.val_not),
        _ => None,
    }
}
