#![allow(dead_code)]

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, Linkage, Module};
use std::collections::HashMap;

use crate::ast::{
    Expr, LambdaExpr, MatchPattern, Opcode, SpannedExpr, SpannedStatement, Statement,
};

type FuncExpr = LambdaExpr;

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
}

impl JIT {
    pub fn new() -> Self {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();
        let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
            panic!("host machine is not supported: {}", msg);
        });
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        let module = JITModule::new(builder);
        Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            data_description: DataDescription::new(),
            module,
        }
    }

    pub fn compile_func(&mut self, func: &FuncExpr) -> Result<*const u8, String> {
        // Then, translate the AST nodes into Cranelift IR.
        self.translate_func(func)?;

        // Next, declare the function to jit. Functions must be declared
        // before they can be called, or defined.
        //
        // TODO: This may be an area where the API should be streamlined; should
        // we have a version of `declare_function` that automatically declares
        // the function?
        let id = self
            .module
            .declare_function(
                &func.name.clone().unwrap(),
                Linkage::Export,
                &self.ctx.func.signature,
            )
            .map_err(|e| e.to_string())?;

        // Define the function to jit. This finishes compilation, although
        // there may be outstanding relocations to perform. Currently, jit
        // cannot finish relocations until all functions to be called are
        // defined. For this toy demo for now, we'll just finalize the
        // function below.
        self.module
            .define_function(id, &mut self.ctx)
            .map_err(|e| e.to_string())?;

        // Now that compilation is finished, we can clear out the context state.
        self.module.clear_context(&mut self.ctx);

        // Finalize the functions which we just defined, which resolves any
        // outstanding relocations (patching in addresses, now that they're
        // available).
        self.module.finalize_definitions().unwrap();

        // We can now retrieve a pointer to the machine code.
        let code = self.module.get_finalized_function(id);

        Ok(code)
    }

    fn translate_func(&mut self, func: &FuncExpr) -> Result<(), String> {
        // Our toy language currently only supports I64 values, though Cranelift
        // supports other types.
        let int = self.module.target_config().pointer_type();

        for _p in &*func.params {
            self.ctx.func.signature.params.push(AbiParam::new(int));
        }

        // Our toy language currently only supports one return value, though
        // Cranelift is designed to support more.
        self.ctx.func.signature.returns.push(AbiParam::new(int));

        // Create the builder to build a function.
        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);

        // Create the entry block, to start emitting code in.
        let entry_block = builder.create_block();

        // Since this is the entry block, add block parameters corresponding to
        // the function's parameters.
        builder.append_block_params_for_function_params(entry_block);

        // Tell the builder to emit code in this block.
        builder.switch_to_block(entry_block);

        // And, tell the builder that this block will have no further
        // predecessors. Since it's the entry block, it won't have any
        // predecessors.
        builder.seal_block(entry_block);

        // Walk the AST and declare all implicitly-declared variables.
        let variables = {
            let mut variable_builder = VariableBuilder::new(int, &mut builder);
            variable_builder.declare_variables_in_func(func, entry_block);
            variable_builder.variables
        };

        builder.seal_all_blocks();

        // Now translate the statements of the function body.
        let mut trans = FunctionTranslator {
            int,
            builder,
            variables,
            module: &mut self.module,
        };
        let return_value = trans.translate_expr(&func.body);

        // Emit the return instruction.
        trans.builder.ins().return_(&[return_value]);

        // Tell the builder we're done with this function.
        trans.builder.finalize();
        Ok(())
    }
}

struct VariableBuilder<'a, 'b> {
    int: types::Type,
    builder: &'a mut FunctionBuilder<'b>,
    variables: HashMap<String, Variable>,
    index: usize,
}

impl<'a, 'b> VariableBuilder<'a, 'b> {
    fn new(int: types::Type, builder: &'a mut FunctionBuilder<'b>) -> Self {
        Self {
            int,
            builder,
            variables: HashMap::new(),
            index: 0,
        }
    }

    fn declare_variables_in_func(&mut self, func: &FuncExpr, entry_block: Block) {
        for (i, pattern) in func.params.iter().enumerate() {
            let name = match pattern.value {
                MatchPattern::Declaration(ref var, _) => &var.name,
                _ => todo!("Pattern matching in functions isn't supported"),
            };
            let val = self.builder.block_params(entry_block)[i];
            let var = self.declare_variable(name);
            self.builder.def_var(var, val);
        }
        // let zero = builder.ins().iconst(int, 0);
        // let return_variable = declare_variable(int, builder, &mut variables, &mut index, the_return);
        // builder.def_var(return_variable, zero);
        // for expr in stmts {}
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
                if let Some(ref else_branch) = else_branch {
                    self.declare_variables_in_expr(else_branch);
                }
            }
            e => todo!("Expression type not supported: {:?}", e),
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

    fn declare_variable(&mut self, name: &str) -> Variable {
        let var = Variable::new(self.index);
        if !self.variables.contains_key(name) {
            self.variables.insert(name.into(), var);
            self.builder.declare_var(var, self.int);
            self.index += 1;
        }
        var
    }
}

struct FunctionTranslator<'a> {
    int: types::Type,
    builder: FunctionBuilder<'a>,
    variables: HashMap<String, Variable>,
    module: &'a mut JITModule,
}

impl<'a> FunctionTranslator<'a> {
    /// When you write out instructions in Cranelift, you get back `Value`s. You
    /// can then use these references in other instructions.
    fn translate_expr(&mut self, expr: &SpannedExpr) -> Value {
        match &expr.value {
            Expr::Variable(ref var) => {
                let variable = self.variables.get(&var.name).expect("variable not defined");
                self.builder.use_var(*variable)
            }
            Expr::Block(b) => {
                for stmt in &b.statements {
                    self.translate_stmt(stmt);
                }
                if let Some(ref expr) = b.expression {
                    return self.translate_expr(expr);
                } else {
                    return self.builder.ins().iconst(self.int, 0);
                }
            }
            Expr::Bool(_) => unimplemented!(),
            Expr::Number(f) => {
                let todo_int = *f as i64;
                self.builder.ins().iconst(self.int, todo_int)
            }
            Expr::Nil => unimplemented!(),
            Expr::Op { op, lhs, rhs } => self.translate_op(*op, lhs, rhs),
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => self.translate_if_else(condition, then_branch, else_branch),
            _ => unimplemented!(),
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
            }
            Statement::PatternAssignment(pattern, rhs) => {
                let rhs_value = self.translate_expr(rhs);
                let (var_ref, _) = pattern.expect_declaration();
                let variable = self
                    .variables
                    .get(&var_ref.name)
                    .expect("variable not defined");
                self.builder.def_var(*variable, rhs_value);
            }
        }
    }

    fn translate_icmp(&mut self, cmp: IntCC, lhs: &SpannedExpr, rhs: &SpannedExpr) -> Value {
        let lhs = self.translate_expr(lhs);
        let rhs = self.translate_expr(rhs);
        self.builder.ins().icmp(cmp, lhs, rhs)
    }

    fn translate_if_else(
        &mut self,
        condition: &SpannedExpr,
        then_body: &SpannedExpr,
        else_body: &Option<Box<SpannedExpr>>,
    ) -> Value {
        let condition_value = self.translate_expr(condition);

        let then_block = self.builder.create_block();
        let else_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        // If-else constructs in the toy language have a return value.
        // In traditional SSA form, this would produce a PHI between
        // the then and else bodies. Cranelift uses block parameters,
        // so set up a parameter in the merge block, and we'll pass
        // the return values to it from the branches.
        self.builder.append_block_param(merge_block, self.int);

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
        let mut else_return = self.builder.ins().iconst(self.int, 0);
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

    fn translate_op(&mut self, op: Opcode, lhs: &SpannedExpr, rhs: &SpannedExpr) -> Value {
        match op {
            Opcode::Add => {
                let lhs = self.translate_expr(lhs);
                let rhs = self.translate_expr(rhs);
                self.builder.ins().iadd(lhs, rhs)
            }
            Opcode::Sub => {
                let lhs = self.translate_expr(lhs);
                let rhs = self.translate_expr(rhs);
                self.builder.ins().isub(lhs, rhs)
            }
            Opcode::Mul => {
                let lhs = self.translate_expr(lhs);
                let rhs = self.translate_expr(rhs);
                self.builder.ins().imul(lhs, rhs)
            }
            Opcode::Div => {
                let lhs = self.translate_expr(lhs);
                let rhs = self.translate_expr(rhs);
                self.builder.ins().udiv(lhs, rhs)
            }
            Opcode::Eq => self.translate_icmp(IntCC::Equal, lhs, rhs),
            Opcode::Neq => self.translate_icmp(IntCC::NotEqual, lhs, rhs),
            Opcode::Lt => self.translate_icmp(IntCC::SignedLessThan, lhs, rhs),
            Opcode::Lte => self.translate_icmp(IntCC::SignedLessThanOrEqual, lhs, rhs),
            Opcode::Gt => self.translate_icmp(IntCC::SignedGreaterThan, lhs, rhs),
            Opcode::Gte => self.translate_icmp(IntCC::SignedGreaterThanOrEqual, lhs, rhs),
            _ => unimplemented!(),
        }
    }
}
