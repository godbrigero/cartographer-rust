//! Tree-walking evaluator for the supported Lua 5.2 subset. Runtime error
//! messages follow Lua 5.2 (`attempt to index a nil value (global 'x')`
//! etc., prefixed with `chunkid:line:`).

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use super::ast::{BinOp, Block, Expr, ExprKind, Field, FuncBody, NameKind, Stat, StatKind, UnOp};
use super::lexer::chunk_id;
use super::parser::parse_chunk;
use super::stdlib;
use super::value::{Function, FunctionRef, LuaClosure, LuaString, TableRef, Value};

/// Maximum depth of nested Lua function calls.
///
/// RUST PORT NOTE: Lua's limit is much higher (it does not recurse on the C
/// stack for Lua-to-Lua calls); this tree-walking interpreter recurses on the
/// Rust stack and therefore uses a limit similar to `LUAI_MAXCCALLS`.
pub const MAX_CALL_DEPTH: usize = 200;

/// Number of positional constructor items stored at once (`LFIELDS_PER_FLUSH`).
const FIELDS_PER_FLUSH: usize = 50;

/// A local variable binding in a lexical scope chain.
pub struct Scope {
    name: Rc<str>,
    cell: RefCell<Value>,
    parent: Env,
}

/// A lexical environment: the innermost local variable binding (or `None`).
pub type Env = Option<Rc<Scope>>;

fn push_scope(env: &Env, name: Rc<str>, value: Value) -> Env {
    Some(Rc::new(Scope {
        name,
        cell: RefCell::new(value),
        parent: env.clone(),
    }))
}

fn lookup<'e>(env: &'e Env, name: &str) -> Option<&'e Scope> {
    let mut scope = env.as_deref();
    while let Some(s) = scope {
        if &*s.name == name {
            return Some(s);
        }
        scope = s.parent.as_deref();
    }
    None
}

/// A Lua error carrying the error object (usually a message string).
#[derive(Clone)]
pub struct LuaError {
    /// The error object passed to `error()` or a runtime error message.
    pub value: Value,
}

impl LuaError {
    /// Creates an error with a message.
    pub fn new(message: impl Into<String>) -> Self {
        LuaError {
            value: Value::String(LuaString::from(message.into())),
        }
    }

    /// The error message (like `lua_tostring` on the error object).
    pub fn message(&self) -> String {
        match self.value.to_lua_string() {
            Some(s) => s.to_str_lossy(),
            None => format!("(error object is a {} value)", self.value.type_name()),
        }
    }
}

impl fmt::Debug for LuaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LuaError({:?})", self.message())
    }
}

impl fmt::Display for LuaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message())
    }
}

impl std::error::Error for LuaError {}

/// A call stack entry; `chunk` is `None` for native functions.
struct CallInfo {
    chunk: Option<Rc<str>>,
    line: u32,
}

struct Ctx {
    env: Env,
    varargs: Rc<Vec<Value>>,
    chunk: Rc<str>,
}

enum Flow {
    Normal,
    Break,
    Return(Vec<Value>),
}

enum Target<'a> {
    Name(&'a Rc<str>),
    Index(Value, Value, &'a Expr, u32),
}

/// A Lua state: global environment and call stack.
pub struct Interpreter {
    globals: TableRef,
    string_library: Option<TableRef>,
    call_stack: Vec<CallInfo>,
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Interpreter {
    fn drop(&mut self) {
        // RUST PORT NOTE: there is no garbage collector; break the `_G._G`
        // cycle so that the globals are freed with the state.
        self.globals.clear();
        if let Some(string_library) = &self.string_library {
            string_library.clear();
        }
    }
}

impl Interpreter {
    /// Creates a state with the standard library opened (`luaL_openlibs`).
    pub fn new() -> Self {
        let mut interpreter = Interpreter {
            globals: TableRef::new(),
            string_library: None,
            call_stack: Vec::new(),
        };
        stdlib::open_libs(&mut interpreter);
        interpreter
    }

    /// The global table (`_G`).
    pub fn globals(&self) -> &TableRef {
        &self.globals
    }

    /// Sets the table used to index string values (`("x"):rep(3)`).
    pub(crate) fn set_string_library(&mut self, table: TableRef) {
        self.string_library = Some(table);
    }

    /// Registers a global Rust function (`lua_register`).
    pub fn register(
        &mut self,
        name: &str,
        f: impl Fn(&mut Interpreter, Vec<Value>) -> Result<Vec<Value>, LuaError> + 'static,
    ) {
        self.globals
            .set_str(name, Value::Function(FunctionRef::native(name, f)));
    }

    /// Compiles a chunk (`luaL_loadbuffer`). `chunk_name` is used like Lua's
    /// chunkname (a string source is shown as `[string "..."]`).
    pub fn load(&mut self, source: &[u8], chunk_name: &str) -> Result<Value, LuaError> {
        let chunk: Rc<str> = Rc::from(chunk_id(chunk_name));
        let body = parse_chunk(source, chunk.clone()).map_err(LuaError::new)?;
        Ok(self.make_closure(body, None, chunk))
    }

    /// Compiles and runs `source`, returning all results.
    pub fn exec(&mut self, source: &[u8], chunk_name: &str) -> Result<Vec<Value>, LuaError> {
        let f = self.load(source, chunk_name)?;
        self.call(&f, Vec::new())
    }

    fn make_closure(&self, body: Rc<FuncBody>, env: Env, chunk: Rc<str>) -> Value {
        Value::Function(FunctionRef(Rc::new(Function::Lua(LuaClosure {
            body,
            env,
            chunk,
        }))))
    }

    /// `luaL_where(L, level)`: `"chunkid:line: "` of the Lua function at
    /// `level` (1 = the function that called the running native function).
    pub fn where_(&self, level: usize) -> String {
        let Some(index) = self.call_stack.len().checked_sub(level + 1) else {
            return String::new();
        };
        let info = &self.call_stack[index];
        match &info.chunk {
            Some(chunk) => format!("{}:{}: ", chunk, info.line),
            None => String::new(),
        }
    }

    /// `luaL_error`: an error message prefixed with the position of the
    /// calling Lua code.
    pub fn error(&self, message: impl AsRef<str>) -> LuaError {
        LuaError::new(format!("{}{}", self.where_(1), message.as_ref()))
    }

    /// Calls a function value with arguments, returning all results.
    pub fn call(&mut self, f: &Value, args: Vec<Value>) -> Result<Vec<Value>, LuaError> {
        let Value::Function(function) = f else {
            return Err(LuaError::new(format!(
                "{}attempt to call a {} value",
                self.where_(0),
                f.type_name()
            )));
        };
        let function = function.clone();
        match &*function.0 {
            Function::Native { f, .. } => {
                if self.call_stack.len() >= MAX_CALL_DEPTH {
                    return Err(LuaError::new(format!("{}stack overflow", self.where_(0))));
                }
                self.call_stack.push(CallInfo {
                    chunk: None,
                    line: 0,
                });
                let result = f(self, args);
                self.call_stack.pop();
                result
            }
            Function::Lua(closure) => self.call_closure(closure, args),
        }
    }

    fn call_closure(
        &mut self,
        closure: &LuaClosure,
        args: Vec<Value>,
    ) -> Result<Vec<Value>, LuaError> {
        if self.call_stack.len() >= MAX_CALL_DEPTH {
            return Err(LuaError::new(format!("{}stack overflow", self.where_(0))));
        }
        let body = &closure.body;
        let mut env = closure.env.clone();
        for (i, param) in body.params.iter().enumerate() {
            env = push_scope(
                &env,
                param.clone(),
                args.get(i).cloned().unwrap_or_default(),
            );
        }
        let varargs = if body.is_vararg && args.len() > body.params.len() {
            args[body.params.len()..].to_vec()
        } else {
            Vec::new()
        };
        let mut ctx = Ctx {
            env,
            varargs: Rc::new(varargs),
            chunk: closure.chunk.clone(),
        };
        self.call_stack.push(CallInfo {
            chunk: Some(closure.chunk.clone()),
            line: body.line,
        });
        let result = self.exec_block_keep_env(&body.body, &mut ctx);
        self.call_stack.pop();
        match result? {
            Flow::Return(values) => Ok(values),
            Flow::Normal | Flow::Break => Ok(Vec::new()),
        }
    }

    fn set_line(&mut self, line: u32) {
        if let Some(top) = self.call_stack.last_mut() {
            top.line = line;
        }
    }

    fn runtime_error(&mut self, ctx: &Ctx, line: u32, message: String) -> LuaError {
        self.set_line(line);
        LuaError::new(format!("{}:{}: {}", ctx.chunk, line, message))
    }

    // ----- Statements -----

    fn exec_block(&mut self, block: &Block, ctx: &mut Ctx) -> Result<Flow, LuaError> {
        let saved = ctx.env.clone();
        let result = self.exec_block_keep_env(block, ctx);
        ctx.env = saved;
        result
    }

    fn exec_block_keep_env(&mut self, block: &Block, ctx: &mut Ctx) -> Result<Flow, LuaError> {
        for stat in &block.stats {
            match self.exec_stat(stat, ctx)? {
                Flow::Normal => {}
                flow => return Ok(flow),
            }
        }
        if let Some((exprs, line)) = &block.ret {
            self.set_line(*line);
            let values = self.eval_list(exprs, ctx)?;
            return Ok(Flow::Return(values));
        }
        Ok(Flow::Normal)
    }

    fn exec_stat(&mut self, stat: &Stat, ctx: &mut Ctx) -> Result<Flow, LuaError> {
        self.set_line(stat.line);
        match &stat.kind {
            StatKind::Call(e) => {
                self.eval_multi(e, ctx)?;
            }
            StatKind::Assign(targets, values) => self.assign(targets, values, ctx)?,
            StatKind::Local(names, exprs) => {
                let mut values = self.eval_list(exprs, ctx)?;
                values.resize(names.len(), Value::Nil);
                for (name, value) in names.iter().zip(values) {
                    ctx.env = push_scope(&ctx.env, name.clone(), value);
                }
            }
            StatKind::LocalFunction(name, body) => {
                ctx.env = push_scope(&ctx.env, name.clone(), Value::Nil);
                let closure = self.make_closure(body.clone(), ctx.env.clone(), ctx.chunk.clone());
                *ctx.env.as_ref().expect("scope").cell.borrow_mut() = closure;
            }
            StatKind::Do(block) => return self.exec_block(block, ctx),
            StatKind::While(cond, body) => loop {
                if !self.eval(cond, ctx)?.truthy() {
                    break;
                }
                match self.exec_block(body, ctx)? {
                    Flow::Break => break,
                    Flow::Return(values) => return Ok(Flow::Return(values)),
                    Flow::Normal => {}
                }
            },
            StatKind::Repeat(body, cond) => loop {
                let saved = ctx.env.clone();
                let flow = self.exec_block_keep_env(body, ctx);
                let flow = match flow {
                    Ok(Flow::Normal) => None,
                    other => Some(other),
                };
                if let Some(flow) = flow {
                    ctx.env = saved;
                    return match flow? {
                        Flow::Break => Ok(Flow::Normal),
                        other => Ok(other),
                    };
                }
                let done = self.eval(cond, ctx);
                ctx.env = saved;
                if done?.truthy() {
                    break;
                }
            },
            StatKind::If(conds, else_block) => {
                for (cond, block) in conds {
                    if self.eval(cond, ctx)?.truthy() {
                        return self.exec_block(block, ctx);
                    }
                }
                if let Some(block) = else_block {
                    return self.exec_block(block, ctx);
                }
            }
            StatKind::NumericFor {
                var,
                start,
                limit,
                step,
                body,
            } => return self.numeric_for(var, start, limit, step.as_ref(), body, stat.line, ctx),
            StatKind::GenericFor { names, exprs, body } => {
                return self.generic_for(names, exprs, body, stat.line, ctx);
            }
            StatKind::Break => return Ok(Flow::Break),
        }
        Ok(Flow::Normal)
    }

    #[allow(clippy::too_many_arguments)]
    fn numeric_for(
        &mut self,
        var: &Rc<str>,
        start: &Expr,
        limit: &Expr,
        step: Option<&Expr>,
        body: &Block,
        line: u32,
        ctx: &mut Ctx,
    ) -> Result<Flow, LuaError> {
        let start_value = self.eval(start, ctx)?;
        let limit_value = self.eval(limit, ctx)?;
        let step_value = match step {
            Some(e) => self.eval(e, ctx)?,
            None => Value::Number(1.),
        };
        let Some(init) = start_value.to_number() else {
            return Err(self.runtime_error(
                ctx,
                line,
                "'for' initial value must be a number".into(),
            ));
        };
        let Some(limit) = limit_value.to_number() else {
            return Err(self.runtime_error(ctx, line, "'for' limit must be a number".into()));
        };
        let Some(step) = step_value.to_number() else {
            return Err(self.runtime_error(ctx, line, "'for' step must be a number".into()));
        };
        let mut index = init - step;
        loop {
            index += step;
            let continue_loop = if 0. < step {
                index <= limit
            } else {
                limit <= index
            };
            if !continue_loop {
                break;
            }
            let saved = ctx.env.clone();
            ctx.env = push_scope(&ctx.env, var.clone(), Value::Number(index));
            let flow = self.exec_block(body, ctx);
            ctx.env = saved;
            match flow? {
                Flow::Break => break,
                Flow::Return(values) => return Ok(Flow::Return(values)),
                Flow::Normal => {}
            }
        }
        Ok(Flow::Normal)
    }

    fn generic_for(
        &mut self,
        names: &[Rc<str>],
        exprs: &[Expr],
        body: &Block,
        line: u32,
        ctx: &mut Ctx,
    ) -> Result<Flow, LuaError> {
        let mut values = self.eval_list(exprs, ctx)?;
        values.resize(3, Value::Nil);
        let f = values[0].clone();
        let state = values[1].clone();
        let mut control = values[2].clone();
        loop {
            if !matches!(f, Value::Function(_)) {
                return Err(self.runtime_error(
                    ctx,
                    line,
                    format!("attempt to call a {} value", f.type_name()),
                ));
            }
            self.set_line(line);
            let mut results = self.call(&f, vec![state.clone(), control.clone()])?;
            results.resize(names.len().max(1), Value::Nil);
            if results[0].is_nil() {
                break;
            }
            control = results[0].clone();
            let saved = ctx.env.clone();
            for (name, value) in names.iter().zip(results) {
                ctx.env = push_scope(&ctx.env, name.clone(), value);
            }
            let flow = self.exec_block(body, ctx);
            ctx.env = saved;
            match flow? {
                Flow::Break => break,
                Flow::Return(values) => return Ok(Flow::Return(values)),
                Flow::Normal => {}
            }
        }
        Ok(Flow::Normal)
    }

    fn assign(&mut self, targets: &[Expr], exprs: &[Expr], ctx: &mut Ctx) -> Result<(), LuaError> {
        let mut resolved = Vec::with_capacity(targets.len());
        for target in targets {
            match &target.kind {
                ExprKind::Name(name, _) => resolved.push(Target::Name(name)),
                ExprKind::Index(object, key) => {
                    let o = self.eval(object, ctx)?;
                    let k = self.eval(key, ctx)?;
                    resolved.push(Target::Index(o, k, object, target.line));
                }
                _ => unreachable!("parser only produces assignable targets"),
            }
        }
        let mut values = self.eval_list(exprs, ctx)?;
        values.resize(targets.len(), Value::Nil);
        // Assign from right to left like the reference implementation.
        for (target, value) in resolved.into_iter().zip(values).rev() {
            match target {
                Target::Name(name) => self.set_variable(name, value, ctx),
                Target::Index(object, key, object_expr, line) => {
                    self.set_index(&object, key, value, object_expr, line, ctx)?;
                }
            }
        }
        Ok(())
    }

    fn set_variable(&mut self, name: &Rc<str>, value: Value, ctx: &Ctx) {
        match lookup(&ctx.env, name) {
            Some(scope) => *scope.cell.borrow_mut() = value,
            None => self
                .globals
                .set(Value::String(LuaString::from(&**name)), value)
                .expect("string keys are valid"),
        }
    }

    fn set_index(
        &mut self,
        object: &Value,
        key: Value,
        value: Value,
        object_expr: &Expr,
        line: u32,
        ctx: &Ctx,
    ) -> Result<(), LuaError> {
        match object {
            Value::Table(t) => t
                .set(key, value)
                .map_err(|e| self.runtime_error(ctx, line, e.to_string())),
            other => Err(self.runtime_error(
                ctx,
                line,
                format!(
                    "attempt to index a {} value{}",
                    other.type_name(),
                    var_info(object_expr)
                ),
            )),
        }
    }

    // ----- Expressions -----

    fn eval_list(&mut self, exprs: &[Expr], ctx: &mut Ctx) -> Result<Vec<Value>, LuaError> {
        let mut values = Vec::with_capacity(exprs.len());
        for (i, e) in exprs.iter().enumerate() {
            if i + 1 == exprs.len() && e.is_multi() {
                values.extend(self.eval_multi(e, ctx)?);
            } else {
                values.push(self.eval(e, ctx)?);
            }
        }
        Ok(values)
    }

    fn eval_multi(&mut self, e: &Expr, ctx: &mut Ctx) -> Result<Vec<Value>, LuaError> {
        match &e.kind {
            ExprKind::Call(f, args) => self.call_expr(f, args, e.line, ctx),
            ExprKind::Method(object, name, args) => {
                self.method_call(object, name, args, e.line, ctx)
            }
            ExprKind::Vararg => Ok(ctx.varargs.to_vec()),
            _ => Ok(vec![self.eval(e, ctx)?]),
        }
    }

    fn call_expr(
        &mut self,
        f: &Expr,
        args: &[Expr],
        line: u32,
        ctx: &mut Ctx,
    ) -> Result<Vec<Value>, LuaError> {
        let function = self.eval(f, ctx)?;
        let args = self.eval_list(args, ctx)?;
        if !matches!(function, Value::Function(_)) {
            return Err(self.runtime_error(
                ctx,
                line,
                format!(
                    "attempt to call a {} value{}",
                    function.type_name(),
                    var_info(f)
                ),
            ));
        }
        self.set_line(line);
        self.call(&function, args)
    }

    fn method_call(
        &mut self,
        object: &Expr,
        name: &LuaString,
        args: &[Expr],
        line: u32,
        ctx: &mut Ctx,
    ) -> Result<Vec<Value>, LuaError> {
        let o = self.eval(object, ctx)?;
        let function = self.index(&o, &Value::String(name.clone()), object, line, ctx)?;
        if !matches!(function, Value::Function(_)) {
            return Err(self.runtime_error(
                ctx,
                line,
                format!(
                    "attempt to call a {} value (method '{}')",
                    function.type_name(),
                    name
                ),
            ));
        }
        let mut all_args = vec![o];
        all_args.extend(self.eval_list(args, ctx)?);
        self.set_line(line);
        self.call(&function, all_args)
    }

    fn index(
        &mut self,
        object: &Value,
        key: &Value,
        object_expr: &Expr,
        line: u32,
        ctx: &Ctx,
    ) -> Result<Value, LuaError> {
        match object {
            Value::Table(t) => Ok(t.get(key)),
            Value::String(_) if self.string_library.is_some() => Ok(self
                .string_library
                .as_ref()
                .expect("string library")
                .get(key)),
            other => Err(self.runtime_error(
                ctx,
                line,
                format!(
                    "attempt to index a {} value{}",
                    other.type_name(),
                    var_info(object_expr)
                ),
            )),
        }
    }

    fn eval(&mut self, e: &Expr, ctx: &mut Ctx) -> Result<Value, LuaError> {
        Ok(match &e.kind {
            ExprKind::Nil => Value::Nil,
            ExprKind::True => Value::Boolean(true),
            ExprKind::False => Value::Boolean(false),
            ExprKind::Number(n) => Value::Number(*n),
            ExprKind::String(s) => Value::String(s.clone()),
            ExprKind::Vararg => ctx.varargs.first().cloned().unwrap_or_default(),
            ExprKind::Function(body) => {
                self.make_closure(body.clone(), ctx.env.clone(), ctx.chunk.clone())
            }
            ExprKind::Table(fields) => self.eval_table(fields, e.line, ctx)?,
            ExprKind::Binary(BinOp::And, a, b) => {
                let v = self.eval(a, ctx)?;
                if !v.truthy() { v } else { self.eval(b, ctx)? }
            }
            ExprKind::Binary(BinOp::Or, a, b) => {
                let v = self.eval(a, ctx)?;
                if v.truthy() { v } else { self.eval(b, ctx)? }
            }
            ExprKind::Binary(op, a, b) => {
                let x = self.eval(a, ctx)?;
                let y = self.eval(b, ctx)?;
                self.binary(*op, x, y, a, b, e.line, ctx)?
            }
            ExprKind::Unary(op, a) => {
                let x = self.eval(a, ctx)?;
                match op {
                    UnOp::Not => Value::Boolean(!x.truthy()),
                    UnOp::Neg => match x.to_number() {
                        Some(n) => Value::Number(-n),
                        None => {
                            return Err(self.runtime_error(
                                ctx,
                                e.line,
                                format!(
                                    "attempt to perform arithmetic on a {} value{}",
                                    x.type_name(),
                                    var_info(a)
                                ),
                            ));
                        }
                    },
                    UnOp::Len => match &x {
                        Value::String(s) => Value::Number(s.len() as f64),
                        Value::Table(t) => Value::Number(t.len() as f64),
                        other => {
                            return Err(self.runtime_error(
                                ctx,
                                e.line,
                                format!(
                                    "attempt to get length of a {} value{}",
                                    other.type_name(),
                                    var_info(a)
                                ),
                            ));
                        }
                    },
                }
            }
            ExprKind::Name(name, _) => match lookup(&ctx.env, name) {
                Some(scope) => scope.cell.borrow().clone(),
                None => self.globals.get(&Value::String(LuaString::from(&**name))),
            },
            ExprKind::Index(object, key) => {
                let o = self.eval(object, ctx)?;
                let k = self.eval(key, ctx)?;
                self.index(&o, &k, object, e.line, ctx)?
            }
            ExprKind::Call(..) | ExprKind::Method(..) => self
                .eval_multi(e, ctx)?
                .into_iter()
                .next()
                .unwrap_or_default(),
            ExprKind::Paren(inner) => self.eval(inner, ctx)?,
        })
    }

    fn eval_table(
        &mut self,
        fields: &[Field],
        line: u32,
        ctx: &mut Ctx,
    ) -> Result<Value, LuaError> {
        let table = TableRef::new();
        let mut pending: Vec<Value> = Vec::new();
        let mut next_index = 1usize;
        let flush = |table: &TableRef, pending: &mut Vec<Value>, next_index: &mut usize| {
            for value in pending.drain(..) {
                table
                    .set(Value::Number(*next_index as f64), value)
                    .expect("integer keys are valid");
                *next_index += 1;
            }
        };
        for (i, field) in fields.iter().enumerate() {
            match field {
                Field::Positional(e) => {
                    if i + 1 == fields.len() && e.is_multi() {
                        pending.extend(self.eval_multi(e, ctx)?);
                    } else {
                        pending.push(self.eval(e, ctx)?);
                    }
                    if pending.len() >= FIELDS_PER_FLUSH {
                        flush(&table, &mut pending, &mut next_index);
                    }
                }
                Field::Keyed(k, v) => {
                    let key = self.eval(k, ctx)?;
                    let value = self.eval(v, ctx)?;
                    table
                        .set(key, value)
                        .map_err(|err| self.runtime_error(ctx, line, err.to_string()))?;
                }
            }
        }
        flush(&table, &mut pending, &mut next_index);
        Ok(Value::Table(table))
    }

    #[allow(clippy::too_many_arguments)]
    fn binary(
        &mut self,
        op: BinOp,
        x: Value,
        y: Value,
        a: &Expr,
        b: &Expr,
        line: u32,
        ctx: &Ctx,
    ) -> Result<Value, LuaError> {
        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Pow => {
                match (x.to_number(), y.to_number()) {
                    (Some(p), Some(q)) => Ok(Value::Number(arith(op, p, q))),
                    (p, _) => {
                        let (culprit, culprit_expr) = if p.is_none() { (&x, a) } else { (&y, b) };
                        Err(self.runtime_error(
                            ctx,
                            line,
                            format!(
                                "attempt to perform arithmetic on a {} value{}",
                                culprit.type_name(),
                                var_info(culprit_expr)
                            ),
                        ))
                    }
                }
            }
            BinOp::Concat => match (x.to_lua_string(), y.to_lua_string()) {
                (Some(p), Some(q)) => {
                    let mut bytes = p.as_bytes().to_vec();
                    bytes.extend_from_slice(q.as_bytes());
                    Ok(Value::String(LuaString::from(bytes)))
                }
                (p, _) => {
                    let (culprit, culprit_expr) = if p.is_some() { (&y, b) } else { (&x, a) };
                    Err(self.runtime_error(
                        ctx,
                        line,
                        format!(
                            "attempt to concatenate a {} value{}",
                            culprit.type_name(),
                            var_info(culprit_expr)
                        ),
                    ))
                }
            },
            BinOp::Eq => Ok(Value::Boolean(x.raw_equals(&y))),
            BinOp::Ne => Ok(Value::Boolean(!x.raw_equals(&y))),
            BinOp::Lt => self
                .less_than(&x, &y)
                .map_err(|m| self.runtime_error(ctx, line, m)),
            BinOp::Le => self
                .less_equal(&x, &y)
                .map_err(|m| self.runtime_error(ctx, line, m)),
            BinOp::Gt => self
                .less_than(&y, &x)
                .map_err(|m| self.runtime_error(ctx, line, m)),
            BinOp::Ge => self
                .less_equal(&y, &x)
                .map_err(|m| self.runtime_error(ctx, line, m)),
            BinOp::And | BinOp::Or => unreachable!("short-circuit operators handled in eval"),
        }
    }

    /// `a < b` (`luaV_lessthan` without metamethods); the error is the
    /// message without position.
    pub fn less_than(&self, a: &Value, b: &Value) -> Result<Value, String> {
        match (a, b) {
            (Value::Number(x), Value::Number(y)) => Ok(Value::Boolean(x < y)),
            (Value::String(x), Value::String(y)) => Ok(Value::Boolean(x.as_bytes() < y.as_bytes())),
            _ => Err(order_error(a, b)),
        }
    }

    /// `a <= b` (`luaV_lessequal` without metamethods).
    pub fn less_equal(&self, a: &Value, b: &Value) -> Result<Value, String> {
        match (a, b) {
            (Value::Number(x), Value::Number(y)) => Ok(Value::Boolean(x <= y)),
            (Value::String(x), Value::String(y)) => {
                Ok(Value::Boolean(x.as_bytes() <= y.as_bytes()))
            }
            _ => Err(order_error(a, b)),
        }
    }
}

fn order_error(a: &Value, b: &Value) -> String {
    let (t1, t2) = (a.type_name(), b.type_name());
    if t1 == t2 {
        format!("attempt to compare two {t1} values")
    } else {
        format!("attempt to compare {t1} with {t2}")
    }
}

/// Lua 5.2 arithmetic on numbers (`luai_num*`).
pub fn arith(op: BinOp, a: f64, b: f64) -> f64 {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
        BinOp::Mod => a - (a / b).floor() * b,
        BinOp::Pow => a.powf(b),
        _ => unreachable!("not an arithmetic operator"),
    }
}

/// `varinfo`: describes the variable an erroneous value came from.
fn var_info(e: &Expr) -> String {
    match &e.kind {
        ExprKind::Name(name, kind) => {
            let kind = match kind {
                NameKind::Local => "local",
                NameKind::Upvalue => "upvalue",
                NameKind::Global => "global",
            };
            format!(" ({kind} '{name}')")
        }
        ExprKind::Index(_, key) => match &key.kind {
            ExprKind::String(s) => format!(" (field '{s}')"),
            _ => String::new(),
        },
        _ => String::new(),
    }
}
