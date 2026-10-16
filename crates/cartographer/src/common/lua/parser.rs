//! Recursive descent parser for the supported Lua 5.2 subset (structure,
//! operator priorities and error messages follow `lparser.c`).

use std::rc::Rc;

use super::ast::{BinOp, Block, Expr, ExprKind, Field, FuncBody, NameKind, Stat, StatKind, UnOp};
use super::lexer::{Lexer, Tok, Token};
use super::value::LuaString;

/// Maximum nesting of syntactic constructs (`LUAI_MAXCCALLS`).
const MAX_LEVELS: usize = 200;
/// Priority of unary operators.
const UNARY_PRIORITY: u8 = 8;

struct FuncState {
    /// Active local variable names, one vector per block.
    blocks: Vec<Vec<Rc<str>>>,
    is_vararg: bool,
    /// Number of enclosing loops (for `break`).
    loops: usize,
}

/// The parser state.
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
    lookahead: Option<Token>,
    functions: Vec<FuncState>,
    level: usize,
}

/// Parses a chunk. Returns the main function (a vararg function without
/// parameters) or a formatted syntax error.
pub fn parse_chunk(source: &[u8], chunk: Rc<str>) -> Result<Rc<FuncBody>, String> {
    let mut lexer = Lexer::new(source, chunk);
    let current = lexer.next_token()?;
    let mut parser = Parser {
        lexer,
        current,
        lookahead: None,
        functions: vec![FuncState {
            blocks: vec![Vec::new()],
            is_vararg: true,
            loops: 0,
        }],
        level: 0,
    };
    let body = parser.block()?;
    if parser.current.tok != Tok::Eof {
        return Err(parser.error_expected(&Tok::Eof));
    }
    Ok(Rc::new(FuncBody {
        params: Vec::new(),
        is_vararg: true,
        body,
        line: 0,
    }))
}

fn binary_op(tok: &Tok) -> Option<(BinOp, u8, u8)> {
    Some(match tok {
        Tok::Char(b'+') => (BinOp::Add, 6, 6),
        Tok::Char(b'-') => (BinOp::Sub, 6, 6),
        Tok::Char(b'*') => (BinOp::Mul, 7, 7),
        Tok::Char(b'/') => (BinOp::Div, 7, 7),
        Tok::Char(b'%') => (BinOp::Mod, 7, 7),
        Tok::Char(b'^') => (BinOp::Pow, 10, 9),
        Tok::Concat => (BinOp::Concat, 5, 4),
        Tok::Eq => (BinOp::Eq, 3, 3),
        Tok::Ne => (BinOp::Ne, 3, 3),
        Tok::Char(b'<') => (BinOp::Lt, 3, 3),
        Tok::Le => (BinOp::Le, 3, 3),
        Tok::Char(b'>') => (BinOp::Gt, 3, 3),
        Tok::Ge => (BinOp::Ge, 3, 3),
        Tok::And => (BinOp::And, 2, 2),
        Tok::Or => (BinOp::Or, 1, 1),
        _ => return None,
    })
}

impl Parser<'_> {
    fn next(&mut self) -> Result<(), String> {
        self.current = match self.lookahead.take() {
            Some(t) => t,
            None => self.lexer.next_token()?,
        };
        Ok(())
    }

    fn peek(&mut self) -> Result<&Token, String> {
        if self.lookahead.is_none() {
            self.lookahead = Some(self.lexer.next_token()?);
        }
        Ok(self.lookahead.as_ref().expect("lookahead"))
    }

    fn line(&self) -> u32 {
        self.current.line
    }

    /// `luaX_syntaxerror`: error at the current token.
    fn syntax_error(&self, msg: &str) -> String {
        format!(
            "{}:{}: {} near {}",
            self.lexer.chunk(),
            self.lexer.line(),
            msg,
            self.current.near()
        )
    }

    fn error_expected(&self, tok: &Tok) -> String {
        self.syntax_error(&format!("{} expected", tok.describe()))
    }

    fn test_next(&mut self, tok: &Tok) -> Result<bool, String> {
        if self.current.tok == *tok {
            self.next()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn check(&self, tok: &Tok) -> Result<(), String> {
        if self.current.tok == *tok {
            Ok(())
        } else {
            Err(self.error_expected(tok))
        }
    }

    fn check_next(&mut self, tok: &Tok) -> Result<(), String> {
        self.check(tok)?;
        self.next()
    }

    fn check_match(&mut self, what: &Tok, who: &Tok, line: u32) -> Result<(), String> {
        if self.test_next(what)? {
            return Ok(());
        }
        if line == self.line() {
            Err(self.error_expected(what))
        } else {
            Err(self.syntax_error(&format!(
                "{} expected (to close {} at line {})",
                what.describe(),
                who.describe(),
                line
            )))
        }
    }

    fn check_name(&mut self) -> Result<Rc<str>, String> {
        match &self.current.tok {
            Tok::Name(n) => {
                let n = n.clone();
                self.next()?;
                Ok(n)
            }
            _ => Err(self.error_expected(&Tok::Name(Rc::from("")))),
        }
    }

    fn enter_level(&mut self) -> Result<(), String> {
        self.level += 1;
        if self.level > MAX_LEVELS {
            return Err(self.syntax_error("chunk has too many syntax levels"));
        }
        Ok(())
    }

    fn leave_level(&mut self) {
        self.level -= 1;
    }

    fn fs(&mut self) -> &mut FuncState {
        self.functions.last_mut().expect("function state")
    }

    fn declare_local(&mut self, name: Rc<str>) {
        self.fs().blocks.last_mut().expect("block").push(name);
    }

    fn resolve(&self, name: &str) -> NameKind {
        for (depth, fs) in self.functions.iter().rev().enumerate() {
            if fs.blocks.iter().any(|b| b.iter().any(|n| &**n == name)) {
                return if depth == 0 {
                    NameKind::Local
                } else {
                    NameKind::Upvalue
                };
            }
        }
        NameKind::Global
    }

    fn block_follow(&self, with_until: bool) -> bool {
        match self.current.tok {
            Tok::Else | Tok::Elseif | Tok::End | Tok::Eof => true,
            Tok::Until => with_until,
            _ => false,
        }
    }

    /// Parses a block in a new scope.
    fn block(&mut self) -> Result<Block, String> {
        self.fs().blocks.push(Vec::new());
        let result = self.statement_list();
        self.fs().blocks.pop();
        result
    }

    fn statement_list(&mut self) -> Result<Block, String> {
        let mut block = Block::default();
        while !self.block_follow(true) {
            if self.current.tok == Tok::Return {
                let line = self.line();
                self.next()?;
                let exprs = if self.block_follow(true) || self.current.tok == Tok::Char(b';') {
                    Vec::new()
                } else {
                    self.expr_list()?
                };
                self.test_next(&Tok::Char(b';'))?;
                block.ret = Some((exprs, line));
                break;
            }
            if let Some(stat) = self.statement()? {
                block.stats.push(stat);
            }
        }
        Ok(block)
    }

    fn statement(&mut self) -> Result<Option<Stat>, String> {
        let line = self.line();
        self.enter_level()?;
        let kind = match self.current.tok {
            Tok::Char(b';') => {
                self.next()?;
                None
            }
            Tok::If => Some(self.if_stat(line)?),
            Tok::While => {
                self.next()?;
                let cond = self.expr()?;
                self.check_next(&Tok::Do)?;
                let body = self.loop_block()?;
                self.check_match(&Tok::End, &Tok::While, line)?;
                Some(StatKind::While(cond, body))
            }
            Tok::Do => {
                self.next()?;
                let body = self.block()?;
                self.check_match(&Tok::End, &Tok::Do, line)?;
                Some(StatKind::Do(body))
            }
            Tok::For => Some(self.for_stat(line)?),
            Tok::Repeat => {
                self.next()?;
                // The condition can see the body's locals.
                self.fs().blocks.push(Vec::new());
                self.fs().loops += 1;
                let body = self.statement_list();
                self.fs().loops -= 1;
                let result = body.and_then(|body| {
                    self.check_match(&Tok::Until, &Tok::Repeat, line)?;
                    let cond = self.expr()?;
                    Ok(StatKind::Repeat(body, cond))
                });
                self.fs().blocks.pop();
                Some(result?)
            }
            Tok::Function => Some(self.func_stat(line)?),
            Tok::Local => {
                self.next()?;
                if self.test_next(&Tok::Function)? {
                    let name = self.check_name()?;
                    self.declare_local(name.clone());
                    let body = self.body(false, line)?;
                    Some(StatKind::LocalFunction(name, body))
                } else {
                    let mut names = vec![self.check_name()?];
                    while self.test_next(&Tok::Char(b','))? {
                        names.push(self.check_name()?);
                    }
                    let values = if self.test_next(&Tok::Char(b'='))? {
                        self.expr_list()?
                    } else {
                        Vec::new()
                    };
                    for name in &names {
                        self.declare_local(name.clone());
                    }
                    Some(StatKind::Local(names, values))
                }
            }
            Tok::DbColon | Tok::Goto => {
                return Err(self.syntax_error("goto and labels are not supported"));
            }
            Tok::Break => {
                self.next()?;
                if self.fs().loops == 0 {
                    return Err(format!(
                        "{}:{}: <break> at line {} not inside a loop",
                        self.lexer.chunk(),
                        self.lexer.line(),
                        line
                    ));
                }
                Some(StatKind::Break)
            }
            _ => Some(self.expr_stat()?),
        };
        self.leave_level();
        Ok(kind.map(|kind| Stat { kind, line }))
    }

    fn loop_block(&mut self) -> Result<Block, String> {
        self.fs().loops += 1;
        let result = self.block();
        self.fs().loops -= 1;
        result
    }

    fn if_stat(&mut self, line: u32) -> Result<StatKind, String> {
        let mut conds = Vec::new();
        // IF cond THEN block
        self.next()?;
        let cond = self.expr()?;
        self.check_next(&Tok::Then)?;
        conds.push((cond, self.block()?));
        while self.current.tok == Tok::Elseif {
            self.next()?;
            let cond = self.expr()?;
            self.check_next(&Tok::Then)?;
            conds.push((cond, self.block()?));
        }
        let else_block = if self.test_next(&Tok::Else)? {
            Some(self.block()?)
        } else {
            None
        };
        self.check_match(&Tok::End, &Tok::If, line)?;
        Ok(StatKind::If(conds, else_block))
    }

    fn for_stat(&mut self, line: u32) -> Result<StatKind, String> {
        self.next()?; // skip 'for'
        let var = self.check_name()?;
        match self.current.tok {
            Tok::Char(b'=') => {
                self.next()?;
                let start = self.expr()?;
                self.check_next(&Tok::Char(b','))?;
                let limit = self.expr()?;
                let step = if self.test_next(&Tok::Char(b','))? {
                    Some(self.expr()?)
                } else {
                    None
                };
                self.check_next(&Tok::Do)?;
                self.fs().blocks.push(vec![var.clone()]);
                let body = self.loop_block();
                self.fs().blocks.pop();
                let body = body?;
                self.check_match(&Tok::End, &Tok::For, line)?;
                Ok(StatKind::NumericFor {
                    var,
                    start,
                    limit,
                    step,
                    body,
                })
            }
            Tok::Char(b',') | Tok::In => {
                let mut names = vec![var];
                while self.test_next(&Tok::Char(b','))? {
                    names.push(self.check_name()?);
                }
                self.check_next(&Tok::In)?;
                let exprs = self.expr_list()?;
                self.check_next(&Tok::Do)?;
                self.fs().blocks.push(names.clone());
                let body = self.loop_block();
                self.fs().blocks.pop();
                let body = body?;
                self.check_match(&Tok::End, &Tok::For, line)?;
                Ok(StatKind::GenericFor { names, exprs, body })
            }
            _ => Err(self.syntax_error("'=' or 'in' expected")),
        }
    }

    fn func_stat(&mut self, line: u32) -> Result<StatKind, String> {
        self.next()?; // skip 'function'
        // funcname: NAME {'.' NAME} [':' NAME]
        let name_line = self.line();
        let name = self.check_name()?;
        let kind = self.resolve(&name);
        let mut target = Expr {
            kind: ExprKind::Name(name, kind),
            line: name_line,
        };
        let mut is_method = false;
        loop {
            let field_line = self.line();
            if self.test_next(&Tok::Char(b'.'))? {
                let key = self.check_name()?;
                target = index_expr(target, &key, field_line);
            } else if self.test_next(&Tok::Char(b':'))? {
                let key = self.check_name()?;
                target = index_expr(target, &key, field_line);
                is_method = true;
                break;
            } else {
                break;
            }
        }
        let body = self.body(is_method, line)?;
        let value = Expr {
            kind: ExprKind::Function(body),
            line,
        };
        Ok(StatKind::Assign(vec![target], vec![value]))
    }

    fn expr_stat(&mut self) -> Result<StatKind, String> {
        let first = self.suffixed_expr()?;
        if matches!(self.current.tok, Tok::Char(b'=') | Tok::Char(b',')) {
            let mut targets = vec![first];
            loop {
                let target = targets.last().expect("target");
                if !matches!(target.kind, ExprKind::Name(..) | ExprKind::Index(..)) {
                    return Err(self.syntax_error("syntax error"));
                }
                if !self.test_next(&Tok::Char(b','))? {
                    break;
                }
                let next = self.suffixed_expr()?;
                targets.push(next);
            }
            self.check_next(&Tok::Char(b'='))?;
            let values = self.expr_list()?;
            Ok(StatKind::Assign(targets, values))
        } else {
            if !matches!(first.kind, ExprKind::Call(..) | ExprKind::Method(..)) {
                return Err(self.syntax_error("syntax error"));
            }
            Ok(StatKind::Call(first))
        }
    }

    fn body(&mut self, is_method: bool, line: u32) -> Result<Rc<FuncBody>, String> {
        self.check_next(&Tok::Char(b'('))?;
        let mut params: Vec<Rc<str>> = Vec::new();
        if is_method {
            params.push(Rc::from("self"));
        }
        let mut is_vararg = false;
        if self.current.tok != Tok::Char(b')') {
            loop {
                match &self.current.tok {
                    Tok::Name(n) => {
                        params.push(n.clone());
                        self.next()?;
                    }
                    Tok::Dots => {
                        self.next()?;
                        is_vararg = true;
                    }
                    _ => return Err(self.syntax_error("<name> expected")),
                }
                if is_vararg || !self.test_next(&Tok::Char(b','))? {
                    break;
                }
            }
        }
        self.check_next(&Tok::Char(b')'))?;
        self.functions.push(FuncState {
            blocks: vec![params.clone()],
            is_vararg,
            loops: 0,
        });
        let body = self.block();
        self.functions.pop();
        let body = body?;
        self.check_match(&Tok::End, &Tok::Function, line)?;
        Ok(Rc::new(FuncBody {
            params,
            is_vararg,
            body,
            line,
        }))
    }

    fn expr_list(&mut self) -> Result<Vec<Expr>, String> {
        let mut list = vec![self.expr()?];
        while self.test_next(&Tok::Char(b','))? {
            list.push(self.expr()?);
        }
        Ok(list)
    }

    /// Parses an expression.
    pub fn expr(&mut self) -> Result<Expr, String> {
        self.sub_expr(0)
    }

    fn sub_expr(&mut self, limit: u8) -> Result<Expr, String> {
        self.enter_level()?;
        let line = self.line();
        let unary = match self.current.tok {
            Tok::Not => Some(UnOp::Not),
            Tok::Char(b'-') => Some(UnOp::Neg),
            Tok::Char(b'#') => Some(UnOp::Len),
            _ => None,
        };
        let mut left = if let Some(op) = unary {
            self.next()?;
            let operand = self.sub_expr(UNARY_PRIORITY)?;
            fold_unary(op, operand, line)
        } else {
            self.simple_expr()?
        };
        while let Some((op, left_priority, right_priority)) = binary_op(&self.current.tok) {
            if left_priority <= limit {
                break;
            }
            let line = self.line();
            self.next()?;
            let right = self.sub_expr(right_priority)?;
            left = Expr {
                kind: ExprKind::Binary(op, Box::new(left), Box::new(right)),
                line,
            };
        }
        self.leave_level();
        Ok(left)
    }

    fn simple_expr(&mut self) -> Result<Expr, String> {
        let line = self.line();
        let kind = match &self.current.tok {
            Tok::Number(n) => ExprKind::Number(*n),
            Tok::String(s) => ExprKind::String(s.clone()),
            Tok::Nil => ExprKind::Nil,
            Tok::True => ExprKind::True,
            Tok::False => ExprKind::False,
            Tok::Dots => {
                if !self.fs().is_vararg {
                    return Err(self.syntax_error("cannot use '...' outside a vararg function"));
                }
                ExprKind::Vararg
            }
            Tok::Char(b'{') => return self.constructor(),
            Tok::Function => {
                self.next()?;
                let body = self.body(false, line)?;
                return Ok(Expr {
                    kind: ExprKind::Function(body),
                    line,
                });
            }
            _ => return self.suffixed_expr(),
        };
        self.next()?;
        Ok(Expr { kind, line })
    }

    fn primary_expr(&mut self) -> Result<Expr, String> {
        let line = self.line();
        match &self.current.tok {
            Tok::Name(n) => {
                let n = n.clone();
                self.next()?;
                let kind = self.resolve(&n);
                Ok(Expr {
                    kind: ExprKind::Name(n, kind),
                    line,
                })
            }
            Tok::Char(b'(') => {
                self.next()?;
                let e = self.expr()?;
                self.check_match(&Tok::Char(b')'), &Tok::Char(b'('), line)?;
                Ok(Expr {
                    kind: ExprKind::Paren(Box::new(e)),
                    line,
                })
            }
            _ => Err(self.syntax_error("unexpected symbol")),
        }
    }

    fn suffixed_expr(&mut self) -> Result<Expr, String> {
        let mut e = self.primary_expr()?;
        loop {
            let line = self.line();
            match &self.current.tok {
                Tok::Char(b'.') => {
                    self.next()?;
                    let key = self.check_name()?;
                    e = index_expr(e, &key, line);
                }
                Tok::Char(b'[') => {
                    self.next()?;
                    let key = self.expr()?;
                    self.check_next(&Tok::Char(b']'))?;
                    e = Expr {
                        kind: ExprKind::Index(Box::new(e), Box::new(key)),
                        line,
                    };
                }
                Tok::Char(b':') => {
                    self.next()?;
                    let name = self.check_name()?;
                    let args = self.func_args(line)?;
                    e = Expr {
                        kind: ExprKind::Method(Box::new(e), LuaString::from(&*name), args),
                        line,
                    };
                }
                Tok::Char(b'(') | Tok::String(_) | Tok::Char(b'{') => {
                    let args = self.func_args(line)?;
                    e = Expr {
                        kind: ExprKind::Call(Box::new(e), args),
                        line,
                    };
                }
                _ => return Ok(e),
            }
        }
    }

    fn func_args(&mut self, line: u32) -> Result<Vec<Expr>, String> {
        match &self.current.tok {
            Tok::Char(b'(') => {
                self.next()?;
                let args = if self.current.tok == Tok::Char(b')') {
                    Vec::new()
                } else {
                    self.expr_list()?
                };
                self.check_match(&Tok::Char(b')'), &Tok::Char(b'('), line)?;
                Ok(args)
            }
            Tok::Char(b'{') => Ok(vec![self.constructor()?]),
            Tok::String(s) => {
                let e = Expr {
                    kind: ExprKind::String(s.clone()),
                    line: self.line(),
                };
                self.next()?;
                Ok(vec![e])
            }
            _ => Err(self.syntax_error("function arguments expected")),
        }
    }

    fn constructor(&mut self) -> Result<Expr, String> {
        let line = self.line();
        self.check_next(&Tok::Char(b'{'))?;
        let mut fields = Vec::new();
        loop {
            if self.current.tok == Tok::Char(b'}') {
                break;
            }
            let field_line = self.line();
            match &self.current.tok {
                Tok::Name(n) => {
                    let n = n.clone();
                    if self.peek()?.tok == Tok::Char(b'=') {
                        self.next()?; // name
                        self.next()?; // '='
                        let key = Expr {
                            kind: ExprKind::String(LuaString::from(&*n)),
                            line: field_line,
                        };
                        fields.push(Field::Keyed(key, self.expr()?));
                    } else {
                        fields.push(Field::Positional(self.expr()?));
                    }
                }
                Tok::Char(b'[') => {
                    self.next()?;
                    let key = self.expr()?;
                    self.check_next(&Tok::Char(b']'))?;
                    self.check_next(&Tok::Char(b'='))?;
                    fields.push(Field::Keyed(key, self.expr()?));
                }
                _ => fields.push(Field::Positional(self.expr()?)),
            }
            if !(self.test_next(&Tok::Char(b','))? || self.test_next(&Tok::Char(b';'))?) {
                break;
            }
        }
        self.check_match(&Tok::Char(b'}'), &Tok::Char(b'{'), line)?;
        Ok(Expr {
            kind: ExprKind::Table(fields),
            line,
        })
    }
}

fn index_expr(object: Expr, key: &str, line: u32) -> Expr {
    let key = Expr {
        kind: ExprKind::String(LuaString::from(key)),
        line,
    };
    Expr {
        kind: ExprKind::Index(Box::new(object), Box::new(key)),
        line,
    }
}

/// Constant folding of unary minus on numerals (like `luaK_prefix`), so
/// that `-0.8` is a single constant.
fn fold_unary(op: UnOp, operand: Expr, line: u32) -> Expr {
    if let (UnOp::Neg, ExprKind::Number(n)) = (op, &operand.kind)
        && *n != 0.
    {
        return Expr {
            kind: ExprKind::Number(-n),
            line,
        };
    }
    Expr {
        kind: ExprKind::Unary(op, Box::new(operand)),
        line,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_err(src: &str) -> String {
        match parse_chunk(src.as_bytes(), Rc::from("[string \"t\"]")) {
            Ok(_) => panic!("expected a syntax error for {src:?}"),
            Err(e) => e,
        }
    }

    #[test]
    fn parses_configuration_like_code() {
        let src = r#"
            include "pose_graph.lua"
            local x, y = 1, 2
            A = { b = { c = 1 }, [1] = 2, "x"; 3, }
            A.b.c = -math.huge
            A["x"], A.y = x .. y, #A
            function A.f(a, ...) return a, ... end
            function A:m() return self end
            local function g(n) if n > 0 then return g(n - 1) else return 0 end end
            for i = 1, 10, 2 do end
            for k, v in pairs(A) do break end
            while false do end
            repeat local z = 1 until z == 1
            do ; end
            return A
        "#;
        parse_chunk(src.as_bytes(), Rc::from("t")).unwrap();
    }

    #[test]
    fn syntax_errors() {
        assert_eq!(parse_err("x"), "[string \"t\"]:1: syntax error near <eof>");
        assert_eq!(
            parse_err("x = "),
            "[string \"t\"]:1: unexpected symbol near <eof>"
        );
        assert_eq!(
            parse_err("return 1 return"),
            "[string \"t\"]:1: <eof> expected near 'return'"
        );
        assert_eq!(
            parse_err("return return"),
            "[string \"t\"]:1: unexpected symbol near 'return'"
        );
        assert_eq!(
            parse_err("x = {1, 2"),
            "[string \"t\"]:1: '}' expected near <eof>"
        );
        assert_eq!(
            parse_err("x = {\n1, 2"),
            "[string \"t\"]:2: '}' expected (to close '{' at line 1) near <eof>"
        );
        assert_eq!(
            parse_err("function f()\n  return 1\n"),
            "[string \"t\"]:3: 'end' expected (to close 'function' at line 1) near <eof>"
        );
        assert_eq!(
            parse_err("local 1 = 2"),
            "[string \"t\"]:1: <name> expected near '1'"
        );
        assert_eq!(
            parse_err("break"),
            "[string \"t\"]:1: <break> at line 1 not inside a loop"
        );
        assert_eq!(
            parse_err("function f() return ... end"),
            "[string \"t\"]:1: cannot use '...' outside a vararg function near '...'"
        );
        assert_eq!(
            parse_err("for x do end"),
            "[string \"t\"]:1: '=' or 'in' expected near 'do'"
        );
        assert_eq!(
            parse_err("f() = 1"),
            "[string \"t\"]:1: syntax error near '='"
        );
        assert_eq!(
            parse_err("x = 1 +"),
            "[string \"t\"]:1: unexpected symbol near <eof>"
        );
        assert_eq!(
            parse_err("goto x"),
            "[string \"t\"]:1: goto and labels are not supported near 'goto'"
        );
        assert_eq!(
            parse_err("x = $"),
            "[string \"t\"]:1: unexpected symbol near '$'"
        );
        assert_eq!(
            parse_err(&format!("x = {}1{}", "(".repeat(300), ")".repeat(300))),
            "[string \"t\"]:1: chunk has too many syntax levels near '('"
        );
    }

    #[test]
    fn name_resolution() {
        let body = parse_chunk(b"local a; function f() return a, b end", Rc::from("t")).unwrap();
        let StatKind::Assign(_, values) = &body.body.stats[1].kind else {
            panic!("expected function assignment");
        };
        let ExprKind::Function(f) = &values[0].kind else {
            panic!("expected function");
        };
        let (ret, _) = f.body.ret.as_ref().unwrap();
        assert!(matches!(ret[0].kind, ExprKind::Name(_, NameKind::Upvalue)));
        assert!(matches!(ret[1].kind, ExprKind::Name(_, NameKind::Global)));
    }
}
