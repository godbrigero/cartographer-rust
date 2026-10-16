//! Abstract syntax tree of the supported Lua 5.2 subset.

use std::rc::Rc;

use super::value::LuaString;

/// Binary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Concat,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

/// Unary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
    Len,
}

/// How a name was resolved by the parser (used in error messages, like
/// Lua's "global 'x'", "local 'x'", "upvalue 'x'").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NameKind {
    Local,
    Upvalue,
    Global,
}

/// An expression with the line it starts on.
#[derive(Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub line: u32,
}

/// Expression kinds.
#[derive(Debug)]
pub enum ExprKind {
    Nil,
    True,
    False,
    Vararg,
    Number(f64),
    String(LuaString),
    Function(Rc<FuncBody>),
    Table(Vec<Field>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    Name(Rc<str>, NameKind),
    Index(Box<Expr>, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Method(Box<Expr>, LuaString, Vec<Expr>),
    Paren(Box<Expr>),
}

impl Expr {
    /// True for expressions that can produce multiple values.
    pub fn is_multi(&self) -> bool {
        matches!(
            self.kind,
            ExprKind::Call(..) | ExprKind::Method(..) | ExprKind::Vararg
        )
    }
}

/// A field of a table constructor.
#[derive(Debug)]
pub enum Field {
    /// `expr` (array item).
    Positional(Expr),
    /// `name = expr` or `[key] = expr`.
    Keyed(Expr, Expr),
}

/// A function body.
#[derive(Debug)]
pub struct FuncBody {
    pub params: Vec<Rc<str>>,
    pub is_vararg: bool,
    pub body: Block,
    pub line: u32,
}

/// A block of statements with an optional trailing `return`.
#[derive(Debug, Default)]
pub struct Block {
    pub stats: Vec<Stat>,
    pub ret: Option<(Vec<Expr>, u32)>,
}

/// A statement with the line it starts on.
#[derive(Debug)]
pub struct Stat {
    pub kind: StatKind,
    pub line: u32,
}

/// Statement kinds.
#[derive(Debug)]
pub enum StatKind {
    /// A function call used as a statement.
    Call(Expr),
    /// `targets = values`.
    Assign(Vec<Expr>, Vec<Expr>),
    /// `local names = values`.
    Local(Vec<Rc<str>>, Vec<Expr>),
    /// `local function name body`.
    LocalFunction(Rc<str>, Rc<FuncBody>),
    Do(Block),
    While(Expr, Block),
    Repeat(Block, Expr),
    If(Vec<(Expr, Block)>, Option<Block>),
    NumericFor {
        var: Rc<str>,
        start: Expr,
        limit: Expr,
        step: Option<Expr>,
        body: Block,
    },
    GenericFor {
        names: Vec<Rc<str>>,
        exprs: Vec<Expr>,
        body: Block,
    },
    Break,
}
