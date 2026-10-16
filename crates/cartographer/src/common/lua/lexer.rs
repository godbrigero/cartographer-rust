//! Lexer for Lua 5.2 source code (a port of the relevant parts of `llex.c`,
//! including its error messages).

use std::rc::Rc;

use super::number::str_to_number;
use super::value::LuaString;

/// Token kinds.
#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    // Reserved words.
    And,
    Break,
    Do,
    Else,
    Elseif,
    End,
    False,
    For,
    Function,
    Goto,
    If,
    In,
    Local,
    Nil,
    Not,
    Or,
    Repeat,
    Return,
    Then,
    True,
    Until,
    While,
    // Multi-character symbols.
    Concat,
    Dots,
    Eq,
    Ge,
    Le,
    Ne,
    DbColon,
    Eof,
    Number(f64),
    Name(Rc<str>),
    String(LuaString),
    /// Single-character token.
    Char(u8),
}

const RESERVED: [(&str, Tok); 22] = [
    ("and", Tok::And),
    ("break", Tok::Break),
    ("do", Tok::Do),
    ("else", Tok::Else),
    ("elseif", Tok::Elseif),
    ("end", Tok::End),
    ("false", Tok::False),
    ("for", Tok::For),
    ("function", Tok::Function),
    ("goto", Tok::Goto),
    ("if", Tok::If),
    ("in", Tok::In),
    ("local", Tok::Local),
    ("nil", Tok::Nil),
    ("not", Tok::Not),
    ("or", Tok::Or),
    ("repeat", Tok::Repeat),
    ("return", Tok::Return),
    ("then", Tok::Then),
    ("true", Tok::True),
    ("until", Tok::Until),
    ("while", Tok::While),
];

impl Tok {
    /// `luaX_token2str`: how a token kind is named in error messages.
    pub fn describe(&self) -> String {
        match self {
            Tok::Char(c) => {
                if c.is_ascii_graphic() || *c == b' ' {
                    format!("'{}'", *c as char)
                } else {
                    format!("char({c})")
                }
            }
            Tok::Concat => "'..'".to_string(),
            Tok::Dots => "'...'".to_string(),
            Tok::Eq => "'=='".to_string(),
            Tok::Ge => "'>='".to_string(),
            Tok::Le => "'<='".to_string(),
            Tok::Ne => "'~='".to_string(),
            Tok::DbColon => "'::'".to_string(),
            Tok::Eof => "<eof>".to_string(),
            Tok::Number(_) => "<number>".to_string(),
            Tok::Name(_) => "<name>".to_string(),
            Tok::String(_) => "<string>".to_string(),
            reserved => {
                let name = RESERVED
                    .iter()
                    .find(|(_, t)| t == reserved)
                    .map(|(n, _)| *n)
                    .expect("reserved word");
                format!("'{name}'")
            }
        }
    }
}

/// A token with its line and, for names/strings/numbers, the source text
/// used in "near '...'" error messages.
#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    pub line: u32,
    pub text: Vec<u8>,
}

impl Token {
    /// `txtToken`: the token as shown in "near ..." error messages.
    pub fn near(&self) -> String {
        match self.tok {
            Tok::Name(_) | Tok::String(_) | Tok::Number(_) => {
                format!("'{}'", String::from_utf8_lossy(&self.text))
            }
            _ => self.tok.describe(),
        }
    }
}

/// Lua's `luaO_chunkid`: the chunk name as shown in error messages.
pub fn chunk_id(source: &str) -> String {
    const LUA_IDSIZE: usize = 60;
    if let Some(rest) = source.strip_prefix('=') {
        return truncate_bytes(rest, LUA_IDSIZE - 1).to_string();
    }
    if let Some(rest) = source.strip_prefix('@') {
        if rest.len() < LUA_IDSIZE {
            return rest.to_string();
        }
        let keep = LUA_IDSIZE - 1 - 3;
        let mut start = rest.len() - keep;
        while !rest.is_char_boundary(start) {
            start += 1;
        }
        return format!("...{}", &rest[start..]);
    }
    // String source: [string "source"].
    let bufflen = LUA_IDSIZE - "[string \"...\"]".len() - 1;
    let newline = source.find('\n');
    if source.len() < bufflen && newline.is_none() {
        format!("[string \"{source}\"]")
    } else {
        let mut l = newline.unwrap_or(source.len());
        if l > bufflen {
            l = bufflen;
        }
        format!("[string \"{}...\"]", truncate_bytes(source, l))
    }
}

fn truncate_bytes(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// The lexer state.
pub struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    line: u32,
    chunk: Rc<str>,
}

const EOZ: i32 = -1;

impl<'a> Lexer<'a> {
    /// Creates a lexer; `chunk` is the chunk id used in error messages.
    pub fn new(src: &'a [u8], chunk: Rc<str>) -> Self {
        let mut lexer = Lexer {
            src,
            pos: 0,
            line: 1,
            chunk,
        };
        // Skip a first line starting with '#' (Unix exec. file), like
        // luaL_loadfilex does for files.
        if lexer.src.first() == Some(&b'#') {
            while lexer.current() != EOZ && !lexer.is_newline() {
                lexer.pos += 1;
            }
        }
        lexer
    }

    /// The current line number.
    pub fn line(&self) -> u32 {
        self.line
    }

    /// The chunk id.
    pub fn chunk(&self) -> &Rc<str> {
        &self.chunk
    }

    fn current(&self) -> i32 {
        self.src.get(self.pos).map_or(EOZ, |&c| i32::from(c))
    }

    fn peek_at(&self, offset: usize) -> i32 {
        self.src
            .get(self.pos + offset)
            .map_or(EOZ, |&c| i32::from(c))
    }

    fn is_newline(&self) -> bool {
        matches!(self.current(), 0x0a | 0x0d)
    }

    fn cur_is(&self, c: u8) -> bool {
        self.current() == i32::from(c)
    }

    fn inc_line_number(&mut self) {
        let old = self.current();
        self.pos += 1; // skip '\n' or '\r'
        if self.is_newline() && self.current() != old {
            self.pos += 1; // skip '\n\r' or '\r\n'
        }
        self.line += 1;
    }

    /// Formats a lexer error like `lexerror`.
    fn error(&self, msg: &str, near: Option<String>) -> String {
        match near {
            Some(near) => format!("{}:{}: {} near {}", self.chunk, self.line, msg, near),
            None => format!("{}:{}: {}", self.chunk, self.line, msg),
        }
    }

    fn near_text(text: &[u8]) -> String {
        format!("'{}'", String::from_utf8_lossy(text))
    }

    /// Returns the next token, or a formatted syntax error message.
    pub fn next_token(&mut self) -> Result<Token, String> {
        loop {
            let c = self.current();
            let line = self.line;
            let simple = |tok: Tok| Token {
                tok,
                line,
                text: Vec::new(),
            };
            match c {
                EOZ => return Ok(simple(Tok::Eof)),
                0x0a | 0x0d => self.inc_line_number(),
                0x20 | 0x0c | 0x09 | 0x0b => self.pos += 1,
                _ => {
                    let c = c as u8;
                    match c {
                        b'-' => {
                            self.pos += 1;
                            if !self.cur_is(b'-') {
                                return Ok(simple(Tok::Char(b'-')));
                            }
                            // A comment.
                            self.pos += 1;
                            if self.cur_is(b'[') {
                                let start = self.pos;
                                let sep = self.skip_sep();
                                if sep >= 0 {
                                    self.read_long_string(sep as usize, false)?;
                                    continue;
                                }
                                self.pos = start;
                            }
                            // Short comment.
                            while !self.is_newline() && self.current() != EOZ {
                                self.pos += 1;
                            }
                        }
                        b'[' => {
                            let start = self.pos;
                            let sep = self.skip_sep();
                            if sep >= 0 {
                                let line = self.line;
                                let (content, raw) = self.read_long_string(sep as usize, true)?;
                                return Ok(Token {
                                    tok: Tok::String(LuaString::from(content)),
                                    line,
                                    text: raw,
                                });
                            } else if sep == -1 {
                                self.pos = start + 1;
                                return Ok(simple(Tok::Char(b'[')));
                            } else {
                                let text = self.src[start..self.pos].to_vec();
                                return Err(self.error(
                                    "invalid long string delimiter",
                                    Some(Self::near_text(&text)),
                                ));
                            }
                        }
                        b'=' => {
                            self.pos += 1;
                            if self.cur_is(b'=') {
                                self.pos += 1;
                                return Ok(simple(Tok::Eq));
                            }
                            return Ok(simple(Tok::Char(b'=')));
                        }
                        b'<' => {
                            self.pos += 1;
                            if self.cur_is(b'=') {
                                self.pos += 1;
                                return Ok(simple(Tok::Le));
                            }
                            return Ok(simple(Tok::Char(b'<')));
                        }
                        b'>' => {
                            self.pos += 1;
                            if self.cur_is(b'=') {
                                self.pos += 1;
                                return Ok(simple(Tok::Ge));
                            }
                            return Ok(simple(Tok::Char(b'>')));
                        }
                        b'~' => {
                            self.pos += 1;
                            if self.cur_is(b'=') {
                                self.pos += 1;
                                return Ok(simple(Tok::Ne));
                            }
                            return Ok(simple(Tok::Char(b'~')));
                        }
                        b':' => {
                            self.pos += 1;
                            if self.cur_is(b':') {
                                self.pos += 1;
                                return Ok(simple(Tok::DbColon));
                            }
                            return Ok(simple(Tok::Char(b':')));
                        }
                        b'"' | b'\'' => return self.read_string(c),
                        b'.' => {
                            if self.peek_at(1) == i32::from(b'.') {
                                if self.peek_at(2) == i32::from(b'.') {
                                    self.pos += 3;
                                    return Ok(simple(Tok::Dots));
                                }
                                self.pos += 2;
                                return Ok(simple(Tok::Concat));
                            }
                            if !(self.peek_at(1) >= 0 && (self.peek_at(1) as u8).is_ascii_digit()) {
                                self.pos += 1;
                                return Ok(simple(Tok::Char(b'.')));
                            }
                            return self.read_numeral();
                        }
                        b'0'..=b'9' => return self.read_numeral(),
                        _ if c.is_ascii_alphabetic() || c == b'_' => {
                            let start = self.pos;
                            while self.current() >= 0 && {
                                let c = self.current() as u8;
                                c.is_ascii_alphanumeric() || c == b'_'
                            } {
                                self.pos += 1;
                            }
                            let text = &self.src[start..self.pos];
                            let name = std::str::from_utf8(text).expect("ascii");
                            if let Some((_, tok)) = RESERVED.iter().find(|(n, _)| *n == name) {
                                return Ok(simple(tok.clone()));
                            }
                            return Ok(Token {
                                tok: Tok::Name(Rc::from(name)),
                                line,
                                text: text.to_vec(),
                            });
                        }
                        _ => {
                            self.pos += 1;
                            return Ok(simple(Tok::Char(c)));
                        }
                    }
                }
            }
        }
    }

    /// `read_numeral`: reads digits, '.', hex digits and exponent markers
    /// with optional signs, then converts with `luaO_str2d`.
    fn read_numeral(&mut self) -> Result<Token, String> {
        let line = self.line;
        let start = self.pos;
        let first = self.current();
        let mut expo: &[u8] = b"Ee";
        self.pos += 1;
        if first == i32::from(b'0') && (self.cur_is(b'x') || self.cur_is(b'X')) {
            self.pos += 1;
            expo = b"Pp";
        }
        loop {
            if self.current() >= 0 && expo.contains(&(self.current() as u8)) {
                self.pos += 1;
                if self.cur_is(b'+') || self.cur_is(b'-') {
                    self.pos += 1;
                }
            }
            let c = self.current();
            if c >= 0 && ((c as u8).is_ascii_hexdigit() || c as u8 == b'.') {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = self.src[start..self.pos].to_vec();
        match str_to_number(&text) {
            Some(n) => Ok(Token {
                tok: Tok::Number(n),
                line,
                text,
            }),
            None => Err(self.error("malformed number", Some(Self::near_text(&text)))),
        }
    }

    /// `skip_sep`: at '[' or ']', counts the '='s. Returns the count if the
    /// bracket is repeated, -1 for a lone bracket without '=' and
    /// `-(count) - 1` otherwise.
    fn skip_sep(&mut self) -> i32 {
        let s = self.current();
        self.pos += 1;
        let mut count = 0;
        while self.cur_is(b'=') {
            self.pos += 1;
            count += 1;
        }
        if self.current() == s {
            count
        } else {
            -count - 1
        }
    }

    /// Reads a long string/comment after the opening `[==[`. Returns the
    /// content and the raw text.
    fn read_long_string(
        &mut self,
        sep: usize,
        is_string: bool,
    ) -> Result<(Vec<u8>, Vec<u8>), String> {
        let open_start = self.pos - sep - 1;
        self.pos += 1; // skip 2nd '['
        if self.is_newline() {
            self.inc_line_number();
        }
        let mut content = Vec::new();
        loop {
            match self.current() {
                EOZ => {
                    let msg = if is_string {
                        "unfinished long string"
                    } else {
                        "unfinished long comment"
                    };
                    return Err(self.error(msg, Some(Tok::Eof.describe())));
                }
                0x5d => {
                    // ']'
                    let start = self.pos;
                    if self.skip_sep() == sep as i32 {
                        self.pos += 1; // skip 2nd ']'
                        let raw = self.src[open_start..self.pos].to_vec();
                        return Ok((content, raw));
                    }
                    content.extend_from_slice(&self.src[start..self.pos]);
                }
                0x0a | 0x0d => {
                    content.push(b'\n');
                    self.inc_line_number();
                }
                c => {
                    content.push(c as u8);
                    self.pos += 1;
                }
            }
        }
    }

    fn read_string(&mut self, delimiter: u8) -> Result<Token, String> {
        let line = self.line;
        self.pos += 1;
        // Buffer like Lua's: delimiter + processed content (+ delimiter).
        let mut buffer = vec![delimiter];
        loop {
            let c = self.current();
            if c == i32::from(delimiter) {
                self.pos += 1;
                break;
            }
            match c {
                EOZ => return Err(self.error("unfinished string", Some(Tok::Eof.describe()))),
                0x0a | 0x0d => {
                    return Err(self.error("unfinished string", Some(Self::near_text(&buffer))));
                }
                0x5c => {
                    // '\\'
                    self.pos += 1;
                    let e = self.current();
                    let value: u8 = match e {
                        0x61 => 0x07, // \a
                        0x62 => 0x08, // \b
                        0x66 => 0x0c, // \f
                        0x6e => b'\n',
                        0x72 => b'\r',
                        0x74 => b'\t',
                        0x76 => 0x0b, // \v
                        0x78 => {
                            // \xXX
                            let mut r = 0u32;
                            let mut read = vec![b'x'];
                            for _ in 0..2 {
                                self.pos += 1;
                                let h = self.current();
                                if h >= 0 {
                                    read.push(h as u8);
                                }
                                match (h >= 0).then(|| (h as u8 as char).to_digit(16)).flatten() {
                                    Some(d) => r = (r << 4) + d,
                                    None => {
                                        return Err(
                                            self.escape_error(&read, "hexadecimal digit expected")
                                        );
                                    }
                                }
                            }
                            self.pos += 1;
                            buffer.push(r as u8);
                            continue;
                        }
                        0x0a | 0x0d => {
                            self.inc_line_number();
                            buffer.push(b'\n');
                            continue;
                        }
                        0x5c | 0x22 | 0x27 => e as u8,
                        EOZ => continue, // will raise an error next loop
                        0x7a => {
                            // \z: skip the following span of spaces
                            self.pos += 1;
                            while self.current() >= 0
                                && (self.current() as u8).is_ascii_whitespace()
                            {
                                if self.is_newline() {
                                    self.inc_line_number();
                                } else {
                                    self.pos += 1;
                                }
                            }
                            continue;
                        }
                        _ => {
                            if !(e >= 0 && (e as u8).is_ascii_digit()) {
                                return Err(
                                    self.escape_error(&[e as u8], "invalid escape sequence")
                                );
                            }
                            // \ddd: up to 3 decimal digits.
                            let mut r = 0u32;
                            let mut read = Vec::new();
                            for _ in 0..3 {
                                let d = self.current();
                                if d >= 0 && (d as u8).is_ascii_digit() {
                                    read.push(d as u8);
                                    r = 10 * r + (d as u32 - u32::from(b'0'));
                                    self.pos += 1;
                                } else {
                                    break;
                                }
                            }
                            if r > 255 {
                                if self.current() >= 0 {
                                    read.push(self.current() as u8);
                                }
                                return Err(self.escape_error(&read, "decimal escape too large"));
                            }
                            buffer.push(r as u8);
                            continue;
                        }
                    };
                    self.pos += 1;
                    buffer.push(value);
                }
                c => {
                    buffer.push(c as u8);
                    self.pos += 1;
                }
            }
        }
        let content = buffer[1..].to_vec();
        buffer.push(delimiter);
        Ok(Token {
            tok: Tok::String(LuaString::from(content)),
            line,
            text: buffer,
        })
    }

    fn escape_error(&self, read: &[u8], msg: &str) -> String {
        let mut text = vec![b'\\'];
        text.extend_from_slice(read);
        self.error(msg, Some(Self::near_text(&text)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(src: &str) -> Result<Vec<Tok>, String> {
        let mut lexer = Lexer::new(src.as_bytes(), Rc::from("[string \"test\"]"));
        let mut result = Vec::new();
        loop {
            let t = lexer.next_token()?;
            if t.tok == Tok::Eof {
                return Ok(result);
            }
            result.push(t.tok);
        }
    }

    fn s(x: &str) -> Tok {
        Tok::String(LuaString::from(x))
    }

    #[test]
    fn numbers() {
        assert_eq!(
            lex("3 3.0 3.25 325e-2 0.325E1 0xff 0x56 .5 5. 1e5 0x1p4 0xA.8").unwrap(),
            vec![
                Tok::Number(3.),
                Tok::Number(3.),
                Tok::Number(3.25),
                Tok::Number(3.25),
                Tok::Number(3.25),
                Tok::Number(255.),
                Tok::Number(86.),
                Tok::Number(0.5),
                Tok::Number(5.),
                Tok::Number(1e5),
                Tok::Number(16.),
                Tok::Number(10.5),
            ]
        );
        assert_eq!(
            lex("x = 3e").unwrap_err(),
            "[string \"test\"]:1: malformed number near '3e'"
        );
        assert_eq!(
            lex("3..2").unwrap_err(),
            "[string \"test\"]:1: malformed number near '3..2'"
        );
    }

    #[test]
    fn strings_and_escapes() {
        assert_eq!(
            lex(r#"'a' "b" "\65\066\x41\n\t\\\"\'" "a\z
                   b" "x\
y""#)
            .unwrap(),
            vec![s("a"), s("b"), s("AB\x41\n\t\\\"'"), s("ab"), s("x\ny")]
        );
        assert_eq!(lex("[[\nfoo]]").unwrap(), vec![s("foo")]);
        assert_eq!(lex("[==[a]]b]=]c]==]").unwrap(), vec![s("a]]b]=]c")]);
        assert_eq!(lex(r#""\0""#).unwrap(), vec![s("\0")]);
        assert_eq!(
            lex("'abc").unwrap_err(),
            "[string \"test\"]:1: unfinished string near <eof>"
        );
        assert_eq!(
            lex("'abc\nx'").unwrap_err(),
            "[string \"test\"]:1: unfinished string near ''abc'"
        );
        assert_eq!(
            lex(r#""\q""#).unwrap_err(),
            "[string \"test\"]:1: invalid escape sequence near '\\q'"
        );
        assert_eq!(
            lex(r#""\300""#).unwrap_err(),
            "[string \"test\"]:1: decimal escape too large near '\\300\"'"
        );
        assert_eq!(
            lex(r#""\xZZ""#).unwrap_err(),
            "[string \"test\"]:1: hexadecimal digit expected near '\\xZ'"
        );
        assert_eq!(
            lex("[[abc").unwrap_err(),
            "[string \"test\"]:1: unfinished long string near <eof>"
        );
        assert_eq!(
            lex("[=x").unwrap_err(),
            "[string \"test\"]:1: invalid long string delimiter near '[='"
        );
    }

    #[test]
    fn comments_and_symbols() {
        assert_eq!(
            lex("-- comment\na --[[ long\ncomment ]] b --[==[ x ]==] c -d").unwrap(),
            vec![
                Tok::Name(Rc::from("a")),
                Tok::Name(Rc::from("b")),
                Tok::Name(Rc::from("c")),
                Tok::Char(b'-'),
                Tok::Name(Rc::from("d")),
            ]
        );
        assert_eq!(
            lex("== ~= <= >= < > = .. ... :: : ; # ^ % [ ]").unwrap(),
            vec![
                Tok::Eq,
                Tok::Ne,
                Tok::Le,
                Tok::Ge,
                Tok::Char(b'<'),
                Tok::Char(b'>'),
                Tok::Char(b'='),
                Tok::Concat,
                Tok::Dots,
                Tok::DbColon,
                Tok::Char(b':'),
                Tok::Char(b';'),
                Tok::Char(b'#'),
                Tok::Char(b'^'),
                Tok::Char(b'%'),
                Tok::Char(b'['),
                Tok::Char(b']'),
            ]
        );
        assert_eq!(
            lex("--[[ open").unwrap_err(),
            "[string \"test\"]:1: unfinished long comment near <eof>"
        );
        // "--[" without a second '[' is a short comment.
        assert_eq!(lex("--[ x\ny").unwrap(), vec![Tok::Name(Rc::from("y"))]);
    }

    #[test]
    fn line_numbers() {
        let mut lexer = Lexer::new(b"a\r\nb\n\rc\rd\n\ne", Rc::from("x"));
        let lines: Vec<u32> = (0..5).map(|_| lexer.next_token().unwrap().line).collect();
        assert_eq!(lines, vec![1, 2, 3, 4, 6]);
    }

    #[test]
    fn chunk_ids() {
        assert_eq!(chunk_id("return {}"), "[string \"return {}\"]");
        assert_eq!(chunk_id("a\nb"), "[string \"a...\"]");
        assert_eq!(
            chunk_id(&"x".repeat(100)),
            format!("[string \"{}...\"]", "x".repeat(45))
        );
        assert_eq!(chunk_id("=stdin"), "stdin");
        assert_eq!(chunk_id("@foo.lua"), "foo.lua");
        assert_eq!(
            chunk_id(&format!("@{}", "y".repeat(100))),
            format!("...{}", "y".repeat(56))
        );
    }
}
