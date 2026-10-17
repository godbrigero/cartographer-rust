//! A minimal JSON reader (the workspace has no serde): enough for the e2e
//! reports, criterion's `estimates.json` and the demo's `metrics.json`.

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(Vec<(String, Value)>),
}

impl Value {
    pub fn parse(text: &str) -> Result<Value, String> {
        let mut p = Parser {
            s: text.as_bytes(),
            i: 0,
        };
        let v = p.value()?;
        p.ws();
        if p.i != p.s.len() {
            return Err(format!("trailing characters at byte {}", p.i));
        }
        Ok(v)
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Obj(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// `a.b.c` lookup through nested objects.
    pub fn path(&self, path: &str) -> Option<&Value> {
        path.split('.').try_fold(self, |v, k| v.get(k))
    }

    /// Number (`null` -> `None`).
    pub fn num(&self) -> Option<f64> {
        match self {
            Value::Num(x) => Some(*x),
            _ => None,
        }
    }

    /// Number or `NaN`.
    pub fn f(&self) -> f64 {
        self.num().unwrap_or(f64::NAN)
    }

    pub fn str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Array elements (empty for non-arrays).
    pub fn arr(&self) -> &[Value] {
        match self {
            Value::Arr(v) => v,
            _ => &[],
        }
    }

    /// Object entries (empty for non-objects).
    pub fn entries(&self) -> &[(String, Value)] {
        match self {
            Value::Obj(v) => v,
            _ => &[],
        }
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn expect(&mut self, c: u8) -> Result<(), String> {
        self.ws();
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", c as char, self.i))
        }
    }

    fn literal(&mut self, word: &str, v: Value) -> Result<Value, String> {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(format!("invalid literal at byte {}", self.i))
        }
    }

    fn value(&mut self) -> Result<Value, String> {
        self.ws();
        match self.peek().ok_or("unexpected end")? {
            b'{' => {
                self.i += 1;
                let mut entries = Vec::new();
                self.ws();
                if self.peek() == Some(b'}') {
                    self.i += 1;
                    return Ok(Value::Obj(entries));
                }
                loop {
                    self.ws();
                    let key = self.string()?;
                    self.expect(b':')?;
                    entries.push((key, self.value()?));
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Value::Obj(entries));
                        }
                        _ => return Err(format!("expected ',' or '}}' at byte {}", self.i)),
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.peek() == Some(b']') {
                    self.i += 1;
                    return Ok(Value::Arr(items));
                }
                loop {
                    items.push(self.value()?);
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Value::Arr(items));
                        }
                        _ => return Err(format!("expected ',' or ']' at byte {}", self.i)),
                    }
                }
            }
            b'"' => Ok(Value::Str(self.string()?)),
            b't' => self.literal("true", Value::Bool(true)),
            b'f' => self.literal("false", Value::Bool(false)),
            b'n' => self.literal("null", Value::Null),
            _ => {
                let start = self.i;
                while self
                    .peek()
                    .is_some_and(|c| c.is_ascii_digit() || b"+-.eE".contains(&c))
                {
                    self.i += 1;
                }
                std::str::from_utf8(&self.s[start..self.i])
                    .ok()
                    .and_then(|t| t.parse().ok())
                    .map(Value::Num)
                    .ok_or_else(|| format!("invalid number at byte {start}"))
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        if self.peek() != Some(b'"') {
            return Err(format!("expected string at byte {}", self.i));
        }
        self.i += 1;
        let mut out = Vec::new();
        loop {
            let c = self.peek().ok_or("unterminated string")?;
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = self.peek().ok_or("unterminated escape")?;
                    self.i += 1;
                    match e {
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'u' => {
                            let hex = std::str::from_utf8(
                                self.s.get(self.i..self.i + 4).ok_or("short \\u escape")?,
                            )
                            .map_err(|e| e.to_string())?;
                            let code = u32::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                            self.i += 4;
                            let ch = char::from_u32(code).unwrap_or('\u{fffd}');
                            let mut buf = [0; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        other => out.push(other),
                    }
                }
                c => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::Value;

    #[test]
    fn parses_nested_values() {
        let v = Value::parse(r#"{"a":[1,-2.5e1,null,true],"b":{"c":"x\"yA"}}"#).unwrap();
        assert_eq!(v.path("b.c").and_then(Value::str), Some("x\"yA"));
        let a = v.get("a").unwrap().arr();
        assert_eq!(a[1].num(), Some(-25.0));
        assert!(a[2].f().is_nan());
        assert_eq!(a[3].bool(), Some(true));
        assert!(Value::parse("[1,]").is_err());
    }
}
