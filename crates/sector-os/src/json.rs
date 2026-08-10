//! JSON output, written by hand.
//!
//! Every command supports `--json` because a search engine gets scripted, and a
//! monitoring check that parses human-readable columns breaks the first time a
//! column is added.
//!
//! # Why not `serde`
//!
//! The workspace has no external dependencies — `Cargo.lock` holds only the
//! `sector-*` crates, and `deny.toml` treats a transitive `alloc` user as a
//! design defect. `serde` plus `serde_json` is roughly a dozen crates and a
//! procedural macro to emit output whose entire shape is under 400 lines of
//! writer here. The trade is deliberate: we own the escaping, and the audit
//! surface stays reviewable by hand.
//!
//! # What this does and does not do
//!
//! Serialization only. There is no parser, because nothing reads JSON: the
//! daemon accepts query vectors as a length-prefixed binary body and as a
//! minimal hand-parsed request line, and the CLI takes flags. A writer cannot be
//! attacked by malformed input the way a parser can, which is the property that
//! makes hand-rolling this defensible at all.
//!
//! Escaping follows RFC 8259: the two mandatory escapes, the five shorthand
//! control escapes, and `\u00XX` for every other C0 byte. Non-ASCII passes
//! through as UTF-8, which is valid JSON and what `serde_json` does by default.
//! `f32`/`f64` values that are not finite are written as `null` rather than the
//! bare `NaN` token, which is not JSON and which every strict parser rejects.

use std::fmt::Write as _;

/// Accumulates a JSON document.
///
/// Structural correctness is the caller's responsibility — this is a writer, not
/// a validator — so the API is shaped to make the common cases hard to get
/// wrong: [`Json::object`] and [`Json::array`] take closures and emit their own
/// delimiters, and the comma placement is handled by the builder rather than by
/// the caller counting fields.
#[derive(Debug, Default)]
pub struct Json {
    out: String,
}

impl Json {
    /// An empty document.
    pub fn new() -> Self {
        Self {
            out: String::with_capacity(256),
        }
    }

    /// The document, consuming the builder.
    pub fn finish(mut self) -> String {
        self.out.push('\n');
        self.out
    }

    /// Write an object, with `body` emitting its fields.
    pub fn object(&mut self, body: impl FnOnce(&mut ObjectWriter<'_>)) {
        self.out.push('{');
        let mut w = ObjectWriter {
            out: &mut self.out,
            first: true,
        };
        body(&mut w);
        self.out.push('}');
    }

    /// Write an array, with `body` emitting its elements.
    pub fn array(&mut self, body: impl FnOnce(&mut ArrayWriter<'_>)) {
        self.out.push('[');
        let mut w = ArrayWriter {
            out: &mut self.out,
            first: true,
        };
        body(&mut w);
        self.out.push(']');
    }
}

/// Emits an object's fields.
#[derive(Debug)]
pub struct ObjectWriter<'a> {
    out: &'a mut String,
    first: bool,
}

impl ObjectWriter<'_> {
    fn key(&mut self, name: &str) {
        if !self.first {
            self.out.push(',');
        }
        self.first = false;
        escape_into(self.out, name);
        self.out.push(':');
    }

    /// A string field.
    pub fn str(&mut self, name: &str, value: &str) {
        self.key(name);
        escape_into(self.out, value);
    }

    /// An integer field.
    pub fn int(&mut self, name: &str, value: i64) {
        self.key(name);
        let _ = write!(self.out, "{value}");
    }

    /// An unsigned field.
    pub fn uint(&mut self, name: &str, value: u64) {
        self.key(name);
        let _ = write!(self.out, "{value}");
    }

    /// A float field. Non-finite values are written as `null`.
    pub fn float(&mut self, name: &str, value: f64) {
        self.key(name);
        if value.is_finite() {
            let _ = write!(self.out, "{value}");
        } else {
            self.out.push_str("null");
        }
    }

    /// A boolean field.
    pub fn bool(&mut self, name: &str, value: bool) {
        self.key(name);
        self.out.push_str(if value { "true" } else { "false" });
    }

    /// A null field.
    pub fn null(&mut self, name: &str) {
        self.key(name);
        self.out.push_str("null");
    }

    /// An `Option<&str>`: the string, or `null`.
    pub fn opt_str(&mut self, name: &str, value: Option<&str>) {
        match value {
            Some(v) => self.str(name, v),
            None => self.null(name),
        }
    }

    /// A nested object.
    pub fn object(&mut self, name: &str, body: impl FnOnce(&mut ObjectWriter<'_>)) {
        self.key(name);
        self.out.push('{');
        let mut w = ObjectWriter {
            out: self.out,
            first: true,
        };
        body(&mut w);
        self.out.push('}');
    }

    /// A nested array.
    pub fn array(&mut self, name: &str, body: impl FnOnce(&mut ArrayWriter<'_>)) {
        self.key(name);
        self.out.push('[');
        let mut w = ArrayWriter {
            out: self.out,
            first: true,
        };
        body(&mut w);
        self.out.push(']');
    }

    /// An array of unsigned integers, the common case for id lists.
    pub fn uints(&mut self, name: &str, values: impl IntoIterator<Item = u64>) {
        self.array(name, |a| {
            for v in values {
                a.uint(v);
            }
        });
    }

    /// An array of signed integers, the common case for score lists.
    pub fn ints(&mut self, name: &str, values: impl IntoIterator<Item = i64>) {
        self.array(name, |a| {
            for v in values {
                a.int(v);
            }
        });
    }
}

/// Emits an array's elements.
#[derive(Debug)]
pub struct ArrayWriter<'a> {
    out: &'a mut String,
    first: bool,
}

impl ArrayWriter<'_> {
    fn sep(&mut self) {
        if !self.first {
            self.out.push(',');
        }
        self.first = false;
    }

    /// A string element.
    pub fn str(&mut self, value: &str) {
        self.sep();
        escape_into(self.out, value);
    }

    /// An integer element.
    pub fn int(&mut self, value: i64) {
        self.sep();
        let _ = write!(self.out, "{value}");
    }

    /// An unsigned element.
    pub fn uint(&mut self, value: u64) {
        self.sep();
        let _ = write!(self.out, "{value}");
    }

    /// A float element. Non-finite values are written as `null`.
    pub fn float(&mut self, value: f64) {
        self.sep();
        if value.is_finite() {
            let _ = write!(self.out, "{value}");
        } else {
            self.out.push_str("null");
        }
    }

    /// An object element.
    pub fn object(&mut self, body: impl FnOnce(&mut ObjectWriter<'_>)) {
        self.sep();
        self.out.push('{');
        let mut w = ObjectWriter {
            out: self.out,
            first: true,
        };
        body(&mut w);
        self.out.push('}');
    }
}

/// Write `s` as a quoted, escaped JSON string.
///
/// RFC 8259 requires escaping `"` and `\`, and forbids unescaped control
/// characters below 0x20. The five shorthand escapes are used where they exist
/// because they are shorter and more readable; everything else in C0 becomes
/// `\u00XX`. Bytes above 0x7F pass through: the input is `&str`, so it is valid
/// UTF-8, and JSON strings are UTF-8.
fn escape_into(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Escape `s` as a standalone JSON string.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    escape_into(&mut out, s);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_object_emits_its_fields_with_correct_commas() {
        let mut j = Json::new();
        j.object(|o| {
            o.str("name", "volume.sector");
            o.uint("vectors", 400);
            o.bool("ok", true);
        });
        assert_eq!(
            j.finish().trim(),
            r#"{"name":"volume.sector","vectors":400,"ok":true}"#
        );
    }

    #[test]
    fn an_empty_object_and_array_are_still_valid() {
        let mut j = Json::new();
        j.object(|_| {});
        assert_eq!(j.finish().trim(), "{}");
        let mut j = Json::new();
        j.array(|_| {});
        assert_eq!(j.finish().trim(), "[]");
    }

    #[test]
    fn nesting_keeps_commas_independent_per_level() {
        // The failure a hand-written writer makes: an inner value consuming the
        // outer separator state and emitting `{"a":1"b":2}`.
        let mut j = Json::new();
        j.object(|o| {
            o.uint("a", 1);
            o.object("inner", |i| {
                i.uint("x", 10);
                i.uint("y", 20);
            });
            o.uint("b", 2);
            o.uints("ids", [7u64, 8, 9]);
        });
        assert_eq!(
            j.finish().trim(),
            r#"{"a":1,"inner":{"x":10,"y":20},"b":2,"ids":[7,8,9]}"#
        );
    }

    #[test]
    fn every_mandatory_escape_is_applied() {
        // A quote or backslash passing through unescaped produces a document no
        // parser accepts, and is the classic hand-rolled-JSON bug.
        assert_eq!(escape(r#"say "hi""#), r#""say \"hi\"""#);
        assert_eq!(escape(r"C:\path"), r#""C:\\path""#);
        assert_eq!(escape("line\nbreak"), r#""line\nbreak""#);
        assert_eq!(escape("tab\there"), r#""tab\there""#);
        assert_eq!(escape("\r"), r#""\r""#);
        assert_eq!(escape("\u{8}\u{c}"), r#""\b\f""#);
    }

    #[test]
    fn other_control_characters_become_unicode_escapes() {
        // 0x00-0x1F must not appear raw. These have no shorthand.
        assert_eq!(escape("\u{0}"), r#""\u0000""#);
        assert_eq!(escape("\u{1}"), r#""\u0001""#);
        assert_eq!(escape("\u{1f}"), r#""\u001f""#);
    }

    #[test]
    fn non_ascii_passes_through_as_utf8() {
        // Valid JSON, and what a conventional serializer does by default.
        assert_eq!(escape("Pi Zero — ARMv6"), "\"Pi Zero — ARMv6\"");
    }

    #[test]
    fn keys_are_escaped_as_well_as_values() {
        // A key is a JSON string and needs the same treatment; forgetting it is
        // easy because keys are usually literals.
        let mut j = Json::new();
        j.object(|o| o.str("a\"b", "v"));
        assert_eq!(j.finish().trim(), r#"{"a\"b":"v"}"#);
    }

    #[test]
    fn non_finite_floats_are_null_not_nan() {
        // `NaN` and `Infinity` are not JSON tokens and every strict parser
        // rejects them, so a latency figure that came out non-finite must not
        // take the whole document down with it.
        let mut j = Json::new();
        j.object(|o| {
            o.float("ok", 1.5);
            o.float("nan", f64::NAN);
            o.float("inf", f64::INFINITY);
            o.float("neg", f64::NEG_INFINITY);
        });
        assert_eq!(
            j.finish().trim(),
            r#"{"ok":1.5,"nan":null,"inf":null,"neg":null}"#
        );
    }

    #[test]
    fn an_array_of_objects_separates_elements() {
        let mut j = Json::new();
        j.array(|a| {
            for i in 0..3u64 {
                a.object(|o| o.uint("i", i));
            }
        });
        assert_eq!(j.finish().trim(), r#"[{"i":0},{"i":1},{"i":2}]"#);
    }

    #[test]
    fn a_document_ends_with_exactly_one_newline() {
        // Line-oriented consumers split on it, and a missing or doubled newline
        // breaks a `jq -c` stream.
        let mut j = Json::new();
        j.object(|o| o.uint("a", 1));
        let s = j.finish();
        assert!(s.ends_with('\n'));
        assert!(!s.ends_with("\n\n"));
    }
}
