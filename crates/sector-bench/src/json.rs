//! Minimal JSON emission.
//!
//! A serialisation dependency would be more surface than this needs: the
//! harness writes flat objects of numbers, strings and arrays, and reads none
//! of them back — the plotting does that.
//!
//! Escaping is real rather than assumed, because a dataset path with a
//! backslash in it would otherwise emit a file that no parser accepts and the
//! failure would surface far from its cause.

use std::fmt::Write as _;

/// A JSON value the harness emits.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A string, escaped on write.
    Str(String),
    /// An integer.
    Int(i64),
    /// A float. Non-finite values are written as `null`, since JSON has no
    /// representation for them and emitting `NaN` produces a file no parser
    /// accepts.
    Num(f64),
    /// A boolean.
    Bool(bool),
    /// An ordered list.
    List(Vec<Value>),
    /// An object, in insertion order so a diff between two runs is readable.
    Obj(Vec<(String, Value)>),
}

impl Value {
    /// Serialise to a JSON string.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, 0);
        out
    }

    fn write(&self, out: &mut String, depth: usize) {
        let pad = "  ".repeat(depth + 1);
        let close_pad = "  ".repeat(depth);
        match self {
            Value::Str(s) => {
                out.push('"');
                escape_into(s, out);
                out.push('"');
            }
            Value::Int(i) => {
                let _ = write!(out, "{i}");
            }
            Value::Num(f) => {
                if f.is_finite() {
                    let _ = write!(out, "{f:.6}");
                } else {
                    out.push_str("null");
                }
            }
            Value::Bool(b) => {
                let _ = write!(out, "{b}");
            }
            Value::List(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push_str("[\n");
                for (i, v) in items.iter().enumerate() {
                    out.push_str(&pad);
                    v.write(out, depth + 1);
                    if i + 1 < items.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                out.push_str(&close_pad);
                out.push(']');
            }
            Value::Obj(fields) => {
                if fields.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push_str("{\n");
                for (i, (k, v)) in fields.iter().enumerate() {
                    out.push_str(&pad);
                    out.push('"');
                    escape_into(k, out);
                    out.push_str("\": ");
                    v.write(out, depth + 1);
                    if i + 1 < fields.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                out.push_str(&close_pad);
                out.push('}');
            }
        }
    }
}

fn escape_into(s: &str, out: &mut String) {
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
}

/// Build an object from key/value pairs.
pub fn obj(fields: Vec<(&str, Value)>) -> Value {
    Value::Obj(
        fields
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

/// A string value.
pub fn s(v: &str) -> Value {
    Value::Str(v.to_string())
}

/// An integer value.
pub fn i(v: i64) -> Value {
    Value::Int(v)
}

/// A float value.
pub fn f(v: f64) -> Value {
    Value::Num(v)
}

/// A list of floats.
pub fn floats(v: &[f64]) -> Value {
    Value::List(v.iter().map(|x| Value::Num(*x)).collect())
}

/// A list of integers.
pub fn ints(v: &[i64]) -> Value {
    Value::List(v.iter().map(|x| Value::Int(*x)).collect())
}

/// Write a value to `measurements/<name>.json`.
pub fn write_measurement(name: &str, value: &Value) -> std::io::Result<std::path::PathBuf> {
    let dir = std::path::Path::new("measurements");
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{name}.json"));
    std::fs::write(&path, value.to_json())?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn objects_keep_insertion_order() {
        // Alphabetised keys would make a diff between two runs unreadable.
        let v = obj(vec![("zebra", i(1)), ("apple", i(2))]);
        let text = v.to_json();
        assert!(
            text.find("zebra").unwrap() < text.find("apple").unwrap(),
            "order not preserved: {text}"
        );
    }

    #[test]
    fn non_finite_floats_become_null_rather_than_invalid_json() {
        // A recall of 0 gives an infinite ratio; emitting `NaN` would produce a
        // file no parser accepts, and the failure would surface far from here.
        assert_eq!(f(f64::NAN).to_json(), "null");
        assert_eq!(f(f64::INFINITY).to_json(), "null");
        assert_eq!(f(1.5).to_json(), "1.500000");
    }

    #[test]
    fn strings_are_escaped() {
        // A dataset path can contain a backslash or a quote.
        let v = s("a\"b\\c\nd\te");
        assert_eq!(v.to_json(), r#""a\"b\\c\nd\te""#);
        // Control characters take the \u form.
        assert_eq!(s("\u{1}").to_json(), r#""\u0001""#);
    }

    #[test]
    fn nested_structures_round_trip_through_a_parser() {
        // Checked by shape rather than by eye: the emitted text must be
        // something a reader can consume.
        let v = obj(vec![
            ("config", obj(vec![("d", i(128)), ("m", i(16))])),
            ("recall", floats(&[0.605, 0.934])),
            ("empty_list", Value::List(vec![])),
            ("empty_obj", Value::Obj(vec![])),
        ]);
        let text = v.to_json();
        assert_eq!(text.matches('{').count(), text.matches('}').count());
        assert_eq!(text.matches('[').count(), text.matches(']').count());
        assert!(text.contains("\"empty_list\": []"));
        assert!(text.contains("\"empty_obj\": {}"));
        assert!(!text.contains(",\n  }"), "trailing comma: {text}");
    }

    #[test]
    fn an_empty_measurement_is_still_valid() {
        assert_eq!(Value::Obj(vec![]).to_json(), "{}");
    }
}
