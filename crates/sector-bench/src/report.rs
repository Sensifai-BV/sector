//! Predicted-versus-measured reporting.
//!
//! Every claim this harness tests has a predicted value from the profile
//! arithmetic and a measured value from the run. Reporting them side by side
//! with an explicit verdict is the point: a measurement without its prediction
//! cannot falsify anything, and a prediction without its measurement is what
//! the report already had.

use crate::json::{self, Value};

/// How a measurement compares to its prediction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Within tolerance.
    Holds,
    /// Outside tolerance. A result to report, not a test to fix.
    Refuted,
    /// The instance could not decide — the quantity was unresolvable or absent.
    Inconclusive,
}

impl Verdict {
    /// Report string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Verdict::Holds => "holds",
            Verdict::Refuted => "REFUTED",
            Verdict::Inconclusive => "inconclusive",
        }
    }
}

/// One predicted-versus-measured comparison.
#[derive(Clone, Debug)]
pub struct Claim {
    /// What is being compared.
    pub name: String,
    /// Value the profile arithmetic predicts.
    pub predicted: f64,
    /// Value the run measured.
    pub measured: f64,
    /// Relative tolerance, as a fraction.
    pub tolerance: f64,
    /// Units, so a number is never reported bare.
    pub unit: String,
}

impl Claim {
    /// Build a claim.
    pub fn new(name: &str, predicted: f64, measured: f64, tolerance: f64, unit: &str) -> Self {
        Self {
            name: name.to_string(),
            predicted,
            measured,
            tolerance,
            unit: unit.to_string(),
        }
    }

    /// Relative error, `|measured - predicted| / predicted`.
    ///
    /// `None` when the prediction is zero: a relative error against zero is
    /// undefined, and reporting infinity as a large error would be wrong in the
    /// case where the measurement is also zero and the claim in fact holds.
    pub fn relative_error(&self) -> Option<f64> {
        if self.predicted == 0.0 {
            return None;
        }
        Some((self.measured - self.predicted).abs() / self.predicted.abs())
    }

    /// Whether the measurement supports the prediction.
    pub fn verdict(&self) -> Verdict {
        match self.relative_error() {
            Some(e) if e <= self.tolerance => Verdict::Holds,
            Some(_) => Verdict::Refuted,
            // Both zero is agreement; a non-zero measurement against a zero
            // prediction is a refutation.
            None if self.measured == 0.0 => Verdict::Holds,
            None => Verdict::Refuted,
        }
    }

    /// JSON form.
    pub fn to_value(&self) -> Value {
        json::obj(vec![
            ("name", json::s(&self.name)),
            ("predicted", json::f(self.predicted)),
            ("measured", json::f(self.measured)),
            ("unit", json::s(&self.unit)),
            (
                "relative_error",
                match self.relative_error() {
                    Some(e) => json::f(e),
                    None => Value::Num(f64::NAN),
                },
            ),
            ("tolerance", json::f(self.tolerance)),
            ("verdict", json::s(self.verdict().as_str())),
        ])
    }
}

/// A set of claims, rendered together.
#[derive(Clone, Debug, Default)]
pub struct Claims {
    /// The claims, in the order they were added.
    pub items: Vec<Claim>,
}

impl Claims {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a claim.
    pub fn push(&mut self, claim: Claim) {
        self.items.push(claim);
    }

    /// Claims the run refuted.
    pub fn refuted(&self) -> Vec<&Claim> {
        self.items
            .iter()
            .filter(|c| c.verdict() == Verdict::Refuted)
            .collect()
    }

    /// JSON form.
    pub fn to_value(&self) -> Value {
        Value::List(self.items.iter().map(|c| c.to_value()).collect())
    }

    /// A fixed-width table for the terminal.
    pub fn table(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "  {:<34} {:>14} {:>14} {:>8}  verdict",
            "claim", "predicted", "measured", "rel.err"
        );
        for c in &self.items {
            let err = match c.relative_error() {
                Some(e) => format!("{:.2}%", e * 100.0),
                None => "-".to_string(),
            };
            let _ = writeln!(
                out,
                "  {:<34} {:>14.3} {:>14.3} {:>8}  {}",
                format!("{} ({})", c.name, c.unit),
                c.predicted,
                c.measured,
                err,
                c.verdict().as_str()
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_measurement_inside_tolerance_holds() {
        let c = Claim::new("peak rss", 51_900.0, 52_400.0, 0.05, "bytes");
        assert_eq!(c.verdict(), Verdict::Holds);
        assert!(c.relative_error().unwrap() < 0.01);
    }

    #[test]
    fn a_measurement_outside_tolerance_is_refuted_not_widened() {
        // The point of the harness: a prediction the run contradicts is a
        // result, and widening the tolerance to make it pass would discard it.
        let c = Claim::new("rerank latency", 1.92, 4.80, 0.10, "ms");
        assert_eq!(c.verdict(), Verdict::Refuted);
        assert!((c.relative_error().unwrap() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn a_zero_prediction_is_handled_without_dividing_by_zero() {
        // Zero predicted and zero measured is agreement, not an error.
        let agree = Claim::new("drops", 0.0, 0.0, 0.05, "candidates");
        assert_eq!(agree.relative_error(), None);
        assert_eq!(agree.verdict(), Verdict::Holds);
        // Zero predicted and non-zero measured is a refutation.
        let disagree = Claim::new("drops", 0.0, 7.0, 0.05, "candidates");
        assert_eq!(disagree.verdict(), Verdict::Refuted);
    }

    #[test]
    fn refuted_claims_are_collected_for_the_summary() {
        let mut cs = Claims::new();
        cs.push(Claim::new("a", 10.0, 10.1, 0.05, "u"));
        cs.push(Claim::new("b", 10.0, 30.0, 0.05, "u"));
        cs.push(Claim::new("c", 10.0, 9.8, 0.05, "u"));
        let refuted = cs.refuted();
        assert_eq!(refuted.len(), 1);
        assert_eq!(refuted[0].name, "b");
    }

    #[test]
    fn the_table_carries_units_so_no_number_is_reported_bare() {
        let mut cs = Claims::new();
        cs.push(Claim::new("codebook", 32_768.0, 32_768.0, 0.0, "bytes"));
        let t = cs.table();
        assert!(t.contains("codebook (bytes)"), "{t}");
        assert!(t.contains("holds"), "{t}");
    }

    #[test]
    fn json_form_carries_the_verdict_and_the_inputs() {
        let c = Claim::new("recall@10", 0.605, 0.412, 0.05, "fraction");
        let text = c.to_value().to_json();
        assert!(text.contains("\"predicted\": 0.605000"), "{text}");
        assert!(text.contains("\"measured\": 0.412000"), "{text}");
        assert!(text.contains("REFUTED"), "{text}");
    }
}
