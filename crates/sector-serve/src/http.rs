//! HTTP/1.1, the subset this daemon serves.
//!
//! Hand-written on `std::io` because the workspace has no external dependencies
//! and the surface is small: three routes, two methods, `Content-Length` bodies.
//! Owning it means owning its failure modes, so this module states its limits
//! rather than leaving them to be discovered.
//!
//! # What is supported
//!
//! `GET` and `POST`, HTTP/1.0 and 1.1, request line plus headers plus an optional
//! `Content-Length` body. Keep-alive on 1.1 unless the client sends
//! `Connection: close`. That is what a monitoring check, a `curl`, and a client
//! library posting JSON need.
//!
//! # What is deliberately not supported
//!
//! - **`Transfer-Encoding: chunked`** — rejected with 411. Chunked decoding is
//!   where hand-written parsers get request smuggling wrong, and a client posting
//!   query vectors always knows their length.
//! - **TLS** — none. The daemon binds a Unix socket by default; a TCP listener is
//!   for a trusted network or behind a reverse proxy, and the documentation says
//!   so rather than implying transport security it does not provide.
//! - **HTTP/2, pipelining, `Expect: 100-continue`, trailers, compression.**
//!
//! # Limits, and why each one exists
//!
//! Every limit here is a defence against a specific denial of service, not a
//! round number:
//!
//! - [`MAX_HEADER_BYTES`] caps the request line and headers together, so a client
//!   cannot stream headers forever and make the server buffer them.
//! - [`MAX_HEADERS`] caps the count, because a million one-byte headers fits
//!   inside a generous byte cap.
//! - [`MAX_BODY_BYTES`] caps the body, sized against the largest legitimate
//!   request: a batch of query vectors. A `Content-Length` above it is refused
//!   with 413 *before* anything is read, so the bytes are never allocated.
//! - Read and write timeouts are set on the socket by the caller. Without them a
//!   connection that opens and sends nothing holds a worker forever, which is the
//!   whole slowloris class — and with a fixed thread pool, holding every worker
//!   is the entire service.
//!
//! A limit that is not enforced is a comment, so each is tested.

use std::io::{BufReader, Read, Write};

/// Largest request line and header block, together.
///
/// 16 KiB is roughly what nginx allows by default and far more than this API's
/// requests need. The cap exists so a client cannot make the server buffer
/// unbounded header bytes.
pub const MAX_HEADER_BYTES: usize = 16 * 1024;

/// Largest number of header lines.
///
/// A byte cap alone permits thousands of tiny headers, each costing a parse and
/// an allocation.
pub const MAX_HEADERS: usize = 64;

/// Largest request body.
///
/// 1 MiB holds roughly 2,000 float32 queries at D=128, well beyond a reasonable
/// batch. A `Content-Length` above this is refused before the body is read, so an
/// attacker cannot cause a large allocation by announcing one.
pub const MAX_BODY_BYTES: usize = 1024 * 1024;

/// A parsed request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    /// Uppercase method.
    pub method: String,
    /// Path, without the query string.
    pub path: String,
    /// Raw query string, without the `?`.
    pub query: String,
    /// Body bytes, at most [`MAX_BODY_BYTES`].
    pub body: Vec<u8>,
    /// Whether the connection should be kept open.
    pub keep_alive: bool,
    /// Value of `Content-Type`, lowercased, when present.
    pub content_type: Option<String>,
}

impl Request {
    /// A query-string parameter's value.
    ///
    /// No percent-decoding: every parameter this API takes is a number or a bare
    /// word, and a decoder is more parsing surface for no gain. A value
    /// containing an escape simply will not match.
    pub fn param(&self, name: &str) -> Option<&str> {
        self.query.split('&').find_map(|kv| {
            let (k, v) = kv.split_once('=')?;
            (k == name).then_some(v)
        })
    }

    /// A numeric query parameter, or `default` when absent.
    ///
    /// A malformed number is an error rather than a silent default: a client
    /// asking for `k=ten` has a bug, and answering with `k=10` hides it.
    pub fn num_param(&self, name: &str, default: usize) -> Result<usize, Status> {
        match self.param(name) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| Status::BadRequest),
        }
    }
}

/// The status codes this server emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// 200.
    Ok,
    /// 400 — malformed request or unparseable parameter.
    BadRequest,
    /// 404 — no such route.
    NotFound,
    /// 405 — wrong method for a known route.
    MethodNotAllowed,
    /// 411 — no `Content-Length`, including the chunked case.
    LengthRequired,
    /// 413 — body above [`MAX_BODY_BYTES`].
    PayloadTooLarge,
    /// 431 — headers above [`MAX_HEADER_BYTES`] or [`MAX_HEADERS`].
    HeadersTooLarge,
    /// 500 — the engine refused a well-formed request.
    ServerError,
    /// 503 — the daemon is shutting down.
    Unavailable,
}

impl Status {
    /// Numeric code.
    pub const fn code(&self) -> u16 {
        match self {
            Self::Ok => 200,
            Self::BadRequest => 400,
            Self::NotFound => 404,
            Self::MethodNotAllowed => 405,
            Self::LengthRequired => 411,
            Self::PayloadTooLarge => 413,
            Self::HeadersTooLarge => 431,
            Self::ServerError => 500,
            Self::Unavailable => 503,
        }
    }

    /// Reason phrase.
    pub const fn reason(&self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::BadRequest => "Bad Request",
            Self::NotFound => "Not Found",
            Self::MethodNotAllowed => "Method Not Allowed",
            Self::LengthRequired => "Length Required",
            Self::PayloadTooLarge => "Payload Too Large",
            Self::HeadersTooLarge => "Request Header Fields Too Large",
            Self::ServerError => "Internal Server Error",
            Self::Unavailable => "Service Unavailable",
        }
    }
}

/// Why a request could not be read.
#[derive(Debug)]
pub enum ReadError {
    /// The connection closed before a complete request arrived.
    ///
    /// Normal at the end of a keep-alive connection, so the caller treats it as
    /// "no more requests" rather than as a failure.
    Closed,
    /// The request was malformed or exceeded a limit. Carries the status to send.
    Rejected(Status),
    /// The socket failed.
    Io(std::io::Error),
}

impl From<std::io::Error> for ReadError {
    fn from(e: std::io::Error) -> Self {
        // A read timeout is how a slow or idle client is disconnected; it is a
        // closed connection from this server's point of view, not an error worth
        // logging as one.
        match e.kind() {
            std::io::ErrorKind::WouldBlock
            | std::io::ErrorKind::TimedOut
            | std::io::ErrorKind::UnexpectedEof
            | std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::BrokenPipe => Self::Closed,
            _ => Self::Io(e),
        }
    }
}

/// Read one request from `stream`.
pub fn read_request<R: Read>(reader: &mut BufReader<R>) -> Result<Request, ReadError> {
    let mut line = String::new();
    let mut header_bytes = 0usize;

    // Request line. An empty first line is a closed connection.
    let n = read_line(reader, &mut line, MAX_HEADER_BYTES)?;
    if n == 0 {
        return Err(ReadError::Closed);
    }
    header_bytes += n;

    let mut parts = line.trim_end().split(' ');
    let method = parts.next().unwrap_or("").to_ascii_uppercase();
    let target = parts.next().unwrap_or("");
    let version = parts.next().unwrap_or("HTTP/1.1");
    if method.is_empty() || target.is_empty() {
        return Err(ReadError::Rejected(Status::BadRequest));
    }
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target.to_string(), String::new()),
    };

    // HTTP/1.1 keeps the connection open by default; 1.0 closes it.
    let mut keep_alive = version.trim() == "HTTP/1.1";
    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    let mut content_type = None;
    let mut count = 0usize;

    loop {
        line.clear();
        let n = read_line(
            reader,
            &mut line,
            MAX_HEADER_BYTES.saturating_sub(header_bytes),
        )?;
        if n == 0 {
            return Err(ReadError::Closed);
        }
        header_bytes += n;
        if header_bytes > MAX_HEADER_BYTES {
            return Err(ReadError::Rejected(Status::HeadersTooLarge));
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        count += 1;
        if count > MAX_HEADERS {
            return Err(ReadError::Rejected(Status::HeadersTooLarge));
        }

        let Some((name, value)) = trimmed.split_once(':') else {
            return Err(ReadError::Rejected(Status::BadRequest));
        };
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        match name.as_str() {
            "content-length" => {
                let len: usize = value
                    .parse()
                    .map_err(|_| ReadError::Rejected(Status::BadRequest))?;
                // Refused on the announced length, before any body byte is read,
                // so a large `Content-Length` cannot cause a large allocation.
                if len > MAX_BODY_BYTES {
                    return Err(ReadError::Rejected(Status::PayloadTooLarge));
                }
                content_length = Some(len);
            }
            "transfer-encoding" => {
                if value.to_ascii_lowercase().contains("chunked") {
                    chunked = true;
                }
            }
            "connection" => {
                let v = value.to_ascii_lowercase();
                if v.contains("close") {
                    keep_alive = false;
                } else if v.contains("keep-alive") {
                    keep_alive = true;
                }
            }
            "content-type" => content_type = Some(value.to_ascii_lowercase()),
            _ => {}
        }
    }

    // Chunked bodies are not decoded. Sending 411 rather than attempting it keeps
    // the request-smuggling surface out of the server entirely.
    if chunked {
        return Err(ReadError::Rejected(Status::LengthRequired));
    }

    let body = match content_length {
        None | Some(0) => Vec::new(),
        Some(len) => {
            let mut buf = vec![0u8; len];
            reader.read_exact(&mut buf)?;
            buf
        }
    };

    Ok(Request {
        method,
        path,
        query,
        body,
        keep_alive,
        content_type,
    })
}

/// Read one CRLF-terminated line, up to `limit` bytes.
///
/// Returns the bytes consumed, or 0 at end of stream. Enforcing the limit here
/// rather than after the fact is what prevents a client from making the server
/// buffer an unbounded line.
fn read_line<R: Read>(
    reader: &mut BufReader<R>,
    out: &mut String,
    limit: usize,
) -> Result<usize, ReadError> {
    out.clear();
    let mut consumed = 0usize;
    let mut byte = [0u8; 1];
    loop {
        let n = reader.read(&mut byte)?;
        if n == 0 {
            return Ok(consumed);
        }
        consumed += 1;
        if consumed > limit.max(1) {
            return Err(ReadError::Rejected(Status::HeadersTooLarge));
        }
        if byte[0] == b'\n' {
            return Ok(consumed);
        }
        // A NUL or a bare CR mid-line is not something a conforming client sends;
        // pushing it through as a char would let a header value carry a line
        // break past the parser.
        if byte[0] == b'\r' {
            continue;
        }
        out.push(byte[0] as char);
    }
}

/// Write a response.
///
/// `Content-Length` is always sent and the body is written in one call, so no
/// chunked encoding is needed on the way out either.
pub fn write_response<W: Write>(
    w: &mut W,
    status: Status,
    content_type: &str,
    body: &[u8],
    keep_alive: bool,
) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {} {}\r\n\
         Content-Type: {}\r\n\
         Content-Length: {}\r\n\
         Connection: {}\r\n\
         \r\n",
        status.code(),
        status.reason(),
        content_type,
        body.len(),
        if keep_alive { "keep-alive" } else { "close" }
    );
    w.write_all(head.as_bytes())?;
    w.write_all(body)?;
    w.flush()
}

/// Write a JSON error body.
pub fn write_error<W: Write>(w: &mut W, status: Status, detail: &str) -> std::io::Result<()> {
    let mut j = sector_os::json::Json::new();
    j.object(|o| {
        o.uint("status", status.code() as u64);
        o.str("error", status.reason());
        o.str("detail", detail);
    });
    // An error closes the connection: after a protocol-level rejection the stream
    // position is not trustworthy, so reusing it could desynchronise the parser.
    write_response(w, status, "application/json", j.finish().as_bytes(), false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn parse(raw: &str) -> Result<Request, ReadError> {
        let mut r = BufReader::new(Cursor::new(raw.as_bytes().to_vec()));
        read_request(&mut r)
    }

    #[test]
    fn a_plain_get_parses() {
        let r = parse("GET /health HTTP/1.1\r\nHost: x\r\n\r\n").expect("parse");
        assert_eq!(r.method, "GET");
        assert_eq!(r.path, "/health");
        assert!(r.query.is_empty());
        assert!(r.keep_alive, "1.1 defaults to keep-alive");
        assert!(r.body.is_empty());
    }

    #[test]
    fn the_query_string_is_split_from_the_path() {
        let r = parse("GET /search?k=5&r=200 HTTP/1.1\r\n\r\n").expect("parse");
        assert_eq!(r.path, "/search");
        assert_eq!(r.param("k"), Some("5"));
        assert_eq!(r.param("r"), Some("200"));
        assert_eq!(r.param("absent"), None);
        assert_eq!(r.num_param("k", 10).unwrap(), 5);
        assert_eq!(r.num_param("absent", 10).unwrap(), 10);
    }

    #[test]
    fn a_malformed_numeric_parameter_is_rejected_not_defaulted() {
        // Answering `k=ten` with k=10 hides a client bug.
        let r = parse("GET /search?k=ten HTTP/1.1\r\n\r\n").expect("parse");
        assert_eq!(r.num_param("k", 10), Err(Status::BadRequest));
    }

    #[test]
    fn a_body_is_read_when_content_length_says_so() {
        let r = parse("POST /search HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello").expect("parse");
        assert_eq!(r.method, "POST");
        assert_eq!(r.body, b"hello");
    }

    #[test]
    fn http_1_0_closes_by_default_and_connection_close_overrides_1_1() {
        assert!(!parse("GET / HTTP/1.0\r\n\r\n").expect("parse").keep_alive);
        let r = parse("GET / HTTP/1.1\r\nConnection: close\r\n\r\n").expect("parse");
        assert!(!r.keep_alive);
        let r = parse("GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n").expect("parse");
        assert!(r.keep_alive);
    }

    #[test]
    fn header_names_are_case_insensitive() {
        // A client sending `CONTENT-LENGTH` is conforming, and treating the name
        // as case-sensitive would silently drop its body.
        let r = parse("POST / HTTP/1.1\r\nCONTENT-LENGTH: 2\r\nCoNtEnT-TyPe: A/B\r\n\r\nhi")
            .expect("parse");
        assert_eq!(r.body, b"hi");
        assert_eq!(r.content_type.as_deref(), Some("a/b"));
    }

    #[test]
    fn a_content_length_above_the_cap_is_refused_before_the_body_is_read() {
        // The allocation defence: refused on the announced length, so the bytes
        // are never reserved. The request here claims a large body and sends none.
        let raw = format!(
            "POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_BODY_BYTES + 1
        );
        assert!(matches!(
            parse(&raw),
            Err(ReadError::Rejected(Status::PayloadTooLarge))
        ));
    }

    #[test]
    fn chunked_encoding_is_refused_rather_than_decoded() {
        // Chunked decoding is where hand-written parsers get smuggling wrong, and
        // a client posting vectors always knows their length.
        assert!(matches!(
            parse("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"),
            Err(ReadError::Rejected(Status::LengthRequired))
        ));
    }

    #[test]
    fn too_many_headers_are_refused_even_when_each_is_tiny() {
        // A byte cap alone permits thousands of one-byte headers.
        let mut raw = String::from("GET / HTTP/1.1\r\n");
        for i in 0..MAX_HEADERS + 5 {
            raw.push_str(&format!("h{i}: v\r\n"));
        }
        raw.push_str("\r\n");
        assert!(matches!(
            parse(&raw),
            Err(ReadError::Rejected(Status::HeadersTooLarge))
        ));
    }

    #[test]
    fn an_overlong_header_block_is_refused() {
        let mut raw = String::from("GET / HTTP/1.1\r\n");
        raw.push_str("x: ");
        raw.push_str(&"a".repeat(MAX_HEADER_BYTES + 100));
        raw.push_str("\r\n\r\n");
        assert!(matches!(
            parse(&raw),
            Err(ReadError::Rejected(Status::HeadersTooLarge))
        ));
    }

    #[test]
    fn an_empty_stream_is_a_closed_connection_not_an_error() {
        // The normal end of a keep-alive connection.
        assert!(matches!(parse(""), Err(ReadError::Closed)));
    }

    #[test]
    fn a_garbage_request_line_is_a_bad_request() {
        assert!(matches!(
            parse("nonsense\r\n\r\n"),
            Err(ReadError::Rejected(Status::BadRequest))
        ));
        assert!(matches!(
            parse("GET / HTTP/1.1\r\nnocolon\r\n\r\n"),
            Err(ReadError::Rejected(Status::BadRequest))
        ));
    }

    #[test]
    fn a_response_carries_content_length_and_the_body() {
        let mut out = Vec::new();
        write_response(&mut out, Status::Ok, "application/json", b"{}", true).expect("write");
        let text = String::from_utf8(out).expect("utf8");
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"), "{text}");
        assert!(text.contains("Content-Length: 2\r\n"), "{text}");
        assert!(text.contains("Connection: keep-alive\r\n"), "{text}");
        assert!(text.ends_with("\r\n\r\n{}"), "{text}");
    }

    #[test]
    fn an_error_response_is_json_and_closes_the_connection() {
        // After a protocol rejection the stream position is not trustworthy, so
        // reusing the connection could desynchronise the parser.
        let mut out = Vec::new();
        write_error(&mut out, Status::PayloadTooLarge, "body too large").expect("write");
        let text = String::from_utf8(out).expect("utf8");
        assert!(text.starts_with("HTTP/1.1 413 "), "{text}");
        assert!(text.contains("Connection: close"), "{text}");
        assert!(text.contains(r#""status":413"#), "{text}");
        assert!(text.contains(r#""detail":"body too large""#), "{text}");
    }

    #[test]
    fn every_status_has_a_distinct_code_and_a_reason() {
        let all = [
            Status::Ok,
            Status::BadRequest,
            Status::NotFound,
            Status::MethodNotAllowed,
            Status::LengthRequired,
            Status::PayloadTooLarge,
            Status::HeadersTooLarge,
            Status::ServerError,
            Status::Unavailable,
        ];
        let mut codes: Vec<u16> = all.iter().map(|s| s.code()).collect();
        codes.sort_unstable();
        let before = codes.len();
        codes.dedup();
        assert_eq!(codes.len(), before, "two statuses share a code");
        assert!(all.iter().all(|s| !s.reason().is_empty()));
    }
}
