//! SIFT1M / GIST1M loaders (`.fvecs` / `.bvecs` / `.ivecs`).
//!
//! Real embeddings are a blocking precondition. Every recall figure this
//! project currently reports comes from a synthetic clustered low-rank corpus
//! at `N = 20,000`, `D = 256`, and the margin distribution — load-bearing in
//! every bound stated — is the property most likely to differ on real data.
//!
//! # Implementation notes
//!
//! Memory-map rather than reading into RAM. GIST1M at `D = 960` f32 is roughly
//! 3.8 GB, and the builder does not need it resident to encode it.
//!
//! Carry the provided ground-truth neighbour sets rather than recomputing them.
//! Both datasets ship exact nearest neighbours; recomputing admits a metric
//! mismatch that would make every recall number wrong in the same direction.
//!
//! Run the full corruption sweep on these datasets, not only a recall
//! comparison. The claim needing real-data evidence is that the
//! label-optimisation gain survives outside synthetic data, and it is listed as
//! a falsification criterion because it may not.

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Why a dataset could not be read.
#[derive(Debug)]
pub enum DatasetError {
    /// The file could not be opened or read.
    Io(io::Error),
    /// The file's length is not a whole number of records.
    ///
    /// The `.?vecs` formats carry no header, so a truncated file is otherwise
    /// indistinguishable from a shorter dataset.
    RaggedFile {
        /// File length in bytes.
        len: u64,
        /// Bytes per record at the dimension read from the first record.
        record_bytes: u64,
    },
    /// Records disagree about dimension.
    InconsistentDimension {
        /// Record index where the disagreement was found.
        at: usize,
        /// Dimension the first record declared.
        expected: u32,
        /// Dimension this record declares.
        found: u32,
    },
    /// A dimension of zero, or one too large to be plausible.
    ImplausibleDimension {
        /// The value read.
        found: u32,
    },
}

impl From<io::Error> for DatasetError {
    fn from(e: io::Error) -> Self {
        DatasetError::Io(e)
    }
}

/// Largest dimension accepted, to catch a misidentified file format.
const MAX_DIM: u32 = 100_000;

/// Component width of a `.?vecs` file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Component {
    /// `.fvecs` — f32 components.
    F32,
    /// `.bvecs` — u8 components.
    U8,
    /// `.ivecs` — i32 components, used for ground-truth neighbour ids.
    I32,
}

impl Component {
    /// Bytes per component.
    pub const fn width(self) -> usize {
        match self {
            Component::F32 | Component::I32 => 4,
            Component::U8 => 1,
        }
    }

    /// Infer from a path extension.
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()? {
            "fvecs" => Some(Component::F32),
            "bvecs" => Some(Component::U8),
            "ivecs" => Some(Component::I32),
            _ => None,
        }
    }
}

/// Geometry of a `.?vecs` file, read from its first record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// Components per vector.
    pub dim: u32,
    /// Vectors in the file.
    pub count: usize,
    /// Component width.
    pub component: Component,
}

impl Layout {
    /// Bytes per record: a 4-byte dimension prefix plus the components.
    pub const fn record_bytes(&self) -> u64 {
        4 + (self.dim as u64) * (self.component.width() as u64)
    }

    /// Byte offset of record `i`.
    pub const fn offset_of(&self, i: usize) -> u64 {
        (i as u64) * self.record_bytes()
    }
}

/// Read a `.?vecs` file's geometry without loading it.
///
/// GIST1M at `D = 960` f32 is roughly 3.8 GB. The builder streams records
/// rather than holding the corpus resident, so the geometry is established
/// separately from the data.
pub fn probe(path: &Path) -> Result<Layout, DatasetError> {
    let component = Component::from_path(path).unwrap_or(Component::F32);
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();

    let mut head = [0u8; 4];
    file.read_exact(&mut head)?;
    let dim = u32::from_le_bytes(head);
    if dim == 0 || dim > MAX_DIM {
        return Err(DatasetError::ImplausibleDimension { found: dim });
    }

    let record_bytes = 4 + (dim as u64) * (component.width() as u64);
    if !len.is_multiple_of(record_bytes) {
        return Err(DatasetError::RaggedFile { len, record_bytes });
    }

    Ok(Layout {
        dim,
        count: (len / record_bytes) as usize,
        component,
    })
}

/// Streaming reader over a `.?vecs` file.
///
/// Holds one record at a time. The builder's memory does not scale with corpus
/// size, so a 3.8 GB dataset is a streaming cost rather than a resident one.
pub struct VecsReader {
    reader: BufReader<File>,
    layout: Layout,
    next: usize,
    /// Raw component bytes for the current record.
    raw: Vec<u8>,
}

impl VecsReader {
    /// Open `path` and read its geometry.
    pub fn open(path: &Path) -> Result<Self, DatasetError> {
        let layout = probe(path)?;
        let file = File::open(path)?;
        let raw = vec![0u8; (layout.dim as usize) * layout.component.width()];
        Ok(Self {
            reader: BufReader::new(file),
            layout,
            next: 0,
            raw,
        })
    }

    /// The file's geometry.
    pub const fn layout(&self) -> Layout {
        self.layout
    }

    /// Vectors in the file.
    pub const fn len(&self) -> usize {
        self.layout.count
    }

    /// Whether the file holds no vectors.
    pub const fn is_empty(&self) -> bool {
        self.layout.count == 0
    }

    /// Seek to record `i`.
    pub fn seek_to(&mut self, i: usize) -> Result<(), DatasetError> {
        self.reader
            .seek(SeekFrom::Start(self.layout.offset_of(i)))?;
        self.next = i;
        Ok(())
    }

    /// Read the next record into `out` as f32, returning its index.
    ///
    /// Returns `None` at end of file. The dimension prefix on every record is
    /// checked against the first: the format repeats it, and a file where they
    /// disagree is not the file it claims to be.
    pub fn next_f32(&mut self, out: &mut [f32]) -> Result<Option<usize>, DatasetError> {
        if self.next >= self.layout.count {
            return Ok(None);
        }
        let mut head = [0u8; 4];
        self.reader.read_exact(&mut head)?;
        let dim = u32::from_le_bytes(head);
        if dim != self.layout.dim {
            return Err(DatasetError::InconsistentDimension {
                at: self.next,
                expected: self.layout.dim,
                found: dim,
            });
        }
        self.reader.read_exact(&mut self.raw)?;

        let width = self.layout.component.width();
        for (i, slot) in out.iter_mut().take(self.layout.dim as usize).enumerate() {
            *slot = match self.layout.component {
                Component::U8 => self.raw.get(i).copied().unwrap_or(0) as f32,
                Component::F32 => {
                    let b = self.raw.get(i * width..(i + 1) * width);
                    match b {
                        Some(b) => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                        None => 0.0,
                    }
                }
                Component::I32 => {
                    let b = self.raw.get(i * width..(i + 1) * width);
                    match b {
                        Some(b) => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
                        None => 0.0,
                    }
                }
            };
        }
        let index = self.next;
        self.next += 1;
        Ok(Some(index))
    }

    /// Read the next record as i32, for `.ivecs` ground truth.
    pub fn next_i32(&mut self, out: &mut [i32]) -> Result<Option<usize>, DatasetError> {
        if self.next >= self.layout.count {
            return Ok(None);
        }
        let mut head = [0u8; 4];
        self.reader.read_exact(&mut head)?;
        let dim = u32::from_le_bytes(head);
        if dim != self.layout.dim {
            return Err(DatasetError::InconsistentDimension {
                at: self.next,
                expected: self.layout.dim,
                found: dim,
            });
        }
        self.reader.read_exact(&mut self.raw)?;
        for (i, slot) in out.iter_mut().take(self.layout.dim as usize).enumerate() {
            let b = self.raw.get(i * 4..(i + 1) * 4);
            *slot = match b {
                Some(b) => i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                None => 0,
            };
        }
        let index = self.next;
        self.next += 1;
        Ok(Some(index))
    }
}

/// Ground-truth neighbour sets shipped with a dataset.
///
/// Carried rather than recomputed. Both SIFT1M and GIST1M ship exact nearest
/// neighbours, and recomputing them admits a metric mismatch that would make
/// every recall number wrong in the same direction — consistently, and
/// therefore invisibly.
#[derive(Clone, Debug)]
pub struct GroundTruth {
    /// Neighbours per query.
    pub k: usize,
    /// Query count.
    pub queries: usize,
    /// Row-major neighbour ids, `queries * k`.
    pub neighbours: Vec<i32>,
}

impl GroundTruth {
    /// Load a `.ivecs` ground-truth file in full.
    ///
    /// Resident, unlike the corpus: at 10,000 queries and `k = 100` this is
    /// 4 MB, and every recall measurement consults it.
    pub fn load(path: &Path) -> Result<Self, DatasetError> {
        let mut reader = VecsReader::open(path)?;
        let k = reader.layout().dim as usize;
        let queries = reader.len();
        let mut neighbours = vec![0i32; queries * k];
        let mut row = vec![0i32; k];
        while let Some(i) = reader.next_i32(&mut row)? {
            let start = i * k;
            if let Some(dst) = neighbours.get_mut(start..start + k) {
                dst.copy_from_slice(&row);
            }
        }
        Ok(Self {
            k,
            queries,
            neighbours,
        })
    }

    /// The neighbour set for `query`.
    pub fn row(&self, query: usize) -> Option<&[i32]> {
        self.neighbours.get(query * self.k..(query + 1) * self.k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("sector_test_{name}"));
        p
    }

    fn write_fvecs(path: &Path, dim: u32, vectors: &[Vec<f32>]) {
        let mut f = File::create(path).unwrap();
        for v in vectors {
            f.write_all(&dim.to_le_bytes()).unwrap();
            for c in v {
                f.write_all(&c.to_le_bytes()).unwrap();
            }
        }
    }

    fn write_ivecs(path: &Path, dim: u32, rows: &[Vec<i32>]) {
        let mut f = File::create(path).unwrap();
        for r in rows {
            f.write_all(&dim.to_le_bytes()).unwrap();
            for c in r {
                f.write_all(&c.to_le_bytes()).unwrap();
            }
        }
    }

    #[test]
    fn geometry_is_read_without_loading_the_file() {
        let p = temp_path("probe.fvecs");
        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|i| (0..4).map(|j| (i * 4 + j) as f32).collect())
            .collect();
        write_fvecs(&p, 4, &vectors);

        let layout = probe(&p).unwrap();
        assert_eq!(layout.dim, 4);
        assert_eq!(layout.count, 10);
        assert_eq!(layout.component, Component::F32);
        // 4-byte prefix + 4 f32 components.
        assert_eq!(layout.record_bytes(), 20);
        assert_eq!(layout.offset_of(3), 60);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn records_round_trip_through_the_streaming_reader() {
        let p = temp_path("stream.fvecs");
        let vectors: Vec<Vec<f32>> = (0..5)
            .map(|i| (0..3).map(|j| (i as f32) * 10.0 + j as f32).collect())
            .collect();
        write_fvecs(&p, 3, &vectors);

        let mut r = VecsReader::open(&p).unwrap();
        assert_eq!(r.len(), 5);
        let mut out = vec![0f32; 3];
        for expected in &vectors {
            let i = r.next_f32(&mut out).unwrap().expect("record");
            assert_eq!(&out, expected, "record {i}");
        }
        assert_eq!(r.next_f32(&mut out).unwrap(), None);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn seeking_lands_on_the_requested_record() {
        let p = temp_path("seek.fvecs");
        let vectors: Vec<Vec<f32>> = (0..8).map(|i| vec![i as f32, 0.0]).collect();
        write_fvecs(&p, 2, &vectors);

        let mut r = VecsReader::open(&p).unwrap();
        r.seek_to(5).unwrap();
        let mut out = vec![0f32; 2];
        assert_eq!(r.next_f32(&mut out).unwrap(), Some(5));
        assert_eq!(out[0], 5.0);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn a_truncated_file_is_refused_not_read_short() {
        // The formats carry no header, so a truncated file is otherwise
        // indistinguishable from a shorter dataset.
        let p = temp_path("truncated.fvecs");
        let vectors: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32; 4]).collect();
        write_fvecs(&p, 4, &vectors);

        // Cut 6 bytes off the end.
        let data = std::fs::read(&p).unwrap();
        std::fs::write(&p, &data[..data.len() - 6]).unwrap();

        assert!(matches!(probe(&p), Err(DatasetError::RaggedFile { .. })));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn a_record_disagreeing_about_dimension_is_caught() {
        let p = temp_path("ragged.fvecs");
        {
            let mut f = File::create(&p).unwrap();
            // Two records of dim 3, then one claiming dim 3 but written as 3.
            for i in 0..3 {
                f.write_all(&3u32.to_le_bytes()).unwrap();
                for j in 0..3 {
                    f.write_all(&((i * 3 + j) as f32).to_le_bytes()).unwrap();
                }
            }
        }
        // Corrupt the second record's dimension prefix in place.
        let mut data = std::fs::read(&p).unwrap();
        data[16..20].copy_from_slice(&7u32.to_le_bytes());
        std::fs::write(&p, &data).unwrap();

        let mut r = VecsReader::open(&p).unwrap();
        let mut out = vec![0f32; 3];
        r.next_f32(&mut out).unwrap();
        assert!(matches!(
            r.next_f32(&mut out),
            Err(DatasetError::InconsistentDimension {
                at: 1,
                expected: 3,
                found: 7
            })
        ));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn an_implausible_dimension_is_refused() {
        // Catches a file that is not in this format at all.
        let p = temp_path("bogus.fvecs");
        std::fs::write(&p, 999_999u32.to_le_bytes()).unwrap();
        assert!(matches!(
            probe(&p),
            Err(DatasetError::ImplausibleDimension { found: 999_999 })
        ));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn byte_datasets_widen_to_f32() {
        // GIST and SIFT ship as .bvecs in some distributions; the reader must
        // present both widths identically to the builder.
        let p = temp_path("bytes.bvecs");
        {
            let mut f = File::create(&p).unwrap();
            for i in 0..4u8 {
                f.write_all(&4u32.to_le_bytes()).unwrap();
                f.write_all(&[i, i + 1, i + 2, i + 3]).unwrap();
            }
        }
        let layout = probe(&p).unwrap();
        assert_eq!(layout.component, Component::U8);
        assert_eq!(layout.record_bytes(), 8);
        assert_eq!(layout.count, 4);

        let mut r = VecsReader::open(&p).unwrap();
        let mut out = vec![0f32; 4];
        r.next_f32(&mut out).unwrap();
        assert_eq!(out, vec![0.0, 1.0, 2.0, 3.0]);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn shipped_ground_truth_is_carried_not_recomputed() {
        let p = temp_path("gt.ivecs");
        let rows: Vec<Vec<i32>> = (0..5)
            .map(|q| (0..10).map(|n| q * 100 + n).collect())
            .collect();
        write_ivecs(&p, 10, &rows);

        let gt = GroundTruth::load(&p).unwrap();
        assert_eq!(gt.k, 10);
        assert_eq!(gt.queries, 5);
        for (q, expected) in rows.iter().enumerate() {
            assert_eq!(gt.row(q).unwrap(), expected.as_slice());
        }
        assert!(gt.row(5).is_none());
        std::fs::remove_file(&p).ok();
    }
}
