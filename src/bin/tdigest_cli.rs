// src/bin/tdigest_cli.rs

use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use serde_json::Value;

use tdigest_rs::tdigest::frontends::{
    ensure_finite_training_values, parse_scale_str, parse_singleton_policy_str,
    validate_quantile_probe, DigestConfig, DigestPrecision, FrontendDigest,
};
use tdigest_rs::tdigest::singleton_policy::SingletonPolicy;
use tdigest_rs::tdigest::wire::WireVersion;
use tdigest_rs::tdigest::ScaleFamily;

#[derive(Debug, Clone, ValueEnum)]
enum ScaleOpt {
    Quad,
    K1,
    K2,
    K3,
}

impl fmt::Display for ScaleOpt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ScaleOpt::Quad => "quad",
            ScaleOpt::K1 => "k1",
            ScaleOpt::K2 => "k2",
            ScaleOpt::K3 => "k3",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PrecisionOpt {
    F32,
    F64,
    Auto,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PolicyOpt {
    Off,
    Use,
    Edges,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum DataFormatOpt {
    Auto,
    Text,
    Csv,
    Json,
    Ndjson,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum WireVersionOpt {
    V1,
    V2,
    V3,
}

impl WireVersionOpt {
    fn into_wire(self) -> WireVersion {
        match self {
            WireVersionOpt::V1 => WireVersion::V1,
            WireVersionOpt::V2 => WireVersion::V2,
            WireVersionOpt::V3 => WireVersion::V3,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputOpt {
    Csv,
    Text,
}

#[derive(Debug, Args)]
struct SourceArgs {
    /// Read training values from stdin.
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "input")]
    stdin: bool,

    /// Read training values from file.
    #[arg(long, value_name = "PATH")]
    input: Option<PathBuf>,

    /// Input format for --stdin/--input.
    #[arg(long, value_enum, default_value_t = DataFormatOpt::Auto)]
    input_format: DataFormatOpt,

    /// Column name or 0-based index when reading CSV/JSON-object data.
    #[arg(long, value_name = "NAME_OR_INDEX")]
    input_column: Option<String>,

    /// Load an existing TDIG bytes file before training/querying.
    #[arg(long, value_name = "PATH")]
    from_digest: Option<PathBuf>,

    /// Merge one or more additional TDIG bytes files.
    #[arg(long, value_name = "PATH")]
    merge_digest: Vec<PathBuf>,
}

#[derive(Debug, Args)]
struct DigestArgs {
    /// TDigest max size.
    #[arg(long = "max-size", default_value_t = 1000)]
    max_size: usize,

    /// Scale family: quad|k1|k2|k3.
    #[arg(long, value_enum, default_value_t = ScaleOpt::K2)]
    scale: ScaleOpt,

    /// Singleton policy: off|use|edges.
    #[arg(long = "singleton-policy", value_enum)]
    singleton_policy: Option<PolicyOpt>,

    /// Edges to pin per side when using 'edges' policy.
    #[arg(long = "pin-per-side")]
    pin_per_side: Option<usize>,

    /// Precision hint for training values: f32|f64|auto.
    #[arg(long, value_enum, default_value_t = PrecisionOpt::F64)]
    precision: PrecisionOpt,
}

#[derive(Debug, Args)]
struct PersistArgs {
    /// Save resulting digest to a TDIG bytes file.
    #[arg(long, value_name = "PATH")]
    to_digest: Option<PathBuf>,

    /// Wire version used with --to-digest.
    #[arg(long, value_enum, default_value_t = WireVersionOpt::V3)]
    wire_version: WireVersionOpt,
}

#[derive(Debug, Args)]
struct QueryArgs {
    #[command(flatten)]
    source: SourceArgs,

    #[command(flatten)]
    digest: DigestArgs,

    #[command(flatten)]
    persist: PersistArgs,
}

#[derive(Debug, Args)]
struct OutputArgs {
    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputOpt::Csv)]
    output: OutputOpt,

    /// Suppress header row when output=csv.
    #[arg(long, action = ArgAction::SetTrue)]
    no_header: bool,
}

#[derive(Debug, Args)]
struct ProbeArgs {
    /// CDF probe(s), accepts repeated or comma-separated values.
    #[arg(long, allow_hyphen_values = true, value_delimiter = ',')]
    x: Vec<String>,

    /// Read CDF probes from stdin.
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "probes_input")]
    probes_stdin: bool,

    /// Read CDF probes from file.
    #[arg(long, value_name = "PATH")]
    probes_input: Option<PathBuf>,

    /// Probe input format for --probes-stdin/--probes-input.
    #[arg(long, value_enum, default_value_t = DataFormatOpt::Auto)]
    probes_format: DataFormatOpt,

    /// Probe column name or 0-based index when reading CSV/JSON-object data.
    #[arg(long, value_name = "NAME_OR_INDEX")]
    probes_column: Option<String>,
}

#[derive(Debug, Args)]
struct BuildCmd {
    #[command(flatten)]
    source: SourceArgs,

    #[command(flatten)]
    digest: DigestArgs,

    /// Save resulting digest to a TDIG bytes file.
    #[arg(long, value_name = "PATH", required = true)]
    to_digest: PathBuf,

    /// Wire version used with --to-digest.
    #[arg(long, value_enum, default_value_t = WireVersionOpt::V3)]
    wire_version: WireVersionOpt,
}

#[derive(Debug, Args)]
struct QuantileCmd {
    #[command(flatten)]
    query: QueryArgs,

    /// Quantile probability/probabilities (comma-separated supported).
    #[arg(
        long,
        allow_hyphen_values = true,
        value_delimiter = ',',
        required = true
    )]
    p: Vec<String>,

    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Debug, Args)]
struct CdfCmd {
    #[command(flatten)]
    query: QueryArgs,

    #[command(flatten)]
    probes: ProbeArgs,

    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Debug, Args)]
struct MedianCmd {
    #[command(flatten)]
    query: QueryArgs,

    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Build a digest and write it to --to-digest.
    Build(BuildCmd),

    /// Query quantiles.
    Quantile(QuantileCmd),

    /// Query CDF values.
    Cdf(CdfCmd),

    /// Query the median.
    Median(MedianCmd),
}

#[derive(Debug, Parser)]
#[command(name = "tdigest", version, disable_help_subcommand = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

fn policy_from_opts(
    opt: Option<PolicyOpt>,
    edges: Option<usize>,
) -> Result<SingletonPolicy, String> {
    let as_str = match opt.unwrap_or(PolicyOpt::Use) {
        PolicyOpt::Off => "off",
        PolicyOpt::Use => "use",
        PolicyOpt::Edges => "edges",
    };
    parse_singleton_policy_str(Some(as_str), edges).map_err(|e| e.to_string())
}

fn scale_from_opt(scale: ScaleOpt) -> Result<ScaleFamily, String> {
    parse_scale_str(Some(&scale.to_string())).map_err(|e| e.to_string())
}

fn detect_format(path: &Path) -> DataFormatOpt {
    let ext = path
        .extension()
        .and_then(|x| x.to_str())
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    match ext.as_str() {
        "csv" => DataFormatOpt::Csv,
        "json" => DataFormatOpt::Json,
        "jsonl" | "ndjson" => DataFormatOpt::Ndjson,
        _ => DataFormatOpt::Text,
    }
}

fn resolve_format(requested: DataFormatOpt, path: Option<&Path>) -> DataFormatOpt {
    if requested != DataFormatOpt::Auto {
        return requested;
    }
    match path {
        Some(p) => detect_format(p),
        None => DataFormatOpt::Text,
    }
}

fn parse_text_values<R: Read>(reader: R) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut rdr = BufReader::new(reader);
    loop {
        buf.clear();
        let n = rdr
            .read_line(&mut buf)
            .map_err(|e| format!("read error: {e}"))?;
        if n == 0 {
            break;
        }
        for tok in buf.split(|c: char| c.is_whitespace() || c == ',' || c == ';') {
            if tok.is_empty() {
                continue;
            }
            if let Ok(v) = tok.parse::<f64>() {
                out.push(v);
            }
        }
    }
    Ok(out)
}

fn resolve_csv_index(
    selector: Option<&str>,
    headers: &[String],
    source_label: &str,
) -> Result<usize, String> {
    match selector {
        None => Ok(0),
        Some(sel) => {
            if let Ok(idx) = sel.parse::<usize>() {
                if idx < headers.len() {
                    return Ok(idx);
                }
                return Err(format!(
                    "csv column index {idx} out of bounds for {source_label} (columns={})",
                    headers.len()
                ));
            }
            let sel_lc = sel.to_ascii_lowercase();
            headers
                .iter()
                .position(|h| h.eq_ignore_ascii_case(sel) || h.to_ascii_lowercase() == sel_lc)
                .ok_or_else(|| format!("csv column {sel:?} not found in {source_label}"))
        }
    }
}

fn split_csv_row(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = line.chars().peekable();
    let mut in_quotes = false;

    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                if in_quotes && matches!(chars.peek(), Some('"')) {
                    cur.push('"');
                    chars.next();
                } else {
                    in_quotes = !in_quotes;
                }
            }
            ',' if !in_quotes => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    out.push(cur.trim().to_string());
    out
}

fn parse_csv_number(cell: &str, source_label: &str, row: usize, idx: usize) -> Result<f64, String> {
    let trimmed = cell.trim();
    if trimmed.is_empty() {
        return Err(format!(
            "csv parse error ({source_label}) at row {row} column {idx}: empty value"
        ));
    }
    trimmed
        .parse::<f64>()
        .map_err(|e| format!("csv parse error ({source_label}) at row {row} column {idx}: {e}"))
}

fn parse_csv_values<R: Read>(
    reader: R,
    column: Option<&str>,
    source_label: &str,
) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    let mut idx: Option<usize> = None;
    let mut saw_first_data_line = false;
    let mut line = String::new();
    let mut rdr = BufReader::new(reader);
    let mut line_no = 0usize;

    loop {
        line.clear();
        let n = rdr
            .read_line(&mut line)
            .map_err(|e| format!("csv read error ({source_label}): {e}"))?;
        if n == 0 {
            break;
        }
        line_no += 1;

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row = split_csv_row(trimmed);
        if row.is_empty() {
            continue;
        }

        if idx.is_none() {
            let selected_idx = resolve_csv_index(column, &row, source_label)?;
            idx = Some(selected_idx);

            if let Some(name) = column {
                if name.parse::<usize>().is_err() {
                    continue;
                }
            }

            let Some(first_cell) = row.get(selected_idx) else {
                continue;
            };
            if let Ok(v) = first_cell.parse::<f64>() {
                out.push(v);
                saw_first_data_line = true;
            }
            continue;
        }

        let selected_idx = idx.expect("csv index initialized");
        let Some(raw_cell) = row.get(selected_idx) else {
            continue;
        };

        if !saw_first_data_line && row.len() > selected_idx && raw_cell.parse::<f64>().is_err() {
            saw_first_data_line = true;
            continue;
        }

        let v = parse_csv_number(raw_cell, source_label, line_no, selected_idx)?;
        out.push(v);
    }
    Ok(out)
}

fn json_value_to_f64(v: &Value, context: &str) -> Result<f64, String> {
    match v {
        Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| format!("json number out of f64 range at {context}")),
        Value::String(s) => s
            .parse::<f64>()
            .map_err(|e| format!("json string parse error at {context}: {e}")),
        _ => Err(format!("expected numeric json value at {context}")),
    }
}

fn push_json_item(
    item: &Value,
    column: Option<&str>,
    context: &str,
    out: &mut Vec<f64>,
) -> Result<(), String> {
    match item {
        Value::Number(_) | Value::String(_) => {
            out.push(json_value_to_f64(item, context)?);
            Ok(())
        }
        Value::Object(map) => {
            let col = column.ok_or_else(|| {
                format!("json object encountered at {context}; pass --input-column/--probes-column")
            })?;
            let col_value = if let Ok(idx) = col.parse::<usize>() {
                let key = map
                    .keys()
                    .nth(idx)
                    .ok_or_else(|| format!("json object index {idx} missing at {context}"))?;
                map.get(key)
                    .ok_or_else(|| format!("json object index {idx} missing at {context}"))?
            } else {
                map.get(col)
                    .or_else(|| {
                        let col_lc = col.to_ascii_lowercase();
                        map.iter()
                            .find(|(k, _)| k.to_ascii_lowercase() == col_lc)
                            .map(|(_, v)| v)
                    })
                    .ok_or_else(|| format!("json object key {col:?} missing at {context}"))?
            };
            out.push(json_value_to_f64(col_value, context)?);
            Ok(())
        }
        _ => Err(format!("expected numeric or object json item at {context}")),
    }
}

fn parse_json_values<R: Read>(
    reader: R,
    column: Option<&str>,
    source_label: &str,
) -> Result<Vec<f64>, String> {
    let value: Value = serde_json::from_reader(reader)
        .map_err(|e| format!("json parse error ({source_label}): {e}"))?;

    let mut out = Vec::new();
    match value {
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                push_json_item(item, column, &format!("{source_label}[{i}]"), &mut out)?;
            }
        }
        Value::Object(map) => {
            let col = column.ok_or_else(|| {
                format!(
                    "top-level json object in {source_label}; pass --input-column/--probes-column"
                )
            })?;
            let selected = map
                .get(col)
                .or_else(|| {
                    let col_lc = col.to_ascii_lowercase();
                    map.iter()
                        .find(|(k, _)| k.to_ascii_lowercase() == col_lc)
                        .map(|(_, v)| v)
                })
                .ok_or_else(|| format!("json key {col:?} missing in {source_label}"))?;
            match selected {
                Value::Array(items) => {
                    for (i, item) in items.iter().enumerate() {
                        push_json_item(
                            item,
                            None,
                            &format!("{source_label}.{col}[{i}]"),
                            &mut out,
                        )?;
                    }
                }
                _ => out.push(json_value_to_f64(selected, source_label)?),
            }
        }
        _ => out.push(json_value_to_f64(&value, source_label)?),
    }
    Ok(out)
}

fn parse_ndjson_values<R: Read>(
    reader: R,
    column: Option<&str>,
    source_label: &str,
) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut rdr = BufReader::new(reader);
    let mut line_no = 0usize;
    loop {
        buf.clear();
        let n = rdr
            .read_line(&mut buf)
            .map_err(|e| format!("ndjson read error ({source_label}): {e}"))?;
        if n == 0 {
            break;
        }
        line_no += 1;
        let trimmed = buf.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(trimmed)
            .map_err(|e| format!("ndjson parse error ({source_label}) line {line_no}: {e}"))?;
        push_json_item(
            &value,
            column,
            &format!("{source_label}:line{line_no}"),
            &mut out,
        )?;
    }
    Ok(out)
}

fn parse_values_from_reader<R: Read>(
    reader: R,
    format: DataFormatOpt,
    column: Option<&str>,
    source_label: &str,
) -> Result<Vec<f64>, String> {
    match format {
        DataFormatOpt::Text => parse_text_values(reader),
        DataFormatOpt::Csv => parse_csv_values(reader, column, source_label),
        DataFormatOpt::Json => parse_json_values(reader, column, source_label),
        DataFormatOpt::Ndjson => parse_ndjson_values(reader, column, source_label),
        DataFormatOpt::Auto => Err("internal error: unresolved auto format".into()),
    }
}

fn parse_values_from_path(
    path: &Path,
    requested: DataFormatOpt,
    column: Option<&str>,
) -> Result<Vec<f64>, String> {
    let fmt = resolve_format(requested, Some(path));
    let file = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    parse_values_from_reader(file, fmt, column, &path.display().to_string())
}

fn parse_values_from_stdin(
    requested: DataFormatOpt,
    column: Option<&str>,
    source_label: &str,
) -> Result<Vec<f64>, String> {
    let fmt = resolve_format(requested, None);
    parse_values_from_reader(io::stdin().lock(), fmt, column, source_label)
}

fn parse_quantile_probe(raw: &str) -> Result<f64, String> {
    let p = raw
        .parse::<f64>()
        .map_err(|_| format!("p must be a finite number in [0,1] (got {raw:?})"))?;
    validate_quantile_probe(p).map_err(|msg| {
        if msg.starts_with('q') {
            msg.replacen('q', "p", 1)
        } else {
            msg.to_string()
        }
    })?;
    Ok(p)
}

fn parse_float_tokens(values: &[String], label: &str) -> Result<Vec<f64>, String> {
    let mut out = Vec::with_capacity(values.len());
    for raw in values {
        let v = raw
            .parse::<f64>()
            .map_err(|_| format!("{label} must be numeric (got {raw:?})"))?;
        out.push(v);
    }
    Ok(out)
}

fn is_representable_in_f32(v: f64) -> bool {
    let as32 = v as f32;
    as32.is_finite() && ((as32 as f64) - v).abs() <= 1e-6 * v.abs().max(1.0)
}

fn choose_precision(flag: PrecisionOpt, values: &[f64]) -> DigestPrecision {
    match flag {
        PrecisionOpt::F32 => DigestPrecision::F32,
        PrecisionOpt::F64 => DigestPrecision::F64,
        PrecisionOpt::Auto => {
            if !values.is_empty()
                && values
                    .iter()
                    .all(|v| v.is_finite() && is_representable_in_f32(*v))
            {
                DigestPrecision::F32
            } else {
                DigestPrecision::F64
            }
        }
    }
}

fn load_digest(path: &Path) -> Result<FrontendDigest, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    FrontendDigest::from_bytes(&bytes).map_err(|e| e.to_string())
}

fn save_digest(
    path: &Path,
    digest: &FrontendDigest,
    version: WireVersionOpt,
) -> Result<(), String> {
    let bytes = digest.to_bytes_with_version(version.into_wire());
    std::fs::write(path, bytes).map_err(|e| format!("failed to write {}: {e}", path.display()))
}

fn read_training_values(source: &SourceArgs) -> Result<(Vec<f64>, bool), String> {
    if source.stdin {
        let values = parse_values_from_stdin(
            source.input_format,
            source.input_column.as_deref(),
            "stdin(training)",
        )?;
        ensure_finite_training_values(&values).map_err(|e| e.to_string())?;
        return Ok((values, true));
    }
    if let Some(path) = source.input.as_deref() {
        let values =
            parse_values_from_path(path, source.input_format, source.input_column.as_deref())?;
        ensure_finite_training_values(&values).map_err(|e| e.to_string())?;
        return Ok((values, true));
    }
    Ok((Vec::new(), false))
}

fn read_cdf_probes(probes: &ProbeArgs, training_values: &[f64]) -> Result<Vec<f64>, String> {
    if !probes.x.is_empty() {
        return parse_float_tokens(&probes.x, "x");
    }
    if probes.probes_stdin {
        return parse_values_from_stdin(
            probes.probes_format,
            probes.probes_column.as_deref(),
            "stdin(probes)",
        );
    }
    if let Some(path) = probes.probes_input.as_deref() {
        return parse_values_from_path(path, probes.probes_format, probes.probes_column.as_deref());
    }
    Ok(training_values.to_vec())
}

fn build_digest(
    source: &SourceArgs,
    digest_args: &DigestArgs,
    training_values: &[f64],
    training_source_provided: bool,
) -> Result<FrontendDigest, String> {
    let scale = scale_from_opt(digest_args.scale.clone())?;
    let policy = policy_from_opts(digest_args.singleton_policy, digest_args.pin_per_side)?;
    let cfg = DigestConfig {
        max_size: digest_args.max_size,
        scale,
        policy,
    };

    let mut merge_iter = source.merge_digest.iter();

    let mut digest = if let Some(path) = source.from_digest.as_deref() {
        load_digest(path)?
    } else if training_source_provided {
        let precision = choose_precision(digest_args.precision, training_values);
        FrontendDigest::from_values(training_values.to_vec(), cfg, precision)
            .map_err(|e| e.to_string())?
    } else if let Some(first_merge) = merge_iter.next() {
        load_digest(first_merge)?
    } else {
        return Err(
            "no digest source provided (use --stdin/--input, --from-digest, or --merge-digest)"
                .into(),
        );
    };

    if source.from_digest.is_some() && training_source_provided && !training_values.is_empty() {
        digest
            .add_values_f64(training_values.to_vec())
            .map_err(|e| e.to_string())?;
    }

    for path in merge_iter {
        let other = load_digest(path)?;
        digest.merge_in_place(&other).map_err(|e| e.to_string())?;
    }

    Ok(digest)
}

fn print_header(output: &OutputArgs, csv_header: &str) {
    if matches!(output.output, OutputOpt::Csv) && !output.no_header {
        println!("{csv_header}");
    }
}

fn maybe_save_digest(persist: &PersistArgs, digest: &FrontendDigest) -> Result<(), String> {
    if let Some(path) = persist.to_digest.as_deref() {
        save_digest(path, digest, persist.wire_version)?;
    }
    Ok(())
}

fn run_build(cmd: BuildCmd) -> Result<(), String> {
    let (training_values, training_source_provided) = read_training_values(&cmd.source)?;
    let digest = build_digest(
        &cmd.source,
        &cmd.digest,
        &training_values,
        training_source_provided,
    )?;
    save_digest(&cmd.to_digest, &digest, cmd.wire_version)
}

fn run_quantile(cmd: QuantileCmd) -> Result<(), String> {
    let (training_values, training_source_provided) = read_training_values(&cmd.query.source)?;
    let digest = build_digest(
        &cmd.query.source,
        &cmd.query.digest,
        &training_values,
        training_source_provided,
    )?;
    maybe_save_digest(&cmd.query.persist, &digest)?;

    print_header(&cmd.output, "p,value");
    for p_raw in &cmd.p {
        let p = parse_quantile_probe(p_raw.trim())?;
        let q = digest.quantile_strict(p).map_err(|e| e.to_string())?;
        match cmd.output.output {
            OutputOpt::Csv => println!("{p},{q}"),
            OutputOpt::Text => println!("{q}"),
        }
    }
    Ok(())
}

fn run_cdf(cmd: CdfCmd) -> Result<(), String> {
    if cmd.query.source.stdin && cmd.probes.probes_stdin {
        return Err("cannot use both --stdin and --probes-stdin in one invocation".into());
    }

    let (training_values, training_source_provided) = read_training_values(&cmd.query.source)?;
    let digest = build_digest(
        &cmd.query.source,
        &cmd.query.digest,
        &training_values,
        training_source_provided,
    )?;
    maybe_save_digest(&cmd.query.persist, &digest)?;

    let probes = read_cdf_probes(&cmd.probes, &training_values)?;
    let ps = digest.cdf(&probes);

    print_header(&cmd.output, "x,p");
    for (x, p) in probes.iter().zip(ps.iter()) {
        match cmd.output.output {
            OutputOpt::Csv => println!("{x},{p}"),
            OutputOpt::Text => println!("{p}"),
        }
    }
    Ok(())
}

fn run_median(cmd: MedianCmd) -> Result<(), String> {
    let (training_values, training_source_provided) = read_training_values(&cmd.query.source)?;
    let digest = build_digest(
        &cmd.query.source,
        &cmd.query.digest,
        &training_values,
        training_source_provided,
    )?;
    maybe_save_digest(&cmd.query.persist, &digest)?;

    print_header(&cmd.output, "value");
    println!("{}", digest.median());
    Ok(())
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Build(cmd) => run_build(cmd),
        Commands::Quantile(cmd) => run_quantile(cmd),
        Commands::Cdf(cmd) => run_cdf(cmd),
        Commands::Median(cmd) => run_median(cmd),
    }
}
