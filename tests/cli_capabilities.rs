use assert_cmd::Command;
use assert_fs::prelude::*;
use tdigest_rs::tdigest::wire::{wire_precision, WirePrecision};

fn run_cli(args: &[&str], stdin_data: Option<&str>) -> String {
    let mut cmd = Command::cargo_bin("tdigest").expect("cli binary");
    cmd.args(args);
    if let Some(data) = stdin_data {
        cmd.write_stdin(data);
    }
    let out = cmd.assert().success().get_output().stdout.clone();
    String::from_utf8(out)
        .expect("utf8 output")
        .trim()
        .to_string()
}

#[test]
fn cli_supports_median_on_stdin_training_data() {
    let out = run_cli(
        &["median", "--stdin", "--no-header", "--output", "csv"],
        Some("0 1 2 3"),
    );
    let median = out.parse::<f64>().expect("median output");
    assert!((median - 1.5).abs() <= 1e-12);
}

#[test]
fn cli_can_train_from_csv_and_probe_from_json() {
    let td = assert_fs::TempDir::new().expect("temp dir");
    let train = td.child("train.csv");
    let probes = td.child("probes.json");

    train
        .write_str("x,y\n0,10\n1,11\n2,12\n3,13\n")
        .expect("write train csv");
    probes.write_str("[0,2,3]").expect("write probes json");

    let out = run_cli(
        &[
            "cdf",
            "--input",
            train.path().to_str().expect("train path"),
            "--input-format",
            "csv",
            "--input-column",
            "x",
            "--probes-input",
            probes.path().to_str().expect("probes path"),
            "--probes-format",
            "json",
            "--no-header",
            "--output",
            "csv",
        ],
        None,
    );

    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 3);

    let mut p_at_2 = None;
    for line in lines {
        let (x_raw, p_raw) = line.split_once(',').expect("csv row");
        let x = x_raw.parse::<f64>().expect("x");
        let p = p_raw.parse::<f64>().expect("p");
        if (x - 2.0).abs() <= 1e-12 {
            p_at_2 = Some(p);
        }
    }

    let p2 = p_at_2.expect("cdf at x=2");
    assert!((p2 - 0.625).abs() <= 1e-9);
}

#[test]
fn cli_can_save_and_load_digest_for_inference() {
    let td = assert_fs::TempDir::new().expect("temp dir");
    let train = td.child("train.json");
    let blob = td.child("model.tdig");

    train.write_str("[0,1,2,3]").expect("write train json");

    let median_out = run_cli(
        &[
            "median",
            "--input",
            train.path().to_str().expect("train path"),
            "--input-format",
            "json",
            "--to-digest",
            blob.path().to_str().expect("blob path"),
            "--wire-version",
            "v3",
            "--no-header",
            "--output",
            "csv",
        ],
        None,
    );
    let median = median_out.parse::<f64>().expect("median");
    assert!((median - 1.5).abs() <= 1e-12);
    blob.assert(predicates::path::exists());

    let q_out = run_cli(
        &[
            "quantile",
            "--from-digest",
            blob.path().to_str().expect("blob path"),
            "--p",
            "0.5",
            "--no-header",
            "--output",
            "csv",
        ],
        None,
    );
    let (p_raw, q_raw) = q_out.split_once(',').expect("quantile csv row");
    let p = p_raw.parse::<f64>().expect("p");
    let q = q_raw.parse::<f64>().expect("q");
    assert!((p - 0.5).abs() <= 1e-12);
    assert!((q - 1.5).abs() <= 1e-12);
}

#[test]
fn cli_precision_auto_uses_f32_when_values_fit_and_f64_when_not() {
    let td = assert_fs::TempDir::new().expect("temp dir");
    let small = td.child("small.json");
    let large = td.child("large.json");
    let small_blob = td.child("small.tdig");
    let large_blob = td.child("large.tdig");

    small.write_str("[0,1,2,3]").expect("write small");
    large.write_str("[1e100,2e100]").expect("write large");

    run_cli(
        &[
            "median",
            "--input",
            small.path().to_str().expect("small path"),
            "--input-format",
            "json",
            "--precision",
            "auto",
            "--to-digest",
            small_blob.path().to_str().expect("small blob"),
            "--no-header",
            "--output",
            "csv",
        ],
        None,
    );
    run_cli(
        &[
            "median",
            "--input",
            large.path().to_str().expect("large path"),
            "--input-format",
            "json",
            "--precision",
            "auto",
            "--to-digest",
            large_blob.path().to_str().expect("large blob"),
            "--no-header",
            "--output",
            "csv",
        ],
        None,
    );

    let small_bytes = std::fs::read(small_blob.path()).expect("read small blob");
    let large_bytes = std::fs::read(large_blob.path()).expect("read large blob");
    assert_eq!(
        wire_precision(&small_bytes).expect("small precision"),
        WirePrecision::F32
    );
    assert_eq!(
        wire_precision(&large_bytes).expect("large precision"),
        WirePrecision::F64
    );
}

#[test]
fn cli_supports_ndjson_with_object_column() {
    let td = assert_fs::TempDir::new().expect("temp dir");
    let train = td.child("train.ndjson");
    train
        .write_str("{\"v\":0}\n{\"v\":1}\n{\"v\":2}\n{\"v\":3}\n")
        .expect("write ndjson");

    let out = run_cli(
        &[
            "quantile",
            "--input",
            train.path().to_str().expect("train path"),
            "--input-format",
            "ndjson",
            "--input-column",
            "v",
            "--p",
            "0.5",
            "--no-header",
            "--output",
            "csv",
        ],
        None,
    );
    let (_, q_raw) = out.split_once(',').expect("quantile row");
    let q = q_raw.parse::<f64>().expect("q");
    assert!((q - 1.5).abs() <= 1e-12);
}
