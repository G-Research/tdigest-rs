//! Time digest construction from unsorted values and measure peak memory on Linux.

use std::process::Command;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use tdigest_rs::tdigest::{ScaleFamily, TDigest};
use testdata::{gen_dataset, DistKind};

#[cfg(target_os = "linux")]
fn rss_peak_kib() -> Option<u64> {
    // ru_maxrss is in KiB on Linux
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        (libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0).then_some(usage.ru_maxrss as u64)
    }
}

#[cfg(not(target_os = "linux"))]
fn rss_peak_kib() -> Option<u64> {
    None
}

fn build_digest_unsorted(vals: Vec<f64>) -> TDigest<f64> {
    TDigest::builder()
        .max_size(1_000)
        .scale(ScaleFamily::K2)
        .build()
        .unwrap()
        .merge_unsorted(vals)
        .expect("finite benchmark data")
}

fn bench_merge_unsorted(c: &mut Criterion) {
    let sizes = [1_000usize, 10_000, 1_000_000, 10_000_000];

    for &n in &sizes {
        let mut group = c.benchmark_group(format!("merge_unsorted/size={n}"));

        if n >= 10_000_000 {
            group
                .sample_size(10)
                .measurement_time(Duration::from_secs(6));
        } else if n >= 1_000_000 {
            group
                .sample_size(15)
                .measurement_time(Duration::from_secs(4));
        } else {
            group
                .sample_size(30)
                .measurement_time(Duration::from_secs(3));
        }
        group.warm_up_time(Duration::from_secs(1));
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("build", n), &n, |b, &nn| {
            b.iter_batched(
                || gen_dataset(DistKind::Mixture, nn, 4242),
                |vals| build_digest_unsorted(vals),
                if nn >= 1_000_000 {
                    BatchSize::LargeInput
                } else {
                    BatchSize::SmallInput
                },
            );
        });

        group.finish();

        // A fresh process measures peak RSS for this input size alone.
        let exe = std::env::current_exe().expect("current exe");
        let out = Command::new(exe)
            .arg("--peak")
            .arg(n.to_string())
            .output()
            .expect("spawn peak child");
        if out.status.success() {
            eprint!("{}", String::from_utf8_lossy(&out.stdout));
        } else {
            eprintln!(
                "[peak] size={} FAILED: {}",
                n,
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
}

fn configure() -> Criterion {
    Criterion::default()
        .without_plots()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(30)
}

fn run_bench_suite() {
    let mut c = configure();
    bench_merge_unsorted(&mut c);
    c.final_summary();
}

fn run_peak_once(n: usize) {
    let vals = gen_dataset(DistKind::Mixture, n, 4242);
    let _td = build_digest_unsorted(vals);
    if let Some(kib) = rss_peak_kib() {
        let mib = kib as f64 / 1024.0;
        println!("[peak] size={n} rss_peak_kib={kib} ({mib:.1} MiB)");
    } else {
        println!("[peak] size={n} peak RSS unavailable");
    }
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().map(|s| s.as_str()) == Some("--peak") {
        match args.get(1).and_then(|s| s.parse::<usize>().ok()) {
            Some(n) => {
                run_peak_once(n);
                return;
            }
            None => {
                eprintln!("[peak] usage: --peak <size>");
                std::process::exit(2);
            }
        }
    }

    run_bench_suite();
}
