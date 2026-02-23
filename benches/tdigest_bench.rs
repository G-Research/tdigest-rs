//! TDigest "merge_unsorted" bench + peak memory (RSS) per size.
//!
//! - Criterion timing of end-to-end build from unsorted Vec<f64>.
//! - Separate one-shot child process per size to print *peak* memory
//!   during the build (includes sort + compress, input Vec, jemalloc arenas, etc).

use std::process::Command;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use tdigest_rs::tdigest::{ScaleFamily, TDigest};
use testdata::{gen_dataset, DistKind};

#[cfg(target_os = "linux")]
fn rss_peak_kib() -> u64 {
    // ru_maxrss is in KiB on Linux
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        usage.ru_maxrss as u64
    }
}

#[cfg(not(target_os = "linux"))]
fn rss_peak_kib() -> u64 {
    // Fallback: unknown platform → 0 (you can add macOS proc reading if you want)
    0
}

fn build_digest_unsorted(vals: Vec<f64>) -> TDigest<f64> {
    TDigest::builder()
        .max_size(1_000)
        .scale(ScaleFamily::K2)
        .build()
        .merge_unsorted(vals)
        .expect("valid benchmark dataset")
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

        // ---- true peak memory: run in a fresh process so ru_maxrss is per-size ----
        let exe = std::env::current_exe().expect("current exe");
        let out = Command::new(exe)
            .arg("--peak")
            .arg(n.to_string())
            .output()
            .expect("spawn peak child");
        if out.status.success() {
            // Child prints a single line: "[peak] size=... rss_peak_kib=... (~MiB)"
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

/* ------------------------ custom harness ------------------------ */

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

fn run_peak_once(n: usize) -> anyhow::Result<()> {
    // Build one digest of size n, then print peak RSS (KiB and MiB)
    let vals = gen_dataset(DistKind::Mixture, n, 4242);
    let _td = build_digest_unsorted(vals);
    let kib = rss_peak_kib();
    let mib = kib as f64 / 1024.0;
    println!("[peak] size={} rss_peak_kib={} (~{:.1} MiB)", n, kib, mib);
    Ok(())
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.get(0).map(|s| s.as_str()) == Some("--peak") {
        match args.get(1).and_then(|s| s.parse::<usize>().ok()) {
            Some(n) => {
                if let Err(e) = run_peak_once(n) {
                    eprintln!("[peak] error: {e}");
                    std::process::exit(1);
                }
                return;
            }
            None => {
                eprintln!("[peak] usage: --peak <size>");
                std::process::exit(2);
            }
        }
    }

    // No --peak: run the Criterion suite.
    run_bench_suite();
}
