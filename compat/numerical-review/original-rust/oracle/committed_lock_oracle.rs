// Dependency-free serializer; this example calls the unmodified original crate.
use tdigest_core::TDigest;

fn quote(text: &str) -> String {
    // Every string emitted here is a fixed ASCII label or a hexadecimal bit string.
    assert!(text.is_ascii());
    format!("{text:?}")
}

fn array(items: impl IntoIterator<Item = String>) -> String {
    format!("[{}]", items.into_iter().collect::<Vec<_>>().join(","))
}

fn object(items: impl IntoIterator<Item = (&'static str, String)>) -> String {
    let entries = items
        .into_iter()
        .map(|(key, value)| format!("{}:{value}", quote(key)));
    format!("{{{}}}", entries.collect::<Vec<_>>().join(","))
}

fn bits(value: f64) -> String {
    quote(&format!("{:016x}", value.to_bits()))
}

macro_rules! cases {
    ($float:ty, $dtype:literal) => {{
        let mut cases = Vec::<String>::new();
        let queries = [
            f64::NEG_INFINITY,
            -0.1,
            0.0,
            0.0001,
            0.1,
            0.5,
            0.9,
            1.0,
            1.1,
            f64::INFINITY,
            f64::NAN,
        ];
        let bounds: [(f64, f64); 6] = [
            (0.0, 1.0),
            (0.2, 0.8),
            (0.49, 0.51),
            (0.9, 0.95),
            (0.8, 0.2),
            (0.5, 0.5),
        ];
        let mut record = |name: &str, input: String, digest: TDigest<$float>| {
            cases.push(object([
                ("name", quote(&format!("{}-{name}", $dtype))),
                ("dtype", quote($dtype)),
                ("input", input),
                ("means", array(digest.means.iter().map(|&x| bits(x as f64)))),
                ("weights", array(digest.weights.iter().map(u32::to_string))),
                ("median", bits(digest.median().unwrap() as f64)),
                (
                    "quantiles",
                    array(queries.iter().map(|&q| {
                        array([bits(q), bits(digest.quantile(q as $float).unwrap() as f64)])
                    })),
                ),
                (
                    "trimmed_means",
                    array(bounds.iter().map(|&(lower, upper)| {
                        array([
                            lower.to_string(),
                            upper.to_string(),
                            bits(
                                digest
                                    .trimmed_mean(lower as $float, upper as $float)
                                    .unwrap() as f64,
                            ),
                        ])
                    })),
                ),
            ]));
        };
        let arrays: Vec<(&str, Vec<$float>)> = vec![
            ("singleton", vec![3.0]),
            ("two", vec![1.0, 2.0]),
            ("ties", vec![3.0, 3.0, 1.0, 0.0, 0.0, -0.0]),
            (
                "nonfinite",
                vec![
                    <$float>::NEG_INFINITY,
                    -2.0,
                    0.0,
                    <$float>::INFINITY,
                    <$float>::NAN,
                ],
            ),
            ("allnan", vec![<$float>::NAN, <$float>::NAN]),
            (
                "tiny",
                vec![-<$float>::MIN_POSITIVE, -0.0, 0.0, <$float>::MIN_POSITIVE],
            ),
            (
                "compressed",
                (0..1000)
                    .map(|i| ((i * 37 % 1000) as $float - 500.0) / 10.0)
                    .collect(),
            ),
        ];
        for (name, values) in &arrays {
            for delta in [10.0, 30.0, 300.0] {
                let digest = TDigest::<$float>::from_array(values, delta as $float).unwrap();
                let input = object([
                    ("op", quote("from_array")),
                    ("dataset", quote(name)),
                    ("delta", delta.to_string()),
                ]);
                record(&format!("{name}-delta-{delta}"), input, digest);
            }
        }
        for (name, values, weights) in [
            ("weighted", vec![0.0, 10.0], vec![90, 10]),
            ("large-weight", vec![1.0], vec![16_777_217]),
            ("uint32-max", vec![1.0], vec![u32::MAX]),
            (
                "uneven",
                vec![-10.0, 0.0, 1.0, 3.0, 100.0],
                vec![1, 5, 900, 7, 2],
            ),
        ] {
            let digest = TDigest::<$float>::from_means_weights(&values, &weights, 300.0).unwrap();
            let input = object([
                ("op", quote("from_means_weights")),
                ("dataset", quote(name)),
                ("weights", array(weights.iter().map(u32::to_string))),
                ("delta", "300.0".into()),
            ]);
            record(name, input, digest);
        }
        let values: Vec<$float> = (0..1000).map(|i| i as $float / 10.0).collect();
        for delta in [10.0, 30.0, 300.0] {
            let left = TDigest::<$float>::from_array(&values[..431], 17.0).unwrap();
            let right = TDigest::<$float>::from_array(&values[431..], 23.0).unwrap();
            for operation in ["merge", "update"] {
                let input = object([
                    ("op", quote(operation)),
                    ("dataset", quote("linear")),
                    ("split", "431".into()),
                    ("delta", "17.0".into()),
                    ("other_delta", "23.0".into()),
                    ("merge_delta", delta.to_string()),
                ]);
                record(
                    &format!("{operation}-{delta}"),
                    input,
                    left.merge(&right, delta as $float).unwrap(),
                );
            }
        }
        for operation in ["copy", "deepcopy", "pickle", "dict"] {
            let digest = TDigest::<$float>::from_array(&values, 17.0).unwrap();
            let restored =
                TDigest::<$float>::from_means_weights(&digest.means, &digest.weights, 300.0)
                    .unwrap();
            let input = object([
                ("op", quote(operation)),
                ("dataset", quote("linear")),
                ("delta", "17.0".into()),
            ]);
            record(operation, input, restored);
        }
        cases
    }};
}

fn main() {
    let mut all = cases!(f32, "float32");
    all.extend(cases!(f64, "float64"));
    assert_eq!(all.len(), 70);
    println!(
        "{}",
        object([
            (
                "upstream_commit",
                quote("8d9cf1fe82c5c9254e5dc9593f01717e317f4333")
            ),
            ("cases", array(all)),
        ])
    );
}
