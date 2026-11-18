// Run against the independent pinned G-Research tdigest_core source.
use serde_json::{json, Value};
use tdigest_core::TDigest;

macro_rules! cases {
    ($float:ty, $dtype:expr) => {{
        let mut cases = Vec::<Value>::new();
        let encode = |xs: &[$float]| xs.iter().map(|x| format!("{:016x}", (*x as f64).to_bits())).collect::<Vec<_>>();
        let queries = [f64::NEG_INFINITY, -0.1, 0.0, 0.0001, 0.1, 0.5, 0.9, 1.0, 1.1, f64::INFINITY, f64::NAN];
        let bounds = [(0.0,1.0),(0.2,0.8),(0.49,0.51),(0.9,0.95),(0.8,0.2),(0.5,0.5)];
        let mut record = |name: &str, input: Value, digest: TDigest<$float>| {
            cases.push(json!({
                "name": format!("{}-{}", $dtype, name), "dtype": $dtype, "input": input,
                "means": encode(&digest.means), "weights": digest.weights,
                "median": format!("{:016x}", (digest.median().unwrap() as f64).to_bits()),
                "quantiles": queries.iter().map(|q| json!([
                    format!("{:016x}", q.to_bits()),
                    format!("{:016x}", (digest.quantile(*q as $float).unwrap() as f64).to_bits())])).collect::<Vec<_>>(),
                "trimmed_means": bounds.iter().map(|(lo,hi)| json!([
                    lo, hi, format!("{:016x}", (digest.trimmed_mean(*lo as $float,*hi as $float).unwrap() as f64).to_bits())])).collect::<Vec<_>>()
            }));
        };
        let arrays: Vec<(&str, Vec<$float>)> = vec![
            ("singleton", vec![3.0]), ("two", vec![1.0,2.0]),
            ("ties", vec![3.0,3.0,1.0,0.0,0.0,-0.0]),
            ("nonfinite", vec![<$float>::NEG_INFINITY,-2.0,0.0,<$float>::INFINITY,<$float>::NAN]),
            ("allnan", vec![<$float>::NAN,<$float>::NAN]),
            ("tiny", vec![-<$float>::MIN_POSITIVE, -0.0, 0.0, <$float>::MIN_POSITIVE]),
            ("compressed", (0..1000).map(|i| ((i * 37 % 1000) as $float - 500.0) / 10.0).collect()),
        ];
        for (name, values) in &arrays {
            for delta in [10.0,30.0,300.0] {
                let d = TDigest::<$float>::from_array(values,delta as $float).unwrap();
                record(&format!("{}-delta-{}", name, delta), json!({"op":"from_array","dataset":name,"delta":delta}), d);
            }
        }
        for (name, values, weights) in [
            ("weighted", vec![0.0,10.0], vec![90,10]),
            ("large-weight", vec![1.0], vec![16_777_217]),
            ("uint32-max", vec![1.0], vec![u32::MAX]),
            ("uneven", vec![-10.0,0.0,1.0,3.0,100.0], vec![1,5,900,7,2]),
        ] {
            let d = TDigest::<$float>::from_means_weights(&values,&weights,300.0).unwrap();
            record(name, json!({"op":"from_means_weights","dataset":name,"weights":weights,"delta":300.0}), d);
        }
        let values: Vec<$float> = (0..1000).map(|i| (i as $float) / 10.0).collect();
        for delta in [10.0,30.0,300.0] {
            let left = TDigest::<$float>::from_array(&values[..431],17.0).unwrap();
            let right = TDigest::<$float>::from_array(&values[431..],23.0).unwrap();
            let merged = left.merge(&right,delta as $float).unwrap();
            let input = json!({"op":"merge","dataset":"linear","split":431,"delta":17.0,"other_delta":23.0,"merge_delta":delta});
            record(&format!("merge-{}",delta), input.clone(), merged);
            let mut update_input = input.clone();
            update_input["op"] = json!("update");
            record(&format!("update-{}",delta), update_input, left.merge(&right,delta as $float).unwrap());
        }
        for op in ["copy", "deepcopy", "pickle", "dict"] {
            let d = TDigest::<$float>::from_array(&values,17.0).unwrap();
            let restored = TDigest::<$float>::from_means_weights(&d.means,&d.weights,300.0).unwrap();
            record(op, json!({"op":op,"dataset":"linear","delta":17.0}),restored);
        }
        cases
    }};
}
fn main() {
    let mut all = cases!(f32,"float32");
    all.extend(cases!(f64,"float64"));
    println!("{}", serde_json::to_string_pretty(&json!({"upstream_commit":"8d9cf1fe82c5c9254e5dc9593f01717e317f4333","cases":all})).unwrap());
}
