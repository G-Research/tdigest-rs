#![cfg(feature = "java")]

use jni::objects::{JByteArray, JClass, JDoubleArray, JFloatArray, JObject, JString, JValue};
use jni::sys::{
    jarray, jboolean, jbyte, jbyteArray, jdouble, jdoubleArray, jint, jlong, JNI_VERSION_1_8,
};
use jni::{JNIEnv, JavaVM};

use std::ffi::c_void;
use std::slice;

use crate::tdigest::frontends::{
    parse_scale_str, policy_from_code_edges, DigestConfig, DigestPrecision, FrontendDigest,
};
use crate::tdigest::singleton_policy::SingletonPolicy;
use crate::tdigest::wire::WireVersion;
use crate::tdigest::ScaleFamily;

#[no_mangle]
pub extern "system" fn JNI_OnLoad(_vm: JavaVM, _reserved: *mut c_void) -> jint {
    JNI_VERSION_1_8
}

fn map_scale(s: &str) -> Result<ScaleFamily, String> {
    parse_scale_str(Some(s)).map_err(|error| error.to_string())
}

fn map_policy(code: jint, edges: jint) -> Result<SingletonPolicy, String> {
    policy_from_code_edges(code, edges).map_err(|e| e.to_string())
}

fn map_wire_version(version: jint) -> Result<WireVersion, String> {
    let v = u8::try_from(version).map_err(|_| format!("unsupported TDIG version: {version}"))?;
    WireVersion::from_u8(v).map_err(|e| e.to_string())
}

fn ensure_non_null_handle(handle: jlong) -> Result<(), &'static str> {
    if handle == 0 {
        Err("TDigest handle is null")
    } else {
        Ok(())
    }
}

fn read_jdouble_array(env: &JNIEnv, arr: &JDoubleArray) -> Result<Vec<f64>, String> {
    let len = env.get_array_length(arr).unwrap_or(0) as usize;
    let mut out = vec![0f64; len];
    if len > 0 {
        env.get_double_array_region(arr, 0, &mut out)
            .map_err(|e| format!("jni: get_double_array_region failed: {e:?}"))?;
    }
    Ok(out)
}

fn read_jfloat_array(env: &JNIEnv, arr: &JFloatArray) -> Result<Vec<f32>, String> {
    let len = env.get_array_length(arr).unwrap_or(0) as usize;
    let mut out = vec![0f32; len];
    if len > 0 {
        env.get_float_array_region(arr, 0, &mut out)
            .map_err(|e| format!("jni: get_float_array_region failed: {e:?}"))?;
    }
    Ok(out)
}

// A catchable Rust panic must never unwind across the JNI ABI. Allocation
// aborts, invalid raw pointers and VM/process failure are not catchable errors.
fn catch_native<T>(operation: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)).map_err(|panic| {
        let message = panic
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("unknown native panic");
        format!("TDigest native operation failed: {message}")
    })
}

fn jni_call<T>(mut env: JNIEnv, fallback: T, operation: impl FnOnce(&mut JNIEnv) -> T) -> T {
    match catch_native(|| operation(&mut env)) {
        Ok(value) => value,
        Err(message) => {
            if !env.exception_check().unwrap_or(true) {
                let _ = env.throw_new("java/lang/IllegalStateException", message);
            }
            fallback
        }
    }
}

fn into_handle(digest: FrontendDigest) -> jlong {
    Box::into_raw(Box::new(digest)) as jlong
}

unsafe fn from_handle<'a>(handle: jlong) -> &'a mut FrontendDigest {
    &mut *(handle as *mut FrontendDigest)
}

unsafe fn from_handle_const<'a>(handle: jlong) -> &'a FrontendDigest {
    &*(handle as *const FrontendDigest)
}

fn throw_illegal_arg(env: &mut JNIEnv, message: impl ToString) {
    let _ = env.throw_new("java/lang/IllegalArgumentException", message.to_string());
}

#[no_mangle]
pub extern "system" fn Java_com_gresearch_tdigest_TDigestNative_fromBytes(
    env: JNIEnv,
    _cls: JClass,
    bytes: JByteArray,
) -> jlong {
    jni_call(env, 0, |env| {
        let len = env.get_array_length(&bytes).unwrap_or(0) as usize;
        let mut buf = vec![0i8; len];
        if len > 0 {
            if let Err(e) = env.get_byte_array_region(&bytes, 0, &mut buf) {
                throw_illegal_arg(env, format!("jni: get_byte_array_region failed: {e:?}"));
                return 0;
            }
        }

        let ubuf: &[u8] = unsafe { slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len()) };

        match FrontendDigest::from_bytes(ubuf) {
            Ok(digest) => into_handle(digest),
            Err(error) => {
                throw_illegal_arg(env, format!("jni: {error}"));
                0
            }
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_com_gresearch_tdigest_TDigestNative_toBytes(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
) -> jbyteArray {
    jni_call(env, std::ptr::null_mut(), |env| {
        // Canonical TDIG wire format — width follows backend precision:
        // - F32 → f32 means on the wire
        // - F64 → f64 means on the wire
        let bytes: Vec<u8> = if handle == 0 {
            Vec::new()
        } else {
            let d = unsafe { from_handle_const(handle) };
            d.to_bytes()
        };

        let arr = env
            .new_byte_array(bytes.len() as jint)
            .expect("new_byte_array");
        if !bytes.is_empty() {
            let ptr = bytes.as_ptr() as *const jbyte;
            let slice_i8 = unsafe { slice::from_raw_parts(ptr, bytes.len()) };
            env.set_byte_array_region(&arr, 0, slice_i8)
                .expect("set_byte_array_region");
        }
        arr.into_raw()
    })
}

#[no_mangle]
pub extern "system" fn Java_com_gresearch_tdigest_TDigestNative_toBytesVersion(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    version: jint,
) -> jbyteArray {
    jni_call(env, std::ptr::null_mut(), |env| {
        let arr = env.new_byte_array(0).expect("new_byte_array");
        if handle == 0 {
            return arr.into_raw();
        }

        let version = match map_wire_version(version) {
            Ok(v) => v,
            Err(msg) => {
                throw_illegal_arg(env, msg);
                return arr.into_raw();
            }
        };

        let d = unsafe { from_handle_const(handle) };
        if let Err(error) = d.validate_wire_version(version) {
            throw_illegal_arg(env, error);
            return arr.into_raw();
        }
        let bytes = d.to_bytes_with_version(version);

        let out = env
            .new_byte_array(bytes.len() as jint)
            .expect("new_byte_array");
        if !bytes.is_empty() {
            let ptr = bytes.as_ptr() as *const jbyte;
            let slice_i8 = unsafe { slice::from_raw_parts(ptr, bytes.len()) };
            env.set_byte_array_region(&out, 0, slice_i8)
                .expect("set_byte_array_region");
        }
        out.into_raw()
    })
}

#[no_mangle]
pub extern "system" fn Java_com_gresearch_tdigest_TDigestNative_fromArray(
    env: JNIEnv,
    _cls: JClass,
    values_obj: JObject,
    max_size: jint,
    scale: JString,
    policy_code: jint,
    edges: jint,
    f32mode: jboolean,
    delta: jdouble,
) -> jlong {
    jni_call(env, 0, |env| {
        let xs: Vec<f64>;

        if env.is_instance_of(&values_obj, "[D").unwrap_or(false) {
            let raw: jarray = env
                .new_local_ref(&values_obj)
                .expect("new_local_ref")
                .as_raw();
            let darr = unsafe { JDoubleArray::from_raw(raw) };
            match read_jdouble_array(env, &darr) {
                Ok(v) => xs = v,
                Err(msg) => {
                    throw_illegal_arg(env, msg);
                    return 0;
                }
            }
        } else if env.is_instance_of(&values_obj, "[F").unwrap_or(false) {
            let raw: jarray = env
                .new_local_ref(&values_obj)
                .expect("new_local_ref")
                .as_raw();
            let farr = unsafe { JFloatArray::from_raw(raw) };
            match read_jfloat_array(env, &farr) {
                Ok(v) => xs = v.into_iter().map(|x| x as f64).collect(),
                Err(msg) => {
                    throw_illegal_arg(env, msg);
                    return 0;
                }
            }
        } else if env
            .is_instance_of(&values_obj, "java/util/List")
            .unwrap_or(false)
        {
            let size = env
                .call_method(&values_obj, "size", "()I", &[])
                .and_then(|v| v.i())
                .unwrap_or(0);
            let mut out = Vec::with_capacity(size as usize);
            for i in 0..size {
                let elem = env
                    .call_method(
                        &values_obj,
                        "get",
                        "(I)Ljava/lang/Object;",
                        &[JValue::from(i)],
                    )
                    .ok()
                    .and_then(|v| v.l().ok());
                if let Some(obj) = elem {
                    if let Ok(dval) = env.call_method(obj, "doubleValue", "()D", &[]) {
                        if let Ok(d) = dval.d() {
                            out.push(d);
                        }
                    }
                }
            }
            xs = out;
        } else {
            throw_illegal_arg(env, "fromArray expects a numeric array");
            return 0;
        }

        let scale_str = env
            .get_string(&scale)
            .ok()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "k2".to_string());
        let sf = match map_scale(&scale_str) {
            Ok(scale) => scale,
            Err(error) => {
                throw_illegal_arg(env, error);
                return 0;
            }
        };
        if max_size <= 0 {
            throw_illegal_arg(env, "max_size must be positive");
            return 0;
        }
        let policy = match map_policy(policy_code, edges) {
            Ok(p) => p,
            Err(msg) => {
                throw_illegal_arg(env, msg);
                return 0;
            }
        };
        let config = DigestConfig {
            max_size: max_size as usize,
            scale: sf,
            policy,
            delta: (delta != 0.0).then_some(delta),
        };
        let precision = if f32mode != 0 {
            DigestPrecision::F32
        } else {
            DigestPrecision::F64
        };
        match FrontendDigest::from_values(xs, config, precision) {
            Ok(digest) => into_handle(digest),
            Err(e) => {
                throw_illegal_arg(env, e);
                0
            }
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_com_gresearch_tdigest_TDigestNative_free(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
) {
    jni_call(env, (), |_env| {
        if handle != 0 {
            unsafe { drop(Box::from_raw(handle as *mut FrontendDigest)) };
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_com_gresearch_tdigest_TDigestNative_mergeArrayF64(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    values: JDoubleArray,
) {
    jni_call(env, (), |env| {
        if let Err(msg) = ensure_non_null_handle(handle) {
            throw_illegal_arg(env, msg);
            return;
        }
        let xs = match read_jdouble_array(env, &values) {
            Ok(v) => v,
            Err(msg) => {
                throw_illegal_arg(env, msg);
                return;
            }
        };

        let d = unsafe { from_handle(handle) };
        if let Err(e) = d.add_values_f64(xs) {
            throw_illegal_arg(env, e);
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_com_gresearch_tdigest_TDigestNative_mergeArrayF32(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    values: JFloatArray,
) {
    jni_call(env, (), |env| {
        if let Err(msg) = ensure_non_null_handle(handle) {
            throw_illegal_arg(env, msg);
            return;
        }
        let xs32 = match read_jfloat_array(env, &values) {
            Ok(v) => v,
            Err(msg) => {
                throw_illegal_arg(env, msg);
                return;
            }
        };
        let xs: Vec<f64> = xs32.into_iter().map(|v| v as f64).collect();

        let d = unsafe { from_handle(handle) };
        if let Err(e) = d.add_values_f64(xs) {
            throw_illegal_arg(env, e);
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_com_gresearch_tdigest_TDigestNative_mergeWeightedF64(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    values: JDoubleArray,
    weights: JDoubleArray,
) {
    jni_call(env, (), |env| {
        if let Err(msg) = ensure_non_null_handle(handle) {
            throw_illegal_arg(env, msg);
            return;
        }
        let xs = match read_jdouble_array(env, &values) {
            Ok(v) => v,
            Err(msg) => {
                throw_illegal_arg(env, msg);
                return;
            }
        };
        let ws = match read_jdouble_array(env, &weights) {
            Ok(v) => v,
            Err(msg) => {
                throw_illegal_arg(env, msg);
                return;
            }
        };

        let d = unsafe { from_handle(handle) };
        if let Err(e) = d.add_weighted_f64(xs, ws) {
            throw_illegal_arg(env, e);
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_com_gresearch_tdigest_TDigestNative_mergeWeightedF32(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    values: JFloatArray,
    weights: JDoubleArray,
) {
    jni_call(env, (), |env| {
        if let Err(msg) = ensure_non_null_handle(handle) {
            throw_illegal_arg(env, msg);
            return;
        }
        let xs32 = match read_jfloat_array(env, &values) {
            Ok(v) => v,
            Err(msg) => {
                throw_illegal_arg(env, msg);
                return;
            }
        };
        let ws = match read_jdouble_array(env, &weights) {
            Ok(v) => v,
            Err(msg) => {
                throw_illegal_arg(env, msg);
                return;
            }
        };
        let xs: Vec<f64> = xs32.into_iter().map(|v| v as f64).collect();

        let d = unsafe { from_handle(handle) };
        if let Err(e) = d.add_weighted_f64(xs, ws) {
            throw_illegal_arg(env, e);
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_com_gresearch_tdigest_TDigestNative_mergeDigest(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    other_handle: jlong,
) {
    jni_call(env, (), |env| {
        if let Err(msg) = ensure_non_null_handle(handle) {
            throw_illegal_arg(env, msg);
            return;
        }
        if let Err(msg) = ensure_non_null_handle(other_handle) {
            throw_illegal_arg(env, msg);
            return;
        }

        let other_owned = unsafe { from_handle_const(other_handle).clone() };
        let d = unsafe { from_handle(handle) };
        if let Err(e) = d.merge_in_place(&other_owned) {
            throw_illegal_arg(env, e);
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_com_gresearch_tdigest_TDigestNative_scaleWeights(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    factor: jdouble,
) {
    jni_call(env, (), |env| {
        if let Err(msg) = ensure_non_null_handle(handle) {
            throw_illegal_arg(env, msg);
            return;
        }
        let d = unsafe { from_handle(handle) };
        if let Err(e) = d.scale_weights(factor) {
            throw_illegal_arg(env, e);
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_com_gresearch_tdigest_TDigestNative_scaleValues(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    factor: jdouble,
) {
    jni_call(env, (), |env| {
        if let Err(msg) = ensure_non_null_handle(handle) {
            throw_illegal_arg(env, msg);
            return;
        }
        let d = unsafe { from_handle(handle) };
        if let Err(e) = d.scale_values(factor) {
            throw_illegal_arg(env, e);
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_com_gresearch_tdigest_TDigestNative_castPrecision(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    f32mode: jboolean,
) -> jlong {
    jni_call(env, 0, |env| {
        if let Err(msg) = ensure_non_null_handle(handle) {
            throw_illegal_arg(env, msg);
            return 0;
        }
        let target = if f32mode != 0 {
            DigestPrecision::F32
        } else {
            DigestPrecision::F64
        };
        let d = unsafe { from_handle_const(handle) };
        match d.cast_precision(target) {
            Ok(casted) => into_handle(casted),
            Err(error) => {
                throw_illegal_arg(env, error);
                0
            }
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_com_gresearch_tdigest_TDigestNative_cdf(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    values: JDoubleArray,
) -> jdoubleArray {
    jni_call(env, std::ptr::null_mut(), |env| {
        let n = env.get_array_length(&values).unwrap_or(0) as usize;
        let out = env.new_double_array(n as jint).expect("new_double_array");
        if handle == 0 || n == 0 {
            return out.into_raw();
        }
        let d = unsafe { from_handle(handle) };

        let mut in_buf = vec![0f64; n];
        if let Err(e) = env.get_double_array_region(&values, 0, &mut in_buf) {
            eprintln!("jni: get_double_array_region failed: {e:?}");
            return out.into_raw();
        }

        let ps = d.cdf(&in_buf);
        env.set_double_array_region(&out, 0, &ps)
            .expect("set_double_array_region");
        out.into_raw()
    })
}

#[no_mangle]
pub extern "system" fn Java_com_gresearch_tdigest_TDigestNative_quantile(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
    q: jdouble,
) -> jdouble {
    jni_call(env, f64::NAN, |env| {
        if handle == 0 {
            return f64::NAN;
        }
        let d = unsafe { from_handle_const(handle) };
        match d.quantile_strict(q) {
            Ok(v) => v,
            Err(msg) => {
                throw_illegal_arg(env, msg);
                f64::NAN
            }
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_com_gresearch_tdigest_TDigestNative_median(
    env: JNIEnv,
    _cls: JClass,
    handle: jlong,
) -> jdouble {
    jni_call(env, f64::NAN, |_env| {
        if handle == 0 {
            return f64::NAN;
        }
        let d = unsafe { from_handle_const(handle) };
        d.quantile_strict(0.5).unwrap_or(f64::NAN)
    })
}

#[cfg(test)]
mod boundary_tests {
    #[test]
    fn catches_native_panics_without_unwinding_across_jni() {
        assert_eq!(super::catch_native(|| 42).unwrap(), 42);
        assert!(super::catch_native(|| panic!("controlled boundary test"))
            .unwrap_err()
            .contains("controlled boundary test"));
    }
}
