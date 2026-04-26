package com.gresearch.tdigest;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.NotSerializableException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.lang.ref.Cleaner;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Locale;
import java.util.Objects;

/**
 * Java wrapper for the native Rust TDigest.
 *
 * Queries:
 *   - cdf(double[] x) -> double[]
 *   - quantile(double q) -> double                 // strict: q ∈ [0,1]
 *   - quantile(double[] qs) -> double[]            // strict: all q ∈ [0,1]
 *   - median() -> double
 *
 * Owns a native handle; frees it via Cleaner or explicit close().
 */
public final class TDigest implements AutoCloseable, Serializable {
  private static final long serialVersionUID = 1L;

  /* ======================= Type-safe enums for builder ======================= */

  public enum Scale {
    QUAD, K1, K2, K3;
    String asWire() { return name().toLowerCase(Locale.ROOT); }
  }

  public enum SingletonPolicy {
    OFF, USE, USE_WITH_PROTECTED_EDGES;

    int toNativeCode() {
      switch (this) {
        case OFF:  return 0;
        case USE:  return 1;
        case USE_WITH_PROTECTED_EDGES: return 2;
        default:   return 1;
      }
    }
  }

  public enum Precision {
    F64, F32, AUTO
  }

  /* ======================= Backing native ======================= */

  private static String norm(String s) {
    if (s == null) throw new NullPointerException();
    return s.trim().toLowerCase(Locale.ROOT);
  }

  private static String parseScale(String s) {
    if (s == null) return "k2";
    String v = norm(s);
    switch (v) {
      case "quad":
      case "k1":
      case "k2":
      case "k3":
        return v;
      default:
        throw new IllegalArgumentException(
            "invalid scale: " + s + " (expected 'quad', 'k1', 'k2', or 'k3')");
    }
  }

  private static int parsePolicyCode(String kind) {
    if (kind == null) return 1; // default "use"
    String v = norm(kind);
    switch (v) {
      case "off":   return 0;
      case "use":   return 1;
      case "edges":
      case "usewithprotectededges":
        return 2;
      default:
        throw new IllegalArgumentException(
            "invalid singleton_policy: " + kind + " (expected 'off', 'use', or 'edges')");
    }
  }

  private static int coalesceEdges(Integer edges) {
    return edges == null ? 3 : Math.max(0, edges);
  }

  private static int policyCode(SingletonPolicy p) {
    switch (p) {
      case OFF:  return 0;
      case USE:  return 1;
      case USE_WITH_PROTECTED_EDGES:return 2;
      default:   return 1;
    }
  }

  /* ======================= Cleaner & state ======================= */

  private static final Cleaner CLEANER = Cleaner.create();

  private static final class State implements Runnable {
    private long handle;
    State(long handle) { this.handle = handle; }

    @Override
    public void run() {
      if (handle != 0) {
        try {
          TDigestNative.free(handle);
        } finally {
          handle = 0;
        }
      }
    }
  }

  private transient State state;
  private transient Cleaner.Cleanable cleanable;

  private TDigest(long handle) {
    if (handle == 0) {
      throw new IllegalStateException("Failed to create TDigest (null native handle)");
    }
    this.state = new State(handle);
    this.cleanable = CLEANER.register(this, state);
  }

  /* ======================= Builder (fluent) ======================= */

  public static final class Builder {
    private int maxSize = 1000;
    private Scale scale = Scale.K2;
    private SingletonPolicy policy = SingletonPolicy.USE;
    private int edgesPerSide = 3;
    private Precision precision = Precision.F64; // keep default as-is to avoid surprises

    public Builder maxSize(int n) {
      if (n <= 0) throw new IllegalArgumentException("maxSize must be > 0");
      this.maxSize = n;
      return this;
    }

    public Builder scale(Scale s) {
      this.scale = Objects.requireNonNull(s, "scale");
      return this;
    }

    public Builder singletonPolicy(SingletonPolicy p) {
      this.policy = Objects.requireNonNull(p, "policy");
      return this;
    }

    public Builder precision(Precision p) {
      this.precision = Objects.requireNonNull(p, "precision");
      return this;
    }

    public Builder keep(int k) { this.edgesPerSide = Math.max(0, k); return this; }
    public Builder edgesPerSide(int k) { this.edgesPerSide = Math.max(0, k); return this; }

    public TDigest build(double[] values) {
      Objects.requireNonNull(values, "values");
      final boolean f32 =
          (precision == Precision.F32) ? true :
          (precision == Precision.AUTO) ? false /* auto: double[] → f64 */ :
          false;
      final int edges = (policy == SingletonPolicy.USE_WITH_PROTECTED_EDGES) ? edgesPerSide : 0;
      long h = TDigestNative.fromArray(
          values, maxSize, scale.asWire(), policy.toNativeCode(), edges, f32);
      return new TDigest(h);
    }

    public TDigest build(float[] values) {
      Objects.requireNonNull(values, "values");
      final boolean f32 =
          (precision == Precision.F32) ? true :
          (precision == Precision.AUTO) ? true /* auto: float[] → f32 */ :
          false;
      final int edges = (policy == SingletonPolicy.USE_WITH_PROTECTED_EDGES) ? edgesPerSide : 0;
      long h = TDigestNative.fromArray(
          values, maxSize, scale.asWire(), policy.toNativeCode(), edges, f32);
      return new TDigest(h);
    }
  }

  public static Builder builder() { return new Builder(); }

  /* ======================= Construction (stringly-typed) ======================= */

  public static TDigest fromArray(
      double[] values, Integer max_size, String scale, Boolean f32_mode, String singleton_policy, Integer edges
  ) {
    Objects.requireNonNull(values, "values");
    final int maxSize = (max_size == null) ? 1000 : max_size;
    if (maxSize <= 0) throw new IllegalArgumentException("max_size must be > 0");
    final String sc = parseScale(scale);
    final int policyCode = parsePolicyCode(singleton_policy);
    final int keep = coalesceEdges(edges);
    final boolean f32 = (f32_mode != null) && f32_mode;
    long h = TDigestNative.fromArray(values, maxSize, sc, policyCode, keep, f32);
    return new TDigest(h);
  }

  public static TDigest fromArray(
      float[] values, Integer max_size, String scale, Boolean f32_mode, String singleton_policy, Integer edges
  ) {
    Objects.requireNonNull(values, "values");
    final int maxSize = (max_size == null) ? 1000 : max_size;
    if (maxSize <= 0) throw new IllegalArgumentException("max_size must be > 0");
    final String sc = parseScale(scale);
    final int policyCode = parsePolicyCode(singleton_policy);
    final int keep = coalesceEdges(edges);
    final boolean f32 = (f32_mode != null) && f32_mode;
    long h = TDigestNative.fromArray(values, maxSize, sc, policyCode, keep, f32);
    return new TDigest(h);
  }

  public static TDigest fromBytes(byte[] bytes) {
    Objects.requireNonNull(bytes, "bytes");
    long h = TDigestNative.fromBytes(bytes);
    return new TDigest(h);
  }

  public static TDigest create(int maxSize, String scale) {
    return TDigest.fromArray(new double[0], maxSize, scale, false, "use", 3);
  }

  public static TDigest fromValues(double[] values, int maxSize, String scale) {
    return TDigest.fromArray(values, maxSize, scale, false, "use", 3);
  }

  /* ======================= Queries ======================= */

  /**
   * CDF evaluated at array-like x (returns double[]).
   * Semantics are implemented in Rust and shared across all frontends.
   */
  public double[] cdf(double[] values) {
    ensureOpen();
    Objects.requireNonNull(values, "values");
    return TDigestNative.cdf(state.handle, values);
  }

  /** Vector quantile (strict): throws if any q is NaN/±inf or outside [0,1]. */
  public double[] quantile(double[] qs) {
    ensureOpen();
    Objects.requireNonNull(qs, "qs");
    double[] out = new double[qs.length];
    for (int i = 0; i < qs.length; i++) {
      out[i] = TDigestNative.quantile(state.handle, qs[i]);
    }
    return out;
  }

  /** Scalar quantile (strict): q must be finite and within [0,1]. */
  public double quantile(double q) {
    ensureOpen();
    return TDigestNative.quantile(state.handle, q);
  }

  public double median() {
    ensureOpen();
    return TDigestNative.median(state.handle);
  }

  public TDigest add(double[] values) {
    ensureOpen();
    Objects.requireNonNull(values, "values");
    TDigestNative.mergeArrayF64(state.handle, values);
    return this;
  }

  public TDigest add(float[] values) {
    ensureOpen();
    Objects.requireNonNull(values, "values");
    TDigestNative.mergeArrayF32(state.handle, values);
    return this;
  }

  public TDigest add(double value) {
    return add(new double[] { value });
  }

  public TDigest add(float value) {
    return add(new float[] { value });
  }

  public TDigest addWeighted(double[] values, double[] weights) {
    ensureOpen();
    Objects.requireNonNull(values, "values");
    Objects.requireNonNull(weights, "weights");
    TDigestNative.mergeWeightedF64(state.handle, values, weights);
    return this;
  }

  public TDigest addWeighted(float[] values, double[] weights) {
    ensureOpen();
    Objects.requireNonNull(values, "values");
    Objects.requireNonNull(weights, "weights");
    TDigestNative.mergeWeightedF32(state.handle, values, weights);
    return this;
  }

  public TDigest addWeighted(double value, double weight) {
    return addWeighted(new double[] { value }, new double[] { weight });
  }

  public TDigest addWeighted(float value, double weight) {
    return addWeighted(new float[] { value }, new double[] { weight });
  }

  public TDigest merge(TDigest other) {
    ensureOpen();
    Objects.requireNonNull(other, "other");
    other.ensureOpen();
    TDigestNative.mergeDigest(state.handle, other.state.handle);
    return this;
  }

  /** Scale centroid weights by a positive finite factor. */
  public TDigest scaleWeights(double factor) {
    ensureOpen();
    TDigestNative.scaleWeights(state.handle, factor);
    return this;
  }

  /** Scale centroid means/value-axis by a positive finite factor. */
  public TDigest scaleValues(double factor) {
    ensureOpen();
    TDigestNative.scaleValues(state.handle, factor);
    return this;
  }

  public TDigest castPrecision(Precision target) {
    ensureOpen();
    Objects.requireNonNull(target, "target");
    if (target == Precision.AUTO) {
      throw new IllegalArgumentException("castPrecision target must be F32 or F64");
    }
    long h = TDigestNative.castPrecision(state.handle, target == Precision.F32);
    return new TDigest(h);
  }

  public static TDigest mergeAll(TDigest first, TDigest second, TDigest... rest) {
    Objects.requireNonNull(first, "first");
    Objects.requireNonNull(second, "second");
    ArrayList<TDigest> all = new ArrayList<>();
    all.add(first);
    all.add(second);
    if (rest != null) {
      all.addAll(Arrays.asList(rest));
    }
    return mergeAll(all);
  }

  public static TDigest mergeAll(Iterable<TDigest> digests) {
    Objects.requireNonNull(digests, "digests");
    Iterator<TDigest> it = digests.iterator();
    if (!it.hasNext()) {
      return TDigest.builder()
          .maxSize(1000)
          .scale(Scale.K2)
          .singletonPolicy(SingletonPolicy.USE)
          .precision(Precision.F64)
          .build(new double[] {});
    }

    TDigest first = Objects.requireNonNull(it.next(), "mergeAll contains null digest");
    TDigest acc = TDigest.fromBytes(first.toBytes());
    try {
      while (it.hasNext()) {
        TDigest d = Objects.requireNonNull(it.next(), "mergeAll contains null digest");
        acc.merge(d);
      }
      return acc;
    } catch (RuntimeException e) {
      acc.close();
      throw e;
    }
  }

  /* ======================= Serialization & lifecycle ======================= */

  public byte[] toBytes() {
    ensureOpen();
    return TDigestNative.toBytes(state.handle);
  }

  public byte[] toBytes(int version) {
    ensureOpen();
    if (version < 1 || version > 3) {
      throw new IllegalArgumentException("version must be in {1,2,3}");
    }
    return TDigestNative.toBytesVersion(state.handle, version);
  }

  public void writeTo(OutputStream out) throws IOException {
    Objects.requireNonNull(out, "out");
    byte[] bytes = toBytes();
    DataOutputStream dout = (out instanceof DataOutputStream)
        ? (DataOutputStream) out
        : new DataOutputStream(out);
    dout.writeInt(bytes.length);
    dout.write(bytes);
  }

  public static TDigest readFrom(InputStream in) throws IOException {
    Objects.requireNonNull(in, "in");
    DataInputStream din = (in instanceof DataInputStream)
        ? (DataInputStream) in
        : new DataInputStream(in);
    final int n;
    try {
      n = din.readInt();
    } catch (EOFException e) {
      throw new EOFException("tdigest: missing byte length prefix");
    }
    if (n < 0) {
      throw new IOException("tdigest: negative byte length: " + n);
    }
    byte[] bytes = new byte[n];
    din.readFully(bytes);
    return fromBytes(bytes);
  }

  private void writeObject(ObjectOutputStream out) throws IOException {
    if (isClosed()) {
      throw new NotSerializableException("TDigest is closed");
    }
    writeTo(out);
  }

  private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
    TDigest restored = readFrom(in);
    this.state = restored.state;
    this.cleanable = restored.cleanable;
    restored.state = null;
    restored.cleanable = null;
  }

  private void readObjectNoData() throws IOException {
    throw new IOException("tdigest: missing serialized data");
  }

  @Override
  public void close() {
    if (state != null && state.handle != 0) {
      if (cleanable != null) {
        cleanable.clean();
      } else {
        TDigestNative.free(state.handle);
        state.handle = 0;
      }
      state = null;
      cleanable = null;
    }
  }

  public boolean isClosed() {
    return state == null || state.handle == 0;
  }

  private void ensureOpen() {
    if (isClosed()) {
      throw new IllegalStateException("TDigest is closed");
    }
  }

  @Override
  public String toString() {
    return "TDigest(handle=" + (state == null ? 0 : state.handle) + ")";
  }
}
