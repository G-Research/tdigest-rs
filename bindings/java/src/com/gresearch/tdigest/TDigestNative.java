package com.gresearch.tdigest;

/**
 * JNI bridge for the native tdigest_rs library.
 *
 * Loading strategy:
 *  - Prefer explicit path via -Dcom.gresearch.tdigest.native=/abs/path/libtdigest_rs.so
 *  - Otherwise load from META-INF/native/<os>-<arch>/… inside the JAR
 *  - Finally, fall back to System.loadLibrary("tdigest_rs")
 *
 * Note: On JDK 22+, run with --enable-native-access=ALL-UNNAMED.
 */
final class TDigestNative {
  static {
    // Load the native library (triggers JNI_OnLoad in the .so)
    Natives.load();

    // Sanity check: resolve one native symbol without doing any work.
    // free(0) is a no-op in native code, but ensures RegisterNatives ran.
    try {
      free(0L);
    } catch (UnsatisfiedLinkError e) {
      throw new UnsatisfiedLinkError(
          "Failed to bind tdigest_rs natives. Ensure JNI_OnLoad is exported and the library was "
          + "loaded with native access on JDK 22+. Original: " + e.getMessage());
    }
  }

  // ---- Native methods (implemented in Rust via JNI) ----
  static native long     fromBytes(byte[] bytes);
  static native byte[]   toBytes(long handle);
  static native byte[]   toBytesVersion(long handle, int version);
  static native long     fromArray(Object values, int maxSize, String scale, int policyCode, int edges, boolean f32mode);
  static native void     mergeArrayF64(long handle, double[] values);
  static native void     mergeArrayF32(long handle, float[] values);
  static native void     mergeWeightedF64(long handle, double[] values, double[] weights);
  static native void     mergeWeightedF32(long handle, float[] values, double[] weights);
  static native void     mergeDigest(long handle, long otherHandle);
  static native void     scaleWeights(long handle, double factor);
  static native void     scaleValues(long handle, double factor);
  static native long     castPrecision(long handle, boolean f32mode);
  static native void     free(long handle);
  static native double[] cdf(long handle, double[] values);
  static native double   quantile(long handle, double q);
  static native double   median(long handle);

  private TDigestNative() {}
}
