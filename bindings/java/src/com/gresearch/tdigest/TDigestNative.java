package com.gresearch.tdigest;

/** Native entry points for the Rust digest. See {@link Natives} for library loading. */
final class TDigestNative {
  static {
    Natives.load();

    // A null handle resolves the exported symbol without allocating a digest.
    try {
      free(0L);
    } catch (UnsatisfiedLinkError e) {
      throw new UnsatisfiedLinkError(
          "Failed to bind tdigest_rs natives. Ensure JNI_OnLoad is exported and the library was "
          + "loaded with native access on JDK 22+. Original: " + e.getMessage());
    }
  }

  static native long     fromBytes(byte[] bytes);
  static native byte[]   toBytes(long handle);
  static native byte[]   toBytesVersion(long handle, int version);
  static native long     fromArray(Object values, int maxSize, String scale, int policyCode, int edges, boolean f32mode, double delta);
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
