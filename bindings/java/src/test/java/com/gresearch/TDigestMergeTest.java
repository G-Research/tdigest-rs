package com.gresearch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.gresearch.tdigest.TDigest;
import com.gresearch.tdigest.TDigest.Precision;
import com.gresearch.tdigest.TDigest.Scale;
import com.gresearch.tdigest.TDigest.SingletonPolicy;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

class TDigestMergeTest {
  @Test
  void addDoubleArrayAndScalarRaisesMedianForLargerValues() {
    try (TDigest d = TDigest.builder()
        .maxSize(256)
        .scale(Scale.K2)
        .singletonPolicy(SingletonPolicy.USE)
        .precision(Precision.F64)
        .build(new double[] {0.0, 1.0, 2.0, 3.0})) {
      double before = d.quantile(0.5);
      d.add(new double[] {10.0, 11.0, 12.0}).add(13.0);
      double after = d.quantile(0.5);
      assertTrue(after > before, "expected median to increase after adding larger values");
    }
  }

  @Test
  void addFloatArrayAndScalarRaisesMedianForLargerValues() {
    try (TDigest d = TDigest.builder()
        .maxSize(256)
        .scale(Scale.K2)
        .singletonPolicy(SingletonPolicy.USE)
        .precision(Precision.F32)
        .build(new float[] {0.0f, 1.0f, 2.0f, 3.0f})) {
      double before = d.quantile(0.5);
      d.add(new float[] {10.0f, 11.0f, 12.0f}).add(13.0f);
      double after = d.quantile(0.5);
      assertTrue(after > before, "expected median to increase after adding larger float values");
    }
  }

  @Test
  void mergeRejectsMixedPrecision() {
    try (TDigest f64 = TDigest.builder().precision(Precision.F64).build(new double[] {1.0, 2.0, 3.0});
         TDigest f32 = TDigest.builder().precision(Precision.F32).build(new float[] {1.0f, 2.0f, 3.0f})) {
      assertThrows(IllegalArgumentException.class, () -> f64.merge(f32));
      assertThrows(IllegalArgumentException.class, () -> TDigest.mergeAll(f64, f32));
    }
  }

  @Test
  void mergeAllIterableMatchesVarargs() {
    try (TDigest a = TDigest.builder().precision(Precision.F64).build(new double[] {0.0, 1.0, 2.0});
         TDigest b = TDigest.builder().precision(Precision.F64).build(new double[] {10.0, 11.0, 12.0});
         TDigest c = TDigest.builder().precision(Precision.F64).build(new double[] {20.0, 21.0, 22.0});
         TDigest byVarargs = TDigest.mergeAll(a, b, c);
         TDigest byIterable = TDigest.mergeAll(Arrays.asList(a, b, c))) {
      assertEquals(byVarargs.quantile(0.5), byIterable.quantile(0.5), 1e-12);
    }
  }

  @Test
  void scaleWeightsAndValuesSmoke() {
    try (TDigest d = TDigest.builder()
        .maxSize(256)
        .scale(Scale.K2)
        .singletonPolicy(SingletonPolicy.USE)
        .precision(Precision.F64)
        .build(new double[] {0.0, 1.0, 2.0, 3.0})) {
      double q0 = d.quantile(0.5);
      double c0 = d.cdf(new double[] {1.5})[0];

      d.scaleWeights(2.0);
      assertEquals(q0, d.quantile(0.5), 1e-12);
      assertEquals(c0, d.cdf(new double[] {1.5})[0], 1e-12);

      d.scaleValues(3.0);
      assertEquals(q0 * 3.0, d.quantile(0.5), 1e-12);
      assertEquals(q0 * 3.0, d.median(), 1e-12);
      assertEquals(c0, d.cdf(new double[] {1.5 * 3.0})[0], 1e-12);
    }
  }

  @Test
  void scaleRejectsInvalidFactor() {
    try (TDigest d = TDigest.builder()
        .maxSize(64)
        .scale(Scale.K2)
        .singletonPolicy(SingletonPolicy.USE)
        .precision(Precision.F64)
        .build(new double[] {0.0, 1.0, 2.0})) {
      assertThrows(IllegalArgumentException.class, () -> d.scaleWeights(0.0));
      assertThrows(IllegalArgumentException.class, () -> d.scaleWeights(-1.0));
      assertThrows(IllegalArgumentException.class, () -> d.scaleWeights(Double.NaN));
      assertThrows(IllegalArgumentException.class, () -> d.scaleWeights(Double.POSITIVE_INFINITY));

      assertThrows(IllegalArgumentException.class, () -> d.scaleValues(0.0));
      assertThrows(IllegalArgumentException.class, () -> d.scaleValues(-1.0));
      assertThrows(IllegalArgumentException.class, () -> d.scaleValues(Double.NaN));
      assertThrows(IllegalArgumentException.class, () -> d.scaleValues(Double.POSITIVE_INFINITY));
    }
  }

  @Test
  void addWeightedAndCastPrecisionSmoke() {
    try (TDigest d = TDigest.builder()
        .maxSize(128)
        .scale(Scale.K2)
        .singletonPolicy(SingletonPolicy.USE)
        .precision(Precision.F64)
        .build(new double[] {0.0, 1.0, 2.0})) {
      d.addWeighted(new double[] {10.0, 20.0}, new double[] {2.0, 3.0});
      assertTrue(Double.isFinite(d.quantile(0.5)));

      try (TDigest d32 = d.castPrecision(Precision.F32);
           TDigest d64 = d32.castPrecision(Precision.F64)) {
        assertTrue(Double.isFinite(d32.quantile(0.5)));
        assertEquals(d.quantile(0.5), d64.quantile(0.5), 1e-4);
      }
    }
  }

  @Test
  void addWeightedRejectsInvalidInputsAndCastRejectsAuto() {
    try (TDigest d = TDigest.builder()
        .maxSize(128)
        .scale(Scale.K2)
        .singletonPolicy(SingletonPolicy.USE)
        .precision(Precision.F64)
        .build(new double[] {0.0, 1.0, 2.0})) {
      assertThrows(IllegalArgumentException.class, () -> d.addWeighted(new double[] {1.0, 2.0}, new double[] {1.0}));
      assertThrows(IllegalArgumentException.class, () -> d.addWeighted(new double[] {1.0}, new double[] {0.0}));
      assertThrows(IllegalArgumentException.class, () -> d.castPrecision(Precision.AUTO));
    }
  }
}
