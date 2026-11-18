package com.gresearch;

import com.gresearch.tdigest.TDigest;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class TDigestConsumerTest {
  private static double mass(TDigest d) {
    return ByteBuffer.wrap(d.toBytes()).order(ByteOrder.LITTLE_ENDIAN).getDouble(20);
  }

  @Test void arrayMergeRetainsMassAndOwnsIndependentState() {
    try (TDigest a = TDigest.builder().build(new double[] {0});
         TDigest b = TDigest.builder().build(new double[] {})) {
      b.addWeighted(10.0, 100.0);
      byte[] before = a.toBytes();
      try (TDigest result = TDigest.mergeAll(new TDigest[] {a, b})) {
        assertEquals(101.0, mass(result));
        assertEquals(10.0, result.median());
        result.add(20.0);
        assertArrayEquals(before, a.toBytes());
        assertEquals(100.0, mass(b));
      }
    }
  }

  @Test void weightedMutationsAreObservableAndFailureAtomic() {
    try (TDigest d = TDigest.builder().build(new double[] {})) {
      assertSame(d, d.addWeighted(new double[] {0, 10}, new double[] {9, 1}));
      assertEquals(0.0, d.median());
      assertArrayEquals(new double[] {.45, .95}, d.cdf(new double[] {0, 10}));
      byte[] before = d.toBytes();
      assertThrows(IllegalArgumentException.class, () -> d.addWeighted(new double[] {1, 2}, new double[] {1, -1}));
      assertArrayEquals(before, d.toBytes());
      assertThrows(IllegalArgumentException.class, () -> d.scaleValues(Double.MAX_VALUE));
      assertArrayEquals(before, d.toBytes());
      d.scaleWeights(.5);
      assertEquals(5.0, mass(d));
      assertEquals(0.0, d.median());
    }
  }

  @Test void smallPrecisionMeansKeepDoubleMassAndDeltaIsModernConfiguration() {
    try (TDigest d = TDigest.builder().precision(TDigest.Precision.F32).build(new float[] {})) {
      d.addWeighted(1.0f, 16777216.0).addWeighted(1.0f, 1.0);
      assertEquals(16777217.0, mass(d));
    }
    try (TDigest d = TDigest.builder().delta(300).build(new double[] {0, 1, 2})) {
      assertEquals(1.0, d.median());
      try (TDigest restored = TDigest.fromBytes(d.toBytes())) {
        assertArrayEquals(d.toBytes(), restored.toBytes());
      }
    }
    assertThrows(IllegalArgumentException.class, () -> TDigest.builder().delta(300).scale(TDigest.Scale.K2).build(new double[] {1}));
  }

  @Test void incompatibleEmptyConfigurationIsRejectedWithoutChangingReceiver() {
    try (TDigest d = TDigest.builder().maxSize(200).build(new double[] {1});
         TDigest empty = TDigest.builder().maxSize(1000).build(new double[] {})) {
      byte[] before = d.toBytes();
      assertThrows(IllegalArgumentException.class, () -> d.merge(empty));
      assertArrayEquals(before, d.toBytes());
    }
  }

  @Test void multipleSerializedDigestsSurviveOriginalCloseAndCollection() throws Exception {
    byte[] serialized;
    try (TDigest a = TDigest.builder().build(new double[] {1});
         TDigest b = TDigest.builder().build(new double[] {2})) {
      ByteArrayOutputStream bytes = new ByteArrayOutputStream();
      try (ObjectOutputStream output = new ObjectOutputStream(bytes)) {
        output.writeObject(new TDigest[] {a, b});
      }
      serialized = bytes.toByteArray();
    }
    TDigest[] restored;
    try (ObjectInputStream input = new ObjectInputStream(new ByteArrayInputStream(serialized))) {
      restored = (TDigest[]) input.readObject();
    }
    System.gc();
    assertEquals(1.0, restored[0].median());
    assertEquals(2.0, restored[1].median());
    restored[0].close();
    restored[0].close();
    assertThrows(IllegalStateException.class, () -> restored[0].median());
    assertEquals(2.0, restored[1].median());
    restored[1].close();
  }

  @Test void exactTinyPointSurvivesBytesAndScaling() {
    try (TDigest d = TDigest.builder().build(new double[] {})) {
      d.addWeighted(Double.MIN_VALUE, .1);
      assertEquals(Double.MIN_VALUE, d.median());
      assertEquals(.5, d.cdf(new double[] {Double.MIN_VALUE})[0]);
      try (TDigest restored = TDigest.fromBytes(d.toBytes())) {
        restored.scaleWeights(10.0);
        assertEquals(1.0, mass(restored));
        assertEquals(Double.MIN_VALUE, restored.median());
      }
    }
  }

  @Test void f32TrainingRejectsUnrepresentableDoubleWithoutMutation() {
    assertThrows(IllegalArgumentException.class,
      () -> TDigest.builder().precision(TDigest.Precision.F32).build(new double[] {Double.MIN_VALUE}));
    try (TDigest d = TDigest.builder().precision(TDigest.Precision.F32).build(new double[] {1})) {
      byte[] before = d.toBytes();
      assertThrows(IllegalArgumentException.class, () -> d.add(new double[] {2, Double.MIN_VALUE}));
      assertArrayEquals(before, d.toBytes());
      assertThrows(IllegalArgumentException.class,
        () -> d.addWeighted(new double[] {2, Double.MIN_VALUE}, new double[] {1, 1}));
      assertArrayEquals(before, d.toBytes());
    }
  }
}
