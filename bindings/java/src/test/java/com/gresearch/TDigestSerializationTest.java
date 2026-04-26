package com.gresearch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.gresearch.tdigest.TDigest;
import com.gresearch.tdigest.TDigest.Precision;
import com.gresearch.tdigest.TDigest.Scale;
import com.gresearch.tdigest.TDigest.SingletonPolicy;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.NotSerializableException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import org.junit.jupiter.api.Test;

class TDigestSerializationTest {
  @Test
  void javaObjectSerializationRoundTripPreservesExactWireBytes() throws Exception {
    TDigest restored;
    byte[] originalBytes;
    try (TDigest d = TDigest.builder()
        .maxSize(256)
        .scale(Scale.K2)
        .singletonPolicy(SingletonPolicy.USE)
        .precision(Precision.F64)
        .build(new double[] {0.0, 1.0, 2.0, 3.0})) {
      d.add(new double[] {10.0, 11.0, 12.0});
      originalBytes = d.toBytes();

      ByteArrayOutputStream bout = new ByteArrayOutputStream();
      try (ObjectOutputStream oos = new ObjectOutputStream(bout)) {
        oos.writeObject(d);
      }
      try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(bout.toByteArray()))) {
        restored = (TDigest) ois.readObject();
      }
    }

    try (TDigest d2 = restored) {
      assertArrayEquals(originalBytes, d2.toBytes());
      double q50 = d2.quantile(0.5);
      assertTrue(Double.isFinite(q50));
      d2.add(13.0);
      assertTrue(d2.quantile(0.5) >= q50);
    }
  }

  @Test
  void streamReadWriteRoundTrip() throws Exception {
    try (TDigest d = TDigest.builder()
        .maxSize(256)
        .scale(Scale.K2)
        .singletonPolicy(SingletonPolicy.USE)
        .precision(Precision.F32)
        .build(new float[] {0.0f, 1.0f, 2.0f, 3.0f})) {
      d.add(new float[] {4.0f, 5.0f, 6.0f});
      double before = d.quantile(0.5);
      byte[] expectedWire = d.toBytes();

      ByteArrayOutputStream bout = new ByteArrayOutputStream();
      d.writeTo(bout);
      byte[] blob = bout.toByteArray();

      try (TDigest restored = TDigest.readFrom(new ByteArrayInputStream(blob))) {
        double after = restored.quantile(0.5);
        assertEquals(before, after, 1e-6);
        assertArrayEquals(expectedWire, restored.toBytes());
      }
    }
  }

  @Test
  void streamReadWriteSupportsMultipleDigestsInSequence() throws Exception {
    byte[] aBytes;
    byte[] bBytes;
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (TDigest a = TDigest.builder().precision(Precision.F64).build(new double[] {1.0, 2.0, 3.0});
         TDigest b = TDigest.builder().precision(Precision.F32).build(new float[] {10.0f, 20.0f, 30.0f})) {
      a.add(new double[] {4.0, 5.0});
      b.add(new float[] {40.0f, 50.0f});
      aBytes = a.toBytes();
      bBytes = b.toBytes();
      a.writeTo(out);
      b.writeTo(out);
    }

    ByteArrayInputStream in = new ByteArrayInputStream(out.toByteArray());
    try (TDigest a2 = TDigest.readFrom(in);
         TDigest b2 = TDigest.readFrom(in)) {
      assertArrayEquals(aBytes, a2.toBytes());
      assertArrayEquals(bBytes, b2.toBytes());
    }
    assertEquals(-1, in.read());
  }

  @Test
  void serializingClosedDigestFails() throws Exception {
    byte[] bytes;
    TDigest d = TDigest.builder().precision(Precision.F64).build(new double[] {1.0, 2.0, 3.0});
    d.close();
    try (ByteArrayOutputStream bout = new ByteArrayOutputStream();
         ObjectOutputStream oos = new ObjectOutputStream(bout)) {
      assertThrows(NotSerializableException.class, () -> oos.writeObject(d));
      bytes = bout.toByteArray();
    }
    assertTrue(bytes.length > 0);
  }

  @Test
  void explicitWireVersionsRoundTrip() throws Exception {
    try (TDigest d = TDigest.builder()
        .maxSize(256)
        .scale(Scale.K2)
        .singletonPolicy(SingletonPolicy.USE)
        .precision(Precision.F64)
        .build(new double[] {0.0, 1.0, 2.0, 3.0})) {
      d.addWeighted(new double[] {10.0, 20.0}, new double[] {2.0, 3.0});
      double q0 = d.quantile(0.5);

      for (int version : new int[] {1, 2, 3}) {
        byte[] blob = d.toBytes(version);
        try (TDigest rt = TDigest.fromBytes(blob)) {
          double qRt = rt.quantile(0.5);
          assertTrue(Double.isFinite(qRt));
          if (version >= 2) {
            assertEquals(q0, qRt, 1e-4);
          }
        }
      }
    }
  }
}
