package com.gresearch.tdigest.example

import com.gresearch.tdigest.TDigest
import org.apache.spark.{SparkConf, SparkContext, SparkEnv, HashPartitioner}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.serializer.KryoRegistrator
import com.esotericsoftware.kryo.{Kryo, Serializer}
import com.esotericsoftware.kryo.io.{Input, Output}

final class TDigestStateSerializer extends Serializer[TDigest] {
  override def write(kryo: Kryo, out: Output, digest: TDigest): Unit = {
    val bytes = digest.toBytes()
    out.writeInt(bytes.length, true)
    out.writeBytes(bytes)
  }
  override def read(kryo: Kryo, in: Input, kind: Class[TDigest]): TDigest = {
    val size = in.readInt(true)
    require(size >= 0 && size <= 64 * 1024 * 1024, "invalid digest byte length")
    TDigest.fromBytes(in.readBytes(size))
  }
}

final class TDigestStateRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo): Unit = {
    kryo.register(classOf[TDigest], new TDigestStateSerializer)
  }
}

object TDigestSparkCheck {
  private def mass(d: TDigest): Double = {
    val bytes = d.toBytes()
    require(bytes(4) == 3, "qualification expects the current TDIG v3 format")
    java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN).getDouble(20)
  }
  def main(args: Array[String]): Unit = {
    require(args.length == 3, "usage: checkpoint-directory java|kryo expected-native-sha256")
    val conf = new SparkConf().setAppName("tdigest-rs-public-cluster-check")
      .set("spark.ui.enabled", "false")
      .set("spark.executor.cores", "1")
      .set("spark.executor.memory", "512m")
      .set("spark.task.maxFailures", "1")
      .set("spark.rpc.netty.dispatcher.numThreads", "2")
    if (conf.get("spark.master", "").startsWith("local-cluster")) {
      conf.set("spark.driver.host", "127.0.0.1").set("spark.driver.bindAddress", "127.0.0.1")
    }
    if (args(1) == "kryo") {
      conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      conf.set("spark.kryo.registrator", classOf[TDigestStateRegistrator].getName)
    } else {
      require(args(1) == "java")
      conf.set("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
    }
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    try {
      sc.setCheckpointDir(args(0))
      val original = TDigest.builder().maxSize(1000).buildWeighted(Array(0.0, 10.0), Array(9.0, 1.0))
      val broadcast = sc.broadcast(original)
      original.close()
      val executors = sc.parallelize(0 until 16, 8).mapPartitions { values =>
        val d = broadcast.value
        require(mass(d) == 10.0 && d.median() == 0.0)
        require(d.cdf(Array(0.0))(0) == 0.45)
        val independent = TDigest.fromBytes(d.toBytes())
        try { independent.add(20.0); require(mass(independent) == 11.0 && mass(d) == 10.0) }
        finally independent.close()
        values.foreach(_ => Thread.sleep(10L))
        Iterator((SparkEnv.get.executorId, java.lang.management.ManagementFactory.getRuntimeMXBean.getName, TDigest.runtimeInfo()))
      }.collect().distinct
      require(executors.forall(_._3.contains("sha256=" + args(2))), "executor loaded a different native payload")
      require(executors.length >= 2 && executors.map(_._1).distinct.length >= 2, "need two real executor processes")
      val driver = java.lang.management.ManagementFactory.getRuntimeMXBean.getName
      require(executors.forall(_._2 != driver), "executor ran in the driver process")
      val built = sc.longAccumulator("public-digest-builds")
      val shards = sc.parallelize(0 until 40, 40).map { i =>
        built.add(1L)
        (i % 4, TDigest.builder().maxSize(1000).build(Array(i.toDouble)))
      }
      val merged = shards.groupByKey(new HashPartitioner(4)).mapValues { input =>
        val digests = input.toArray
        try TDigest.mergeAll(digests) finally digests.foreach(_.close())
      }.persist(StorageLevel.DISK_ONLY)
      merged.checkpoint()
      require(merged.count() == 4 && merged.isCheckpointed)
      val builds = built.value
      require(builds == 40L)
      merged.unpersist(blocking = true)
      val restored = merged.collect().sortBy(_._1)
      require(built.value == builds, "checkpoint read unexpectedly rebuilt source digests")
      try {
        restored.foreach { case (key, digest) =>
          require(mass(digest) == 10.0)
          require(digest.quantile(0.0) == key.toDouble && digest.quantile(1.0) == key + 36.0)
          require(digest.median() == key + 18.0)
        }
        val all = TDigest.mergeAll(restored.map(_._2))
        try { require(mass(all) == 40.0 && all.quantile(0.0) == 0.0 && all.quantile(1.0) == 39.0) }
        finally all.close()
      } finally restored.foreach(_._2.close())
      broadcast.destroy()
      println("TDIGEST_SPARK_OK spark=" + sc.version + " scala=" + util.Properties.versionNumberString + " serializer=" + args(1) + " driver=" + driver + " executorProcesses=" + executors.map(_._2).distinct.mkString(",") + " checkpoint=" + merged.getCheckpointFile.getOrElse("missing") + " runtime=" + TDigest.runtimeInfo())
    } finally sc.stop()
  }
}
