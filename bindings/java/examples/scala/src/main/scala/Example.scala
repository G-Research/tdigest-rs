import com.gresearch.tdigest.TDigest

object Example extends App {
  val a = TDigest.builder().maxSize(1000).buildWeighted(Array(0.0, 10.0), Array(9.0, 1.0))
  val b = TDigest.builder().maxSize(1000).build(Array(20.0))
  try {
    assert(a.median() == 0.0)
    val merged = TDigest.mergeAll(Array(a, b))
    try {
      val restored = TDigest.fromBytes(merged.toBytes())
      try { assert(restored.median() == merged.median()) }
      finally { restored.close() }
    } finally { merged.close() }
    println(TDigest.runtimeInfo())
    println("SCALA_CONSUMER_OK")
  } finally { a.close(); b.close() }
}
