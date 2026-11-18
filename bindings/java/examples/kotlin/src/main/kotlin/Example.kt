package demo
import com.gresearch.tdigest.TDigest

fun main() {
    TDigest.builder().buildWeighted(doubleArrayOf(0.0, 10.0), doubleArrayOf(9.0, 1.0)).use { a ->
        check(a.median() == 0.0)
        TDigest.mergeAll(arrayOf(a)).use { merged ->
            check(merged.median() == a.median())
            check(merged !== a)
        }
        println(TDigest.runtimeInfo())
        println("KOTLIN_CONSUMER_OK")
    }
}
