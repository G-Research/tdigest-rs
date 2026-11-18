# Building and loading the Java binding

The JAR contains Java 11 classes and the native library for the build platform. The release jobs exercise JDK 11, 17, 21 and 25. With an installed JDK, use the Gradle wrapper from the repository root.

```sh
bindings/java/gradlew --no-daemon -p bindings/java clean test jar smoke verifyJava11Bytecode
```

The default wrapper uses Gradle 8.14.5. For JDK 25, use `TDIGEST_GRADLE_HOME` to select an installed Gradle 9.1.0 or set both `TDIGEST_GRADLE_URL` and `TDIGEST_GRADLE_SHA256`. The Gradle 9.1.0 binary distribution has SHA256 `a17ddd85a26b6a7f5ddb71ff8b05fc5104c0202c6e64782429790c933686c806`. On JDK 25, use `--enable-native-access=ALL-UNNAMED` when launching the application.

## Internal repositories and offline builds

The optional bootstrap accepts archive URLs or local archive paths and verifies their checksums. It requires Python 3.10 or later. Tar archives containing links also require the tarfile data filter. Prepare the tools and dependency caches while the configured repositories are reachable.

| Setting | Purpose |
| --- | --- |
| `JAVA_HOME` | Select an installed JDK. `TDIGEST_JDK_URL` with `TDIGEST_JDK_SHA256` selects a verified archive instead. |
| `TDIGEST_DOWNLOAD_USERNAME`, `TDIGEST_DOWNLOAD_PASSWORD` | Authenticate tool downloads. |
| `TDIGEST_MAVEN_REPOSITORY_URL` | Select the exclusive Maven dependency source. Use `TDIGEST_MAVEN_USERNAME` and `TDIGEST_MAVEN_PASSWORD` when it requires credentials. |
| `TDIGEST_TOOLCHAIN_CACHE`, `GRADLE_USER_HOME` | Set the tool installation cache and Gradle dependency cache. |
| `CARGO_HOME`, `CARGO_TARGET_DIR` | Set the Rust source cache and native build output. |

Run `make java-prepare` to stage tools. Run `make build-java java-test` once to populate dependencies, then use `TDIGEST_OFFLINE=1 make build-java java-test` with the same repository and Gradle settings. To publish to an internal Maven repository, configure the publishing destination in the Gradle build separately from the dependency source.

## Running a packaged JAR

The loader checks an explicit library override, then the platform resource in the JAR. If the resource is absent, it tries the Java library path and finally a checkout. Use `-Dcom.gresearch.tdigest.native=path/to/library.so` for an override or `-Dcom.gresearch.tdigest.extractDir=/writable/location` when temporary files cannot be executed. Set `TDIGEST_SOURCE_SHA` at build time to record the source revision alongside the native library hash. Use independent digests for concurrent work or coordinate access to shared mutable objects.

## Scala, Kotlin and Spark

The [Kotlin example](examples/kotlin/src/main/kotlin/Example.kt) resolves the published Maven artifact. Run `bindings/java/gradlew -p bindings/java/examples/kotlin -PtdigestVersion=1.0.0 run` from the repository root. The [Scala example](examples/scala/src/main/scala/Example.scala) uses sbt with JDK 17 and targets Java 11. Run `sbt -Dtdigest.version=1.0.0 run` in `bindings/java/examples/scala`. For private repositories set the Maven URL and credentials above. The sbt launcher may also need its own repository, realm and boot credentials.

The [Spark check](examples/spark/check.py) transfers digest state between two executor processes using a shuffle, a broadcast and a checkpoint. Run the script with `--help` for the Spark distribution, compiler classpath, JAR, serializer and output arguments. Its Kryo example registers a serializer for TDIG bytes. On a cluster, provide a shared checkpoint URI and run the check with the actual executor image. The [recorded run](../../compat/spark-20261021/receipt.json) identifies the Spark and JDK versions, artifacts and serializers tested.
