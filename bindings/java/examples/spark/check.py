"""Compile and run the Spark consumer check with staged tools and one exact JAR.

No dependencies are downloaded. Local mode uses two separate executor JVMs.
The output directory must be new so earlier qualification evidence is retained.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spark-home", type=Path, required=True)
    parser.add_argument("--jar", type=Path, required=True)
    parser.add_argument("--scala-compiler-classpath", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master", default="local-cluster[2,1,512]")
    parser.add_argument("--checkpoint", help="Shared checkpoint URI; required for a remote cluster")
    parser.add_argument("--serializer", choices=["java", "kryo"], default="java")
    parser.add_argument("--properties-file", type=Path)
    parser.add_argument("--conf", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory exists; choose a new receipt directory")
    if not args.master.startswith("local-cluster") and not args.checkpoint:
        parser.error("a remote check needs a checkpoint URI shared by driver and executors")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    java_home = Path(os.environ["JAVA_HOME"]).resolve()
    spark_home, jar = args.spark_home.resolve(), args.jar.resolve()
    source = Path(__file__).with_name("TDigestSparkCheck.scala")
    with zipfile.ZipFile(jar) as archive:
        native = [name for name in archive.namelist() if name.endswith("libtdigest_rs.so")]
        if len(native) != 1:
            parser.error("expected one Linux native payload in the candidate JAR")
        native_hash = hashlib.sha256(archive.read(native[0])).hexdigest()
        manifest = archive.read("META-INF/MANIFEST.MF").decode()
    args.output.mkdir(parents=True)
    record = {
        "jar_sha256": digest(jar),
        "jar_manifest": manifest,
        "native_sha256": native_hash,
        "scala_source_sha256": digest(source),
        "runner_sha256": digest(Path(__file__)),
        "java_home": str(java_home),
        "spark_home": str(spark_home),
        "master": args.master,
        "serializer": args.serializer,
        "commands": [],
        "scope": "Public consumer exercise; actual executor image, trust, serializers and workload need owner qualification",
        "status": "RUNNING",
    }

    def save():
        target = args.output / "receipt.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(temporary, target)

    def run(name, command, env, cwd):
        command = list(map(str, command))
        log = args.output / (name + ".log")
        with log.open("w") as stream:
            process = subprocess.Popen(
                command,
                env=env,
                cwd=cwd,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
        record["commands"].append(
            {
                "command": command,
                "exit_code": code,
                "log": log.name,
                "log_sha256": digest(log),
            }
        )
        save()
        if code:
            raise RuntimeError(f"{name} failed ({code}); inspect {log}")
        return log.read_text()

    save()
    try:
        # Spark shell launchers mishandle some paths containing spaces. Stage aliases
        # and the exact JAR in a temporary directory without changing its bytes.
        with tempfile.TemporaryDirectory(prefix="tdigest-spark-") as temporary:
            stage = Path(temporary)
            home = stage / "spark"
            home.symlink_to(spark_home, target_is_directory=True)
            staged_jar = stage / jar.name
            shutil.copyfile(jar, staged_jar)
            classes = stage / "classes"
            classes.mkdir()
            env = os.environ.copy()
            env.update(
                SPARK_HOME=str(home),
                SPARK_WORKER_DIR=str(stage / "workers"),
                SPARK_LOCAL_DIRS=str(stage / "scratch"),
            )
            if args.master.startswith("local-cluster"):
                env["SPARK_LOCAL_IP"] = "127.0.0.1"
            java = java_home / "bin/java"
            record["compiler"] = run(
                "compiler-version",
                [
                    java,
                    "-cp",
                    args.scala_compiler_classpath,
                    "scala.tools.nsc.Main",
                    "-version",
                ],
                env,
                stage,
            ).strip()
            run(
                "compile",
                [
                    java,
                    "-Xmx1g",
                    "-cp",
                    args.scala_compiler_classpath,
                    "scala.tools.nsc.Main",
                    "-target:jvm-1.8",
                    "-classpath",
                    str(home / "jars/*") + os.pathsep + str(staged_jar),
                    "-d",
                    classes,
                    source,
                ],
                env,
                stage,
            )
            app = stage / "tdigest-spark-check.jar"
            run(
                "package",
                [java_home / "bin/jar", "cf", app, "-C", classes, "."],
                env,
                stage,
            )
            record["application_sha256"] = digest(app)
            shutil.copyfile(app, args.output / app.name)
            checkpoint = args.checkpoint or (stage / "checkpoint").as_uri()
            record["checkpoint"] = checkpoint
            record["checkpoint_retention"] = (
                "owner-managed" if args.checkpoint else "temporary local filesystem; removed after check"
            )
            command = [
                home / "bin/spark-submit",
                "--class",
                "com.gresearch.tdigest.example.TDigestSparkCheck",
                "--master",
                args.master,
                "--driver-memory",
                "512m",
            ]
            if args.properties_file:
                command += ["--properties-file", args.properties_file.resolve()]
            for config in args.conf:
                command += ["--conf", config]
            command += [
                "--jars",
                staged_jar,
                app,
                checkpoint,
                args.serializer,
                native_hash,
            ]
            output = run("spark", command, env, stage)
            marker = next(
                (line for line in output.splitlines() if line.startswith("TDIGEST_SPARK_OK ")),
                None,
            )
            if marker is None:
                raise RuntimeError("Spark exited without a successful state/identity check")
            record.update(status="PASS", summary=marker)
    except Exception as error:
        record.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    print(record["summary"])


if __name__ == "__main__":
    main()
