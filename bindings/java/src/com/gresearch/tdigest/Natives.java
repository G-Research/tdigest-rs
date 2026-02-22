package com.gresearch.tdigest;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;

final class Natives {
  private static volatile boolean loaded = false;

  static synchronized void load() {
    if (loaded) return;

    final String osName  = System.getProperty("os.name").toLowerCase(Locale.ROOT);
    final String osArch0 = System.getProperty("os.arch").toLowerCase(Locale.ROOT);

    final String platform = osName.contains("win") ? "windows"
                       : osName.contains("mac") ? "macos"
                                                : "linux";

    final String arch = (osArch0.equals("x86_64") || osArch0.equals("amd64")) ? "x86_64"
                    : (osArch0.equals("aarch64") || osArch0.equals("arm64")) ? "aarch64"
                    : (osArch0.equals("x86") || osArch0.equals("i386") || osArch0.equals("i686")) ? "x86"
                    : osArch0;

    // Canonical base name for System.loadLibrary("tdigest_rs")
    final String base = "tdigest_rs";
    final String lib =
        platform.equals("windows") ? base + ".dll"
      : platform.equals("macos")   ? "lib" + base + ".dylib"
                                   : "lib" + base + ".so";

    final String resource = "/META-INF/native/" + platform + "-" + arch + "/" + lib;

    // 0) Explicit override: -Dcom.gresearch.tdigest.native=/abs/path/to/libtdigest_rs.so
    final String overridePath = System.getProperty("com.gresearch.tdigest.native");
    if (overridePath != null && !overridePath.isEmpty()) {
      System.load(overridePath);
      loaded = true;
      return;
    }

    // 1) Preferred: load from JAR resource (production packaging path)
    try (InputStream in = Natives.class.getResourceAsStream(resource)) {
      if (in != null) {
        Path tmp = Files.createTempFile(base + "-", lib);
        // Best effort cleanup; the OS temp directory may purge on reboot anyway.
        tmp.toFile().deleteOnExit();
        Files.copy(in, tmp, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        System.load(tmp.toAbsolutePath().toString());
        loaded = true;
        return;
      }
    } catch (IOException e) {
      // Fall through to other strategies.
    }

    // 2) Dev fallback: rely on -Djava.library.path (e.g., target/release)
    try {
      System.loadLibrary(base);
      loaded = true;
      return;
    } catch (UnsatisfiedLinkError ignored) {
      // Keep falling through.
    }

    // 3) Last-ditch: common local build path when running from classes dir
    try {
      Path local = Paths.get("target", "release", lib);
      if (Files.exists(local)) {
        System.load(local.toAbsolutePath().toString());
        loaded = true;
        return;
      }
    } catch (UnsatisfiedLinkError ignored) {
      // Give a richer error below.
    }

    throw new UnsatisfiedLinkError(
        "Failed loading native " + base + ": no resource " + resource +
        " on classpath, and loadLibrary/load() fallbacks failed. " +
        "Tips: set -Djava.library.path=target/release, or " +
        "-Dcom.gresearch.tdigest.native=/abs/path/to/" + lib);
  }

  private Natives() {}
}
