package com.gresearch.tdigest;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Locale;

/** Loads one native payload. Explicit overrides never silently fall back. */
final class Natives {
  private static boolean loaded;
  private static String identity;

  static synchronized void load() {
    if (loaded) return;
    String os = System.getProperty("os.name").toLowerCase(Locale.ROOT);
    String arch = System.getProperty("os.arch").toLowerCase(Locale.ROOT);
    if (arch.equals("amd64")) arch = "x86_64";
    if (arch.equals("arm64")) arch = "aarch64";
    String platform = os.contains("win") ? "windows" : os.contains("mac") ? "macos" : "linux";
    String lib = platform.equals("windows") ? "tdigest_rs.dll"
        : platform.equals("macos") ? "libtdigest_rs.dylib" : "libtdigest_rs.so";
    String resource = "/META-INF/native/" + platform + "-" + arch + "/" + lib;
    String override = System.getProperty("com.gresearch.tdigest.native");
    if (override != null) {
      if (override.isBlank()) throw failure("explicit native override is blank", null);
      // Spark distributed-file paths may be relative to the executor working dir.
      Path selected = Paths.get(override).toAbsolutePath().normalize();
      loadPath(selected, "explicit override");
      return;
    }
    try (InputStream input = Natives.class.getResourceAsStream(resource)) {
      if (input != null) {
        String configured = System.getProperty("com.gresearch.tdigest.extractDir");
        Path directory = configured == null
            ? Paths.get(System.getProperty("java.io.tmpdir")) : Paths.get(configured);
        directory = directory.toAbsolutePath().normalize();
        if (!Files.isDirectory(directory)) throw new IOException("extraction directory does not exist: " + directory);
        Path temporary = Files.createTempDirectory(directory, "tdigest-rs-");
        temporary.toFile().deleteOnExit();
        Path payload = temporary.resolve(lib);
        payload.toFile().deleteOnExit();
        Files.copy(input, payload, StandardCopyOption.REPLACE_EXISTING);
        loadPath(payload, "bundled resource " + resource);
        return;
      }
    } catch (IOException error) {
      throw failure("cannot extract " + resource + "; set -Dcom.gresearch.tdigest.extractDir to an existing writable directory on a mount permitting native loading", error);
    }
    // Development-only paths are used only when no bundled resource exists.
    try {
      System.loadLibrary("tdigest_rs");
      identity = "java.library.path=" + System.getProperty("java.library.path") + " (payload hash unavailable)";
      loaded = true;
      return;
    } catch (UnsatisfiedLinkError error) {
      Path local = Paths.get("target", "release", lib).toAbsolutePath().normalize();
      if (Files.isRegularFile(local)) { loadPath(local, "checkout fallback"); return; }
      throw failure("no bundled " + resource + "; set -Dcom.gresearch.tdigest.native to the distributed native file (relative paths allowed)", error);
    }
  }

  private static void loadPath(Path path, String source) {
    try {
      if (!Files.isRegularFile(path)) throw new IOException("native file does not exist: " + path);
      String hash = sha256(path);
      System.load(path.toString());
      identity = source + "; path=" + path + "; sha256=" + hash;
      loaded = true;
    } catch (IOException | UnsatisfiedLinkError error) {
      throw failure(source + " failed at " + path + "; check architecture, glibc requirements and directory mount permissions", error);
    }
  }

  private static String sha256(Path path) throws IOException {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      try (InputStream input = Files.newInputStream(path)) {
        byte[] buffer = new byte[65536];
        int read;
        while ((read = input.read(buffer)) != -1) digest.update(buffer, 0, read);
      }
      StringBuilder result = new StringBuilder();
      for (byte value : digest.digest()) result.append(String.format("%02x", value & 0xff));
      return result.toString();
    } catch (NoSuchAlgorithmException impossible) { throw new AssertionError(impossible); }
  }

  private static UnsatisfiedLinkError failure(String message, Throwable cause) {
    UnsatisfiedLinkError error = new UnsatisfiedLinkError("TDigest native load: " + message);
    if (cause != null) error.initCause(cause);
    return error;
  }

  static synchronized String identity() { load(); return identity; }
  private Natives() {}
}
