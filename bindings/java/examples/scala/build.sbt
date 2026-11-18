scalaVersion := "2.12.16"
scalacOptions ++= Seq("-release", "11")
resolvers := { sys.env.get("TDIGEST_MAVEN_REPOSITORY_URL") match {
  case Some(url) => Seq(("configuredDependencies" at url).withAllowInsecureProtocol(true))
  case None => Seq(Resolver.mavenCentral)
}}
externalResolvers := resolvers.value
credentials ++= sys.env.get("TDIGEST_MAVEN_USERNAME").toSeq.map { username =>
  Credentials(sys.env.getOrElse("TDIGEST_MAVEN_REALM", "Artifactory Realm"),
    new java.net.URI(sys.env("TDIGEST_MAVEN_REPOSITORY_URL")).getHost,
    username, sys.env("TDIGEST_MAVEN_PASSWORD"))
}
libraryDependencies += "com.gresearch.tdigest" % "tdigest-rs" % sys.props("tdigest.version")
Compile / run / fork := true

Compile / run / javaHome := sys.env.get("TDIGEST_RUNTIME_JAVA_HOME").map(file)
