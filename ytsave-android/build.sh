#!/usr/bin/env bash
# Builds dist/ytsave.apk from scratch without the Android Gradle plugin or sdkmanager.
#
# Host requirements (Ubuntu 24.04): JDK 17+, curl, git, unzip, zip, and the Debian
# packaged Android build tools:   sudo apt install aapt zipalign apksigner
# Everything else (android.jar, D8, NewPipeExtractor sources, jars) is fetched into .tools/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="${YTSAVE_TOOLS:-$ROOT/.tools}"
BUILD="$ROOT/build"
DIST="$ROOT/dist"

NPE_TAG="v0.26.5"                                        # NewPipeExtractor release
NANOJSON_REV="e9d656ddb49a412a5a0a5d5ef20ca7ef09549996"  # pinned by NewPipeExtractor
R8_VERSION="8.11.18"
COMPILE_API=34   # 35 uses a resources.arsc format the Debian aapt2 cannot read
MIN_SDK=26
TARGET_SDK=34
VERSION_CODE=1
VERSION_NAME="1.0"

MAVEN="https://repo1.maven.org/maven2"
JSOUP="1.22.2"; JSR305="3.0.2"; PROTOBUF="4.35.1"; RHINO="1.8.1"

KEYSTORE="${YTSAVE_KEYSTORE:-$ROOT/keys/ytsave.jks}"
KS_PASS="${YTSAVE_KS_PASS:-ytsave123}"
KEY_ALIAS="${YTSAVE_KEY_ALIAS:-ytsave}"

for tool in java javac jar curl git unzip zip aapt2 zipalign apksigner; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done

fetch() { # url dest
  if [ ! -s "$2" ]; then
    echo "fetch $1"
    curl -fsSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
  fi
}

mkdir -p "$TOOLS" "$DIST"
rm -rf "$BUILD"
mkdir -p "$BUILD"/{proto-gen,lib-classes,app-classes,gen,dex,extra,res}

# ------------------------------------------------------------------ toolchain + deps
ANDROID_JAR="$TOOLS/android-$COMPILE_API.jar"
R8_JAR="$TOOLS/r8lib-$R8_VERSION.jar"
fetch "https://raw.githubusercontent.com/Sable/android-platforms/master/android-$COMPILE_API/android.jar" "$ANDROID_JAR"
fetch "https://storage.googleapis.com/r8-releases/raw/$R8_VERSION/r8lib.jar" "$R8_JAR"
fetch "$MAVEN/org/jsoup/jsoup/$JSOUP/jsoup-$JSOUP.jar" "$TOOLS/jsoup.jar"
fetch "$MAVEN/com/google/code/findbugs/jsr305/$JSR305/jsr305-$JSR305.jar" "$TOOLS/jsr305.jar"
fetch "$MAVEN/com/google/protobuf/protobuf-javalite/$PROTOBUF/protobuf-javalite-$PROTOBUF.jar" "$TOOLS/protobuf-javalite.jar"
fetch "$MAVEN/org/mozilla/rhino/$RHINO/rhino-$RHINO.jar" "$TOOLS/rhino.jar"
fetch "$MAVEN/org/mozilla/rhino-engine/$RHINO/rhino-engine-$RHINO.jar" "$TOOLS/rhino-engine.jar"
fetch "$MAVEN/com/google/protobuf/protoc/$PROTOBUF/protoc-$PROTOBUF-linux-x86_64.exe" "$TOOLS/protoc"
chmod +x "$TOOLS/protoc"

if [ ! -d "$TOOLS/NewPipeExtractor" ]; then
  git -c advice.detachedHead=false clone -q --depth 1 --branch "$NPE_TAG" https://github.com/TeamNewPipe/NewPipeExtractor "$TOOLS/NewPipeExtractor"
fi
if [ ! -d "$TOOLS/nanojson" ]; then
  git clone -q https://github.com/TeamNewPipe/nanojson "$TOOLS/nanojson"
  git -C "$TOOLS/nanojson" checkout -q "$NANOJSON_REV"
fi

DEP_JARS="$TOOLS/jsoup.jar:$TOOLS/jsr305.jar:$TOOLS/protobuf-javalite.jar:$TOOLS/rhino.jar:$TOOLS/rhino-engine.jar"

# ------------------------------------------------------------------ 1. NewPipeExtractor (plain javac)
echo "== compiling NewPipeExtractor $NPE_TAG"
NPE="$TOOLS/NewPipeExtractor"
"$TOOLS/protoc" -I "$NPE/extractor/src/main/proto" --java_out=lite:"$BUILD/proto-gen" \
  $(find "$NPE/extractor/src/main/proto" -name '*.proto')
find "$TOOLS/nanojson/src/main/java" "$NPE/extractor/src/main/java" "$NPE/timeago-parser/src/main/java" \
  "$BUILD/proto-gen" -name '*.java' > "$BUILD/lib-sources.txt"
javac -nowarn -encoding UTF-8 --release 11 -proc:none -cp "$DEP_JARS" -d "$BUILD/lib-classes" @"$BUILD/lib-sources.txt"
(cd "$BUILD/lib-classes" && jar cf "$BUILD/extractor.jar" .)

# ------------------------------------------------------------------ 2. resources
echo "== resources"
aapt2 compile --dir "$ROOT/res" -o "$BUILD/res.zip"
aapt2 link -o "$BUILD/app-res.apk" -I "$ANDROID_JAR" \
  --manifest "$ROOT/AndroidManifest.xml" --java "$BUILD/gen" \
  --min-sdk-version "$MIN_SDK" --target-sdk-version "$TARGET_SDK" \
  --version-code "$VERSION_CODE" --version-name "$VERSION_NAME" \
  "$BUILD/res.zip"

# ------------------------------------------------------------------ 3. app code
echo "== compiling app"
find "$ROOT/src" "$BUILD/gen" -name '*.java' > "$BUILD/app-sources.txt"
# android.jar goes on the classpath (not the bootclasspath): its stubs lack LambdaMetafactory,
# which javac needs to compile lambdas. D8 desugars them for the device afterwards.
javac -nowarn -encoding UTF-8 --release 11 -proc:none \
  -cp "$ANDROID_JAR:$BUILD/extractor.jar:$DEP_JARS" -d "$BUILD/app-classes" @"$BUILD/app-sources.txt"
(cd "$BUILD/app-classes" && jar cf "$BUILD/app.jar" .)

# ------------------------------------------------------------------ 4. dex
echo "== dexing"
java -cp "$R8_JAR" com.android.tools.r8.D8 --release --min-api "$MIN_SDK" --lib "$ANDROID_JAR" \
  --output "$BUILD/dex" "$BUILD/app.jar" "$BUILD/extractor.jar" ${DEP_JARS//:/ }

# ------------------------------------------------------------------ 5. package, align, sign
echo "== packaging"
cp "$BUILD/app-res.apk" "$BUILD/app-unaligned.apk"
(cd "$BUILD/dex" && zip -q "$BUILD/app-unaligned.apk" classes*.dex)
# Rhino loads its error-message bundles from the classpath at runtime.
(cd "$BUILD/extra" && unzip -q -o "$TOOLS/rhino.jar" 'org/mozilla/javascript/resources/*' \
  && zip -q -r "$BUILD/app-unaligned.apk" org)
zipalign -p -f 4 "$BUILD/app-unaligned.apk" "$BUILD/app-aligned.apk"
apksigner sign --ks "$KEYSTORE" --ks-pass "pass:$KS_PASS" --ks-key-alias "$KEY_ALIAS" \
  --out "$DIST/ytsave.apk" "$BUILD/app-aligned.apk"
apksigner verify --print-certs "$DIST/ytsave.apk" | head -3
ls -la "$DIST/ytsave.apk"
