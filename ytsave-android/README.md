# YT Save (Android)

Share a YouTube link to **Save to gallery (YT Save)** and the video is saved into
your gallery (`Movies/YT Save`) at **480p**, or the closest quality available.

How it works:

1. `ShareActivity` receives the shared link (share sheet, or "open with" on a YouTube URL).
2. `DownloadService` (foreground service with a progress notification) resolves the
   streams with [NewPipeExtractor](https://github.com/TeamNewPipe/NewPipeExtractor).
3. `StreamPicker` picks the 480p H.264 video-only MP4 stream (exact 480p, else the largest
   below it, else the smallest above) plus the best M4A audio track.
4. Both are downloaded in ranged chunks and merged on the device with `MediaMuxer`
   (no re-encoding). If any step fails, it falls back to the progressive MP4 stream that
   already contains audio (typically 360p).
5. The result is inserted into `MediaStore` so it shows up in the gallery immediately.

Requires Android 8.0 (API 26) or newer. No Gradle: `./build.sh` fetches the toolchain
(android.jar, D8, extractor sources and jars) and produces `dist/ytsave.apk`. It needs
`apt install aapt zipalign apksigner` plus a JDK 17+.

The extractor library is GPL-3.0-or-later, so this app is distributed under the same
license. The signing key in `keys/` is a self-signed key generated for this app.
