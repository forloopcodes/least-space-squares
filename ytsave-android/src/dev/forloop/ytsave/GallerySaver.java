package dev.forloop.ytsave;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/** Puts a finished MP4 into the device gallery (Movies/YT Save). */
public final class GallerySaver {
    public static final String FOLDER = "YT Save";
    private static final String MIME = "video/mp4";

    private GallerySaver() {
    }

    public static String safeFileName(String title) {
        String s = title == null ? "" : title.replaceAll("[\\\\/:*?\"<>|\\p{Cntrl}]", "_").trim();
        if (s.length() > 80) {
            s = s.substring(0, 80).trim();
        }
        return s.isEmpty() ? "video" : s;
    }

    /** Returns a content Uri (API 29+) or a file Uri (older devices). */
    public static Uri saveVideo(Context ctx, File src, String fileName) throws IOException {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            return saveViaMediaStore(ctx, src, fileName);
        }
        return saveToPublicMovies(ctx, src, fileName);
    }

    private static Uri saveViaMediaStore(Context ctx, File src, String fileName) throws IOException {
        ContentResolver resolver = ctx.getContentResolver();
        ContentValues values = new ContentValues();
        values.put(MediaStore.MediaColumns.DISPLAY_NAME, fileName);
        values.put(MediaStore.MediaColumns.MIME_TYPE, MIME);
        values.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_MOVIES + "/" + FOLDER);
        values.put(MediaStore.MediaColumns.IS_PENDING, 1);

        Uri collection = MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY);
        Uri uri = resolver.insert(collection, values);
        if (uri == null) {
            throw new IOException("MediaStore refused to create the video entry");
        }
        try (OutputStream os = resolver.openOutputStream(uri);
             InputStream in = new FileInputStream(src)) {
            if (os == null) {
                throw new IOException("Could not open gallery file for writing");
            }
            copy(in, os);
        } catch (IOException | RuntimeException e) {
            resolver.delete(uri, null, null);
            throw e;
        }
        values.clear();
        values.put(MediaStore.MediaColumns.IS_PENDING, 0);
        resolver.update(uri, values, null, null);
        return uri;
    }

    private static Uri saveToPublicMovies(Context ctx, File src, String fileName) throws IOException {
        File dir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES), FOLDER);
        if (!dir.isDirectory() && !dir.mkdirs()) {
            throw new IOException("Cannot create " + dir);
        }
        File dest = uniqueFile(dir, fileName);
        try (InputStream in = new FileInputStream(src); OutputStream os = new FileOutputStream(dest)) {
            copy(in, os);
        }
        MediaScannerConnection.scanFile(ctx.getApplicationContext(),
                new String[] {dest.getAbsolutePath()}, new String[] {MIME}, null);
        return Uri.fromFile(dest);
    }

    private static File uniqueFile(File dir, String fileName) {
        File f = new File(dir, fileName);
        if (!f.exists()) {
            return f;
        }
        int dot = fileName.lastIndexOf('.');
        String base = dot > 0 ? fileName.substring(0, dot) : fileName;
        String ext = dot > 0 ? fileName.substring(dot) : "";
        for (int i = 1; ; i++) {
            f = new File(dir, base + " (" + i + ")" + ext);
            if (!f.exists()) {
                return f;
            }
        }
    }

    private static void copy(InputStream in, OutputStream os) throws IOException {
        byte[] buf = new byte[256 * 1024];
        int n;
        while ((n = in.read(buf)) > 0) {
            os.write(buf, 0, n);
        }
        os.flush();
    }
}
