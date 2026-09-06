package dev.forloop.ytsave;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.net.Uri;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.util.Log;
import android.widget.Toast;

import org.schabi.newpipe.extractor.stream.StreamInfo;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Foreground service that extracts, downloads, merges and saves one video per start request.
 * Requests are processed one at a time; the service stops itself when the queue is empty.
 */
public class DownloadService extends Service {
    public static final String EXTRA_URL = "dev.forloop.ytsave.URL";

    private static final String TAG = "YtSave";
    private static final String CHANNEL = "downloads";
    private static final int NOTIF_PROGRESS = 1;
    private static final long CHUNK_BYTES = 10L * 1024 * 1024;
    private static final int IO_TIMEOUT_MS = 30_000;
    private static final long PROGRESS_INTERVAL_MS = 700;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final AtomicInteger active = new AtomicInteger();
    private final AtomicInteger nextResultId = new AtomicInteger(100);
    private NotificationManager notifications;
    private volatile Notification lastProgress;

    @Override
    public void onCreate() {
        super.onCreate();
        notifications = getSystemService(NotificationManager.class);
        NotificationChannel channel = new NotificationChannel(CHANNEL,
                getString(R.string.channel_name), NotificationManager.IMPORTANCE_LOW);
        notifications.createNotificationChannel(channel);
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        final String url = intent == null ? null : intent.getStringExtra(EXTRA_URL);
        Notification n = lastProgress != null ? lastProgress
                : buildProgress(getString(R.string.preparing), null, -1);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIF_PROGRESS, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
        } else {
            startForeground(NOTIF_PROGRESS, n);
        }
        if (url == null) {
            if (active.get() == 0) {
                stopNow();
            }
            return START_NOT_STICKY;
        }

        active.incrementAndGet();
        executor.execute(() -> {
            try {
                handle(url);
            } catch (Throwable t) {
                Log.e(TAG, "download failed for " + url, t);
                reportFailure(t);
            } finally {
                if (active.decrementAndGet() == 0) {
                    mainHandler.post(this::stopNow);
                }
            }
        });
        return START_NOT_STICKY;
    }

    private void stopNow() {
        if (active.get() != 0) {
            return;
        }
        lastProgress = null;
        stopForeground(Service.STOP_FOREGROUND_REMOVE);
        stopSelf();
    }

    @Override
    public void onDestroy() {
        executor.shutdownNow();
        super.onDestroy();
    }

    // ---------------------------------------------------------------- pipeline

    private void handle(String url) throws Exception {
        progress(getString(R.string.fetching), null, -1);
        StreamInfo info = StreamInfo.getInfo(url);
        String title = info.getName();
        if (title == null || title.trim().isEmpty()) {
            title = "YouTube " + info.getId();
        }

        List<StreamPicker.Plan> plans = StreamPicker.plans(info);
        if (plans.isEmpty()) {
            throw new IOException(getString(R.string.no_streams));
        }

        File dir = new File(getCacheDir(), "ytsave");
        if (!dir.isDirectory() && !dir.mkdirs()) {
            throw new IOException("Cannot create cache directory");
        }

        Exception last = null;
        for (StreamPicker.Plan plan : plans) {
            File out = new File(dir, "out-" + System.nanoTime() + ".mp4");
            try {
                produce(plan, title, dir, out);
                progress(title, getString(R.string.saving), -1);
                String fileName = GallerySaver.safeFileName(title) + ".mp4";
                Uri uri = GallerySaver.saveVideo(this, out, fileName);
                reportSuccess(title, plan.label, uri);
                return;
            } catch (Exception e) {
                Log.w(TAG, "plan " + plan.label + " failed, trying next", e);
                last = e;
            } finally {
                //noinspection ResultOfMethodCallIgnored
                out.delete();
            }
        }
        throw last;
    }

    private void produce(StreamPicker.Plan plan, String title, File dir, File out) throws IOException {
        if (!plan.needsMerge()) {
            download(plan.video.getContent(), out, title,
                    getString(R.string.downloading_video, plan.label), 0, 95);
            return;
        }
        File video = new File(dir, "v-" + System.nanoTime() + ".mp4");
        File audio = new File(dir, "a-" + System.nanoTime() + ".m4a");
        try {
            download(plan.video.getContent(), video, title,
                    getString(R.string.downloading_video, plan.label), 0, 70);
            download(plan.audio.getContent(), audio, title,
                    getString(R.string.downloading_audio), 70, 90);
            progress(title, getString(R.string.merging), -1);
            Mp4Merger.merge(video, audio, out);
        } finally {
            //noinspection ResultOfMethodCallIgnored
            video.delete();
            //noinspection ResultOfMethodCallIgnored
            audio.delete();
        }
    }

    /** Downloads in 10 MB ranged chunks (YouTube throttles or rejects huge single requests). */
    private void download(String url, File dest, String title, String label,
                          int pctFrom, int pctTo) throws IOException {
        long pos = 0;
        long total = -1;
        long lastNotify = 0;
        byte[] buf = new byte[64 * 1024];
        try (OutputStream os = new FileOutputStream(dest)) {
            while (total < 0 || pos < total) {
                long end = pos + CHUNK_BYTES - 1;
                HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
                c.setConnectTimeout(IO_TIMEOUT_MS);
                c.setReadTimeout(IO_TIMEOUT_MS);
                c.setInstanceFollowRedirects(true);
                c.setRequestProperty("User-Agent", HttpDownloader.USER_AGENT);
                c.setRequestProperty("Range", "bytes=" + pos + "-" + end);
                try {
                    int code = c.getResponseCode();
                    boolean wholeBody;
                    if (code == 206) {
                        wholeBody = false;
                        total = parseTotal(c.getHeaderField("Content-Range"), total);
                    } else if (code == 200) {
                        if (pos != 0) {
                            throw new IOException("Server ignored the byte range");
                        }
                        wholeBody = true;
                        total = c.getContentLengthLong();
                    } else {
                        throw new IOException("HTTP " + code + " while downloading " + label);
                    }

                    long got = 0;
                    try (InputStream in = c.getInputStream()) {
                        int n;
                        while ((n = in.read(buf)) > 0) {
                            os.write(buf, 0, n);
                            pos += n;
                            got += n;
                            long now = System.currentTimeMillis();
                            if (total > 0 && now - lastNotify > PROGRESS_INTERVAL_MS) {
                                lastNotify = now;
                                int pct = (int) (pctFrom + (pctTo - pctFrom) * pos / total);
                                progress(title, label, pct);
                            }
                        }
                    }
                    if (wholeBody) {
                        break;
                    }
                    if (got == 0) {
                        throw new IOException("Empty response while downloading " + label);
                    }
                    if (total < 0 && got < (end - (pos - got) + 1)) {
                        break; // short chunk with unknown size: end of file
                    }
                } finally {
                    c.disconnect();
                }
            }
        }
        if (total > 0 && pos < total) {
            throw new IOException("Download incomplete (" + pos + "/" + total + ")");
        }
    }

    private static long parseTotal(String contentRange, long fallback) {
        if (contentRange == null) {
            return fallback;
        }
        int slash = contentRange.lastIndexOf('/');
        if (slash < 0) {
            return fallback;
        }
        try {
            return Long.parseLong(contentRange.substring(slash + 1).trim());
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    // ---------------------------------------------------------------- notifications & toasts

    private Notification buildProgress(String title, String text, int pct) {
        Notification.Builder b = new Notification.Builder(this, CHANNEL)
                .setSmallIcon(R.drawable.ic_stat_download)
                .setContentTitle(title)
                .setOnlyAlertOnce(true)
                .setOngoing(true);
        if (text != null) {
            b.setContentText(text);
        }
        if (pct < 0) {
            b.setProgress(0, 0, true);
        } else {
            b.setProgress(100, Math.min(100, pct), false);
        }
        return b.build();
    }

    private void progress(String title, String text, int pct) {
        Notification n = buildProgress(title, text, pct);
        lastProgress = n;
        notifications.notify(NOTIF_PROGRESS, n);
    }

    private void reportSuccess(String title, String label, Uri uri) {
        toast(getString(R.string.saved_toast, title, label));
        Notification.Builder b = new Notification.Builder(this, CHANNEL)
                .setSmallIcon(R.drawable.ic_stat_download)
                .setContentTitle(getString(R.string.saved))
                .setContentText(title + " (" + label + ")")
                .setAutoCancel(true);
        if (uri != null && "content".equals(uri.getScheme())) {
            Intent view = new Intent(Intent.ACTION_VIEW)
                    .setDataAndType(uri, "video/mp4")
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_ACTIVITY_NEW_TASK);
            b.setContentIntent(PendingIntent.getActivity(this, nextResultId.get(), view,
                    PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE));
        }
        notifications.notify(nextResultId.getAndIncrement(), b.build());
    }

    private void reportFailure(Throwable t) {
        String msg = t.getMessage() == null || t.getMessage().isEmpty()
                ? t.getClass().getSimpleName() : t.getMessage();
        toast(getString(R.string.failed_toast, msg));
        Notification n = new Notification.Builder(this, CHANNEL)
                .setSmallIcon(R.drawable.ic_stat_download)
                .setContentTitle(getString(R.string.failed))
                .setContentText(msg)
                .setStyle(new Notification.BigTextStyle().bigText(msg))
                .setAutoCancel(true)
                .build();
        notifications.notify(nextResultId.getAndIncrement(), n);
    }

    private void toast(final String msg) {
        mainHandler.post(() -> Toast.makeText(getApplicationContext(), msg, Toast.LENGTH_LONG).show());
    }
}
