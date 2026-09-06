package dev.forloop.ytsave;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;

/** Invisible share target: grabs the link, asks for permissions if needed, hands off to the service. */
public class ShareActivity extends Activity {
    private static final int REQ_PERMISSIONS = 1;
    private String url;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        url = resolveUrl(getIntent());
        if (url == null) {
            Toast.makeText(this, R.string.no_link, Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        List<String> missing = new ArrayList<>();
        if (Build.VERSION.SDK_INT >= 33
                && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            missing.add(Manifest.permission.POST_NOTIFICATIONS);
        }
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q
                && checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            missing.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
        }
        if (missing.isEmpty()) {
            startDownload();
        } else {
            requestPermissions(missing.toArray(new String[0]), REQ_PERMISSIONS);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q
                && checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, R.string.storage_denied, Toast.LENGTH_LONG).show();
            finish();
            return;
        }
        // Notifications being denied only hides progress; the download still runs.
        startDownload();
    }

    private void startDownload() {
        Intent i = new Intent(this, DownloadService.class).putExtra(DownloadService.EXTRA_URL, url);
        startForegroundService(i);
        Toast.makeText(this, R.string.queued, Toast.LENGTH_SHORT).show();
        finish();
    }

    static String resolveUrl(Intent intent) {
        if (intent == null) {
            return null;
        }
        String found = null;
        if (Intent.ACTION_SEND.equals(intent.getAction())) {
            found = Links.extractYoutubeUrl(intent.getStringExtra(Intent.EXTRA_TEXT));
            if (found == null) {
                found = Links.extractYoutubeUrl(intent.getStringExtra(Intent.EXTRA_SUBJECT));
            }
        }
        if (found == null && intent.getData() != null) {
            found = Links.extractYoutubeUrl(intent.getDataString());
        }
        return found;
    }
}
