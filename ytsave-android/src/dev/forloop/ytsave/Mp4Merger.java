package dev.forloop.ytsave;

import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.media.MediaMuxer;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;

/** Muxes a video-only MP4 and an M4A audio file into one MP4 without re-encoding. */
public final class Mp4Merger {
    private static final int BUFFER_SIZE = 8 * 1024 * 1024;

    private Mp4Merger() {
    }

    public static void merge(File video, File audio, File out) throws IOException {
        MediaExtractor ve = new MediaExtractor();
        MediaExtractor ae = new MediaExtractor();
        MediaMuxer muxer = null;
        try {
            ve.setDataSource(video.getAbsolutePath());
            ae.setDataSource(audio.getAbsolutePath());
            int vi = findTrack(ve, "video/");
            int ai = findTrack(ae, "audio/");
            if (vi < 0) {
                throw new IOException("No video track in downloaded video stream");
            }
            if (ai < 0) {
                throw new IOException("No audio track in downloaded audio stream");
            }
            ve.selectTrack(vi);
            ae.selectTrack(ai);
            MediaFormat vf = ve.getTrackFormat(vi);
            MediaFormat af = ae.getTrackFormat(ai);

            muxer = new MediaMuxer(out.getAbsolutePath(), MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);
            int vt = muxer.addTrack(vf);
            int at = muxer.addTrack(af);
            muxer.start();
            copySamples(ve, muxer, vt, bufferSize(vf));
            copySamples(ae, muxer, at, bufferSize(af));
            muxer.stop();
        } catch (RuntimeException e) {
            throw new IOException("Merging failed: " + e.getMessage(), e);
        } finally {
            ve.release();
            ae.release();
            if (muxer != null) {
                try {
                    muxer.release();
                } catch (RuntimeException ignored) {
                    // already reported
                }
            }
        }
    }

    private static int bufferSize(MediaFormat f) {
        int size = BUFFER_SIZE;
        if (f.containsKey(MediaFormat.KEY_MAX_INPUT_SIZE)) {
            size = Math.max(size, f.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE));
        }
        return size;
    }

    private static int findTrack(MediaExtractor ex, String mimePrefix) {
        for (int i = 0; i < ex.getTrackCount(); i++) {
            String mime = ex.getTrackFormat(i).getString(MediaFormat.KEY_MIME);
            if (mime != null && mime.startsWith(mimePrefix)) {
                return i;
            }
        }
        return -1;
    }

    private static void copySamples(MediaExtractor ex, MediaMuxer muxer, int track, int bufSize) {
        ByteBuffer buf = ByteBuffer.allocateDirect(bufSize);
        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
        while (true) {
            buf.clear();
            int n = ex.readSampleData(buf, 0);
            if (n < 0) {
                break;
            }
            info.offset = 0;
            info.size = n;
            info.presentationTimeUs = ex.getSampleTime();
            info.flags = ex.getSampleFlags();
            muxer.writeSampleData(track, buf, info);
            ex.advance();
        }
    }
}
