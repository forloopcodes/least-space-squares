package dev.forloop.ytsave;

import org.schabi.newpipe.extractor.MediaFormat;
import org.schabi.newpipe.extractor.stream.AudioStream;
import org.schabi.newpipe.extractor.stream.AudioTrackType;
import org.schabi.newpipe.extractor.stream.DeliveryMethod;
import org.schabi.newpipe.extractor.stream.Stream;
import org.schabi.newpipe.extractor.stream.StreamInfo;
import org.schabi.newpipe.extractor.stream.VideoStream;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Chooses which streams to download. Target is 480p; the plans are tried in order:
 * <ol>
 *   <li>480p (or the closest resolution) MP4 video-only stream + M4A audio, merged on device</li>
 *   <li>a progressive MP4 stream that already contains audio (usually 360p) as a fallback</li>
 * </ol>
 */
public final class StreamPicker {
    public static final int TARGET_HEIGHT = 480;

    public static final class Plan {
        public final VideoStream video;
        public final AudioStream audio; // null => progressive stream, no merge needed
        public final String label;

        Plan(VideoStream video, AudioStream audio, String label) {
            this.video = video;
            this.audio = audio;
            this.label = label;
        }

        public boolean needsMerge() {
            return audio != null;
        }
    }

    private StreamPicker() {
    }

    public static List<Plan> plans(StreamInfo info) {
        List<Plan> plans = new ArrayList<>();

        VideoStream videoOnly = pickVideoOnly(info.getVideoOnlyStreams());
        AudioStream audio = pickAudio(info.getAudioStreams());
        if (videoOnly != null && audio != null) {
            plans.add(new Plan(videoOnly, audio, resolutionLabel(videoOnly)));
        }

        VideoStream progressive = pickProgressive(info.getVideoStreams());
        if (progressive != null) {
            plans.add(new Plan(progressive, null, resolutionLabel(progressive)));
        }
        return plans;
    }

    private static boolean usable(Stream s) {
        return s != null && s.isUrl() && s.getContent() != null && !s.getContent().isEmpty()
                && s.getDeliveryMethod() == DeliveryMethod.PROGRESSIVE_HTTP;
    }

    static int heightOf(VideoStream v) {
        if (v.getHeight() > 0) {
            return v.getHeight();
        }
        String res = v.getResolution();
        if (res == null) {
            return 0;
        }
        int end = 0;
        while (end < res.length() && Character.isDigit(res.charAt(end))) {
            end++;
        }
        try {
            return end == 0 ? 0 : Integer.parseInt(res.substring(0, end));
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    static String resolutionLabel(VideoStream v) {
        String res = v.getResolution();
        if (res != null && !res.isEmpty()) {
            return res;
        }
        int h = heightOf(v);
        return h > 0 ? h + "p" : "unknown";
    }

    private static boolean isAvc(VideoStream v) {
        String codec = v.getCodec();
        return codec != null && codec.toLowerCase(Locale.ROOT).startsWith("avc");
    }

    static VideoStream pickVideoOnly(List<VideoStream> streams) {
        List<VideoStream> avc = new ArrayList<>();
        List<VideoStream> other = new ArrayList<>();
        for (VideoStream v : streams) {
            if (!usable(v) || v.getFormat() != MediaFormat.MPEG_4 || heightOf(v) <= 0) {
                continue;
            }
            (isAvc(v) ? avc : other).add(v);
        }
        // H.264 in MP4 is what MediaMuxer handles most reliably; try it first.
        VideoStream best = closestToTarget(avc);
        return best != null ? best : closestToTarget(other);
    }

    static VideoStream pickProgressive(List<VideoStream> streams) {
        List<VideoStream> candidates = new ArrayList<>();
        for (VideoStream v : streams) {
            if (usable(v) && v.getFormat() == MediaFormat.MPEG_4 && !v.isVideoOnly()
                    && heightOf(v) > 0) {
                candidates.add(v);
            }
        }
        return closestToTarget(candidates);
    }

    /** Exact target height first, otherwise the largest below it, otherwise the smallest above. */
    static VideoStream closestToTarget(List<VideoStream> candidates) {
        VideoStream exact = null;
        VideoStream below = null;
        VideoStream above = null;
        for (VideoStream v : candidates) {
            int h = heightOf(v);
            if (h == TARGET_HEIGHT) {
                if (exact == null || preferable(v, exact)) {
                    exact = v;
                }
            } else if (h < TARGET_HEIGHT) {
                if (below == null || h > heightOf(below)
                        || (h == heightOf(below) && preferable(v, below))) {
                    below = v;
                }
            } else {
                if (above == null || h < heightOf(above)
                        || (h == heightOf(above) && preferable(v, above))) {
                    above = v;
                }
            }
        }
        if (exact != null) {
            return exact;
        }
        return below != null ? below : above;
    }

    /** Among streams of the same height prefer standard frame rate (smaller file). */
    private static boolean preferable(VideoStream candidate, VideoStream current) {
        int cf = candidate.getFps() <= 0 ? 30 : candidate.getFps();
        int xf = current.getFps() <= 0 ? 30 : current.getFps();
        if (cf != xf) {
            return cf < xf;
        }
        return candidate.getBitrate() > current.getBitrate();
    }

    static AudioStream pickAudio(List<AudioStream> streams) {
        AudioStream best = null;
        for (AudioStream a : streams) {
            if (!usable(a) || a.getFormat() != MediaFormat.M4A) {
                continue;
            }
            if (best == null || audioBetter(a, best)) {
                best = a;
            }
        }
        return best;
    }

    private static boolean isOriginalTrack(AudioStream a) {
        AudioTrackType t = a.getAudioTrackType();
        return t == null || t == AudioTrackType.ORIGINAL;
    }

    private static boolean audioBetter(AudioStream candidate, AudioStream current) {
        boolean co = isOriginalTrack(candidate);
        boolean xo = isOriginalTrack(current);
        if (co != xo) {
            return co;
        }
        return candidate.getAverageBitrate() > current.getAverageBitrate();
    }
}
