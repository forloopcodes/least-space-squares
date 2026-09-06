package dev.forloop.ytsave;

import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Finds a YouTube URL inside shared text. */
public final class Links {
    private static final Pattern URL = Pattern.compile("https?://[^\\s<>\"']+");
    private static final String TRAILING = ".,;:!?)]}>";

    private Links() {
    }

    public static String extractYoutubeUrl(String text) {
        if (text == null) {
            return null;
        }
        Matcher m = URL.matcher(text);
        while (m.find()) {
            String u = m.group();
            while (u.length() > 0 && TRAILING.indexOf(u.charAt(u.length() - 1)) >= 0) {
                u = u.substring(0, u.length() - 1);
            }
            String lower = u.toLowerCase(Locale.ROOT);
            if (lower.contains("youtube.com/") || lower.contains("youtu.be/")
                    || lower.contains("youtube-nocookie.com/")) {
                return u;
            }
        }
        return null;
    }
}
