package dev.forloop.ytsave;

import org.schabi.newpipe.extractor.downloader.Downloader;
import org.schabi.newpipe.extractor.downloader.Request;
import org.schabi.newpipe.extractor.downloader.Response;
import org.schabi.newpipe.extractor.exceptions.ReCaptchaException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPInputStream;

/** NewPipeExtractor {@link Downloader} backed by HttpURLConnection (no extra dependencies). */
public class HttpDownloader extends Downloader {
    public static final String USER_AGENT =
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0";
    private static final int TIMEOUT_MS = 30_000;

    @Override
    public Response execute(Request request) throws IOException, ReCaptchaException {
        HttpURLConnection c = (HttpURLConnection) new URL(request.url()).openConnection();
        try {
            c.setConnectTimeout(TIMEOUT_MS);
            c.setReadTimeout(TIMEOUT_MS);
            c.setInstanceFollowRedirects(true);
            c.setRequestMethod(request.httpMethod());
            c.setRequestProperty("User-Agent", USER_AGENT);

            for (Map.Entry<String, List<String>> e : request.headers().entrySet()) {
                List<String> values = e.getValue();
                if (values == null || values.isEmpty()) {
                    continue;
                }
                c.setRequestProperty(e.getKey(), values.get(0));
                for (int i = 1; i < values.size(); i++) {
                    c.addRequestProperty(e.getKey(), values.get(i));
                }
            }

            byte[] data = request.dataToSend();
            if (data != null) {
                c.setDoOutput(true);
                c.setFixedLengthStreamingMode(data.length);
                try (OutputStream os = c.getOutputStream()) {
                    os.write(data);
                }
            }

            int code = c.getResponseCode();
            if (code == 429) {
                throw new ReCaptchaException("reCaptcha Challenge requested", request.url());
            }

            InputStream in = code >= 400 ? c.getErrorStream() : c.getInputStream();
            String body = "";
            if (in != null) {
                if ("gzip".equalsIgnoreCase(c.getContentEncoding())) {
                    in = new GZIPInputStream(in);
                }
                body = readAll(in);
            }

            Map<String, List<String>> headers = new LinkedHashMap<>();
            for (Map.Entry<String, List<String>> e : c.getHeaderFields().entrySet()) {
                if (e.getKey() != null) {
                    headers.put(e.getKey(), e.getValue());
                }
            }
            return new Response(code, c.getResponseMessage(), headers, body, c.getURL().toString());
        } finally {
            c.disconnect();
        }
    }

    private static String readAll(InputStream in) throws IOException {
        try (InputStream is = in) {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            byte[] buf = new byte[16 * 1024];
            int n;
            while ((n = is.read(buf)) > 0) {
                bos.write(buf, 0, n);
            }
            return new String(bos.toByteArray(), StandardCharsets.UTF_8);
        }
    }
}
