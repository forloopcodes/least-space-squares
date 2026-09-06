package dev.forloop.ytsave;

import android.app.Application;

import org.schabi.newpipe.extractor.NewPipe;
import org.schabi.newpipe.extractor.localization.ContentCountry;
import org.schabi.newpipe.extractor.localization.Localization;
import org.schabi.newpipe.extractor.services.youtube.extractors.YoutubeStreamExtractor;

import java.util.Locale;

public class YtSaveApp extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        String country = Locale.getDefault().getCountry();
        ContentCountry contentCountry = (country == null || country.isEmpty())
                ? ContentCountry.DEFAULT : new ContentCountry(country);
        NewPipe.init(new HttpDownloader(), Localization.DEFAULT, contentCountry);
        // Also query the iOS client: gives an extra set of stream URLs to fall back on.
        YoutubeStreamExtractor.setFetchIosClient(true);
    }
}
