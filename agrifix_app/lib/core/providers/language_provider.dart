import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

// Locales the app supports — add more here as needed
const List<Locale> kSupportedLocales = [
  Locale('en'),
  Locale('hi'),
  Locale('pa'),
];

const Map<String, String> kLocaleDisplayNames = {
  'en': 'English',
  'hi': 'हिन्दी',
  'pa': 'ਪੰਜਾਬੀ',
};

const Map<String, String> kLocaleFlags = {
  'en': '🇬🇧',
  'hi': '🇮🇳',
  'pa': '🌾',
};

const String _kPrefKey = 'app_locale_language_code';

class LanguageProvider extends ChangeNotifier {
  Locale _locale = const Locale('en');

  Locale get locale       => _locale;
  String get languageCode => _locale.languageCode;
  String get displayName  => kLocaleDisplayNames[_locale.languageCode] ?? 'English';
  String get flag         => kLocaleFlags[_locale.languageCode] ?? '🌐';

  Future<void> loadSavedLocale() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getString(_kPrefKey);
    if (saved != null && kSupportedLocales.any((l) => l.languageCode == saved)) {
      _locale = Locale(saved);
      notifyListeners();
    }
  }

  Future<void> setLocale(Locale locale) async {
    if (_locale == locale) return;
    _locale = locale;
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_kPrefKey, locale.languageCode);
  }
}
