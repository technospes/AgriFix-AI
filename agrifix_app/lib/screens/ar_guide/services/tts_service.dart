// lib/screens/ar_guide/services/tts_service.dart
//
// Single responsibility: own the FlutterTts instance and expose speak/stop/init.
// No guidance logic here — that belongs in ARController._speakGuidance().
import 'package:flutter_tts/flutter_tts.dart';

class TtsService {
  final FlutterTts _tts = FlutterTts();

  /// Maps app locale code → BCP-47 TTS tag.
  static String langTagFor(String code) {
    switch (code) {
      case 'hi': return 'hi-IN';
      case 'pa': return 'pa-IN';
      default:   return 'en-US';
    }
  }

  /// Initialise (or re-initialise) TTS for [langCode].
  ///
  /// Calls stop() first — flutter_tts caches the previous engine and
  /// setLanguage() alone is a no-op on several Android OEM builds.
  Future<void> init(
    String langCode, {
    required void Function() onComplete,
    required void Function() onCancel,
  }) async {
    await _tts.stop();
    await _tts.setLanguage(langTagFor(langCode));
    await _tts.setSpeechRate(0.48);
    _tts.setCompletionHandler(onComplete);
    _tts.setCancelHandler(onCancel);
  }

  Future<void> speak(String text) => _tts.speak(text);
  Future<void> stop()             => _tts.stop();
}
