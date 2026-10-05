// lib/screens/ar_guide/services/guidance_service.dart
//
// Predictive Guidance Engine — spoken + visual camera instructions.
//
// Phase 1 (pre-detection): rule-based hints from area_hint/requiredPart metadata.
//   Fires once per session so the farmer gets instant direction without spam.
// Phase 2 (bbox visible, conf ≥ 0.87): combined directional hints.
//   "Move camera down and right", "Move camera left", etc.
// Phase 3 (centred): "Good — hold camera steady."
//
// Enforces:
//   • TTS cooldown — 5 s between spoken instructions
//   • Deduplication — same phrase never spoken back-to-back
//   • Confidence gate — directional hint only when conf ≥ 0.87

import '../models/bbox.dart';
import 'tts_service.dart';

class GuidanceService {
  final TtsService _tts;

  /// Called when the visual guidance chip text should update (no cooldown).
  final void Function(String text) onVisualUpdate;

  /// Called when TTS active status changes.
  final void Function(bool active) onVoiceActiveChanged;

  // ── Thresholds (identical to monolith) ───────────────────────────────────
  static const kTtsCooldownMs         = 5000;
  static const kGuidanceConfThreshold = 0.87;
  static const _kBboxLeftThresh       = 0.35;
  static const _kBboxRightThresh      = 0.65;
  static const _kBboxTopThresh        = 0.35;
  static const _kBboxBottomThresh     = 0.65;

  // ── Internal state ────────────────────────────────────────────────────────
  DateTime? _lastGuidanceSpoke;
  String    _lastGuidanceText      = '';
  bool      _preDetectionHintFired = false;
  bool      _voiceActive           = false;
  bool      _isHindi               = false;  // set by controller each tick

  /// Called by ARController before every speak call so the correct
  /// language string is selected without threading isHindi through every call site.
  void setLanguage(String langCode) => _isHindi = langCode == 'hi';

  GuidanceService({
    required TtsService tts,
    required this.onVisualUpdate,
    required this.onVoiceActiveChanged,
  }) : _tts = tts;

  // ── Reset helpers ─────────────────────────────────────────────────────────

  void resetForNewSession() {
    _lastGuidanceText      = '';
    _preDetectionHintFired = false;
  }

  void resetOnDetectionLost() {
    _lastGuidanceText      = '';
    _preDetectionHintFired = false;
  }

  void resetOnStepChange() {
    _lastGuidanceText      = '';
    _preDetectionHintFired = false;
  }

  /// Clear dedup so the next speakGuidance call always fires (unique events).
  void bypassDedup() => _lastGuidanceText = '';

  void onTtsComplete() {
    _voiceActive = false;
    onVoiceActiveChanged(false);
  }

  // ── Phase 1: pre-detection hint (fires once per session) ─────────────────
  Future<void> speakPreDetectionHint(
      String requiredPart, String areaHint, {bool isHindi = false}) async {
    if (_preDetectionHintFired) return;
    final hint = _preDetectionHint(requiredPart, areaHint);
    if (hint == null) return;
    _preDetectionHintFired = true;
    await speakGuidance(hint.en, hint.hi, isHindi: isHindi);
  }

  // ── Phase 2: directional hint from bbox centre ────────────────────────────
  Future<void> speakDirectionalHint(NormBbox bbox, {bool isHindi = false}) async {
    final cx = bbox.cx;
    final cy = bbox.cy;
    final centredX = cx >= _kBboxLeftThresh && cx <= _kBboxRightThresh;
    final centredY = cy >= _kBboxTopThresh  && cy <= _kBboxBottomThresh;

    if (centredX && centredY) {
      await speakGuidance('Good — hold camera steady', 'अच्छा — कैमरा स्थिर रखें',
          isHindi: isHindi);
      return;
    }

    // Combined instruction — both axes simultaneously
    String hEn = '', hHi = '', vEn = '', vHi = '';
    if (!centredX) {
      if (cx < _kBboxLeftThresh) { hEn = 'right'; hHi = 'दाईं ओर'; }
      else                       { hEn = 'left';  hHi = 'बाईं ओर'; }
    }
    if (!centredY) {
      if (cy < _kBboxTopThresh) { vEn = 'down'; vHi = 'नीचे'; }
      else                      { vEn = 'up';   vHi = 'ऊपर';  }
    }

    final String en, hi;
    if (hEn.isNotEmpty && vEn.isNotEmpty) {
      en = 'Move camera $vEn and $hEn';   hi = 'कैमरा $vHi और $hHi ले जाएं';
    } else if (hEn.isNotEmpty) {
      en = 'Move camera slightly $hEn';  hi = 'कैमरा थोड़ा $hHi ले जाएं';
    } else {
      en = 'Move camera $vEn';           hi = 'कैमरा $vHi करें';
    }
    await speakGuidance(en, hi, isHindi: isHindi);
  }

  // ── Core speak: cooldown + dedup + visual chip + TTS ─────────────────────
  Future<void> speakGuidance(String en, String hi,
      {bool? isHindi}) async {
    // Use caller's override if provided, else fall back to field set by setLanguage()
    final text = (isHindi ?? _isHindi) ? hi : en;
    if (text.isEmpty) return;
    if (text == _lastGuidanceText) return;

    final now = DateTime.now();
    if (_lastGuidanceSpoke != null &&
        now.difference(_lastGuidanceSpoke!).inMilliseconds < kTtsCooldownMs) {
      return;
    }

    _lastGuidanceSpoke = now;
    _lastGuidanceText  = text;
    onVisualUpdate(text);

    if (!_voiceActive) {
      _voiceActive = true;
      onVoiceActiveChanged(true);
      await _tts.speak(text);
    }
  }

  // ── Pre-detection lookup table ────────────────────────────────────────────
  static ({String en, String hi})? _preDetectionHint(
      String part, String area) {
    const Map<String, ({String en, String hi})> areaMap = {
      'transmission_area': (
        en: 'Look near the floor under the steering wheel',
        hi: 'स्टीयरिंग व्हील के नीचे फर्श के पास देखें'
      ),
      'engine_compartment': (
        en: 'Open the hood and point camera at the engine',
        hi: 'बोनट खोलें और इंजन पर कैमरा करें'
      ),
      'undercarriage': (
        en: 'Point camera underneath the machine',
        hi: 'मशीन के नीचे कैमरा करें'
      ),
      'fuel_system': (
        en: 'Look near the fuel tank and fuel lines',
        hi: 'ईंधन टंकी और ईंधन पाइपों के पास देखें'
      ),
      'electrical_panel': (
        en: 'Look for the fuse box or wiring panel',
        hi: 'फ्यूज बॉक्स या वायरिंग पैनल ढूंढें'
      ),
      'hydraulic_system': (
        en: 'Look near the hydraulic pump and pipes',
        hi: 'हाइड्रोलिक पंप और पाइपों के पास देखें'
      ),
      'cooling_system': (
        en: 'Point camera at the radiator at the front',
        hi: 'सामने रेडिएटर पर कैमरा करें'
      ),
      'air_intake': (
        en: 'Look for the air filter box near the engine top',
        hi: 'इंजन के ऊपर एयर फिल्टर बॉक्स ढूंढें'
      ),
      'exhaust_system': (
        en: 'Look for the exhaust pipe at the side or back',
        hi: 'साइड या पीछे एग्जॉस्ट पाइप ढूंढें'
      ),
      'pto_area': (
        en: 'Look at the rear of the tractor near the PTO shaft',
        hi: 'ट्रैक्टर के पीछे पीटीओ शाफ्ट के पास देखें'
      ),
      'front_axle': (
        en: 'Point camera at the front wheels and axle',
        hi: 'अगले पहियों और एक्सल पर कैमरा करें'
      ),
      'rear_axle': (
        en: 'Point camera at the rear wheels and axle',
        hi: 'पिछले पहियों और एक्सल पर कैमरा करें'
      ),
    };
    const Map<String, ({String en, String hi})> partMap = {
      'clutch_pedal': (
        en: 'Look at the pedals on the left side of the floor',
        hi: 'फर्श के बाईं ओर पेडल देखें'
      ),
      'clutch_linkage': (
        en: 'Look underneath near where the pedal connects',
        hi: 'पेडल के नीचे लिंकेज ढूंढें'
      ),
      'fuel_filter': (
        en: 'Look for a small cylindrical filter near the fuel line',
        hi: 'ईंधन लाइन के पास छोटा बेलनाकार फ़िल्टर ढूंढें'
      ),
      'air_filter': (
        en: 'Look for a round filter box on top of the engine',
        hi: 'इंजन के ऊपर गोल फ़िल्टर बॉक्स ढूंढें'
      ),
      'battery': (
        en: 'Look for the battery near the engine, usually has thick cables',
        hi: 'इंजन के पास बैटरी ढूंढें — मोटी तारें होंगी'
      ),
      'radiator': (
        en: 'Point camera at the front grille of the machine',
        hi: 'मशीन के सामने ग्रिल पर कैमरा करें'
      ),
    };
    final partHint = partMap[part.toLowerCase().replaceAll(' ', '_')];
    if (partHint != null) return partHint;
    return areaMap[area.toLowerCase().replaceAll(' ', '_')];
  }
}