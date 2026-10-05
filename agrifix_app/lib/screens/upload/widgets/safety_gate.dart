// lib/screens/upload/widgets/safety_gate.dart
// ignore_for_file: deprecated_member_use
//
// SafetyGateSheet — mandatory pre-diagnosis safety confirmation.
//
// Architecture:
//   • Full-screen custom PageRoute — fixes half-screen blur bug (v1)
//   • Route owns the single slide animation — fixes double-slide jitter (v1)
//   • Self-contained FlutterTts — fires automatically after slide-in completes
//     so the user hears the checklist read aloud without any extra caller code
//
// TTS behaviour:
//   • Speaks an intro line, then each of the 4 checklist items in sequence
//   • Starts AFTER the slide-in animation finishes (380 ms) so audio never
//     overlaps with the mechanical slide sound
//   • Stops immediately when the user taps Continue or Cancel
//   • Language matches the languageCode passed to showSafetyGate()
//   • No changes required in upload_screen.dart or ar_guide_screen.dart

import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:google_fonts/google_fonts.dart';

// ── Public entry point ────────────────────────────────────────────────────────
//
// Returns true  → user confirmed all 4 safety checks, proceed.
// Returns false → user cancelled / pressed back, do not proceed.
//
// Caller code is unchanged from v1 — no new parameters needed.
//
// Usage:
//   final ok = await showSafetyGate(context, languageCode: 'hi');
//   if (!ok) return;

Future<bool> showSafetyGate(
  BuildContext context, {
  String languageCode = 'en',
}) async {
  final result = await Navigator.of(context).push<bool>(
    _SafetyGateRoute(languageCode: languageCode),
  );
  return result == true;
}

// ── Custom full-screen transparent page route ─────────────────────────────────

class _SafetyGateRoute extends PageRoute<bool> {
  final String languageCode;

  _SafetyGateRoute({required this.languageCode})
      : super(settings: const RouteSettings(name: '/safety_gate'));

  @override bool get opaque           => false;
  @override bool get barrierDismissible => false;
  @override bool get maintainState    => true;
  @override Color? get barrierColor   => null;
  @override String? get barrierLabel  => null;

  @override
  Duration get transitionDuration => const Duration(milliseconds: 380);

  @override
  Duration get reverseTransitionDuration => const Duration(milliseconds: 280);

  @override
  Widget buildPage(
    BuildContext context,
    Animation<double> animation,
    Animation<double> secondaryAnimation,
  ) {
    // Pass the route animation into the overlay so the sheet can wait for
    // slide-in to complete before starting TTS.
    return _SafetyGateOverlay(
      languageCode:   languageCode,
      routeAnimation: animation,
    );
  }

  @override
  Widget buildTransitions(
    BuildContext context,
    Animation<double> animation,
    Animation<double> secondaryAnimation,
    Widget child,
  ) {
    final curved = CurvedAnimation(
      parent:       animation,
      curve:        Curves.easeOutCubic,
      reverseCurve: Curves.easeInCubic,
    );
    return FadeTransition(
      opacity: curved,
      child: SlideTransition(
        position: Tween<Offset>(
          begin: const Offset(0, 0.12),
          end:   Offset.zero,
        ).animate(curved),
        child: child,
      ),
    );
  }
}

// ── Full-screen overlay ───────────────────────────────────────────────────────

class _SafetyGateOverlay extends StatelessWidget {
  final String            languageCode;
  final Animation<double> routeAnimation;

  const _SafetyGateOverlay({
    required this.languageCode,
    required this.routeAnimation,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.transparent,
      child: Stack(
        fit: StackFit.expand,
        children: [
          // Full-screen blur + scrim — covers every pixel
          BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
            child: Container(color: Colors.black.withOpacity(0.40)),
          ),

          // Sheet anchored to bottom — no animation of its own
          Align(
            alignment: Alignment.bottomCenter,
            child: _SafetyGateSheet(
              languageCode:   languageCode,
              routeAnimation: routeAnimation,
            ),
          ),
        ],
      ),
    );
  }
}

// ── Sheet content ─────────────────────────────────────────────────────────────

class _SafetyGateSheet extends StatefulWidget {
  final String            languageCode;
  final Animation<double> routeAnimation;

  const _SafetyGateSheet({
    required this.languageCode,
    required this.routeAnimation,
  });

  @override
  State<_SafetyGateSheet> createState() => _SafetyGateSheetState();
}

class _SafetyGateSheetState extends State<_SafetyGateSheet> {

  // ── State ─────────────────────────────────────────────────────────────────
  final List<bool> _checked  = [false, false, false, false];
  bool             _speaking = false;   // true while TTS engine is active

  // Self-contained TTS — this widget owns its own FlutterTts instance.
  // No dependency on TtsService (AR layer) — keeps this widget portable.
  final FlutterTts _tts = FlutterTts();

  bool get _allChecked => _checked.every((c) => c);
  bool get _isHindi    => widget.languageCode == 'hi';

  // ── BCP-47 language tag map ───────────────────────────────────────────────
  // Mirrors TtsService.langTagFor() without importing the AR service layer.
  static String _ttsTag(String code) {
    switch (code) {
      case 'hi': return 'hi-IN';
      case 'pa': return 'pa-IN';
      default:   return 'en-US';
    }
  }

  // ── Localised strings ─────────────────────────────────────────────────────

  String get _title       => _isHindi ? 'काम शुरू करने से पहले'               : 'Before You Begin';
  String get _subtitle    => _isHindi ? 'आगे बढ़ने के लिए सभी बातें जांचें:'  : 'Confirm all safety checks before proceeding:';
  String get _footer      => _isHindi ? 'चलती या चालू मशीन को कभी न छुएं।'   : 'Do not inspect moving or powered machinery.';
  String get _continueLbl => _isHindi ? 'पुष्टि करें और जारी रखें'            : 'Confirm & Continue';
  String get _cancelLbl   => _isHindi ? 'रद्द करें'                            : 'Cancel';

  // TTS intro spoken before the list — concise, imperative tone
  String get _ttsIntro => _isHindi
      ? 'आगे बढ़ने से पहले कृपया ये चार बातें जांचें:'
      : 'Before proceeding, please confirm these four safety checks:';

  // TTS outro spoken after all items — confirms what to do next
  String get _ttsOutro => _isHindi
      ? 'सभी बातें जांचने के बाद, "पुष्टि करें और जारी रखें" दबाएं।'
      : 'Once all checks are complete, tap Confirm and Continue.';

  List<({String label, String icon, String ttsLabel})> get _items => _isHindi
      ? [
          (label: 'मशीन पूरी तरह बंद है',               icon: '🛑', ttsLabel: 'एक: मशीन पूरी तरह बंद है।'),
          (label: 'इंजन / बिजली बंद है',                 icon: '⚡', ttsLabel: 'दो: इंजन और बिजली बंद है।'),
          (label: 'ब्रेक / अलगाव लगाया गया है',          icon: '🔒', ttsLabel: 'तीन: ब्रेक या अलगाव लगाया गया है।'),
          (label: 'क्षेत्र पास जाने के लिए सुरक्षित है', icon: '✅', ttsLabel: 'चार: क्षेत्र पास जाने के लिए सुरक्षित है।'),
        ]
      : [
          (label: 'Machine is fully stopped',  icon: '🛑', ttsLabel: 'One. Machine is fully stopped.'),
          (label: 'Engine / power is OFF',     icon: '⚡', ttsLabel: 'Two. Engine and power is OFF.'),
          (label: 'Brake / isolation applied', icon: '🔒', ttsLabel: 'Three. Brake or isolation is applied.'),
          (label: 'Area is safe to approach',  icon: '✅', ttsLabel: 'Four. Area is safe to approach.'),
        ];

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  @override
  void initState() {
    super.initState();
    _initTtsAndScheduleReadout();
  }

  @override
  void dispose() {
    _tts.stop();
    super.dispose();
  }

  // ── TTS setup + scheduled readout ─────────────────────────────────────────

  Future<void> _initTtsAndScheduleReadout() async {
    // Configure TTS — same rates used in TtsService across the app
    await _tts.setLanguage(_ttsTag(widget.languageCode));
    await _tts.setSpeechRate(0.48);
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);

    _tts.setStartHandler(()      { if (mounted) setState(() => _speaking = true);  });
    _tts.setCompletionHandler(() { if (mounted) setState(() => _speaking = false); });
    _tts.setCancelHandler(()    { if (mounted) setState(() => _speaking = false); });
    _tts.setErrorHandler((_)    { if (mounted) setState(() => _speaking = false); });

    // Wait for the route slide-in to finish before speaking.
    // routeAnimation goes from 0.0 → 1.0 over transitionDuration (380 ms).
    // Listening for status == completed fires exactly once when it reaches 1.0.
    // This prevents speech from starting while the mechanical slide is still
    // playing, which would feel jarring.
    void _onAnimationStatus(AnimationStatus status) {
      if (status == AnimationStatus.completed) {
        widget.routeAnimation.removeStatusListener(_onAnimationStatus);
        // Small extra gap so the sheet settles visually before speech starts
        Future.delayed(const Duration(milliseconds: 120), _speakChecklist);
      }
    }

    // If the route animation is already complete (e.g. hot-reload / test),
    // speak immediately rather than waiting forever.
    if (widget.routeAnimation.isCompleted) {
      Future.delayed(const Duration(milliseconds: 120), _speakChecklist);
    } else {
      widget.routeAnimation.addStatusListener(_onAnimationStatus);
    }
  }

  // ── Speak the full checklist ───────────────────────────────────────────────
  //
  // Strategy: queue a single concatenated string rather than four separate
  // speak() calls. flutter_tts handles sentence pauses via punctuation,
  // and this avoids any gap/click between items that multiple calls would
  // introduce on some Android TTS engines.

  Future<void> _speakChecklist() async {
    if (!mounted) return;

    final items    = _items.map((i) => i.ttsLabel).join('  ');
    final fullText = '$_ttsIntro  $items  $_ttsOutro';

    await _tts.stop();           // clear any pending queue
    await _tts.speak(fullText);
  }

  // ── Interaction handlers ───────────────────────────────────────────────────

  void _onToggle(int i) {
    HapticFeedback.lightImpact();
    setState(() => _checked[i] = !_checked[i]);
  }

  Future<void> _onContinue() async {
    if (!_allChecked) return;
    HapticFeedback.mediumImpact();
    await _tts.stop();
    if (mounted) Navigator.of(context).pop(true);
  }

  Future<void> _onCancel() async {
    await _tts.stop();
    if (mounted) Navigator.of(context).pop(false);
  }

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final bottomPad = MediaQuery.of(context).padding.bottom;

    return Container(
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
        boxShadow: [
          BoxShadow(
            color: Colors.black26,
            blurRadius: 32,
            offset: Offset(0, -6),
          ),
        ],
      ),
      padding: EdgeInsets.fromLTRB(24, 12, 24, 20 + bottomPad),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [

          // ── Drag handle + TTS indicator ──────────────────────────────────
          // Show a subtle animated dot when TTS is active so the user knows
          // something is being spoken — same visual language as SolutionScreen.
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                width: 36, height: 4,
                decoration: BoxDecoration(
                  color: const Color(0xFFD1D5DB),
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              if (_speaking) ...[
                const SizedBox(width: 10),
                const Icon(Icons.graphic_eq_rounded,
                    size: 16, color: Color(0xFF1E9E55)),
              ],
            ],
          ),
          const SizedBox(height: 20),

          // ── Header: warning badge + title + subtitle ─────────────────────
          Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Container(
                width: 48, height: 48,
                decoration: BoxDecoration(
                  color: const Color(0xFFFFF3E0),
                  shape: BoxShape.circle,
                  border: Border.all(
                    color: const Color(0xFFF59E0B).withOpacity(0.35),
                    width: 1.5,
                  ),
                ),
                child: const Center(
                  child: Text('⚠️', style: TextStyle(fontSize: 22)),
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      _title,
                      style: GoogleFonts.inter(
                        fontSize: 19,
                        fontWeight: FontWeight.w700,
                        color: const Color(0xFF111827),
                        height: 1.2,
                      ),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      _subtitle,
                      style: GoogleFonts.inter(
                        fontSize: 13,
                        color: const Color(0xFF6B7280),
                        height: 1.4,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),

          // ── Safety checklist card ────────────────────────────────────────
          Container(
            decoration: BoxDecoration(
              color: const Color(0xFFF9FAFB),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: const Color(0xFFE5E7EB), width: 1),
            ),
            child: Column(
              children: List.generate(_items.length, (i) {
                return _CheckRow(
                  icon:     _items[i].icon,
                  label:    _items[i].label,
                  checked:  _checked[i],
                  isLast:   i == _items.length - 1,
                  onToggle: () => _onToggle(i),
                );
              }),
            ),
          ),
          const SizedBox(height: 14),

          // ── Footer disclaimer ────────────────────────────────────────────
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Padding(
                padding: EdgeInsets.only(top: 1),
                child: Icon(Icons.info_outline_rounded,
                    size: 14, color: Color(0xFF9CA3AF)),
              ),
              const SizedBox(width: 6),
              Expanded(
                child: Text(
                  _footer,
                  style: GoogleFonts.inter(
                    fontSize: 12,
                    color: const Color(0xFF9CA3AF),
                    height: 1.4,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),

          // ── Continue button ──────────────────────────────────────────────
          AnimatedOpacity(
            duration: const Duration(milliseconds: 200),
            opacity: _allChecked ? 1.0 : 0.42,
            child: GestureDetector(
              onTap: _allChecked ? _onContinue : null,
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 180),
                width:  double.infinity,
                height: 54,
                decoration: BoxDecoration(
                  color: _allChecked
                      ? const Color(0xFF1E9E55)
                      : const Color(0xFF9CA3AF),
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: _allChecked
                      ? [
                          BoxShadow(
                            color: const Color(0xFF1E9E55).withOpacity(0.30),
                            blurRadius: 16,
                            offset: const Offset(0, 6),
                          ),
                        ]
                      : [],
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(Icons.shield_rounded,
                        color: Colors.white, size: 20),
                    const SizedBox(width: 8),
                    Text(
                      _continueLbl,
                      style: GoogleFonts.inter(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        color: Colors.white,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(height: 10),

          // ── Cancel link ──────────────────────────────────────────────────
          Center(
            child: GestureDetector(
              onTap: _onCancel,
              behavior: HitTestBehavior.opaque,
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 6),
                child: Text(
                  _cancelLbl,
                  style: GoogleFonts.inter(
                    fontSize: 14,
                    color: const Color(0xFF9CA3AF),
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ── Individual checklist row ──────────────────────────────────────────────────

class _CheckRow extends StatelessWidget {
  final String     icon;
  final String     label;
  final bool       checked;
  final bool       isLast;
  final VoidCallback onToggle;

  const _CheckRow({
    required this.icon,
    required this.label,
    required this.checked,
    required this.isLast,
    required this.onToggle,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onToggle,
      behavior: HitTestBehavior.opaque,
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            child: Row(
              children: [

                SizedBox(
                  width: 28,
                  child: Text(icon, style: const TextStyle(fontSize: 18)),
                ),
                const SizedBox(width: 10),

                Expanded(
                  child: Text(
                    label,
                    style: GoogleFonts.inter(
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                      color: checked
                          ? const Color(0xFF1F2937)
                          : const Color(0xFF4B5563),
                      height: 1.3,
                    ),
                  ),
                ),
                const SizedBox(width: 10),

                AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  width: 24, height: 24,
                  decoration: BoxDecoration(
                    color: checked
                        ? const Color(0xFF1E9E55)
                        : Colors.white,
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(
                      color: checked
                          ? const Color(0xFF1E9E55)
                          : const Color(0xFFD1D5DB),
                      width: 1.8,
                    ),
                  ),
                  child: checked
                      ? const Icon(Icons.check_rounded,
                          color: Colors.white, size: 15)
                      : null,
                ),
              ],
            ),
          ),
          if (!isLast)
            const Divider(height: 1, color: Color(0xFFE5E7EB), indent: 54),
        ],
      ),
    );
  }
}