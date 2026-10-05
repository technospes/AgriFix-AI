// ignore_for_file: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import 'package:go_router/go_router.dart';                  // AR nav
import '../../core/providers/diagnosis_provider.dart';
import '../../core/providers/language_provider.dart';
import '../../core/router.dart';                      // AppRoutes
import '../../core/providers/agent_session_provider.dart';
// ═══════════════════════════════════════════════════════════════════════════
// SolutionScreen  —  "AI Repair Guide"  (Text Solution Scene)
//
// NEW in this version:
//   • Prev/Next row pinned ABOVE the AR button (not inside scroll)
//   • Voice button: tap → speaks current step body (TTS)
//                   tap again → stops speech
//                   tap again → speaks current step again
//   • Voice icon changes: volume_up (unmuted/speaking) ↔ volume_off (muted)
//   • Auto-speaks when step changes while already speaking
// ═══════════════════════════════════════════════════════════════════════════
class SolutionScreen extends StatefulWidget {
  const SolutionScreen({super.key});

  @override
  State<SolutionScreen> createState() => _SolutionScreenState();
}

class _SolutionScreenState extends State<SolutionScreen>
    with TickerProviderStateMixin {
  int _currentStep = 0;
  final ScrollController _scroll = ScrollController();
  final List<GlobalKey> _stepKeys = [];

  // Entrance animation replays every time active step changes
  late final AnimationController _entranceCtrl;

  // ── TTS ──────────────────────────────────────────────────────────────
  final FlutterTts _tts = FlutterTts();
  bool _isSpeaking = false;   // true while TTS engine is playing

  @override
  void initState() {
    super.initState();
    _entranceCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 320),
    )..forward();

    _initTts();
  }

  // ── TTS language helpers ──────────────────────────────────────────────

  /// Maps app locale code → BCP-47 TTS tag supported by flutter_tts.
  static String _ttsLangFor(String code) {
    switch (code) {
      case 'hi': return 'hi-IN';
      case 'pa': return 'pa-IN';
      default:   return 'en-US';
    }
  }

  /// Registers callbacks and sets initial TTS parameters.
  /// Language is NOT set here — it is set in _applyTtsLanguage() which is
  /// called both from initState (via addPostFrameCallback so context is ready)
  /// and from didChangeDependencies on every locale change.
  Future<void> _initTts() async {
    await _tts.setSpeechRate(0.48);   // slightly slower — easier for farmers
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);

    _tts.setStartHandler(()      { if (mounted) setState(() => _isSpeaking = true);  });
    _tts.setCompletionHandler(() { if (mounted) setState(() => _isSpeaking = false); });
    _tts.setCancelHandler(()    { if (mounted) setState(() => _isSpeaking = false); });
    _tts.setErrorHandler((_)    { if (mounted) setState(() => _isSpeaking = false); });
  }

  /// Flushes the cached voice engine then applies the new language.
  ///
  /// flutter_tts keeps the previous TTS engine loaded in memory. On many
  /// Android OEM builds (Samsung, Xiaomi, Oppo) calling setLanguage() without
  /// stop() first is silently ignored — the old language keeps playing.
  /// stop() forces a full engine reset; setLanguage() then loads the correct
  /// locale from scratch.
  Future<void> _applyTtsLanguage(String langCode) async {
    await _tts.stop();
    await _tts.setLanguage(_ttsLangFor(langCode));
  }

  // Called on first frame AND whenever LanguageProvider emits a locale change.
  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final langCode = context.read<LanguageProvider>().languageCode;
    _applyTtsLanguage(langCode);   // async — fire-and-forget is safe here
  }

  /// Speak a string, stopping any current speech first.
  Future<void> _speak(String text) async {
    await _tts.stop();
    await _tts.speak(text);
  }

  /// Toggle: if currently speaking → stop; if muted → speak current step.
  Future<void> _toggleVoice(List<StepData> steps) async {
    HapticFeedback.lightImpact();
    
    if (_isSpeaking) {
      // 🔥 OPTIMISTIC UPDATE: Force UI to false instantly
      setState(() => _isSpeaking = false); 
      await _tts.stop();
    } else {
      if (_currentStep < steps.length) {
        // 🔥 OPTIMISTIC UPDATE: Force UI to true instantly
        setState(() => _isSpeaking = true); 
        final text = _buildSpeakText(steps[_currentStep], _currentStep + 1);
        await _tts.stop(); // Clear any stuck queues
        await _tts.speak(text);
      }
    }
  }
  /// Full spoken text for a step: "Step 1. [title]. [body]"
  String _buildSpeakText(StepData step, int stepNum) {
    final title = _stepTitle(step);
    final body  = _stepBody(step);
    return 'Step $stepNum. $title. $body';
  }

  @override
  void dispose() {
    _tts.stop();
    _scroll.dispose();
    _entranceCtrl.dispose();
    super.dispose();
  }

  // ── Data helpers ──────────────────────────────────────────────────────
  List<StepData> _stepsFromProvider(DiagnosisProvider prov) =>
      prov.solution?.steps ?? [];

  String _problemTitleFromProvider(DiagnosisProvider prov) {
    final isHindi = context.read<LanguageProvider>().languageCode == 'hi';
    return prov.problemDescription ??
        prov.solution!.getLocalizedProblem(isHindi);
  }

  String _stepBody(StepData step) {
    // Read locale from the tree — context is available because _stepBody is
    // only ever called from build() or helpers called from build().
    final isHindi = context.read<LanguageProvider>().languageCode == 'hi';
    // getLocalizedText: returns textHi when isHindi && textHi non-empty,
    // otherwise falls back to textEn, then text — matching StepData contract.
    return step.getLocalizedText(isHindi);
  }

  String _stepTitle(StepData step) {
    final isHindi = context.read<LanguageProvider>().languageCode == 'hi';

    // 1. Prefer the server-supplied localized title
    final serverTitle = step.getLocalizedTitle(isHindi);
    if (serverTitle.isNotEmpty) return serverTitle;
    if (step.action.isNotEmpty) {
      return step.displayAction;
    }
    
    // 4. Last resort: truncate the step body
    final body = _stepBody(step);
    if (body.isEmpty) return isHindi ? 'निरीक्षण करें' : 'Perform inspection';
    final cut = body.lastIndexOf(' ', 44);
    return cut > 10 ? '${body.substring(0, cut)}…'
                    : '${body.substring(0, body.length.clamp(0, 44))}…';
  }

  String _stepEmoji(StepData step) {
    const m = {
      'ignition_key': '🔑', 'battery_terminal': '🔋', 'fuel_cap': '⛽',
      'fan_belt': '⚙️',     'clutch_pedal': '🦶',      'clutch_cable': '🔗',
      'gear_lever': '🎚️',  'air_filter': '💨',          'spark_plug': '⚡',
      'radiator_cap': '🌡️','engine_oil_dipstick': '🛢️','wiring_harness': '🔌',
      'hydraulic_pump': '💧','fuel_filter': '🔧',        'drain_plug': '📷',
    };
    return m[step.visualCue ?? ''] ?? '🔧';
  }

  // ── i18n helper ───────────────────────────────────────────────────────────
  void _jumpTo(int index, List<StepData> steps) {
    if (index < 0 || index >= steps.length) return;
    HapticFeedback.lightImpact();
    setState(() => _currentStep = index);
    _entranceCtrl..reset()..forward();

    // If voice was active, auto-read the new step
    if (_isSpeaking) {
      _speak(_buildSpeakText(steps[index], index + 1));
    }

    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (index < _stepKeys.length) {
        final ctx = _stepKeys[index].currentContext;
        if (ctx != null) {
          Scrollable.ensureVisible(ctx,
              duration: const Duration(milliseconds: 420),
              curve: Curves.easeOutCubic,
              alignment: 0.08);
        }
      }
    });
  }

  // ── Build ─────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    final prov    = context.watch<DiagnosisProvider>();
    final steps   = _stepsFromProvider(prov);
    final problem = _problemTitleFromProvider(prov);
    final isHindi = context.watch<LanguageProvider>().languageCode == 'hi';

    while (_stepKeys.length < steps.length) _stepKeys.add(GlobalKey());

    final bottomPad = MediaQuery.of(context).padding.bottom;
    // Height of the fixed bottom bar: PrevNext(56) + gap(12) + AR(64) + gap(20) + safeArea
    final fixedBarH = 56 + 12 + 64 + 20 + bottomPad;

    return Scaffold(
      backgroundColor: _SC.bg,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // ── Scrollable content ─────────────────────────────────────
          SafeArea(
            bottom: false,
            child: SingleChildScrollView(
              controller: _scroll,
              padding: EdgeInsets.only(
                left: _SS.outer, right: _SS.outer,
                bottom: fixedBarH + 8,
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 8),

                  // Header  (voice + share live here)
                  _HeaderRow(
                    isHindi: isHindi,
                    isSpeaking: _isSpeaking,
                    onBack:  () => Navigator.of(context).pop(),
                    onVoice: () => _toggleVoice(steps),
                  ).animate().fadeIn(duration: 280.ms),

                  const SizedBox(height: 20),

                  _ProgressBars(
                    total: steps.isEmpty ? 4 : steps.length,
                    current: _currentStep,
                  ).animate().fadeIn(duration: 340.ms, delay: 40.ms),

                  const SizedBox(height: 20),

                  _SolutionCard(problem: problem, isHindi: isHindi,
                      cacheHit: prov.solution?.cacheHit ?? false)
                      .animate()
                      .fadeIn(duration: 380.ms, delay: 80.ms)
                      .slideY(begin: 0.06, end: 0,
                              duration: 380.ms, delay: 80.ms,
                              curve: Curves.easeOutCubic),

                  // ── Unsafe scene warning (non-blocking) ───────────────
                  // Shown only when backend detected unsafe-scene evidence
                  // from existing frame/transcript analysis (no extra Gemini
                  // call). Farmer sees this before reading repair steps.
                  if (prov.unsafeSceneSuspected) ...[
                    const SizedBox(height: 12),
                    _UnsafeSceneCard(
                      message: prov.unsafeSceneMessage,
                      isHindi: isHindi,
                    ).animate().fadeIn(duration: 340.ms, delay: 120.ms),
                  ],

                  const SizedBox(height: 20),

                                  if (steps.isEmpty)
                    if (prov.solution?.status == 'escalate')
                      _EscalationCard(
                        isHindi: isHindi,
                        problem: prov.solution?.problemIdentifiedEn ?? problem,
                        safetyWarnings: prov.solution?.safetyWarnings ?? [],
                      )
                    else
                      _EmptyState(isHindi: isHindi)
                  else
                    ...List.generate(steps.length, (i) {
                      return Padding(
                        key: _stepKeys[i],
                        padding: const EdgeInsets.only(bottom: 16),
                        child: _StepCard(
                          stepNumber:  i + 1,
                          title:       _stepTitle(steps[i]),
                          emoji:       _stepEmoji(steps[i]),
                          body:        _stepBody(steps[i]),
                          stepType:    steps[i].stepType,
                          isActive:    i == _currentStep,
                          isPast:      i < _currentStep,
                          isHindi:     isHindi,
                          entranceCtrl: i == _currentStep ? _entranceCtrl : null,
                        ),
                      );
                    }),

                  const SizedBox(height: 16),
                ],
              ),
            ),
          ),

          // ── Fixed bottom bar: PrevNext + AR button ─────────────────
          Positioned(
            left: 0, right: 0,
            bottom: 0,
            child: _FixedBottomBar(
              currentStep: _currentStep,
              totalSteps:  steps.length,
              bottomPad:   bottomPad,
              isHindi:     isHindi,
              onPrev: () => _jumpTo(_currentStep - 1, steps),
              onNext: () => _jumpTo(_currentStep + 1, steps),
              onAR: () async {
              final prov = context.read<DiagnosisProvider>();
              final agentProv = context.read<AgentSessionProvider>();
              final steps = prov.solution?.steps ?? [];
              
              // Convert StepData to raw dicts for the backend
              final diagnosisSteps = steps.map((s) => {
                  'step_id': s.stepId,
                  'action': s.textEn.isNotEmpty ? s.textEn : s.text,
                  'description': s.textEn.isNotEmpty ? s.textEn : s.text,
                  'required_part': s.requiredPart.isNotEmpty ? s.requiredPart : 'unknown',
                  'area_hint': s.areaHint.isNotEmpty ? s.areaHint : 'engine_compartment',
                  'step_type': s.stepType == StepType.visual ? 'verification' : s.stepType.name,
              }).toList();
              
              await agentProv.startSession(
                machineType: prov.solution?.machineType ?? 'water_pump',
                problemDescription: prov.problemDescription ?? '',
                language: context.read<LanguageProvider>().languageCode,
                diagnosisSteps: diagnosisSteps,
              );
              
              if (agentProv.sessionId != null && mounted) {
                context.push(AppRoutes.arGuide);  // no initialStep param
              }
            },
            )
            .animate()
            .slideY(begin: 1.0, end: 0,
                    duration: 480.ms, delay: 200.ms,
                    curve: Curves.easeOutCubic)
            .fadeIn(duration: 380.ms, delay: 200.ms),
          ),
        ],
      ),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// _FixedBottomBar  — PrevNext row + AR button, pinned at bottom
// ═══════════════════════════════════════════════════════════════════════════
class _FixedBottomBar extends StatelessWidget {
  final int currentStep;
  final int totalSteps;
  final double bottomPad;
  final bool isHindi;
  final VoidCallback onPrev;
  final VoidCallback onNext;
  final VoidCallback onAR;

  const _FixedBottomBar({
    required this.currentStep,
    required this.totalSteps,
    required this.bottomPad,
    required this.isHindi,
    required this.onPrev,
    required this.onNext,
    required this.onAR,
  });

  @override
  Widget build(BuildContext context) {
    final canPrev = currentStep > 0;
    final canNext = currentStep < totalSteps - 1;
    final lblPrev = isHindi ? 'पिछला' : 'Prev';
    final lblNext = isHindi ? 'अगला'  : 'Next';

    return Container(
      // Frosted white strip
      decoration: BoxDecoration(
        color: _SC.bg.withOpacity(0.96),
        border: const Border(
          top: BorderSide(color: Color(0xFFE5E7EB), width: 1),
        ),
      ),
      padding: EdgeInsets.fromLTRB(
        _SS.outer, 14, _SS.outer, 20 + bottomPad),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // ── Prev / Next ────────────────────────────────────────────
          if (totalSteps > 0) ...[
            Row(children: [
              // Prev
              Opacity(
                opacity: canPrev ? 1.0 : 0.28,
                child: GestureDetector(
                  onTap: canPrev ? onPrev : null,
                  behavior: HitTestBehavior.opaque,
                  child: Padding(
                    padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
                    child: Row(mainAxisSize: MainAxisSize.min, children: [
                      const Icon(Icons.arrow_back_rounded,
                          size: 18, color: Color(0xFF6B7280)),
                      const SizedBox(width: 5),
                      Text(lblPrev,
                        style: GoogleFonts.inter(
                          fontSize: 15, fontWeight: FontWeight.w500,
                          color: const Color(0xFF6B7280))),
                    ]),
                  ),
                ),
              ),

              const Spacer(),

              // Step counter in centre  "2 / 5"
              Text(
                totalSteps > 0 ? '${currentStep + 1} / $totalSteps' : '',
                style: GoogleFonts.inter(
                  fontSize: 13, fontWeight: FontWeight.w500,
                  color: const Color(0xFF9CA3AF)),
              ),

              const Spacer(),

              // Next pill
              AnimatedOpacity(
                duration: const Duration(milliseconds: 200),
                opacity: canNext ? 1.0 : 0.28,
                child: GestureDetector(
                  onTap: canNext ? onNext : null,
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 22, vertical: 11),
                    decoration: BoxDecoration(
                      color: const Color(0xFFF3F4F6),
                      borderRadius: BorderRadius.circular(24),
                    ),
                    child: Row(mainAxisSize: MainAxisSize.min, children: [
                      Text(lblNext,
                        style: GoogleFonts.inter(
                          fontSize: 15, fontWeight: FontWeight.w600,
                          color: _SC.textDark)),
                      const SizedBox(width: 5),
                      const Icon(Icons.arrow_forward_rounded,
                          size: 18, color: _SC.textDark),
                    ]),
                  ),
                ),
              ),
            ]),
            const SizedBox(height: 12),
          ],

          // ── AR button ──────────────────────────────────────────────
          _ARButton(onTap: onAR, isHindi: isHindi),
        ],
      ),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// _HeaderRow  —  back · title · voice toggle · share
// ═══════════════════════════════════════════════════════════════════════════
class _HeaderRow extends StatelessWidget {
  final bool isHindi;
  final bool isSpeaking;
  final VoidCallback onBack;
  final VoidCallback onVoice;

  const _HeaderRow({
    required this.isHindi,
    required this.isSpeaking,
    required this.onBack,
    required this.onVoice,
  });

  @override
  Widget build(BuildContext context) {
    final title = isHindi ? 'AI मरम्मत गाइड' : 'AI Repair Guide';
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        GestureDetector(
          onTap: onBack,
          behavior: HitTestBehavior.opaque,
          child: const Padding(
            padding: EdgeInsets.only(right: 8, top: 4, bottom: 4),
            child: Icon(Icons.arrow_back_rounded,
                size: 24, color: _SC.textDark),
          ),
        ),

        Expanded(
          child: Text(title,
            textAlign: TextAlign.center,
            style: GoogleFonts.inter(
              fontSize: 21, fontWeight: FontWeight.w600,
              color: _SC.textDark, letterSpacing: -0.3)),
        ),

        // ── Voice toggle button ──────────────────────────────────────
        GestureDetector(
          onTap: onVoice,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            width: 44, height: 44,
            decoration: BoxDecoration(
              // Speaking → solid green; muted → light green tint
              color: isSpeaking
                  ? _SC.primary
                  : const Color(0xFFE3F5EC),
              shape: BoxShape.circle,
            ),
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 180),
              child: Icon(
                isSpeaking
                    ? Icons.volume_up_rounded    // actively speaking
                    : Icons.volume_off_rounded,  // muted / stopped
                key: ValueKey(isSpeaking),
                size: 22,
                color: isSpeaking ? Colors.white : _SC.primary,
              ),
            ),
          ),
        ),

        const SizedBox(width: 10),

        GestureDetector(
          onTap: () {},
          child: const Icon(Icons.share_outlined,
              size: 20, color: Color(0xFF6B7280)),
        ),
      ],
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// _ProgressBars
// ═══════════════════════════════════════════════════════════════════════════
class _ProgressBars extends StatelessWidget {
  final int total;
  final int current;
  const _ProgressBars({required this.total, required this.current});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: List.generate(total, (i) {
        final active = i == current;
        final done   = i < current;
        return Expanded(
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 360),
            curve: Curves.easeInOut,
            margin: EdgeInsets.only(right: i < total - 1 ? 6 : 0),
            height: 5,
            decoration: BoxDecoration(
              color: active ? _SC.primary
                  : done   ? const Color(0xFF86D7A8)
                           : const Color(0xFFE2E4E8),
              borderRadius: BorderRadius.circular(10),
            ),
          ),
        );
      }),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// _SolutionCard  (warm beige)
// ═══════════════════════════════════════════════════════════════════════════
class _SolutionCard extends StatelessWidget {
  final String problem;
  final bool isHindi;
  final bool cacheHit;
  const _SolutionCard({required this.problem, required this.isHindi, this.cacheHit = false});

  @override
  Widget build(BuildContext context) {
    final label = isHindi ? 'समाधान मिला' : 'SOLUTION FOUND';
    return Container(
      padding: const EdgeInsets.symmetric(
          horizontal: _SS.cardPad, vertical: 16),
      decoration: BoxDecoration(
        color: const Color(0xFFF3E9DD),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFFE6D2B8), width: 1),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Container(
                width: 42, height: 42,
                decoration: const BoxDecoration(
                    color: Color(0xFFFF9F3F), shape: BoxShape.circle),
                child: const Icon(Icons.check_rounded,
                    color: Colors.white, size: 22),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(label,
                      style: GoogleFonts.inter(
                        fontSize: 11, fontWeight: FontWeight.w600,
                        color: const Color(0xFF8C8C8C), letterSpacing: 1.3)),
                    const SizedBox(height: 3),
                    Text(problem,
                      style: GoogleFonts.inter(
                        fontSize: 16, fontWeight: FontWeight.w600,
                        color: _SC.textDark, height: 1.3),
                      maxLines: 2, overflow: TextOverflow.ellipsis),
                  ],
                ),
              ),
            ],
          ),

          // Cache-hit badge — visible only when served from plan cache
          if (cacheHit) ...[
            const SizedBox(height: 10),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
              decoration: BoxDecoration(
                color: const Color(0xFFE6F6EC),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                    color: const Color(0xFF86D7A8).withOpacity(0.40), width: 1)),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.bolt_rounded,
                      size: 13, color: Color(0xFF22C55E)),
                  const SizedBox(width: 4),
                  Text(
                    isHindi ? 'त्वरित उत्तर (कैश)' : 'Instant result  ·  Cached plan',
                    style: GoogleFonts.inter(
                      fontSize: 11, fontWeight: FontWeight.w600,
                      color: const Color(0xFF22C55E))),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// _StepCard
// ═══════════════════════════════════════════════════════════════════════════
class _StepCard extends StatelessWidget {
  final int stepNumber;
  final String title;
  final String emoji;
  final String body;
  final StepType stepType;     // ← new
  final bool isActive;
  final bool isPast;
  final bool isHindi;
  final AnimationController? entranceCtrl;

  const _StepCard({
    required this.stepNumber,
    required this.title,
    required this.emoji,
    required this.body,
    required this.stepType,
    required this.isActive,
    required this.isPast,
    required this.isHindi,
    this.entranceCtrl,
  });

  // Step type icon + colour + label
  (IconData, Color, String) _typeInfo(bool isHindi) {
    switch (stepType) {
      case StepType.inspection:
        return (Icons.touch_app_rounded,   const Color(0xFFF59E0B),
                isHindi ? 'निरीक्षण'  : 'Inspection');
      case StepType.action:
        return (Icons.build_rounded,        const Color(0xFF6366F1),
                isHindi ? 'कार्य'      : 'Action');
      case StepType.observation:
        return (Icons.hearing_rounded,      const Color(0xFF0EA5E9),
                isHindi ? 'अवलोकन'   : 'Observation');
      case StepType.visual:
        return (Icons.camera_alt_rounded,   _SC.primary,
                isHindi ? 'दृश्य'     : 'Visual');
    }
  }

  @override
  Widget build(BuildContext context) {
    final lblStep    = isHindi ? 'चरण $stepNumber'    : 'Step $stepNumber';
    final lblCurrent = isHindi ? 'वर्तमान कार्य' : 'Current Task';
    final opacity = isActive ? 1.0 : (isPast ? 0.50 : 0.38);
    final (typeIcon, typeColor, typeLabel) = _typeInfo(isHindi);

    Widget card = AnimatedOpacity(
      duration: const Duration(milliseconds: 300),
      opacity: opacity,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeOutCubic,
        padding: const EdgeInsets.all(_SS.cardPad),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(24),
          boxShadow: isActive
              ? [BoxShadow(
                  color: Colors.black.withOpacity(0.05),
                  blurRadius: 24, offset: const Offset(0, 8))]
              : [],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: [
              // Step number badge
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
                decoration: BoxDecoration(
                  color: isActive
                      ? const Color(0xFFE6F6EC)
                      : const Color(0xFFF3F4F6),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Text(lblStep,
                  style: GoogleFonts.inter(
                    fontSize: 13, fontWeight: FontWeight.w600,
                    color: isActive ? _SC.primary : const Color(0xFF9CA3AF))),
              ),

              const SizedBox(width: 8),

              // Step type badge  ← NEW
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
                decoration: BoxDecoration(
                  color: typeColor.withOpacity(0.10),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: typeColor.withOpacity(0.25), width: 1),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(typeIcon, size: 11, color: typeColor),
                    const SizedBox(width: 4),
                    Text(typeLabel,
                      style: GoogleFonts.inter(
                        fontSize: 11, fontWeight: FontWeight.w600,
                        color: typeColor)),
                  ],
                ),
              ),

              const Spacer(),
              if (isActive) ...[
                Container(width: 7, height: 7,
                    decoration: const BoxDecoration(
                        color: _SC.primary, shape: BoxShape.circle)),
                const SizedBox(width: 5),
                Text(lblCurrent,
                  style: GoogleFonts.inter(
                    fontSize: 13, fontWeight: FontWeight.w500,
                    color: const Color(0xFF86D7A8))),
              ],
            ]),

            const SizedBox(height: 14),

            Text('$title $emoji',
              style: GoogleFonts.inter(
                fontSize: 22, fontWeight: FontWeight.w700,
                color: _SC.textDark, height: 1.3)),

            const SizedBox(height: 16),

            Container(
              padding: const EdgeInsets.symmetric(
                  horizontal: 14, vertical: 13),
              decoration: BoxDecoration(
                color: const Color(0xFFF2F4F7),
                borderRadius: BorderRadius.circular(18),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    width: 26, height: 26,
                    margin: const EdgeInsets.only(top: 1),
                    decoration: BoxDecoration(
                      color: isActive
                          ? const Color(0xFFFF9F3F)
                          : const Color(0xFFD1D5DB),
                      shape: BoxShape.circle,
                    ),
                    child: Icon(Icons.info_outline_rounded,
                      size: 15,
                      color: isActive
                          ? Colors.white
                          : const Color(0xFF9CA3AF)),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(body,
                      style: GoogleFonts.inter(
                        fontSize: 15, fontWeight: FontWeight.w400,
                        color: const Color(0xFF6B7280), height: 1.55)),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );

    if (isActive && entranceCtrl != null) {
      card = AnimatedBuilder(
        animation: entranceCtrl!,
        builder: (_, child) {
          final t = CurvedAnimation(
              parent: entranceCtrl!, curve: Curves.easeOutCubic).value;
          return Opacity(
            opacity: t,
            child: Transform.translate(
                offset: Offset(0, 14 * (1 - t)), child: child),
          );
        },
        child: card,
      );
    }
    return card;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// _ARButton
// ═══════════════════════════════════════════════════════════════════════════
class _ARButton extends StatefulWidget {
  final VoidCallback onTap;
  final bool isHindi;
  const _ARButton({required this.onTap, required this.isHindi});

  @override
  State<_ARButton> createState() => _ARButtonState();
}

class _ARButtonState extends State<_ARButton> {
  bool _pressed = false;

  @override
  Widget build(BuildContext context) {
    final label = widget.isHindi ? 'AR गाइड शुरू करें' : 'Start AR Guide';
    return GestureDetector(
      onTapDown: (_) => setState(() => _pressed = true),
      onTapUp:   (_) { setState(() => _pressed = false); widget.onTap(); },
      onTapCancel: () => setState(() => _pressed = false),
      child: AnimatedScale(
        scale: _pressed ? 0.97 : 1.0,
        duration: const Duration(milliseconds: 90),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 140),
          width: double.infinity,
          height: 64,
          decoration: BoxDecoration(
            color: _pressed ? const Color(0xFF18A34A) : _SC.primary,
            borderRadius: BorderRadius.circular(28),
            boxShadow: _pressed ? [] : [
              BoxShadow(
                color: _SC.primary.withOpacity(0.30),
                blurRadius: 24, offset: const Offset(0, 10)),
            ],
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.camera_alt_rounded,
                  color: Colors.white, size: 22),
              const SizedBox(width: 10),
              Text(label,
                style: GoogleFonts.inter(
                  fontSize: 18, fontWeight: FontWeight.w600,
                  color: Colors.white, letterSpacing: 0.15)),
            ],
          ),
        ),
      ),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// _EmptyState
// ═══════════════════════════════════════════════════════════════════════════
class _EmptyState extends StatelessWidget {
  final bool isHindi;
  const _EmptyState({required this.isHindi});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(
          color: Colors.white, borderRadius: BorderRadius.circular(24)),
      child: Column(children: [
        const Icon(Icons.build_circle_outlined,
            size: 48, color: Color(0xFFD1D5DB)),
        const SizedBox(height: 16),
        Text(isHindi ? 'मरम्मत के चरण उपलब्ध नहीं' : 'No repair steps available',
          style: GoogleFonts.inter(
            fontSize: 16, fontWeight: FontWeight.w600,
            color: const Color(0xFF6B7280))),
        const SizedBox(height: 8),
        Text(
          isHindi
              ? 'निदान पूर्ण हुआ लेकिन कोई गाइड नहीं बनी।\nकृपया दोबारा अपलोड करें।'
              : 'The diagnosis completed but no guide was generated.\nPlease try uploading again.',
          textAlign: TextAlign.center,
          style: GoogleFonts.inter(
            fontSize: 14, color: const Color(0xFF9CA3AF), height: 1.55)),
      ]),
    );
  }
}
class _EscalationCard extends StatelessWidget {
  final bool isHindi;
  final String problem;
  final List<String> safetyWarnings;
  
  const _EscalationCard({
    required this.isHindi,
    required this.problem,
    required this.safetyWarnings,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: const Color(0xFFFFFBEB),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: const Color(0xFFF59E0B).withOpacity(0.5)),
      ),
      child: Column(
        children: [
          const Icon(Icons.engineering_rounded, size: 48, color: Color(0xFFF59E0B)),
          const SizedBox(height: 16),
          Text(
            isHindi ? 'प्रमाणित मैकेनिक आवश्यक' : 'Professional Mechanic Required',
            style: GoogleFonts.inter(
              fontSize: 18, fontWeight: FontWeight.w700,
              color: const Color(0xFF92400E)),
          ),
          const SizedBox(height: 12),
          Text(
            problem,
            textAlign: TextAlign.center,
            style: GoogleFonts.inter(
              fontSize: 14, color: const Color(0xFF92400E), height: 1.5),
          ),
                    if (safetyWarnings.isNotEmpty) ...[
            const SizedBox(height: 16),
            for (final w in safetyWarnings)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Icon(Icons.warning_amber_rounded, size: 16, color: Color(0xFFF59E0B)),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        w,
                        style: GoogleFonts.inter(
                          fontSize: 13,
                          color: const Color(0xFF92400E),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
          ],
          const SizedBox(height: 20),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: () => Navigator.of(context).pop(),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFFF59E0B),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 14),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
              ),
              child: Text(
                isHindi ? 'दूसरी समस्या आज़माएं' : 'Try Another Problem',
                style: GoogleFonts.inter(fontSize: 16, fontWeight: FontWeight.w600),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
// ═══════════════════════════════════════════════════════════════════════════
// Design tokens
// ═══════════════════════════════════════════════════════════════════════════
abstract class _SC {
  static const bg      = Color(0xFFF5F6F7);
  static const primary = Color(0xFF22C55E);
  static const textDark = Color(0xFF111827);
}

abstract class _SS {
  static const double outer   = 24;
  static const double cardPad = 20;
}

// ═══════════════════════════════════════════════════════════════════════════
// _UnsafeSceneCard  — amber warning shown when backend flags unsafe scene
// ═══════════════════════════════════════════════════════════════════════════
//
// Displayed between _SolutionCard and the step list on the solution screen.
// Non-blocking: the farmer can still read steps and start AR — the card
// just makes the warning visible before they begin physical inspection.
// Derived from backend pipeline, no extra Gemini call.
class _UnsafeSceneCard extends StatelessWidget {
  final String message;
  final bool   isHindi;
  const _UnsafeSceneCard({required this.message, required this.isHindi});

  @override
  Widget build(BuildContext context) {
    final text = message.isNotEmpty
        ? message
        : (isHindi
            ? 'असुरक्षित दृश्य संदिग्ध — आगे बढ़ने से पहले दोबारा जांचें कि मशीन पूरी तरह बंद है।'
            : 'Unsafe scene suspected. Recheck that the machine is fully stopped before proceeding.');

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: const Color(0xFFFFFBEB),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFF59E0B).withOpacity(0.50), width: 1.2),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(
            padding: EdgeInsets.only(top: 1),
            child: Icon(Icons.warning_amber_rounded,
                color: Color(0xFFF59E0B), size: 20),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  isHindi ? 'सावधान' : 'CAUTION',
                  style: GoogleFonts.inter(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 1.0,
                    color: const Color(0xFFD97706),
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  text,
                  style: GoogleFonts.inter(
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                    color: const Color(0xFF92400E),
                    height: 1.45,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}