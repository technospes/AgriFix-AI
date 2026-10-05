// lib/screens/ar_guide/widgets/scanning_indicator.dart
// ignore_for_file: deprecated_member_use
//
// UI chrome widgets — everything except the AR camera overlay:
//   • TopBar                 — close + voice buttons + centre pill
//   • LocatingPill           — "Scanning for part…" spinner pill
//   • SpeakingPill           — "Speaking…" animated pill
//   • CircleBtn              — generic circle icon button
//   • ToastPositioned        — animated toast wrapper
//   • ToastCard              — toast content (all 6 kinds)
//   • NextPartButton         — green gradient "Next Part" CTA
//   • BottomPanel            — collapsible step info panel
//   • DangerPanel            — full-screen danger overlay
//   • StepBadge              — "STEP N" gold/green pill
//   • InstructionHeading     — rich-text step title with keyword highlight
//   • ProgressBar            — thin step-completion bar
//   • InspectionPanel        — inspection/action/observation bottom sheet
//   • PanelButton            — coloured answer/action button

import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import '../models/inspection_panel_model.dart';
import '../../../core/providers/diagnosis_provider.dart';
import '../../../core/providers/language_provider.dart';
import '../../../core/models/agent_models.dart';
import '../models/ar_state.dart';

// ── Top bar ───────────────────────────────────────────────────────────────
class TopBar extends StatelessWidget {
  final bool         voiceActive;
  final ARState      arState;
  final bool         isHindi;
  final VoidCallback onClose;
  final VoidCallback onVoice;
  const TopBar({
    required this.voiceActive, required this.arState,
    required this.isHindi,     required this.onClose, required this.onVoice,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      alignment: Alignment.center,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            CircleBtn(icon: Icons.close_rounded, onTap: onClose),
            CircleBtn(
              icon: voiceActive ? Icons.volume_up_rounded : Icons.volume_off_rounded,
              onTap: onVoice,
              activeDot: voiceActive,
            ),
          ],
        ),
        if (voiceActive)
          SpeakingPill(isHindi: isHindi)
        else if (arState == ARState.locating)
          LocatingPill(isHindi: isHindi),
      ],
    );
  }
}

// ── Locating pill ─────────────────────────────────────────────────────────
class LocatingPill extends StatelessWidget {
  final bool isHindi;
  const LocatingPill({required this.isHindi});

  @override
  Widget build(BuildContext context) {
    final label = isHindi ? 'भाग ढूंढ रहा है…' : 'Scanning for part…';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.60),
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: C.arrowBlue.withOpacity(0.35), width: 1)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: 14, height: 14,
            child: CircularProgressIndicator(
              strokeWidth: 2,
              valueColor: const AlwaysStoppedAnimation(C.arrowBlue),
            ),
          ),
          const SizedBox(width: 7),
          Text(label,
            style: GoogleFonts.inter(
              fontSize: 12, fontWeight: FontWeight.w500,
              color: const Color(0xFFEAEAEA))),
        ],
      ),
    );
  }
}

// ── Speaking pill ─────────────────────────────────────────────────────────
class SpeakingPill extends StatelessWidget {
  final bool isHindi;
  const SpeakingPill({required this.isHindi});

  @override
  Widget build(BuildContext context) {
    final label = isHindi ? 'बोल रहा है...' : 'Speaking...';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.60),
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: C.primary.withOpacity(0.35), width: 1)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.graphic_eq_rounded, color: C.primary, size: 16),
          const SizedBox(width: 6),
          Text(label,
            style: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w500, color: C.textSoft)),
        ],
      ),
    );
  }
}

// ── Circle button ─────────────────────────────────────────────────────────
class CircleBtn extends StatelessWidget {
  final IconData     icon;
  final VoidCallback onTap;
  final bool         activeDot;
  const CircleBtn({required this.icon, required this.onTap, this.activeDot = false});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Stack(
        clipBehavior: Clip.none,
        children: [
          Container(
            width: 44, height: 44,
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.50),
              shape: BoxShape.circle,
              border: Border.all(color: Colors.white.withOpacity(0.12), width: 1)),
            child: Icon(icon, color: Colors.white, size: 20),
          ),
          if (activeDot)
            Positioned(
              right: 1, bottom: 1,
              child: Container(
                width: 8, height: 8,
                decoration: const BoxDecoration(color: C.primary, shape: BoxShape.circle),
              ),
            ),
        ],
      ),
    );
  }
}

// ── Toast ─────────────────────────────────────────────────────────────────
class ToastPositioned extends StatelessWidget {
  final Animation<double>   slideAnim;
  final Animation<double>   fadeAnim;
  final ToastKind           kind;
  final AnimationController spinnerCtrl;
  final double              topOffset;
  final String              dynamicFeedback;
  final bool                isHindi;

  const ToastPositioned({
    required this.slideAnim, required this.fadeAnim,
    required this.kind,      required this.spinnerCtrl,
    required this.topOffset,
    this.dynamicFeedback = '', this.isHindi = false,
  });

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: topOffset, left: 16, right: 16,
      child: AnimatedBuilder(
        animation: Listenable.merge([slideAnim, fadeAnim]),
        builder: (_, child) => Opacity(
          opacity: fadeAnim.value,
          child: Transform.translate(offset: Offset(0, slideAnim.value), child: child),
        ),
        child: ToastCard(kind: kind, spinnerCtrl: spinnerCtrl,
            dynamicFeedback: dynamicFeedback, isHindi: isHindi),
      ),
    );
  }
}

class ToastCard extends StatelessWidget {
  final ToastKind           kind;
  final AnimationController spinnerCtrl;
  final String              dynamicFeedback;
  final bool                isHindi;
  const ToastCard({required this.kind, required this.spinnerCtrl,
      this.dynamicFeedback = '', this.isHindi = false});

  @override
  Widget build(BuildContext context) {
    final Color  accent;
    final Widget leadingIcon;
    final String message;

    switch (kind) {
      case ToastKind.analyzing:
        accent = C.primary;
        leadingIcon = SizedBox(
          width: 22, height: 22,
          child: AnimatedBuilder(
            animation: spinnerCtrl,
            builder: (_, __) => Transform.rotate(
              angle: spinnerCtrl.value * 2 * math.pi,
              child: const CircularProgressIndicator(
                strokeWidth: 2.5,
                valueColor: AlwaysStoppedAnimation(C.primary)),
            ),
          ),
        );
        message = isHindi ? 'AI के साथ संसाधित किया जा रहा है…'
                          : 'Processing with AI…'; 
        break;
      case ToastKind.sent:
        accent      = C.primary;
        leadingIcon = const Icon(Icons.check_circle_rounded, color: C.primary, size: 22);
        message = isHindi ? 'छवि भेजी गई — AI प्रतिक्रिया की प्रतीक्षा है…'
                          : 'Image sent successfully — awaiting AI response…';
        break;
      case ToastKind.analyzed:
        accent      = C.primary;
        leadingIcon = const Icon(Icons.check_circle_rounded, color: C.primary, size: 22);
        message = isHindi ? 'AI ने छवि का सफलतापूर्वक विश्लेषण किया'
                          : 'Image analyzed successfully by AI';
        break;
      case ToastKind.resultOk:
        accent      = C.primary;
        leadingIcon = const Icon(Icons.verified_rounded, color: C.primary, size: 22);
        message = isHindi ? 'सही भाग पहचाना गया — कोई खराबी नहीं ✓'
                          : 'Correct part identified — no damage detected ✓';
        break;
      case ToastKind.resultWarn:
        accent      = C.warning;
        leadingIcon = const Icon(Icons.warning_amber_rounded, color: C.warning, size: 22);
        message = dynamicFeedback.isNotEmpty
            ? dynamicFeedback
            : (isHindi ? 'छवि अस्पष्ट है — नीचे संकेत देखें'
                       : 'Image unclear or wrong part captured — see hint below for guidance');
        break;
      case ToastKind.error:
        accent      = C.danger;
        leadingIcon = const Icon(Icons.error_outline_rounded, color: C.danger, size: 22);
        message = isHindi
            ? 'कनेक्शन त्रुटि — इंटरनेट जांचें और दोबारा प्रयास करें'
            : 'Connection error — please check your internet and try again';
        break;

      case ToastKind.verifying:
        accent = C.arrowBlue;
        leadingIcon = SizedBox(
          width: 22, height: 22,
          child: AnimatedBuilder(
            animation: spinnerCtrl,
            builder: (_, __) => Transform.rotate(
              angle: spinnerCtrl.value * 2 * math.pi,
              child: const CircularProgressIndicator(
                strokeWidth: 2.5,
                valueColor: AlwaysStoppedAnimation(C.arrowBlue)),
            ),
          ),
        );
        message = isHindi ? 'भाग की पुष्टि हो रही है…' : 'Verifying part…';
        break;

      case ToastKind.inspecting:
        accent = C.gold;
        leadingIcon = SizedBox(
          width: 22, height: 22,
          child: AnimatedBuilder(
            animation: spinnerCtrl,
            builder: (_, __) => Transform.rotate(
              angle: spinnerCtrl.value * 2 * math.pi,
              child: const CircularProgressIndicator(
                strokeWidth: 2.5,
                valueColor: AlwaysStoppedAnimation(C.gold)),
            ),
          ),
        );
        message = isHindi ? 'क्षति का निरीक्षण हो रहा है…' : 'Inspecting for damage…';
        break;

    }

    return Container(
      decoration: BoxDecoration(
        color: C.toastBg,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: accent.withOpacity(0.28), width: 1),
        boxShadow: [
          BoxShadow(color: Colors.black.withOpacity(0.50),
              blurRadius: 20, offset: const Offset(0, 5)),
        ],
      ),
      child: IntrinsicHeight(
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Container(
              width: 5,
              decoration: BoxDecoration(
                color: accent,
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(16), bottomLeft: Radius.circular(16))),
            ),
            const SizedBox(width: 12),
            Center(child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 14),
              child: leadingIcon,
            )),
            const SizedBox(width: 10),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 2),
                child: Text(message,
                  softWrap: true,
                  overflow: TextOverflow.visible,
                  style: GoogleFonts.inter(
                    fontSize: 14, fontWeight: FontWeight.w500,
                    color: C.textSoft, height: 1.45, letterSpacing: 0.1)),
              ),
            ),
            const SizedBox(width: 12),
          ],
        ),
      ),
    );
  }
}

// ── Next Part button ──────────────────────────────────────────────────────
class NextPartButton extends StatefulWidget {
  final VoidCallback onTap;
  final bool         isHindi;
  const NextPartButton({required this.onTap, required this.isHindi});
  @override
  State<NextPartButton> createState() => _NextPartButtonState();
}

class _NextPartButtonState extends State<NextPartButton> {
  bool _pressed = false;
  @override
  Widget build(BuildContext context) {
    final label = widget.isHindi ? 'अगला भाग' : 'Next Part';
    return GestureDetector(
      onTapDown:   (_) => setState(() => _pressed = true),
      onTapUp:     (_) { setState(() => _pressed = false); widget.onTap(); },
      onTapCancel: () => setState(() => _pressed = false),
      child: AnimatedScale(
        scale: _pressed ? 0.97 : 1.0,
        duration: const Duration(milliseconds: 90),
        child: Container(
          height: 56,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(28),
            gradient: const LinearGradient(colors: [C.arrowGreen, C.primary]),
            boxShadow: [
              BoxShadow(color: C.primary.withOpacity(0.40),
                  blurRadius: 24, offset: const Offset(0, 10)),
            ],
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(label,
                style: GoogleFonts.inter(
                  fontSize: 17, fontWeight: FontWeight.w600, color: Colors.black87)),
              const SizedBox(width: 8),
              const Icon(Icons.chevron_right_rounded, color: Colors.black87, size: 22),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Bottom panel ──────────────────────────────────────────────────────────
// Completion badge copy depends on what was actually verified. Only a
// camera step ran an AI vision damage check — everything else (boolean
// confirmation, numeric measurement, multiple-choice observation) was a
// farmer self-report and must say so, not borrow camera-inspection language.
String _verifiedBadgeText(InteractionType? type, bool isHindi) {
  switch (type) {
    case InteractionType.camera:
      return isHindi ? 'घटक सत्यापित — कोई क्षति नहीं' : 'Component Verified — No Damage Detected';
    case InteractionType.number:
      return isHindi ? 'माप दर्ज किया गया' : 'Measurement Recorded';
    case InteractionType.choice:
      return isHindi ? 'उत्तर दर्ज किया गया' : 'Response Recorded';
    case InteractionType.boolean:
    case InteractionType.none:
    default:
      return isHindi ? 'चरण पूर्ण — पुष्टि की गई' : 'Step Confirmed Complete';
  }
}

class BottomPanel extends StatelessWidget {
  final StepData?       step;
  final NextStepDetail? nextStep;   // agent's real upcoming step — null only while loading
  final bool             isResolved; // true only when the agent actually returned status=resolved
  final int          stepIndex;
  final int          total;
  final ARState      arState;
  final bool         expanded;
  final double       screenH;
  final String       feedbackMsg;
  final VoidCallback onToggle;
  final bool         isHindi;
  // Which interaction produced the current 'verified' state — null while
  // loading. Drives the completion badge copy below; must never be assumed
  // to be camera just because a step was completed.
  final InteractionType? completedInteractionType;

  const BottomPanel({
    required this.step,       required this.nextStep,
    this.isResolved = false,
    required this.stepIndex,  required this.total,
    required this.arState,    required this.expanded,
    required this.screenH,    required this.onToggle,
    this.feedbackMsg = '',    this.isHindi = false,
    this.completedInteractionType,
  });

  @override
  Widget build(BuildContext context) {
    final isVerified = arState == ARState.verified;
    return AnimatedContainer(
      duration: const Duration(milliseconds: 340),
      curve: Curves.easeOutCubic,
      constraints: BoxConstraints(
        minHeight: kPanelHeight,
        maxHeight: expanded ? screenH * 0.60 : kPanelHeight,
      ),
      decoration: BoxDecoration(
        borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
        gradient: const LinearGradient(
          begin: Alignment.topCenter, end: Alignment.bottomCenter,
          colors: [C.panelBg1, C.panelBg2]),
        boxShadow: [
          BoxShadow(color: Colors.black.withOpacity(0.60),
              blurRadius: 40, offset: const Offset(0, -10)),
        ],
      ),
      child: SingleChildScrollView(
        physics: expanded
            ? const BouncingScrollPhysics()
            : const NeverScrollableScrollPhysics(),
        child: Padding(
          padding: EdgeInsets.fromLTRB(20, 14, 20,
              MediaQuery.of(context).padding.bottom + 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Container(
                  width: 36, height: 4,
                  margin: const EdgeInsets.only(bottom: 14),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.16),
                    borderRadius: BorderRadius.circular(2)),
                ),
              ),
              Row(children: [
                StepBadge(stepIndex: stepIndex, verified: isVerified),
                const SizedBox(width: 10),
                if (!isVerified)
                  Flexible(
                    child: Text(
                      (step?.displayPart ?? 'COMPONENT').toUpperCase(),
                      style: GoogleFonts.inter(
                        fontSize: 12, fontWeight: FontWeight.w600,
                        letterSpacing: 1.0, color: C.textMuted)),
                  )
                else ...[ 
                  Text(isHindi ? 'चरण ${stepIndex + 1}: पूर्ण' : 'STEP ${stepIndex + 1}: COMPLETE',
                    style: GoogleFonts.inter(
                      fontSize: 12, fontWeight: FontWeight.w700,
                      letterSpacing: 0.8, color: C.primary)),
                  const Spacer(),
                  if (total > 0)
                    Text('${stepIndex + 1} / $total',
                      style: GoogleFonts.inter(fontSize: 12, color: C.textMuted)),
                ],
              ]),
              const SizedBox(height: 10),
              if (isVerified) ...[
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: C.primary.withOpacity(0.14),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: C.primary.withOpacity(0.30), width: 1)),
                  child: Text(
                    _verifiedBadgeText(completedInteractionType, isHindi),
                    style: GoogleFonts.inter(
                      fontSize: 12, fontWeight: FontWeight.w600, color: C.primary)),
                ),
                const SizedBox(height: 10),
              ],
              GestureDetector(
                onTap: onToggle,
                child: InstructionHeading(
                    step: step, nextStep: nextStep,
                    verified: isVerified, isResolved: isResolved),
              ),
              if (arState == ARState.unclear) ...[
                const SizedBox(height: 10),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                  decoration: BoxDecoration(
                    color: C.warning.withOpacity(0.12),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: C.warning.withOpacity(0.40), width: 1)),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Padding(
                        padding: EdgeInsets.only(top: 2),
                        child: Icon(Icons.photo_camera_rounded, color: C.warning, size: 16)),
                      const SizedBox(width: 8),
                      Expanded(child: Text(
                        feedbackMsg.isNotEmpty ? feedbackMsg
                            : (isHindi
                                ? 'फोन स्थिर रखें और भाग फ्रेम में आए, फिर विश्लेषण करें।'
                                : 'Hold your phone steady and ensure the part fills the frame, then tap Analyze again.'),
                        style: GoogleFonts.inter(
                          fontSize: 13, fontWeight: FontWeight.w500,
                          color: C.warning, height: 1.45))),
                    ],
                  ),
                ),
              ],
              if (expanded && step != null) ...[
                const SizedBox(height: 12),
                Text(
                  step!.getLocalizedText(
                      context.watch<LanguageProvider>().languageCode == 'hi'),
                  style: GoogleFonts.inter(
                      fontSize: 14, color: C.textMuted, height: 1.55)),
                if (step!.safetyWarning != null) ...[
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    decoration: BoxDecoration(
                      color: C.warning.withOpacity(0.12),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: C.warning.withOpacity(0.35), width: 1)),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Padding(
                          padding: EdgeInsets.only(top: 2),
                          child: Icon(Icons.warning_amber_rounded, color: C.warning, size: 16)),
                        const SizedBox(width: 8),
                        Expanded(child: Text(step!.safetyWarning!,
                          style: GoogleFonts.inter(
                            fontSize: 13, color: C.warning,
                            fontWeight: FontWeight.w500, height: 1.45))),
                      ],
                    ),
                  ),
                ],
              ],
              if (isVerified) ...[
                const SizedBox(height: 14),
                ProgressBar(current: stepIndex + 1, total: total),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

// ── Danger panel ──────────────────────────────────────────────────────────
class DangerPanel extends StatelessWidget {
  final String       message;
  final bool         isHindi;
  final VoidCallback onDismiss;
  const DangerPanel({required this.message, required this.isHindi, required this.onDismiss});

  @override
  Widget build(BuildContext context) {
    final lblDanger  = isHindi ? '⚠ खतरा पता चला' : '⚠ DANGER DETECTED';
    final lblDismiss = isHindi ? 'समझ गया — सावधानी से जारी रखें' : 'I understand — Resume safely';
    return Container(
      color: const Color(0xF2180000),
      child: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(28),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.dangerous_rounded, color: Color(0xFFFF4B4B), size: 80),
              const SizedBox(height: 24),
              Text(lblDanger,
                style: GoogleFonts.inter(
                  fontSize: 26, fontWeight: FontWeight.w800,
                  color: const Color(0xFFFF4B4B), letterSpacing: 1.2)),
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
                  color: const Color(0x30FF4B4B),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: const Color(0xFFFF4B4B).withOpacity(0.50), width: 1.5)),
                child: Text(message,
                  textAlign: TextAlign.center,
                  style: GoogleFonts.inter(
                    fontSize: 16, fontWeight: FontWeight.w500,
                    color: Colors.white, height: 1.6)),
              ),
              const SizedBox(height: 36),
              GestureDetector(
                onTap: onDismiss,
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  decoration: BoxDecoration(
                    color: const Color(0xFF2A2A2A),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: Colors.white.withOpacity(0.15))),
                  child: Text(lblDismiss,
                    textAlign: TextAlign.center,
                    style: GoogleFonts.inter(
                      fontSize: 15, fontWeight: FontWeight.w600, color: Colors.white)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Step badge ────────────────────────────────────────────────────────────
class StepBadge extends StatelessWidget {
  final int  stepIndex;
  final bool verified;
  const StepBadge({required this.stepIndex, required this.verified});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: verified ? C.primary.withOpacity(0.16) : C.goldBadgeBg,
        borderRadius: BorderRadius.circular(20)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (verified) ...[
            const Icon(Icons.check_circle_rounded, color: C.primary, size: 13),
            const SizedBox(width: 5),
          ],
          Text('STEP ${stepIndex + 1}',
            style: GoogleFonts.inter(
              fontSize: 12, fontWeight: FontWeight.w700,
              letterSpacing: 1.0,
              color: verified ? C.primary : C.gold)),
        ],
      ),
    );
  }
}

// ── Instruction heading ───────────────────────────────────────────────────
class InstructionHeading extends StatelessWidget {
  final StepData?       step;
  final NextStepDetail? nextStep;   // agent's real upcoming step
  final bool            verified;
  final bool            isResolved; // true only when the agent actually finished
  const InstructionHeading({
    required this.step, required this.nextStep,
    required this.verified, this.isResolved = false,
  });

  @override
  Widget build(BuildContext context) {
    final isHindi = context.watch<LanguageProvider>().languageCode == 'hi';
    if (step == null) {
      return Text(isHindi ? 'घटक ढूंढें' : 'Locate the component',
        maxLines: 4, overflow: TextOverflow.ellipsis,
        style: GoogleFonts.inter(
          fontSize: 19, fontWeight: FontWeight.w700, color: C.textPrimary, height: 1.3));
    }
    final full    = verified ? _nextInstruction(isHindi) : step!.getLocalizedText(isHindi);
    final keyword = verified ? (nextStep?.displayPart ?? '').replaceAll('_', ' ')
                             : (step!.visualCue ?? '').replaceAll('_', ' ');

    if (keyword.isEmpty || !full.contains(keyword)) {
      return Text(full,
        maxLines: 4, overflow: TextOverflow.ellipsis,
        style: GoogleFonts.inter(
          fontSize: 19, fontWeight: FontWeight.w700, color: C.textPrimary, height: 1.3));
    }
    final parts = full.split(keyword);
    return RichText(
      maxLines: 4, overflow: TextOverflow.ellipsis,
      text: TextSpan(
        style: GoogleFonts.inter(
          fontSize: 19, fontWeight: FontWeight.w700, color: C.textPrimary, height: 1.3),
        children: [
          TextSpan(text: parts.first),
          TextSpan(text: keyword,
            style: TextStyle(
              color:           verified ? C.primary : C.danger,
              decoration:      TextDecoration.underline,
              decorationColor: verified ? C.primary : C.danger)),
          if (parts.length > 1) TextSpan(text: parts.sublist(1).join(keyword)),
        ],
      ),
    );
  }

  String _nextInstruction(bool isHindi) {
    if (isResolved) return isHindi ? 'सभी चरण पूर्ण!' : 'All steps complete!';
    if (nextStep == null) {
      return isHindi ? 'अगला चरण लोड हो रहा है…' : 'Loading next step…';
    }
    final vc   = nextStep!.displayPart;
    final body = nextStep!.localizedText(isHindi);
    if (vc.isNotEmpty) {
      final partName = vc.replaceAll('_', ' ');
      return isHindi ? 'शानदार। अब $partName ढूंढें।' : 'Excellent. Now locate the $partName.';
    }
    return isHindi ? 'शानदार। $body' : 'Excellent. $body';
  }
}

// ── Progress bar ──────────────────────────────────────────────────────────
class ProgressBar extends StatelessWidget {
  final int current, total;
  const ProgressBar({required this.current, required this.total});

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(4),
      child: LinearProgressIndicator(
        value: total > 0 ? current / total : 0.0,
        minHeight: 4,
        backgroundColor: Colors.white.withOpacity(0.10),
        valueColor: const AlwaysStoppedAnimation(C.primary),
      ),
    );
  }
}

// ── Inspection panel ──────────────────────────────────────────────────────
class InspectionPanel extends StatefulWidget {
  final InspectionPanelModel model;
  final bool      isHindi;
  final void Function(StepOption) onAnswer;
  final VoidCallback               onDone;
  final VoidCallback               onDismiss;
  const InspectionPanel({
    required this.model,
    required this.isHindi,
    required this.onAnswer,
    required this.onDone,
    required this.onDismiss,
  });

  @override
  State<InspectionPanel> createState() => _InspectionPanelState();
}

class _InspectionPanelState extends State<InspectionPanel>
    with SingleTickerProviderStateMixin {
  late final AnimationController _slideCtrl;
  late final Animation<Offset>   _slideAnim;

  @override
  void initState() {
    super.initState();
    _slideCtrl = AnimationController(vsync: this,
        duration: const Duration(milliseconds: 360));
    _slideAnim = Tween<Offset>(begin: const Offset(0, 1), end: Offset.zero)
        .animate(CurvedAnimation(parent: _slideCtrl, curve: Curves.easeOutCubic));
    _slideCtrl.forward();
  }

  @override
  void dispose() { _slideCtrl.dispose(); super.dispose(); }

  static Color _optionBg(String id, int idx) {
    if (id == 'c') return const Color(0x22FF4B4B);
    if (idx == 0)  return const Color(0x1A34D399);
    return const Color(0x14FFFFFF);
  }
  static Color _optionBorder(String id, int idx) {
    if (id == 'c') return const Color(0x60FF4B4B);
    if (idx == 0)  return const Color(0x5034D399);
    return const Color(0x30FFFFFF);
  }
  static Color _optionText(String id, int idx) {
    if (id == 'c') return const Color(0xFFFF7070);
    if (idx == 0)  return const Color(0xFF34D399);
    return const Color(0xFFEAEAEA);
  }

  @override
  Widget build(BuildContext context) {
    final step     = widget.model;
    final isHindi  = widget.isHindi;
    final isAction = step.showDoneButton;

    final (typeIcon, typeLabel) = switch (step.stepType) {
      StepType.inspection  => (Icons.touch_app_rounded, isHindi ? 'निरीक्षण' : 'INSPECTION'),
      StepType.action      => (Icons.build_rounded,     isHindi ? 'कार्य'     : 'ACTION'),
      StepType.observation => (Icons.hearing_rounded,   isHindi ? 'अवलोकन'   : 'OBSERVATION'),
      _                    => (Icons.camera_alt_rounded, isHindi ? 'दृश्य'    : 'VISUAL'),
    };

    return SlideTransition(
      position: _slideAnim,
      child: Container(
        constraints: BoxConstraints(maxHeight: MediaQuery.of(context).size.height * 0.72),
        decoration: BoxDecoration(
          borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
          gradient: const LinearGradient(
            begin: Alignment.topCenter, end: Alignment.bottomCenter,
            colors: [Color(0xFF1E1E1E), Color(0xFF111111)]),
          boxShadow: [
            BoxShadow(color: Colors.black.withOpacity(0.70),
                blurRadius: 40, offset: const Offset(0, -10)),
          ],
        ),
        child: SafeArea(
          top: false,
          child: SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(20, 0, 20, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Center(
                  child: Container(
                    width: 36, height: 4,
                    margin: const EdgeInsets.symmetric(vertical: 14),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.16),
                      borderRadius: BorderRadius.circular(2)),
                  ),
                ),
                Row(children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                    decoration: BoxDecoration(
                      color: const Color(0xFF2A2A2A),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(color: Colors.white.withOpacity(0.12), width: 1)),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(typeIcon, size: 13, color: C.gold),
                        const SizedBox(width: 5),
                        Text(typeLabel,
                          style: GoogleFonts.inter(
                            fontSize: 11, fontWeight: FontWeight.w700,
                            letterSpacing: 0.8, color: C.gold)),
                      ],
                    ),
                  ),
                  const Spacer(),
                  if (!isAction)
                    GestureDetector(
                      onTap: widget.onDismiss,
                      child: Container(
                        width: 32, height: 32,
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.06),
                          shape: BoxShape.circle),
                        child: const Icon(Icons.close_rounded, color: C.textMuted, size: 16)),
                    ),
                ]),
                const SizedBox(height: 14),
                Text(step.description,
                  style: GoogleFonts.inter(
                    fontSize: 15, color: C.textSoft,
                    fontWeight: FontWeight.w500, height: 1.5)),
                if (step.safetyWarning != null) ...[
                  const SizedBox(height: 10),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    decoration: BoxDecoration(
                      color: C.warning.withOpacity(0.12),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: C.warning.withOpacity(0.35), width: 1)),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Padding(
                          padding: EdgeInsets.only(top: 2),
                          child: Icon(Icons.warning_amber_rounded, color: C.warning, size: 15)),
                        const SizedBox(width: 8),
                        Expanded(child: Text(step.safetyWarning!,
                          style: GoogleFonts.inter(
                            fontSize: 12, color: C.warning,
                            fontWeight: FontWeight.w500, height: 1.4))),
                      ],
                    ),
                  ),
                ],
                const SizedBox(height: 16),
                if (isAction) ...[
                  PanelButton(
                    label:   isHindi ? '✓  कार्य पूर्ण हुआ' : '✓  Action Done',
                    bgColor: const Color(0x1A34D399),
                    border:  const Color(0x5034D399),
                    text:    const Color(0xFF34D399),
                    onTap:   widget.onDone,
                  ),
                ] else ...[
                  if (step.question != null && step.question!.isNotEmpty) ...[
                    Text(step.question!,
                      style: GoogleFonts.inter(
                        fontSize: 16, fontWeight: FontWeight.w700,
                        color: C.textPrimary, height: 1.3)),
                    const SizedBox(height: 12),
                  ],
                  ...List.generate(step.options.length, (i) {
                    final opt = step.options[i];
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 10),
                      child: PanelButton(
                        label:   opt.getLocalizedLabel(isHindi),
                        bgColor: _optionBg(opt.id, i),
                        border:  _optionBorder(opt.id, i),
                        text:    _optionText(opt.id, i),
                        onTap:   () => widget.onAnswer(opt),
                      ),
                    );
                  }),
                ],
                const SizedBox(height: 4),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ── Panel button ──────────────────────────────────────────────────────────
class PanelButton extends StatefulWidget {
  final String label;
  final Color  bgColor, border, text;
  final VoidCallback onTap;
  const PanelButton({
    required this.label, required this.bgColor,
    required this.border, required this.text, required this.onTap});
  @override
  State<PanelButton> createState() => _PanelButtonState();
}

class _PanelButtonState extends State<PanelButton> {
  bool _pressed = false;
  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown:   (_) => setState(() => _pressed = true),
      onTapUp:     (_) { setState(() => _pressed = false); widget.onTap(); },
      onTapCancel: () => setState(() => _pressed = false),
      child: AnimatedScale(
        scale: _pressed ? 0.97 : 1.0,
        duration: const Duration(milliseconds: 80),
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
          decoration: BoxDecoration(
            color: widget.bgColor,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: widget.border, width: 1.2)),
          child: Text(widget.label,
            textAlign: TextAlign.center,
            style: GoogleFonts.inter(
              fontSize: 15, fontWeight: FontWeight.w600,
              color: widget.text, height: 1.3)),
        ),
      ),
    );
  }
}