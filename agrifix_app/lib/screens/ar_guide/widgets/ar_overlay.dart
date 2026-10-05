// lib/screens/ar_guide/widgets/ar_overlay.dart
// ignore_for_file: deprecated_member_use
//
// AR overlay widgets — everything drawn on top of the camera feed:
//   • ARArrowOverlay         — FadeTransition + AnimatedBuilder wrapping ARArrowPainter
//   • CameraGuidanceChip     — text chip above scan box when locating
//   • BlurSurround           — frosted blur outside scan box
//   • GradientOverlay        — top/bottom gradient tint
//   • ScanningIndicator      — top-bar pills (Scanning + Speaking + TARGET)
//   • CameraPreviewStub      — dark gradient while camera loads
//   • CameraPermDeniedFallback — full-screen permission denied UI
//   • CentreArea             — scan box + capture button positioned precisely
//   • SolidScanBox           — clean rounded border box
//   • ComponentVerifiedBadge — green "verified" badge above scan box

import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:permission_handler/permission_handler.dart';

import '../models/ar_state.dart';
import '../models/bbox.dart';
import 'ar_arrow_painter.dart';

// ── AR arrow overlay ───────────────────────────────────────────────────────
class ARArrowOverlay extends StatelessWidget {
  final NormBbox?             bbox; // <--- Make sure this is nullable (NormBbox?)
  final Animation<double>     fadeAnim;
  final AnimationController   pulseCtrl;
  final String                partLabel;
  final bool                  isHindi;
  final double                previewW;
  final double                previewH;
  
  // NEW: Off-screen guidance properties
  final bool                  showOffScreenArrow;
  final String                cloudGuidanceVector;

  const ARArrowOverlay({
    required this.bbox,
    required this.fadeAnim,
    required this.pulseCtrl,
    required this.partLabel,
    required this.isHindi,
    required this.previewW,
    required this.previewH,
    this.showOffScreenArrow = false,
    this.cloudGuidanceVector = '',
  });

  @override
  Widget build(BuildContext context) {
    return Positioned.fill(
      child: FadeTransition(
        opacity: fadeAnim,
        child: AnimatedBuilder(
          animation: pulseCtrl,
          builder: (_, __) => CustomPaint(
            painter: ARArrowPainter(
              bbox:                bbox,
              previewW:            previewW,
              previewH:            previewH,
              pulseValue:          pulseCtrl.value,
              partLabel:           partLabel,
              isHindi:             isHindi,
              showOffScreenArrow:  showOffScreenArrow,
              cloudGuidanceVector: cloudGuidanceVector,
            ),
          ),
        ),
      ),
    );
  }
}

// ── Camera guidance chip ───────────────────────────────────────────────────
class CameraGuidanceChip extends StatelessWidget {
  final String message;
  final bool   isHindi;
  const CameraGuidanceChip({required this.message, required this.isHindi});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.78),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: C.arrowBlue.withOpacity(0.55), width: 1.2)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.arrow_upward_rounded, color: C.arrowBlue, size: 16),
          const SizedBox(width: 8),
          Flexible(
            child: Text(
              message,
              textAlign: TextAlign.center,
              style: GoogleFonts.inter(
                fontSize: 13, fontWeight: FontWeight.w600,
                color: C.arrowBlue, height: 1.35)),
          ),
        ],
      ),
    );
  }
}

// ── Blur surround ─────────────────────────────────────────────────────────
class BlurSurround extends StatelessWidget {
  final Rect   boxRect;
  final double cornerRadius;
  const BlurSurround({required this.boxRect, required this.cornerRadius});

  @override
  Widget build(BuildContext context) {
    return Positioned.fill(
      child: ClipPath(
        clipper: _SurroundClipper(boxRect: boxRect, cornerRadius: cornerRadius),
        child: BackdropFilter(
          filter: ui.ImageFilter.blur(sigmaX: 6, sigmaY: 6),
          child: Container(color: Colors.black.withOpacity(0.45)),
        ),
      ),
    );
  }
}

class _SurroundClipper extends CustomClipper<Path> {
  final Rect   boxRect;
  final double cornerRadius;
  const _SurroundClipper({required this.boxRect, required this.cornerRadius});

  @override
  Path getClip(Size size) {
    final outer = Path()..addRect(Rect.fromLTWH(0, 0, size.width, size.height));
    final hole  = Path()..addRRect(RRect.fromRectAndRadius(
        boxRect, Radius.circular(cornerRadius)));
    return Path.combine(PathOperation.difference, outer, hole);
  }

  @override
  bool shouldReclip(_SurroundClipper old) =>
      old.boxRect != boxRect || old.cornerRadius != cornerRadius;
}

// ── Gradient overlay ──────────────────────────────────────────────────────
class GradientOverlay extends StatelessWidget {
  final ARState arState;
  const GradientOverlay({required this.arState});

  @override
  Widget build(BuildContext context) {
    final isVerified = arState == ARState.verified;
    return AnimatedContainer(
      duration: const Duration(milliseconds: 400),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end:   Alignment.bottomCenter,
          stops: const [0.0, 0.30, 0.70, 1.0],
          colors: isVerified
              ? [
                  const Color(0xFF1A3A2A).withOpacity(0.80),
                  Colors.transparent,
                  Colors.transparent,
                  const Color(0xFF0D2016).withOpacity(0.92),
                ]
              : [
                  Colors.black.withOpacity(0.60),
                  Colors.transparent,
                  Colors.transparent,
                  Colors.black.withOpacity(0.72),
                ],
        ),
      ),
    );
  }
}

// ── Camera stubs ──────────────────────────────────────────────────────────
class CameraPreviewStub extends StatelessWidget {
  const CameraPreviewStub();
  @override
  Widget build(BuildContext context) => Container(
    decoration: const BoxDecoration(
      gradient: LinearGradient(
        begin: Alignment.topCenter, end: Alignment.bottomCenter,
        colors: [Color(0xFF2A2A2A), Color(0xFF1A1A1A)],
      ),
    ),
  );
}

class CameraPermDeniedFallback extends StatelessWidget {
  final bool isHindi;
  const CameraPermDeniedFallback({required this.isHindi});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: C.bg,
      child: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 40),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.no_photography_rounded, color: C.textMuted, size: 56),
              const SizedBox(height: 20),
              Text(isHindi ? 'कैमरा एक्सेस आवश्यक है' : 'Camera Access Required',
                textAlign: TextAlign.center,
                style: GoogleFonts.inter(fontSize: 20,
                    fontWeight: FontWeight.w700, color: C.textPrimary)),
              const SizedBox(height: 10),
              Text(
                isHindi
                    ? 'AR गाइड उपयोग करने के लिए डिवाइस सेटिंग में कैमरा एक्सेस दें।'
                    : 'Please allow camera access in your device settings to use the AR guide.',
                textAlign: TextAlign.center,
                style: GoogleFonts.inter(fontSize: 14, color: C.textMuted, height: 1.55)),
              const SizedBox(height: 28),
              GestureDetector(
                onTap: openAppSettings,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 14),
                  decoration: BoxDecoration(
                    color: C.primary,
                    borderRadius: BorderRadius.circular(16)),
                  child: Text(isHindi ? 'सेटिंग खोलें' : 'Open Settings',
                    style: GoogleFonts.inter(fontSize: 15,
                        fontWeight: FontWeight.w600, color: Colors.black87)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Centre area (scan box + capture button) ───────────────────────────────
class CentreArea extends StatelessWidget {
  final ARState          arState;
  final Animation<double> verifiedFade;
  final VoidCallback      onCapture;
  final Rect              boxRect;
  final String            captureLabel;
  final bool              isHindi;

  const CentreArea({
    required this.arState,
    required this.verifiedFade,
    required this.onCapture,
    required this.boxRect,
    required this.captureLabel,
    required this.isHindi,
  });

  @override
  Widget build(BuildContext context) {
    final lblTarget = isHindi ? 'लक्ष्य' : 'TARGET';
    final boxColor = switch (arState) {
      ARState.guiding  => C.arrowGreen,
      ARState.locating => C.arrowBlue,
      ARState.unclear  => C.warning,
      ARState.verified => C.primary,
      _                => Colors.white,
    };
    return Stack(
      children: [
        if (arState == ARState.verified)
          Positioned(
            top: boxRect.top - 54, left: 0, right: 0,
            child: FadeTransition(
              opacity: verifiedFade,
              child: Center(child: ComponentVerifiedBadge(isHindi: isHindi)),
            ),
          ),
        Positioned(
          top: boxRect.top, left: boxRect.left,
          child: arState == ARState.verified
              ? FadeTransition(
                  opacity: verifiedFade,
                  child: const SolidScanBox(color: C.primary),
                )
              : SolidScanBox(
                  color: boxColor,
                  label: arState == ARState.scanning ? lblTarget : null,
                ),
        ),
        if (arState != ARState.verified && arState != ARState.danger)
          Positioned(
            top: boxRect.bottom + 28, left: 0, right: 0,
            child: Column(
              children: [
                CaptureButton(
                  onTap:   onCapture,
                  enabled: arState == ARState.scanning ||
                           arState == ARState.guiding  ||
                           arState == ARState.unclear,
                ),
                const SizedBox(height: 10),
                AnimatedDefaultTextStyle(
                  duration: const Duration(milliseconds: 200),
                  style: GoogleFonts.inter(
                    fontSize: 17, fontWeight: FontWeight.w400,
                    color: (arState == ARState.scanning || arState == ARState.guiding)
                        ? C.textSoft.withOpacity(0.85)
                        : C.textMuted.withOpacity(0.40)),
                  child: Text(captureLabel),
                ),
              ],
            ),
          ),
      ],
    );
  }
}

// ── Solid scan box ────────────────────────────────────────────────────────
class SolidScanBox extends StatelessWidget {
  final Color   color;
  final String? label;
  const SolidScanBox({required this.color, this.label});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: kBoxSize, height: kBoxSize,
      child: Stack(
        clipBehavior: Clip.none,
        alignment: Alignment.topCenter,
        children: [
          CustomPaint(
            size: const Size(kBoxSize, kBoxSize),
            painter: _SolidBoxPainter(color: color),
          ),
          if (label != null)
            Positioned(
              top: -38,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 5),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.52),
                  borderRadius: BorderRadius.circular(20)),
                child: Text(label!,
                  style: GoogleFonts.inter(
                    fontSize: 11, fontWeight: FontWeight.w700,
                    letterSpacing: 1.4,
                    color: Colors.white.withOpacity(0.82))),
              ),
            ),
        ],
      ),
    );
  }
}

class _SolidBoxPainter extends CustomPainter {
  final Color color;
  const _SolidBoxPainter({required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    const double radius  = 20.0;
    const double strokeW = 2.5;
    final rect  = Rect.fromLTWH(0, 0, size.width, size.height);
    final rrect = RRect.fromRectAndRadius(rect, const Radius.circular(radius));
    canvas.drawRRect(rrect, Paint()
      ..color       = color.withOpacity(0.70)
      ..style       = PaintingStyle.stroke
      ..strokeWidth = strokeW);
  }

  @override
  bool shouldRepaint(_SolidBoxPainter old) => old.color != color;
}

// ── Component verified badge ──────────────────────────────────────────────
class ComponentVerifiedBadge extends StatelessWidget {
  final bool isHindi;
  const ComponentVerifiedBadge({required this.isHindi});

  @override
  Widget build(BuildContext context) {
    final label = isHindi ? 'घटक सत्यापित' : 'COMPONENT VERIFIED';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.60),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: C.primary.withOpacity(0.30), width: 1)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.verified_rounded, color: C.primary, size: 18),
          const SizedBox(width: 8),
          Text(label,
            style: GoogleFonts.inter(
              fontSize: 12, fontWeight: FontWeight.w700,
              letterSpacing: 1.0, color: C.primary)),
        ],
      ),
    );
  }
}

// ── Capture button ────────────────────────────────────────────────────────
class CaptureButton extends StatefulWidget {
  final VoidCallback onTap;
  final bool         enabled;
  const CaptureButton({required this.onTap, required this.enabled});
  @override
  State<CaptureButton> createState() => _CaptureButtonState();
}

class _CaptureButtonState extends State<CaptureButton> {
  bool _pressed = false;
  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown:   widget.enabled ? (_) => setState(() => _pressed = true)  : null,
      onTapUp:     widget.enabled ? (_) { setState(() => _pressed = false); widget.onTap(); } : null,
      onTapCancel: widget.enabled ? () => setState(() => _pressed = false)  : null,
      child: AnimatedScale(
        scale: _pressed ? 0.94 : 1.0,
        duration: const Duration(milliseconds: 80),
        child: AnimatedOpacity(
          duration: const Duration(milliseconds: 200),
          opacity: widget.enabled ? 1.0 : 0.35,
          child: Container(
            width: kCaptureSize, height: kCaptureSize,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(color: Colors.white, width: 3),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.35),
                  blurRadius: 16, offset: const Offset(0, 6)),
              ],
            ),
            child: Center(
              child: Container(
                width: kCaptureSize - 14, height: kCaptureSize - 14,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: Colors.white.withOpacity(widget.enabled ? 0.22 : 0.08)),
              ),
            ),
          ),
        ),
      ),
    );
  }
}