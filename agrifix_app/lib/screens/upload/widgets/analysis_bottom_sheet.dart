import 'dart:ui';
import 'package:flutter/material.dart';

// ── Status enum ──────────────────────────────────────────────────
enum StepStatus { pending, active, completed }

class AnalysisStep {
  final String label;
  StepStatus status;
  AnalysisStep({required this.label, this.status = StepStatus.pending});
}

// ── Public entry point ───────────────────────────────────────────
Future<void> showAnalysisSheet(
  BuildContext context, {
  required Future<void> Function(
    void Function(int stepIndex) markActive,
    void Function(int stepIndex) markDone,
    void Function() markCacheHit,          // ← NEW: triggers instant flash
  ) runAnalysis,
  required VoidCallback onCompleted,
}) async {
  await showModalBottomSheet<void>(
    context: context,
    isScrollControlled: true,
    backgroundColor: Colors.transparent,
    barrierColor: Colors.transparent,
    isDismissible: false,
    enableDrag: false,
    builder: (_) => _AnalysisSheetWithOverlay(
      runAnalysis: runAnalysis,
      onCompleted: onCompleted,
    ),
  );
}

// ── Overlay wrapper ──────────────────────────────────────────────
class _AnalysisSheetWithOverlay extends StatelessWidget {
  final Future<void> Function(
    void Function(int) markActive,
    void Function(int) markDone,
    void Function()    markCacheHit,
  ) runAnalysis;
  final VoidCallback onCompleted;

  const _AnalysisSheetWithOverlay({
    required this.runAnalysis,
    required this.onCompleted,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8),
          child: Container(color: Colors.black.withValues(alpha: 0.35)),
        ),
        Align(
          alignment: Alignment.bottomCenter,
          child: _AnalysisSheet(
            runAnalysis: runAnalysis,
            onCompleted: onCompleted,
          ),
        ),
      ],
    );
  }
}

// ── Bottom Sheet ─────────────────────────────────────────────────
class _AnalysisSheet extends StatefulWidget {
  final Future<void> Function(
    void Function(int) markActive,
    void Function(int) markDone,
    void Function()    markCacheHit,
  ) runAnalysis;
  final VoidCallback onCompleted;

  const _AnalysisSheet({
    required this.runAnalysis,
    required this.onCompleted,
  });

  @override
  State<_AnalysisSheet> createState() => _AnalysisSheetState();
}

class _AnalysisSheetState extends State<_AnalysisSheet>
    with TickerProviderStateMixin {

  static const _stepLabels = [
    'Identifying your machine from the video...',
    'Understanding your voice complaint...',
    'Searching repair manuals for your issue...',
    'Preparing your step-by-step repair guide...',
  ];

  late final List<AnalysisStep> _steps;
  late final List<AnimationController> _checkControllers;
  late final List<Animation<double>> _checkScales;
  late final List<AnimationController> _spinControllers;

  double _progress = 0.0;
  bool   _hasError = false;
  bool   _cacheHit = false;   // true when stage-3 served from plan cache

  int _doneCount    = 0;
  // int _displayHead  = 0;   

  @override
  void initState() {
    super.initState();

    _steps = _stepLabels.map((l) => AnalysisStep(label: l)).toList();

    _checkControllers = List.generate(
      _steps.length,
      (_) => AnimationController(
        vsync: this, duration: const Duration(milliseconds: 280)),
    );
    _checkScales = _checkControllers.map((c) =>
      Tween<double>(begin: 0.6, end: 1.0).animate(
        CurvedAnimation(parent: c, curve: Curves.easeOutBack))).toList();

    _spinControllers = List.generate(
      _steps.length,
      (_) => AnimationController(
        vsync: this, duration: const Duration(milliseconds: 1400)),
    );

    WidgetsBinding.instance.addPostFrameCallback((_) {
      _activateVisualRow(0);

      widget.runAnalysis(_onBackendStageStart, _onBackendStageDone, _onCacheHit)
          .catchError((error) {
        if (!mounted) return;
        for (final c in _spinControllers) c.stop();
        setState(() => _hasError = true);
      });
    });
  }

  @override
  void dispose() {
    for (final c in _checkControllers) c.dispose();
    for (final c in _spinControllers) c.dispose();
    super.dispose();
  }

  void _onBackendStageStart(int backendIndex) {
    // No-op
  }

  /// Called when the server returns cache_hit: true on stage-3.
  /// Instantly completes all remaining steps with a brief flash delay
  /// so the farmer sees the checklist complete quickly (feels snappy, not broken).
  void _onCacheHit() {
    if (!mounted) return;
    setState(() => _cacheHit = true);
    // Complete all pending steps with a short stagger
    for (int i = _doneCount; i < _steps.length; i++) {
      final delay = Duration(milliseconds: 80 * (i - _doneCount + 1));
      Future.delayed(delay, () {
        if (mounted) _onBackendStageDone(i);
      });
    }
  }

  void _onBackendStageDone(int backendIndex) {
    if (!mounted) return;

    _doneCount++;
    final visualRow = _doneCount - 1; 

    if (visualRow >= _steps.length) return;
    if (_steps[visualRow].status == StepStatus.completed) return;

    setState(() {
      _steps[visualRow].status = StepStatus.completed;
      _spinControllers[visualRow].stop();
      _progress = _doneCount / _steps.length;
    });
    _checkControllers[visualRow].forward();

    final nextVisual = _doneCount; 
    if (nextVisual < _steps.length) {
      Future.delayed(const Duration(milliseconds: 160), () {
        if (mounted) _activateVisualRow(nextVisual);
      });
    } else {
      Future.delayed(const Duration(milliseconds: 600), () {
        if (mounted) {
          Navigator.of(context).pop();
          widget.onCompleted();
        }
      });
    }
  }

  void _activateVisualRow(int i) {
    if (!mounted || i >= _steps.length) return;
    
    // 🚨 THE FIX: Prevent Race Condition!
    // If the backend completes stages faster than the 160ms visual delay,
    // this row might have already been marked as 'completed'. 
    // We must NOT revert it back to 'active' (PROCESSING).
    if (_steps[i].status == StepStatus.completed) return;

    setState(() {
      // _displayHead = i;
      _steps[i].status = StepStatus.active;
    });
    _spinControllers[i].repeat();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
        boxShadow: [
          BoxShadow(
            color: Colors.black26,
            blurRadius: 28,
            offset: Offset(0, -6),
          ),
        ],
      ),
      padding: const EdgeInsets.fromLTRB(24, 12, 24, 40),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 36,
            height: 4,
            decoration: BoxDecoration(
              color: const Color(0xFFD1D5DB),
              borderRadius: BorderRadius.circular(8),
            ),
          ),
          const SizedBox(height: 24),
          const Text(
            'AI Analysis in Progress',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.w700,
              color: Color(0xFF1F2937),
            ),
          ),
          const SizedBox(height: 8),
          if (_cacheHit)
            Container(
              margin: const EdgeInsets.only(bottom: 6),
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
              decoration: BoxDecoration(
                color: const Color(0xFFE6F6EC),
                borderRadius: BorderRadius.circular(20),
              ),
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.bolt_rounded, size: 14, color: Color(0xFF22C55E)),
                  SizedBox(width: 4),
                  Text('Instant — Cached Plan',
                    style: TextStyle(
                      fontSize: 12, fontWeight: FontWeight.w600,
                      color: Color(0xFF22C55E))),
                ],
              ),
            ),
          const Text(
            'This takes about 15–30 seconds. Please keep the app open.',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w400,
              color: Color(0xFF6B7280),
            ),
          ),
          const SizedBox(height: 28),

          if (_hasError) ...[
            const Icon(
              Icons.error_outline_rounded,
              color: Color(0xFFEF4444),
              size: 40,
            ),
            const SizedBox(height: 12),
            const Text(
              'Analysis failed. Please try again.',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w500,
                color: Color(0xFFEF4444),
              ),
            ),
            const SizedBox(height: 20),
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Dismiss'),
            ),
          ] else ...[
            ...List.generate(
              _steps.length,
              (i) => _StepRow(
                step: _steps[i],
                spinController: _spinControllers[i],
                checkScale: _checkScales[i],
              ),
            ),
            const SizedBox(height: 24),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: TweenAnimationBuilder<double>(
                tween: Tween(begin: 0, end: _progress),
                duration: const Duration(milliseconds: 450),
                curve: Curves.easeInOut,
                builder: (_, val, __) => LinearProgressIndicator(
                  value: val,
                  minHeight: 6,
                  backgroundColor: const Color(0xFFE5E7EB),
                  valueColor: const AlwaysStoppedAnimation(Color(0xFF10B981)),
                ),
              ),
            ),
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerRight,
              child: Text(
                '${(_progress * 100).toInt()}%',
                style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: Color(0xFF10B981),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

// ── Single Step Row ──────────────────────────────────────────────
class _StepRow extends StatelessWidget {
  final AnalysisStep step;
  final AnimationController spinController;
  final Animation<double> checkScale;

  const _StepRow({
    required this.step,
    required this.spinController,
    required this.checkScale,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 10),
      child: Row(
        children: [
          SizedBox(width: 32, height: 32, child: _icon()),
          const SizedBox(width: 14),
          Expanded(
            child: Text(
              step.label,
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w500,
                color: step.status == StepStatus.pending
                    ? const Color(0xFF9CA3AF)
                    : const Color(0xFF1F2937),
              ),
            ),
          ),
          _tag(),
        ],
      ),
    );
  }

  Widget _icon() {
    switch (step.status) {
      case StepStatus.pending:
        return Container(
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            border: Border.all(color: const Color(0xFFD1D5DB), width: 2),
          ),
        );

      case StepStatus.active:
        return RotationTransition(
          turns: CurvedAnimation(
            parent: spinController,
            curve: Curves.linear,
          ),
          child: const Icon(
            Icons.timelapse_rounded,
            color: Color(0xFF10B981),
            size: 24,
          ),
        );

      case StepStatus.completed:
        return ScaleTransition(
          scale: checkScale,
          child: Container(
            decoration: const BoxDecoration(
              shape: BoxShape.circle,
              color: Color(0xFF10B981),
            ),
            child: const Icon(
              Icons.check_rounded,
              color: Colors.white,
              size: 20,
            ),
          ),
        );
    }
  }

  Widget _tag() {
    switch (step.status) {
      case StepStatus.pending:
        return const SizedBox(width: 76);

      case StepStatus.active:
        return SizedBox(
          width: 76,
          child: Text(
            'PROCESSING',
            textAlign: TextAlign.right,
            style: TextStyle(
              fontSize: 10,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.3,
              color: Colors.orange.shade700,
            ),
          ),
        );

      case StepStatus.completed:
        return const SizedBox(
          width: 76,
          child: Text(
            'DONE',
            textAlign: TextAlign.right,
            style: const TextStyle(
              fontSize: 10,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.3,
              color: Color(0xFF10B981),
            ),
          ),
        );
    }
  }
}