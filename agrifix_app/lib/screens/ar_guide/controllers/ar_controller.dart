// lib/screens/ar_guide/controllers/ar_controller.dart
// ignore_for_file: deprecated_member_use
//
// AR pipeline orchestration layer.
//
// Owns: locate loop scheduling, Gemini timing, state transitions,
//       stability counting, detection locking, re-acquire logic,
//       verify-step flow, step navigation, camera lifecycle.
//
// Does NOT own: UI layout, widget trees, CustomPainter drawing.
// Services it delegates to: TrackingService, GuidanceService, TtsService.

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;
import '../models/inspection_panel_model.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';

import '../../../core/providers/diagnosis_provider.dart';
import '../../../core/providers/language_provider.dart';
import '../../../services/api_service.dart';
import '../models/ar_state.dart';
import '../models/bbox.dart';
import '../services/tracking_service.dart';
import '../services/guidance_service.dart';
import '../services/tts_service.dart';
import '../../../core/models/agent_models.dart';
import '../../../core/providers/agent_session_provider.dart';
import '../../../core/models/inspection_snapshot.dart';

// ── Frame quality gate ─────────────────────────────────────────────────────
// Pure-Dart blur+brightness check. Runs before every Gemini call.
class ARQualityGate {
  static const _kLaplacianMin  = 80.0;
  static const _kBrightnessMin = 30.0;
  static const _kBrightnessMax = 230.0;

  static Future<({bool ok, String message})> check(Uint8List jpeg) async {
    try {
      final codec = await ui.instantiateImageCodec(
          jpeg, targetWidth: 32, targetHeight: 24);
      final frame = await codec.getNextFrame();
      final data  = await frame.image.toByteData(
          format: ui.ImageByteFormat.rawRgba);
      frame.image.dispose();
      if (data == null) return (ok: true, message: '');

      final px = data.buffer.asUint8List();
      const int imgW = 32, imgH = 24, n = imgW * imgH;

      final luma = List<double>.generate(n, (i) {
        final b = i * 4;
        return 0.299 * px[b] + 0.587 * px[b + 1] + 0.114 * px[b + 2];
      });

      var lumaSum = 0.0;
      for (final v in luma) lumaSum += v;
      final brightness = lumaSum / n;

      if (brightness < _kBrightnessMin) {
        return (ok: false, message: 'Too dark — move to better light');
      }
      if (brightness > _kBrightnessMax) {
        return (ok: false,
            message: 'Too bright — avoid direct sunlight on the part');
      }

      var lapSum = 0.0, lapSumSq = 0.0;
      var lapCount = 0;
      for (var y = 1; y < imgH - 1; y++) {
        for (var x = 1; x < imgW - 1; x++) {
          final i = y * imgW + x;
          final v = luma[i - imgW] + luma[i + imgW] +
                    luma[i - 1]    + luma[i + 1] -
                    4.0 * luma[i];
          lapSum   += v;
          lapSumSq += v * v;
          lapCount++;
        }
      }
      final lapMean     = lapSum / lapCount;
      final lapVariance = (lapSumSq / lapCount) - (lapMean * lapMean);

      if (lapVariance < _kLaplacianMin) {
        return (ok: false,
            message: 'Image blurry — hold still and move closer');
      }
      return (ok: true, message: '');
    } catch (_) {
      return (ok: true, message: '');
    }
  }
}

// ── Crop helper ────────────────────────────────────────────────────────────
class ARCropHelper {
  static const _kContextMargin      = 0.40;
  static const _kLargePartThreshold = 0.40;

  static Future<Uint8List?> cropToBbox(Uint8List jpeg, NormBbox bbox) async {
    if (bbox.w * bbox.h > _kLargePartThreshold) return null;
    try {
      final codec = await ui.instantiateImageCodec(jpeg);
      final frame = await codec.getNextFrame();
      final img   = frame.image;
      final imgW  = img.width.toDouble();
      final imgH  = img.height.toDouble();

      final _longestAxis   = math.max(bbox.w, bbox.h);
      final _dynamicMargin = math.min(
          _kContextMargin,
          (0.5 - _longestAxis / 2) * 0.85,
      ).clamp(0.05, _kContextMargin);

      final expW   = bbox.w + 2 * _dynamicMargin;
      final expH   = bbox.h + 2 * _dynamicMargin;
      final left   = ((bbox.cx - expW / 2) * imgW).clamp(0.0, imgW);
      final top    = ((bbox.cy - expH / 2) * imgH).clamp(0.0, imgH);
      final right  = ((bbox.cx + expW / 2) * imgW).clamp(0.0, imgW);
      final bottom = ((bbox.cy + expH / 2) * imgH).clamp(0.0, imgH);
      final cropW  = right - left;
      final cropH  = bottom - top;

      if (cropW < 112 || cropH < 112) { img.dispose(); return null; }

      final recorder = ui.PictureRecorder();
      final canvas   = ui.Canvas(recorder);
      canvas.drawImageRect(
        img,
        Rect.fromLTWH(left, top, cropW, cropH),
        Rect.fromLTWH(0,    0,    cropW, cropH),
        ui.Paint(),
      );
      img.dispose();

      final picture  = recorder.endRecording();
      final cropImg  = await picture.toImage(cropW.round(), cropH.round());
      final byteData = await cropImg.toByteData(format: ui.ImageByteFormat.png);
      cropImg.dispose();

      if (byteData == null) return null;
      return byteData.buffer.asUint8List();
    } catch (e) {
      debugPrint('ARGuide crop failure (using full frame): $e');
      return null;
    }
  }
}

// ══════════════════════════════════════════════════════════════════════════
// ARController
// ══════════════════════════════════════════════════════════════════════════
class ARController {
  // ── Constructor dependencies ──────────────────────────────────────────────
  final BuildContext Function() getContext;
  final void Function(void Function()) setState;
  final bool Function() isMounted;

  // ── Services ──────────────────────────────────────────────────────────────
  final TrackingService tracking  = TrackingService();
  final TtsService      tts       = TtsService();
  late  GuidanceService guidance;

  // ── Scheduling constants (identical to monolith) ─────────────────────────
  static const _kLocateIntervalMs       = 1000;
  static const _kLocateIntervalGuidedMs = 4000;
  static const _kFrameCooldownMs        = 800;
  static const _kReacquireDelayMs       = 500;
  static const _kMaxLocateAttempts      = 0;

  // ── AR state (UI reads these via getters) ─────────────────────────────────
  ARState   arState         = ARState.scanning;
  bool      voiceActive     = false;
  bool      panelExpanded   = false;
  int       attemptCount    = 0;
  String    dangerMessage   = '';
  String    dynamicFeedback = '';
  String    cameraGuidance  = '';
  String    partDescription = '';
  bool      bboxLocked      = false;
  ToastKind toastKind       = ToastKind.analyzing;

  // Which interaction type produced the current ARState.verified — set right
  // before advance() on every completion path (camera / choice / boolean).
  // The completion badge/copy in BottomPanel must vary on this: a plain
  // yes/no or numeric confirmation was never inspected for "damage", so it
  // must not carry the same vision-verification copy as an actual camera
  // check. Without this, every step — safety, measurement, manual action,
  // or camera — showed the identical "Component Verified — No Damage
  // Detected" badge regardless of what was actually verified.
  InteractionType? lastCompletedInteractionType;

  final List<Map<String, dynamic>> attemptResults = [];

  // ── AR pipeline state ─────────────────────────────────────────────────────
  Timer? _locateTimer;
  bool   _locateRunning  = false;
  bool _captureInProgress = false;
  int    _locateAttempts = 0;
  bool   _partLocked     = false;
  bool   _everDetected   = false;
  DateTime? _lastLocateSent;
  DateTime? _lastCorrectionSent;
  int    _frameId        = 0;
  int    _netFailCount      = 0;  // consecutive network errors
  int    _consecutiveMisses = 0;  // found=False streak during LOCATING
  // One blurry/occluded frame should not wipe stability progress.
  static const _kMissesTolerance = 2;
  String cloudGuidanceVector = '';
  bool showOffScreenArrow = false;
  // Superseded by the /locate_part loop kicked off in maybeStartLocateLoop().
  // Left wired but permanently off — see note there before re-enabling.
  bool _isInBlindSearch = false;
  Timer? _stabilityTimer;
  bool _reticleSteady = false;
  DateTime? _reticleSteadyStart;
  // ── Tracking timer ────────────────────────────────────────────────────────
  Timer?    _trackingTimer;
  bool      _trackingEnabled = false;
  DateTime? _lastBboxTimeTick;
  Timer?    _bboxLockTimer;

  // ── Camera ────────────────────────────────────────────────────────────────
  CameraController? cameraController;
  bool cameraReady      = false;
  bool cameraPermDenied = false;

  // ── Panel state ───────────────────────────────────────────────────────────
  bool inspectionPanelVisible = false;
  InspectionPanelModel? agentPanelModel;

  // ── Inspection state ───────────────────────────────────────────────────────
  InspectionSnapshot? inspectionSnapshot;

  // ── Step navigation ───────────────────────────────────────────────────────
  late int currentStep;

  // ── Animation controllers (set by screen after creation) ─────────────────
  late AnimationController bboxFadeCtrl;
  late AnimationController arrowPulseCtrl;
  late AnimationController verifiedCtrl;
  late AnimationController toastCtrl;
  late AnimationController spinnerCtrl;
  late Animation<double>   bboxFadeAnim;
  late Animation<double>   verifiedFade;
  late Animation<double>   toastSlide;
  late Animation<double>   toastFade;
  Timer? toastTimer;

  // ── Orientation ───────────────────────────────────────────────────────────
  Orientation? lastOrientation;

  ARController({
    required this.getContext,
    required this.setState,
    required this.isMounted,
  }) {
    guidance = GuidanceService(
      tts: tts,
      onVisualUpdate: (text) {
        if (isMounted()) setState(() => cameraGuidance = text);
      },
      onVoiceActiveChanged: (active) {
        if (isMounted()) setState(() => voiceActive = active);
      },
    );
  }

  // ══════════════════════════════════════════════════════════════════════════
  // DIRECTION EXTRACTION
  // ══════════════════════════════════════════════════════════════════════════

  /// Extracts AR arrow direction from Gemini's spatial guidance text.
  /// Handles English and Hindi directional keywords.
  /// Returns null if no clear spatial direction is found.
  static String? _extractDirectionFromGuidance(String guidance) {
    final lower = guidance.toLowerCase();
    if (lower.contains('above') || lower.contains('up') || lower.contains('ऊपर')) {
      return 'up';
    }
    if (lower.contains('below') || lower.contains('down') || lower.contains('नीचे')) {
      return 'down';
    }
    if (lower.contains(' left') || lower.contains('बाएं') || lower.contains('बायें') || lower.contains('बाईं')) {
      return 'left';
    }
    if (lower.contains(' right') || lower.contains('दाएं') || lower.contains('दायें') || lower.contains('दाईं')) {
      return 'right';
    }
    return null;
  }
  // ══════════════════════════════════════════════════════════════════════════
  // CAMERA
  // ══════════════════════════════════════════════════════════════════════════

  Future<void> initCamera() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      if (isMounted()) setState(() => cameraPermDenied = true);
      return;
    }
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) return;
      cameraController = CameraController(
        cameras.first, ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await cameraController!.initialize();
      try {
        await cameraController!.setFocusMode(FocusMode.auto);
        await cameraController!.setExposureMode(ExposureMode.auto);
      } catch (modeErr) {
        debugPrint('ARGuide: focus/exposure mode not supported — $modeErr');
      }
      if (isMounted()) {
        setState(() => cameraReady = true);
        maybeStartLocateLoop();
        checkAgentPanel();
      }
    } catch (e) {
      debugPrint('ARGuide: camera init failed — $e');
    }
  }

  Future<void> pauseCamera()  async {
    if (cameraReady) await cameraController?.pausePreview();
  }
  Future<void> resumeCamera() async {
    if (cameraReady) await cameraController?.resumePreview();
  }
Future<File?> captureFrame() async {
  if (_captureInProgress) return null;

  final controller = cameraController;
  if (controller == null || !controller.value.isInitialized) {
    return null;
  }

  _captureInProgress = true;

  try {
    final xFile = await controller.takePicture();
    return File(xFile.path);
  } catch (e) {
    debugPrint("captureFrame failed: $e");
    return null;
  } finally {
    _captureInProgress = false;
  }
}

  // ══════════════════════════════════════════════════════════════════════════
  // TTS
  // ══════════════════════════════════════════════════════════════════════════

  Future<void> initTts(String langCode) async {
    await tts.init(
      langCode,
      onComplete: () {
        if (isMounted()) setState(() => voiceActive = false);
        guidance.onTtsComplete();
      },
      onCancel: () {
        if (isMounted()) setState(() => voiceActive = false);
        guidance.onTtsComplete();
      },
    );
    // Sync guidance language once — not on every tick.
    // didChangeDependencies calls initTts on start + on locale change,
    // so this fires exactly when it's needed.
    guidance.setLanguage(langCode);
  }

void maybeStartLocateLoop() {
    final ctx = getContext();
    final agentStep = ctx.read<AgentSessionProvider>().current?.nextStep;
    if (agentStep == null) return;
    
    // Only camera steps with a visual target enter the AR flow
    final hasVisualTarget = agentStep.requiredPart.isNotEmpty || 
                            agentStep.visualCue.isNotEmpty || 
                            agentStep.trackingScope == 'assembly';
                            
    if (!hasVisualTarget) return;
    if (arState == ARState.verified) return;

    _stopLocateLoop();
    _locateAttempts = 0;
    tracking.reset();
    tracking.resetKalman();
    _partLocked = false;
    _lastCorrectionSent = null;
    guidance.resetForNewSession();
    _consecutiveMisses = 0;
    cloudGuidanceVector = '';
    showOffScreenArrow = false;

    unawaited(guidance.speakPreDetectionHint(
      agentStep.requiredPart.isNotEmpty ? agentStep.requiredPart : (agentStep.visualCue),
      agentStep.areaHint,
      isHindi: ctx.read<LanguageProvider>().languageCode == 'hi',
    ));

    // ── MANUAL-ONLY MODE ──────────────────────────────────────────────────
    // No auto-detection loop. The farmer reads the step instruction,
    // points the camera at the part, and taps "Analyze Part".
    // That single tap triggers onCapture() → /verify_step → /inspect_part.
    final isHindi = ctx.read<LanguageProvider>().languageCode == 'hi';
    
    if (isMounted()) {
      setState(() {
        arState = ARState.scanning;
        if (agentStep.trackingScope == 'assembly') {
          cameraGuidance = isHindi 
              ? 'कैमरा पूरे ${agentStep.areaHint.replaceAll('_', ' ')} क्षेत्र पर केंद्रित करें'
              : 'Point camera at the entire ${agentStep.areaHint.replaceAll('_', ' ')} area and tap Analyze Part';
        } else {
          final targetName = agentStep.requiredPart.isNotEmpty 
              ? agentStep.requiredPart : agentStep.visualCue;
          cameraGuidance = isHindi 
              ? '${targetName.replaceAll('_', ' ')} पर कैमरा लगाएं और "विश्लेषण करें" दबाएं'
              : 'Point camera at the ${targetName.replaceAll('_', ' ')} and tap Analyze Part';
        }
      });
    }

    // If _kMaxAutoAttempts > 0, the burst-mode auto-loop would start here.
    // Currently set to 0 (manual-only) for production quota efficiency.
  }

  void _stopLocateLoop() {
    _locateTimer?.cancel();
    _locateTimer   = null;
    _locateRunning = false;
    _stabilityTimer?.cancel();
    _stopTrackingTimer();
  }

  // ══════════════════════════════════════════════════════════════════════════
  // TRACKING TIMER (30fps Kalman predict)
  // ══════════════════════════════════════════════════════════════════════════

  void _startTrackingTimer() {
    _trackingTimer?.cancel();
    tracking.velCx = 0; tracking.velCy = 0;
    _lastBboxTimeTick = DateTime.now();
    _trackingTimer = Timer.periodic(
      const Duration(milliseconds: 33),
      (_) => _trackingTick(),
    );
  }

  void _stopTrackingTimer() {
    _trackingTimer?.cancel();
    _trackingTimer = null;
  }

  void _trackingTick() {
    if (!isMounted()) return;
    if (!_trackingEnabled) return;
    if (bboxLocked) return;
    if (tracking.smoothBbox == null) return;
    if (arState != ARState.guiding) return;

    final now = DateTime.now();
    final dt  = _lastBboxTimeTick == null
        ? 0.033
        : now.difference(_lastBboxTimeTick!).inMilliseconds / 1000.0;
    _lastBboxTimeTick = now;

    final clampedDt = dt.clamp(0.0, 1.5);
    final predicted = tracking.predictTick(clampedDt);
    if (predicted != null && isMounted()) {
      setState(() {});  // trigger repaint with updated smoothBbox
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // LOCATE TICK — core detection pipeline
  // ══════════════════════════════════════════════════════════════════════════
    // ignore : unsed_element
  Future<void> _locateTick() async {
    if (_locateRunning) return;
    if (!isMounted()) return;
    if (bboxLocked) return;
    if (!cameraReady || cameraController == null) return;
    if (arState == ARState.verified || arState == ARState.analyzing) return;
    final ctx       = getContext();
    final agentProv = ctx.read<AgentSessionProvider>();
    final agentStep = agentProv.current?.nextStep;
    if (agentStep == null) { _locateRunning = false; return; }
    final langCode  = ctx.read<LanguageProvider>().languageCode;
    final machine   = agentProv.current?.updatedMemory['machine_type'] as String? 
                      ?? ctx.read<DiagnosisProvider>().solution?.machineType 
                      ?? 'tractor';
    _locateRunning = true;
    _locateAttempts++;
    final now = DateTime.now();

    // ── LOCKED phase: tracking only, no Gemini ───────────────────────────
    if (_partLocked) { _locateRunning = false; return; }

    // ── GUIDING phase: drift-triggered correction ────────────────────────
    if (arState == ARState.guiding &&
        tracking.lastAiBbox != null &&
        tracking.smoothBbox != null) {
      if (!tracking.needsCorrection(_lastCorrectionSent)) {
        debugPrint('ARGuide [SKIP_CORRECTION] tracking accurate');
        _locateRunning = false; return;
      }
    }

    // ── Hard cooldown ─────────────────────────────────────────────────────
    if (_lastLocateSent != null &&
        now.difference(_lastLocateSent!).inMilliseconds < _kFrameCooldownMs) {
      _locateRunning = false; return;
    }

    try {
      final File? frame = await captureFrame();
      if (frame == null) { _locateRunning = false; return; }  

      final bytes = await frame.readAsBytes();
      _lastLocateSent = DateTime.now();
      if (arState == ARState.guiding) {
        _lastCorrectionSent = _lastLocateSent;
        debugPrint('ARGuide [CORRECTION] attempt=$_locateAttempts');
      } else {
        debugPrint('ARGuide [DETECT] attempt=$_locateAttempts');
      }

      _frameId++;
      final sentFrameId = _frameId;

      final qResult = await ARQualityGate.check(bytes);
      if (!qResult.ok) {
        if (isMounted()) setState(() => cameraGuidance = qResult.message);
        _locateRunning = false; return;
      }

      final part  = agentStep.requiredPart.isNotEmpty ? agentStep.requiredPart : agentStep.visualCue;
      final area  = agentStep.areaHint;
      final scope = agentStep.trackingScope;

      if (scope == 'assembly') {
        // Assembly mode — DO NOT ping ApiService.locatePart.
        // Wait for manual "Analyze Part" tap from the user.
        _locateRunning = false; 
        return;
      }

      if (part.isEmpty) { _locateRunning = false; return; }

      // Pre-detection hint — fires once
      if (tracking.smoothBbox == null) {
        unawaited(guidance.speakPreDetectionHint(
            part, area, isHindi: langCode == 'hi'));
      }

      final roiHint = tracking.smoothBbox != null
          ? '${tracking.smoothBbox!.cx.toStringAsFixed(3)},'
            '${tracking.smoothBbox!.cy.toStringAsFixed(3)},0.300'
          : '';

      Map<String, dynamic> result;
      try {
        result = await ApiService.locatePart(
          imageFile:    frame,
          requiredPart: part,
          areaHint:     area,
          machineType:  machine,
          attemptCount: _locateAttempts,
          language:     langCode,
          frameId:      _frameId,
          searchRoi:    roiHint,
        );
      } catch (e) {
        debugPrint('ARGuide locate_part error: $e');
        _netFailCount++;
        if (_netFailCount >= 2 && isMounted()) {
          setState(() => cameraGuidance = 'Network error — retrying…');
        }
        _locateRunning = false; return;
      }

      if (!isMounted()) { _locateRunning = false; return; }
      if (sentFrameId != _frameId) { _locateRunning = false; return; }
      final respFrameId = result['frame_id'] as int? ?? sentFrameId;
      if (respFrameId != sentFrameId) { _locateRunning = false; return; }
      _netFailCount = 0;

      final found      = result['found'] == true;
      final confidence = (result['confidence'] as num?)?.toDouble() ?? 0.0;
      final rawBbox    = result['bbox'] as List?;
      final rawGuidance = result['camera_guidance'] as String? ?? '';
      final serverGuidance = rawGuidance.contains('Network error') ? '' : rawGuidance;
      final partDesc   = result['part_description'] as String? ?? '';

      if (found && confidence >= TrackingService.kConfThreshold
          && rawBbox != null && rawBbox.length == 4) {
        final cx = (rawBbox[0] as num).toDouble();
        final cy = (rawBbox[1] as num).toDouble();
        final bw = (rawBbox[2] as num).toDouble();
        final bh = (rawBbox[3] as num).toDouble();

        final newSmooth = tracking.update(
            cx: cx, cy: cy, bw: bw, bh: bh, confidence: confidence);
        if (newSmooth == null) {
          debugPrint('ARGuide [BBOX_REJECTED] cx=$cx cy=$cy w=$bw h=$bh '
              'area=${(bw*bh).toStringAsFixed(4)} — failed client sanity');
          _locateRunning = false; return;
        }

        final isHindi = langCode == 'hi';
        final distHint = tracking.distanceHint(bw * bh, isHindi);

        _everDetected = true;
        _consecutiveMisses = 0;  // valid detection — clear miss streak
        setState(() {
          partDescription = partDesc;
          cameraGuidance  = distHint.isNotEmpty ? distHint : '';
        });

        // ── DIAGNOSTIC: print after every valid detection ─────────────
        debugPrint('═══ ARGuide DETECT: STATE=${arState.name} '
            'STABLE=${tracking.stableFrameCount}/${TrackingService.kStableFramesNeeded} '
            'conf=${confidence.toStringAsFixed(2)} '
            'bbox=(${newSmooth.cx.toStringAsFixed(3)},${newSmooth.cy.toStringAsFixed(3)}) '
            'ARROW=${arState == ARState.guiding ? "VISIBLE" : "waiting for stable"}');
        debugPrint('ARGuide [KALMAN] est=(${newSmooth.cx},${newSmooth.cy})');

        // Directional guidance (confidence-gated)
        if (confidence >= GuidanceService.kGuidanceConfThreshold) {
          unawaited(guidance.speakDirectionalHint(
              newSmooth, isHindi: langCode == 'hi'));
        }

        // ── LOCKED transition ──────────────────────────────────────────────
        if (tracking.stableFrameCount >= TrackingService.kLockFramesNeeded &&
            confidence >= TrackingService.kLockConfThreshold &&
            !_partLocked) {
          _partLocked = true;
          debugPrint('ARGuide [LOCKED] stable=${tracking.stableFrameCount}');
          guidance.bypassDedup();
          unawaited(guidance.speakGuidance(
            'Locked — hold steady and tap Analyze Part',
            'भाग लॉक — स्थिर रखें और "भाग विश्लेषण करें" दबाएं',
          ));
        }

        // ── GUIDING transition ─────────────────────────────────────────────
        if (tracking.stableFrameCount >= TrackingService.kStableFramesNeeded &&
            (arState == ARState.locating || arState == ARState.unclear)) {
          bboxFadeCtrl.forward(from: 0);
          setState(() {
            arState        = ARState.guiding;
            cameraGuidance = '';
          });
          HapticFeedback.mediumImpact();
          _stopLocateLoop();
          _trackingEnabled = true;
          _consecutiveMisses = 0;  // fresh tracking session
          _startTrackingTimer();
          _bboxLockTimer?.cancel();
          _bboxLockTimer = Timer(const Duration(milliseconds: 1500), () {});
          debugPrint('══════════════════════════════════════════════');
          debugPrint('ARGuide [GUIDING] ✅ ARROW SHOULD NOW BE VISIBLE');
          debugPrint('  stable=${tracking.stableFrameCount} locked=$_partLocked');
          debugPrint('  bboxFadeCtrl started — arrow fading in over 350ms');
          debugPrint('══════════════════════════════════════════════');

          final label = partDesc.isNotEmpty
              ? partDesc
              : agentStep.requiredPart.replaceAll('_', ' ');
          guidance.bypassDedup();
          unawaited(guidance.speakGuidance(
            label.isNotEmpty
                ? 'Found it — tap Analyze Part to verify'
                : 'Part found — tap Analyze Part to verify',
            label.isNotEmpty
                ? 'मिल गया — जांच के लिए "भाग विश्लेषण करें" दबाएं'
                : 'भाग मिल गया — "भाग विश्लेषण करें" दबाएं',
          ));
        }

        // ── CORRECTION timer restart ───────────────────────────────────────
        if (arState == ARState.guiding && _locateTimer == null && !_partLocked) {
          _locateTimer = Timer.periodic(
              const Duration(milliseconds: _kLocateIntervalGuidedMs),
              (_) => _locateTick());
        }

      } else {
        // ── Detection lost / re-acquisition ───────────────────────────────
        final wasLocked = _partLocked;
        _consecutiveMisses++;
        final bool shouldResetStability =
            arState == ARState.guiding ||        // always reset if was guiding
            _consecutiveMisses >= _kMissesTolerance; // or after 2 misses
        debugPrint('ARGuide [FOUND=FALSE] STATE=${arState.name} '
            'miss_streak=$_consecutiveMisses '
            'stable=${tracking.stableFrameCount} '
            '→ ${shouldResetStability ? "RESET" : "PRESERVED"}');
        _trackingEnabled = false;
        _partLocked      = false;
        tracking.resetVelocity();
        if (shouldResetStability) tracking.stableFrameCount = 0;
        guidance.resetOnDetectionLost();

        if (wasLocked) {
          debugPrint('ARGuide [REACQUIRE] re-acquiring…');
        }
        // ── Extract direction for off-screen AR arrow ─────────────────
        final direction = serverGuidance.isNotEmpty
            ? _extractDirectionFromGuidance(serverGuidance)
            : null;
        final hasDirection = direction != null;

        setState(() {
          cameraGuidance     = serverGuidance.isNotEmpty ? serverGuidance : '';
          showOffScreenArrow = hasDirection;
          cloudGuidanceVector = direction ?? '';
          if (arState == ARState.guiding) {
            arState          = ARState.locating;
            tracking.smoothBbox = null;
            bboxFadeCtrl.reverse();
            _stopLocateLoop();
            _lastCorrectionSent = null;
            Future.delayed(
              const Duration(milliseconds: _kReacquireDelayMs),
              () {
                if (isMounted() && arState == ARState.locating) {
                  final raMs = _everDetected
                      ? _kLocateIntervalGuidedMs : _kLocateIntervalMs;
                  _locateTimer = Timer.periodic(
                      Duration(milliseconds: raMs), (_) => _locateTick());
                  _locateTick();
                }
              },
            );
          }
        });

        if (_locateAttempts >= _kMaxLocateAttempts) {
          _stopLocateLoop();
          if (isMounted()) {
            final part2 = (agentStep.requiredPart.isNotEmpty 
                ? agentStep.requiredPart : agentStep.visualCue)
                .replaceAll('_', ' ');
            final area2 = agentStep.areaHint.replaceAll('_', ' ');
            final hindi = langCode == 'hi';

            setState(() {
              dynamicFeedback = hindi
                  ? 'स्वचालित रूप से नहीं मिला।\n'
                    '👉 देखें: $part2${area2.isNotEmpty ? " → $area2" : ""}\n'
                    'कैमरा वहाँ ले जाएं और "भाग विश्लेषण करें" दबाएं।'
                  : 'Could not locate automatically.\n'
                    '👉 Look for: $part2${area2.isNotEmpty ? " at $area2" : ""}\n'
                    'Point camera there and tap Analyze Part.';
              arState = ARState.scanning;
            });
          }
        }
      }
    } finally {
      _locateRunning = false;
    }
  }
  /// Called every frame when in blind search mode.
  /// If camera is steady for 2.5s, triggers a cloud call to check what's visible.
  void _checkReticleSteady() {
    if (!_isInBlindSearch) return;
    if (!isMounted()) return;
    
    final tracking = this.tracking;
    
    // Widen the threshold slightly to 0.05 for budget device sensor noise.
    final isSteady = tracking.velCx.abs() < 0.05 && tracking.velCy.abs() < 0.05;
    
    if (isSteady && !_reticleSteady) {
      // User just stopped moving — start the stability countdown
      _reticleSteady = true;
      _reticleSteadyStart = DateTime.now();
      setState(() => cameraGuidance = 'Hold steady — analyzing...');
    } else if (isSteady && _reticleSteady) {
      // Check if 2.5 seconds have passed
      final elapsed = DateTime.now().difference(_reticleSteadyStart!).inMilliseconds;
      if (elapsed >= 2500) {
        // Trigger a cloud call!
        _reticleSteady = false;
        _reticleSteadyStart = null;
        _triggerBlindSearchCloudCall();
      }
    } else if (!isSteady) {
      // User is moving — reset countdown
      _reticleSteady = false;
      _reticleSteadyStart = null;
    }
  }

  /// Fires ONE cloud call to check what the user is looking at.
  /// VLM returns: verified, detected_part, guidance_vector (if wrong part)
  Future<void> _triggerBlindSearchCloudCall() async {
    if (!isMounted()) return;
    
    final ctx = getContext();
    final agentProv = ctx.read<AgentSessionProvider>();
    final agentStep = agentProv.current?.nextStep;
    if (agentStep == null) return;
    final machine = agentProv.current?.updatedMemory['machine_type'] as String?
                    ?? ctx.read<DiagnosisProvider>().solution?.machineType
                    ?? 'water_pump';
    final isHindi = ctx.read<LanguageProvider>().languageCode == 'hi';
    
    setState(() => arState = ARState.analyzing);
    
    try {
      final frame = await captureFrame();
      if (frame == null) return;
      final bytes = await frame.readAsBytes();
      
      // Quality check first
      final qResult = await ARQualityGate.check(bytes);
      if (!qResult.ok) {
        setState(() => cameraGuidance = qResult.message);
        transitionTo(ARState.scanning);
        return;
      }
      
    final result = await ApiService.verifyStep(
      imageFile: frame, 
      stepText: agentStep.localizedText(isHindi),
      requiredPart: agentStep.requiredPart,
      areaHint: agentStep.areaHint,
      machineType: machine,
      problemContext: '',
      attemptCount: 1,
      isBlindSearch: true,
    );
      
      final verified = result['verified'] == true;
      
      if (verified) {
        // Phase 3: Part found! Activate Kalman lock
        _isInBlindSearch = false;
        cloudGuidanceVector = '';
        setState(() => showOffScreenArrow = false);
        
        // Initialize Kalman with VLM's bbox
        final bbox = result['bbox'] as List?;
        if (bbox != null && bbox.length == 4) {
          tracking.update(
            cx: ((bbox[0] as num) + (bbox[2] as num)) / 2.0,
            cy: ((bbox[1] as num) + (bbox[3] as num)) / 2.0,
            bw: ((bbox[2] as num) - (bbox[0] as num)).toDouble(),
            bh: ((bbox[3] as num) - (bbox[1] as num)).toDouble(),
            confidence: (result['confidence'] as num?)?.toDouble() ?? 0.85,
          );
          transitionTo(ARState.guiding);
          HapticFeedback.mediumImpact();
        }
      } else {
        // Phase 2: Wrong part — extract direction from Gemini's spatial guidance
        final feedback = isHindi
            ? (result['feedback_hi'] as String? ?? 'गलत भाग — पुनः प्रयास करें')
            : (result['feedback_en'] as String? ?? 'Wrong part — try again');

        // Try guidance_vector first, then extract from feedback text
        final vectorFromApi = result['guidance_vector'] as String?;
        final extractedDir = vectorFromApi != null && vectorFromApi.isNotEmpty
            ? vectorFromApi
            : _extractDirectionFromGuidance(feedback);
        
        cloudGuidanceVector = extractedDir ?? '';
        showOffScreenArrow = extractedDir != null && extractedDir.isNotEmpty;
            
        setState(() {
          dynamicFeedback = feedback;
          cameraGuidance = feedback;
        });
        
        // Speak the correction
        await tts.stop();
        await tts.speak(isHindi ? result['feedback_hi'] ?? feedback : result['feedback_en'] ?? feedback);
        
        transitionTo(ARState.scanning);
      }
    } catch (e) {
      debugPrint('Blind search cloud call failed: $e');
      transitionTo(ARState.scanning);
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // TOAST
  // ══════════════════════════════════════════════════════════════════════════

  Duration _dismissDuration(ToastKind k) {
    switch (k) {
      case ToastKind.analyzing:  return const Duration(seconds: 30);
      case ToastKind.sent:       return const Duration(seconds: 5);
      case ToastKind.analyzed:   return const Duration(seconds: 5);
      case ToastKind.resultOk:   return const Duration(seconds: 6);
      case ToastKind.resultWarn: return const Duration(seconds: 8);
      case ToastKind.error:      return const Duration(seconds: 8);
      case ToastKind.verifying:  return const Duration(seconds: 15);
      case ToastKind.inspecting: return const Duration(seconds: 15);      
    }
  }

  Future<void> showToast(ToastKind kind) async {
    if (!isMounted()) return;
    if (toastCtrl.isAnimating || toastCtrl.isCompleted) {
      await toastCtrl.reverse();
    }
    if (!isMounted()) return;
    setState(() => toastKind = kind);
    toastCtrl.forward(from: 0);
    _scheduleToastDismiss(_dismissDuration(kind));
  }

  void _scheduleToastDismiss(Duration delay) {
    toastTimer?.cancel();
    toastTimer = Timer(delay, () async {
      if (!isMounted()) return;
      await toastCtrl.reverse();
      if (isMounted() && arState == ARState.unclear) {
        transitionTo(ARState.scanning);
      }
    });
  }

  // ══════════════════════════════════════════════════════════════════════════
  // STATE TRANSITIONS
  // ══════════════════════════════════════════════════════════════════════════

  void transitionTo(ARState next) {
    if (!isMounted()) return;
    setState(() => arState = next);
    switch (next) {
      case ARState.scanning:
      case ARState.locating:
        toastTimer?.cancel();
        toastCtrl.reverse();
        break;
      case ARState.guiding:
        break;
      case ARState.analyzing:
        break;
      case ARState.verifying:
      case ARState.inspecting:
      case ARState.repairing:
        break;
      case ARState.unclear:
        break;
      case ARState.verified:
        _stopLocateLoop();
        _partLocked         = false;
        _lastCorrectionSent = null;
        toastTimer?.cancel();
        toastCtrl.reverse();
        verifiedCtrl.forward(from: 0);
        break;
      case ARState.danger:
        _stopLocateLoop();
        toastTimer?.cancel();
        toastCtrl.reset();
        break;
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // AGENT STATUS HANDLER
  // ══════════════════════════════════════════════════════════════════════════

  void handleAgentStatus(AgentStatus? status) {
    if (status == null || !isMounted()) return;
    switch (status) {
      case AgentStatus.resolved:
      case AgentStatus.escalate:
        // Navigate back — SolutionScreen will show escalation card
        if (isMounted()) {
          Navigator.of(getContext()).pop();
        }
        break;
      case AgentStatus.unsafe:
        setState(() {
          dangerMessage = 'The agent detected an unsafe condition. Stopping repair.';
        });
        transitionTo(ARState.danger);
        break;
      default:
        break;
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // CAPTURE / VERIFY
  // ══════════════════════════════════════════════════════════════════════════

  Future<void> onCapture() async {
    if (arState == ARState.analyzing) return;
    if (!cameraReady || cameraController == null) return;
    final ctx   = getContext();
    final agentProv = ctx.read<AgentSessionProvider>();
    final agentStep = agentProv.current?.nextStep;
    if (agentStep == null) return;
    final machine  = agentProv.current?.updatedMemory['machine_type'] as String?
                     ?? ctx.read<DiagnosisProvider>().solution?.machineType
                     ?? 'tractor';
    final isHindi  = ctx.read<LanguageProvider>().languageCode == 'hi';
    final problem  = ctx.read<DiagnosisProvider>().solution?.getLocalizedProblem(isHindi) ?? '';
    
    final interactionType = agentStep.interaction?.type;
    
    if (interactionType == InteractionType.camera) {
        // ── Camera flow: AR locate → verify → inspect ─────────────────
        _stopLocateLoop();
        _partLocked         = false;
        _lastCorrectionSent = null;
        setState(() => bboxLocked = true);
        HapticFeedback.mediumImpact();
        attemptCount++;
        await pauseCamera();
        transitionTo(ARState.analyzing);
        await showToast(ToastKind.analyzing);

        File? tempFile;
        try {
          tempFile = await captureFrame();
        } catch (_) {
          await resumeCamera();
          transitionTo(ARState.locating);
          maybeStartLocateLoop();
          return;
        }

        if (tempFile == null) {
          await resumeCamera();
          transitionTo(ARState.locating);
          maybeStartLocateLoop();
          return;
        }
        
        final File imageFile = tempFile;
        await showToast(ToastKind.sent);

        final verifyBytes = await imageFile.readAsBytes();
        final vq = await ARQualityGate.check(verifyBytes);
        if (!vq.ok) {
          await resumeCamera();
          setState(() {
            bboxLocked      = false;
            dynamicFeedback = vq.message;
          });
          await showToast(ToastKind.resultWarn);
          transitionTo(ARState.unclear);
          Future.delayed(const Duration(seconds: 2), maybeStartLocateLoop);
          return;
        }

        final cropBytes = tracking.smoothBbox != null
            ? await ARCropHelper.cropToBbox(verifyBytes, tracking.smoothBbox!)
            : null;

        final stepText = agentStep.textEn.isNotEmpty
            ? agentStep.textEn : '';
        final reqPart  = agentStep.requiredPart.isNotEmpty
            ? agentStep.requiredPart : agentStep.visualCue;
        final areaHint = agentStep.areaHint;

        Map<String, dynamic> result;
        try {
          result = await ApiService.verifyStep(
            imageFile:      imageFile,
            imageCropBytes: cropBytes,
            stepText:       stepText,
            machineType:    machine,
            problemContext: problem,
            attemptCount:   attemptCount,
            requiredPart:   reqPart, // Flutter guarantees this is a string (''), never null
            areaHint:       areaHint,
            trackingScope:  agentStep.trackingScope,
            previousSteps:  jsonEncode(attemptResults),
          );
        } on Exception catch (e) {
          debugPrint('ARGuide verifyStep error: $e');
          await showToast(ToastKind.error);
          await resumeCamera();
          setState(() => bboxLocked = false);
          transitionTo(ARState.locating);
          Future.delayed(const Duration(seconds: 2), maybeStartLocateLoop);
          return;
        }

        await showToast(ToastKind.analyzed);

        final isDangerous = result['danger'] == true ||
            result['status'] == 'danger' ||
            (result['severity'] as String? ?? '').toLowerCase() == 'critical';

        if (isDangerous) {
          if (isMounted()) {
            setState(() {
              dangerMessage = result['danger_message'] as String? ??
                  'STOP — Critical safety hazard detected!\n\n'
                  'The machine appears to be running or there is an immediate risk.\n'
                  'Do NOT proceed until the machine is fully off and safe.';
            });
          }
          transitionTo(ARState.danger);
          HapticFeedback.vibrate();
          return;
        }

        final verified = result['verified'] == true ||
            result['status'] == 'verified';

        if (verified) {
          await _runInspection(verifyBytes, reqPart, areaHint, machine, isHindi);
        } else {
          attemptResults.add({
            'attempt_count': result['attempt_count'] ?? attemptCount,
            'status':        result['status']        ?? 'unclear',
            'detected_part': result['detected_part'] ?? '',
            'feedback':      result['feedback']      ?? '',
          });
          if (isMounted()) {
            setState(() {
              final raw = isHindi
                  ? (result['feedback_hi'] ?? result['feedback'])
                  : result['feedback'];
              dynamicFeedback = raw ??
                  result['ai_observation'] ??
                  'Image unclear or wrong part captured — see hint below for guidance.';
            });
          }
          await showToast(ToastKind.resultWarn);
          await resumeCamera();
          setState(() => bboxLocked = false);
          transitionTo(ARState.unclear);
          Future.delayed(const Duration(seconds: 2), maybeStartLocateLoop);
        }
    } else {
        // ── Non-camera flow: open the inspection panel ─────────────────
        agentPanelModel = InspectionPanelModel.fromAgentStep(agentStep);
        setState(() => inspectionPanelVisible = true);
    }
}
    /// Runs AFTER verification confirms the correct part.
  /// Sends the frozen frame to /inspect_part for damage analysis.
  Future<void> _runInspection(
    Uint8List imageBytes,
    String requiredPart,
    String areaHint,
    String machineType,
    bool isHindi,
  ) async {
    transitionTo(ARState.inspecting);
    
    try {
      final result = await ApiService.inspectPart(
        imageBytes: imageBytes,
        machineType: machineType,
        requiredPart: requiredPart,
        areaHint: areaHint,
        language: isHindi ? 'hi' : 'en',
      );
      
      inspectionSnapshot = InspectionSnapshot.fromJson(result, imageBytes.toList());
      
      final outcome = inspectionSnapshot!.outcome;
      
      if (outcome == InspectionOutcome.damaged) {
        // Damage found — stay frozen for review
        HapticFeedback.heavyImpact();
        
        final desc = inspectionSnapshot!.getLocalizedDescription(isHindi);
        if (desc.isNotEmpty) {
          await tts.stop();
          await tts.speak(desc);
        }
        // Stay in inspecting state — UI shows InspectionOverlay
        
      } else if (outcome == InspectionOutcome.healthy) {
        // No damage — advance agent
        final ctx = getContext();
        final agentProv = ctx.read<AgentSessionProvider>();
        lastCompletedInteractionType = InteractionType.camera;
        await agentProv.advance({
          'status': 'inspected',
          'outcome': 'healthy',
          'observations': inspectionSnapshot!.observations,
        });
        handleAgentStatus(agentProv.current?.status);
        await showToast(ToastKind.resultOk);
        transitionTo(ARState.verified);
        HapticFeedback.heavyImpact();
        
      } else {
        // Unclear/hidden/dirty — let user retry
        setState(() {
          dynamicFeedback = inspectionSnapshot!.getLocalizedDescription(isHindi);
        });
        await showToast(ToastKind.resultWarn);
        await resumeCamera();
        setState(() => bboxLocked = false);
        transitionTo(ARState.unclear);
        Future.delayed(const Duration(seconds: 2), maybeStartLocateLoop);
      }
      
    } catch (e) {
      debugPrint('Inspection failed: $e');
      await resumeCamera();
      setState(() => bboxLocked = false);
      transitionTo(ARState.unclear);
      Future.delayed(const Duration(seconds: 2), maybeStartLocateLoop);
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // STEP NAVIGATION
  // ══════════════════════════════════════════════════════════════════════════

  // NOTE: does NOT call agentProv.advance() — the step shown here was
  // already fetched by whichever handler (onActionDone / onInspectionAnswer /
void nextPart(List<StepData> steps) {
    HapticFeedback.lightImpact();
    _stopLocateLoop();
    resumeCamera();

    final ctx = getContext();
    final agentProv = ctx.read<AgentSessionProvider>();

    handleAgentStatus(agentProv.current?.status);

    setState(() {
      currentStep++;
      attemptCount = 0;
      panelExpanded = false;
      dynamicFeedback = '';
      cameraGuidance = '';
      showOffScreenArrow = false;
      cloudGuidanceVector = '';
      partDescription = '';
      bboxLocked = false;
      _partLocked = false;
      _lastCorrectionSent = null;
      _netFailCount = 0;
      _everDetected = false;
      _trackingEnabled = false;
      inspectionPanelVisible = false;
    });
    tracking.reset();
    guidance.resetOnStepChange();
    attemptResults.clear();
    verifiedCtrl.reset();
    bboxFadeCtrl.reset();

    transitionTo(ARState.scanning);
    final nextStep = agentProv.current?.nextStep;
    
    // FIX: Route to camera if there's ANY visual target
    if (nextStep != null && (nextStep.requiredPart.isNotEmpty || nextStep.visualCue.isNotEmpty)) {
      maybeStartLocateLoop();
    }
  }

void onInspectionAnswer(StepOption option, List<StepData> steps) {
    HapticFeedback.mediumImpact();
    setState(() => inspectionPanelVisible = false);
    transitionTo(ARState.analyzing);           // ← ADD THIS
    showToast(ToastKind.analyzing);            // ← ADD THIS

    final ctx = getContext();
    final agentProv = ctx.read<AgentSessionProvider>();

    final interactionOption = InteractionOption(
      id: option.id,
      label: option.getLocalizedLabel(false),
      nextState: option.nextStep,
    );
    final result = AgentResultBuilder.fromChoice(interactionOption);
    lastCompletedInteractionType = agentProv.current?.nextStep.interaction?.type;
    
    agentProv.advance(result).then((_) {
      if (!isMounted()) return;
      
      final status = agentProv.current?.status; 
      handleAgentStatus(status);
      
      if (status == AgentStatus.continueFlow) {
        transitionTo(ARState.verified);
        HapticFeedback.heavyImpact();
      }
    });
}

void onActionDone() {
    HapticFeedback.heavyImpact();
    setState(() => inspectionPanelVisible = false);
    transitionTo(ARState.analyzing);           // ← ADD THIS
    showToast(ToastKind.analyzing);            // ← ADD THIS

    final ctx = getContext();
    final agentProv = ctx.read<AgentSessionProvider>();
    lastCompletedInteractionType = agentProv.current?.nextStep.interaction?.type;
    
    agentProv.advance(AgentResultBuilder.fromBoolean(true, 'done')).then((_) {
      if (!isMounted()) return;
      final status = agentProv.current?.status; 
      handleAgentStatus(status);
      if (status == AgentStatus.continueFlow) {
        transitionTo(ARState.verified);
        HapticFeedback.heavyImpact();
      }
    });
}

void checkAgentPanel() {
      if (inspectionPanelVisible) return;
      if (arState != ARState.scanning) return;
      
      final ctx = getContext();
      final agentStep = ctx.read<AgentSessionProvider>().current?.nextStep;
      if (agentStep == null) return;
      final interaction = agentStep.interaction;
      if (interaction == null) return;
      
      // FIX: If there is a part to look at, DO NOT auto-open the panel.
      // We must force the user to point the camera and press "Analyze Part" first.
      if (agentStep.requiredPart.isNotEmpty || agentStep.visualCue.isNotEmpty) {
        return;
      }
      
      if (interaction.type == InteractionType.camera) return;
      
      _stopLocateLoop();
      agentPanelModel = InspectionPanelModel.fromAgentStep(agentStep);
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (isMounted()) setState(() => inspectionPanelVisible = true);
      });
  }

  void maybeShowInspectionPanel(StepData? step) {
    if (step == null) return;
    if ((step.requiresDecisionPanel || step.isActionStep) &&
        !inspectionPanelVisible &&
        arState == ARState.scanning) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (isMounted()) setState(() => inspectionPanelVisible = true);
      });
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // ORIENTATION CHANGE
  // ══════════════════════════════════════════════════════════════════════════

  void onOrientationChange() {
    setState(() {
      tracking.smoothBbox       = null;
      tracking.prevBbox         = null;
      tracking.stableFrameCount = 0;
      bboxLocked                = false;
      if (arState == ARState.guiding) arState = ARState.locating;
    });
    _stopTrackingTimer();
    _partLocked         = false;
    _lastCorrectionSent = null;
    tracking.resetKalman();
    tracking.resetVelocity();
    bboxFadeCtrl.reverse();
    if (arState != ARState.verified) maybeStartLocateLoop();
  }

  // ══════════════════════════════════════════════════════════════════════════
  // DISPOSE
  // ══════════════════════════════════════════════════════════════════════════

  void dispose() {
    _locateTimer?.cancel();
    _bboxLockTimer?.cancel();
    _trackingTimer?.cancel();
    toastTimer?.cancel();
    tts.stop();
    cameraController?.dispose();
  }
}

// ── Demo steps fallback ────────────────────────────────────────────────────
final demoSteps = List.generate(12, (i) => StepData(
  stepId:      's${i + 1}',
  stepType:    i == 2 ? StepType.inspection
             : i == 6 ? StepType.action
             : i == 9 ? StepType.observation
             : StepType.visual,
  stepTitleEn: 'Step ${i + 1}',
  stepTitleHi: 'चरण ${i + 1}',
  text:      'Inspect component ${i + 1} carefully before proceeding.',
  textEn:    'Inspect component ${i + 1} carefully before proceeding.',
  textHi:    '',
  action:    'Inspect component ${i + 1}',
  visualCue: i == 3 ? 'red_cable'
           : i == 4 ? 'fuse_box'
           : i == 5 ? 'battery_terminal'
           : null,
  requiredPart: i == 3 ? 'red_cable'
              : i == 4 ? 'fuse_box'
              : i == 5 ? 'battery_terminal'
              : '',
  areaHint:     i >= 3 && i <= 5 ? 'electrical_panel' : '',
  questionEn: i == 2 ? 'What is the condition of the component?' : null,
  questionHi: i == 2 ? 'घटक की क्या स्थिति है?' : null,
  options: i == 2 ? [
    StepOption(id: 'a', labelEn: 'No damage visible',   labelHi: 'कोई क्षति नहीं', nextStep: 's4'),
    StepOption(id: 'b', labelEn: 'Minor damage',         labelHi: 'मामूली क्षति',   nextStep: 's4'),
    StepOption(id: 'c', labelEn: 'Significant damage',   labelHi: 'गंभीर क्षति',    nextStep: 's_replace'),
    StepOption(id: 'd', labelEn: "Haven't checked yet",  labelHi: 'अभी नहीं देखा', nextStep: 's3'),
  ] : [],
));