// ignore_for_file: unnecessary_non_null_assertion
import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';
import 'package:record/record.dart';
import 'package:video_player/video_player.dart';
import 'package:video_thumbnail/video_thumbnail.dart';
import 'widgets/analysis_bottom_sheet.dart';
import 'widgets/safety_gate.dart';
import '../../core/theme.dart';
import '../../core/router.dart';
import '../../core/providers/diagnosis_provider.dart';
import '../../core/providers/language_provider.dart';
import '../../services/api_service.dart';
import '../../l10n/app_localizations.dart';

// ─────────────────────────────────────────────────────────────────────────────
// State enums
// ─────────────────────────────────────────────────────────────────────────────
enum UploadState { idle, uploading, transcribing, analyzing, generating, ready, error }
enum RecordState { idle, recording, paused, done }

// ─────────────────────────────────────────────────────────────────────────────
// Media limits
// Single source of truth — change here, enforced everywhere.
// ─────────────────────────────────────────────────────────────────────────────
const int    kVideoMaxSec = 20;          // live video auto-stops at 20 s
const int    kAudioMaxSec = 20;          // live audio auto-stops at 20 s
const int    kVideoMaxMB  = 20;          // gallery video rejected above 20 MB
const int    kAudioMaxMB  = 5;           // gallery audio rejected above 5 MB

// ─────────────────────────────────────────────────────────────────────────────
// UploadScreen
// ─────────────────────────────────────────────────────────────────────────────
class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key});
  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen>
    with TickerProviderStateMixin {

  // ── Media state ───────────────────────────────────────────────────────────
  String?                _videoPath;
  String?                _videoName;
  VideoPlayerController? _videoThumbCtrl;
  bool                   _videoThumbReady = false;

  String? _audioPath;
  String? _audioName;

  final _picker = ImagePicker();

  // ─────────────────────────────────────────────────────────────────────────
  // Lifecycle
  // ─────────────────────────────────────────────────────────────────────────

  @override
  void initState() {
    super.initState();
  }

  @override
  void dispose() {
    _videoThumbCtrl?.dispose();
    super.dispose();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Helpers
  // ─────────────────────────────────────────────────────────────────────────

  bool get _bothSelected => _videoPath != null && _audioPath != null;

  void _safeSetState(VoidCallback fn) {
    if (mounted) setState(fn);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Video helpers
  // ─────────────────────────────────────────────────────────────────────────

  Future<void> _loadVideoThumbnail(String path) async {
    _videoThumbCtrl?.dispose();
    try {
      final ctrl = VideoPlayerController.file(File(path));
      await ctrl.initialize();
      await ctrl.seekTo(Duration.zero);
      if (mounted) {
        setState(() {
          _videoThumbCtrl  = ctrl;
          _videoThumbReady = true;
        });
      }
    } catch (_) {
      // If thumbnail fails, still mark video as selected — just no preview
    }
  }

  void _setVideo(String path, String name) {
    _safeSetState(() {
      _videoPath       = path;
      _videoName       = name;
      _videoThumbReady = false;
    });
    _loadVideoThumbnail(path);
  }

  void _clearVideo() {
    _videoThumbCtrl?.dispose();
    _safeSetState(() {
      _videoPath       = null;
      _videoName       = null;
      _videoThumbCtrl  = null;
      _videoThumbReady = false;
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Frame extraction helper (fast-path upload)
  // ─────────────────────────────────────────────────────────────────────────
  // Extracts 3 JPEG frames at 15% / 40% / 70% of video duration using the
  // video_thumbnail package (timeMs parameter). Falls back to full-video
  // upload if extraction fails on any frame.
  //
  // Output size: 3 × ~50 KB JPEG = ~150 KB  (vs 20–35 MB for the raw .mp4)
  // ─────────────────────────────────────────────────────────────────────────
  Future<List<String>?> _extractVideoFrames(String videoPath) async {
    try {
      final videoFile = File(videoPath);
      if (!videoFile.existsSync()) return null;

      // Get video duration via VideoPlayerController (already in pubspec)
      final ctrl = VideoPlayerController.file(videoFile);
      await ctrl.initialize();
      final totalMs = ctrl.value.duration.inMilliseconds;
      await ctrl.dispose();

      if (totalMs < 300) return null; // degenerate / corrupt video

      // ── Why video_thumbnail instead of video_compress ─────────────────────
      // video_compress.getFileThumbnail ignores the `position` parameter on
      // Android — MediaMetadataRetriever always returns frame 0 regardless of
      // the value passed, which is why all 3 frames had identical blur=3224.3
      // bright=145.8 in logs. video_thumbnail.thumbnailFile uses `timeMs`
      // which is correctly forwarded to MediaMetadataRetriever on Android and
      // AVAssetImageGenerator on iOS — both reliably seek to the right frame.
      // ─────────────────────────────────────────────────────────────────────
      final tmpDir = await getTemporaryDirectory();
      final labels = ['early', 'mid', 'late'];
      final timesMs = [
        (totalMs * 0.15).round(),   // 15% — early
        (totalMs * 0.40).round(),   // 40% — mid
        (totalMs * 0.70).round(),   // 70% — late
      ];

      // Deduplicate for very short videos (< 3 s): force >= 100 ms apart
      for (int i = 1; i < 3; i++) {
        if (timesMs[i] <= timesMs[i - 1]) {
          timesMs[i] = timesMs[i - 1] + 100;
        }
        timesMs[i] = timesMs[i].clamp(0, totalMs - 1);
      }

      final framePaths = <String>[];
      for (int i = 0; i < 3; i++) {
        final destPath =
            '${tmpDir.path}/agrifix_frame_${labels[i]}_$hashCode.jpg';
        final result = await VideoThumbnail.thumbnailFile(
          video:         videoPath,
          thumbnailPath: tmpDir.path,   // directory — plugin writes the file
          imageFormat:   ImageFormat.JPEG,
          timeMs:        timesMs[i],    // milliseconds, reliable on Android + iOS
          quality:       85,
        );
        if (result == null) return null; // platform failed → fall back
        await File(result).copy(destPath);
        framePaths.add(destPath);
      }

      return framePaths.length == 3 ? framePaths : null;
    } catch (e) {
      // Any failure → fall back to original full-video upload path silently
      return null;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Audio helpers
  // ─────────────────────────────────────────────────────────────────────────

  void _setAudio(String path, String name) {
    _safeSetState(() {
      _audioPath = path;
      _audioName = name;
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Snackbar helper
  // ─────────────────────────────────────────────────────────────────────────

  void _showSizeError(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(msg, style: GoogleFonts.inter(fontSize: 13, color: Colors.white)),
      backgroundColor: Colors.redAccent,
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      margin: const EdgeInsets.fromLTRB(16, 0, 16, 16),
    ));
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Bottom-sheet launchers
  // ─────────────────────────────────────────────────────────────────────────

  void _showVideoSheet() {
    final l10n = AppLocalizations.of(context);
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (_) => _MediaBottomSheet(
        galleryLabel: l10n.uploadFromGallery,
        liveLabel:    l10n.uploadRecordLiveCamera,
        onGallery: () async {
          Navigator.pop(context);
          // Pass maxDuration so the OS-level picker enforces the cap too
          final file = await _picker.pickVideo(
            source: ImageSource.gallery,
            maxDuration: const Duration(seconds: kVideoMaxSec),
          );
          if (file == null || !mounted) return;
          final bytes = await file.length();
          if (bytes > kVideoMaxMB * 1024 * 1024) {
            _showSizeError('Video too large (max ${kVideoMaxMB} MB). Please pick a shorter clip.');
            return;
          }
          _setVideo(file.path, file.name);
        },
        onLive: () {
          Navigator.pop(context);
          _openLiveVideoPanel();
        },
      ),
    );
  }

  void _showAudioSheet() {
    final l10n = AppLocalizations.of(context);
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (_) => _MediaBottomSheet(
        galleryLabel: l10n.uploadFromGallery,
        liveLabel:    l10n.uploadRecordLiveMic,
        onGallery: () async {
          Navigator.pop(context);
          final file = await _picker.pickMedia();
          if (file == null || !mounted) return;
          final bytes = await file.length();
          if (bytes > kAudioMaxMB * 1024 * 1024) {
            _showSizeError('Audio too large (max ${kAudioMaxMB} MB). Please pick a shorter clip.');
            return;
          }
          _setAudio(file.path, file.name);
        },
        onLive: () {
          Navigator.pop(context);
          _openAudioRecorderPanel();
        },
      ),
    );
  }

  // Opens the live-video countdown panel (our own UI that wraps image_picker)
  void _openLiveVideoPanel() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      isDismissible: false,
      enableDrag: false,
      builder: (_) => _LiveVideoPanel(
        onSave:   (path) { Navigator.pop(context); _setVideo(path, 'Live Recording.mp4'); },
        onCancel: ()     { Navigator.pop(context); },
      ),
    );
  }

  // Opens the mic recorder panel (uses `record` package with full controls)
  void _openAudioRecorderPanel() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      isDismissible: false,
      enableDrag: false,
      builder: (_) => _AudioRecorderPanel(
        onSave:   (path) { Navigator.pop(context); _setAudio(path, 'Live Recording.m4a'); },
        onCancel: ()     { Navigator.pop(context); },
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Find Solution
  // ─────────────────────────────────────────────────────────────────────────

  Future<void> _onFindSolution() async {
    if (!_bothSelected) return;

    // ── Mandatory safety gate ─────────────────────────────────────────────────
    // Show before clearing state so the farmer confirms safety before any work
    // begins. Uses language from LanguageProvider. Zero Gemini calls.
    final langCode = context.read<LanguageProvider>().languageCode;
    final safetyOk = await showSafetyGate(context, languageCode: langCode);
    if (!safetyOk || !mounted) return;

    final diagProv = context.read<DiagnosisProvider>();
    diagProv.clear();
    // Record that safety was confirmed so AR screen can verify it.
    diagProv.confirmSafety();

    await showAnalysisSheet(
      context,
      runAnalysis: (markActive, markDone, markCacheHit) async {
        // ── Fast path: extract 3 JPEG frames locally, upload ~350 KB ─────────
        // Falls back to the original full-video upload if extraction fails.
        // The fast path is transparent — all stage events, accuracy, and
        // token usage are identical to the original path.
        final framePaths = await _extractVideoFrames(_videoPath!);

        final DiagnosisResult result;
        if (framePaths != null) {
          // Fast path: ~350 KB upload instead of ~25 MB
          result = await ApiService.uploadAndDiagnoseStreamingFast(
            framePaths: framePaths,
            audioPath: _audioPath!,
            onStageStart: (stageIndex) => markActive(stageIndex),
            onStageComplete: (stageIndex, data) {
              markDone(stageIndex);
              if (stageIndex == 3 && data.containsKey('result')) {
                final res = data['result'] as Map<String, dynamic>;
                if (data['cache_hit'] == true || res['cache_hit'] == true) {
                  markCacheHit();
                }
                diagProv.setDiagnosis(res);
              }
            },
          );
        } else {
          // Fallback: original full-video path (unchanged)
          result = await ApiService.uploadAndDiagnoseStreaming(
            videoPath: _videoPath!,
            audioPath: _audioPath!,
            onStageStart: (stageIndex) => markActive(stageIndex),
            onStageComplete: (stageIndex, data) {
              markDone(stageIndex);
              if (stageIndex == 3 && data.containsKey('result')) {
                final res = data['result'] as Map<String, dynamic>;
                if (data['cache_hit'] == true || res['cache_hit'] == true) {
                  markCacheHit();
                }
                diagProv.setDiagnosis(res);
              }
            },
          );
        }
        if (!diagProv.hasSolution) diagProv.setDiagnosis(result.raw);
      },
      onCompleted: () {
        if (mounted) context.push(AppRoutes.solution);
      },
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Build
  // ─────────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;

    return Scaffold(
      backgroundColor: AppColors.pageBg,
      body: Stack(
        fit: StackFit.expand,
        children: [

          // Layer 1 — scrollable content
          SafeArea(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(
                  horizontal: AppSpacing.screenPadding, vertical: AppSpacing.sm),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _Header(l10n: l10n),
                  const SizedBox(height: 16),
                  _StepChip(l10n: l10n),
                  const SizedBox(height: 12),
                  Text(l10n.uploadHeading, style: AppTextStyles.uploadHeading),
                  const SizedBox(height: 8),
                  Text(l10n.uploadSubtext, style: AppTextStyles.uploadSubtext),
                  const SizedBox(height: 24),

                  // ── Video thumbnail — expands in smoothly above the card ──
                  AnimatedSize(
                    duration: const Duration(milliseconds: 320),
                    curve: Curves.easeOutCubic,
                    alignment: Alignment.topCenter,
                    child: _videoThumbReady && _videoThumbCtrl != null
                        ? _VideoThumbnail(
                            controller: _videoThumbCtrl!,
                            onRemove:   _clearVideo,
                          )
                        : const SizedBox.shrink(),
                  ),

                  // ── Video card ───────────────────────────────────────────
                  // limitHint shown as small muted line inside the card body
                  _MediaCard(
                    icon:                Icons.videocam_outlined,
                    title:               l10n.uploadRecordVideo,
                    subtitle:            _videoName ?? l10n.uploadRecordVideoSub,
                    limitHint:           'Max ${kVideoMaxSec}s · Max ${kVideoMaxMB} MB',
                    isSelected:          _videoPath != null,
                    selectedBorderColor: AppColors.uploadGreen,
                    selectedBg:          AppColors.uploadGreenBg,
                    onTap:               _showVideoSheet,
                  ),
                  const SizedBox(height: 20),

                  // ── Audio card ───────────────────────────────────────────
                  _MediaCard(
                    icon:                Icons.mic_outlined,
                    title:               l10n.uploadRecordAudio,
                    subtitle:            _audioName ?? l10n.uploadRecordAudioSub,
                    limitHint:           'Max ${kAudioMaxSec}s duration',
                    isSelected:          _audioPath != null,
                    selectedBorderColor: AppColors.uploadBlue,
                    selectedBg:          AppColors.uploadBlueBg,
                    onTap:               _showAudioSheet,
                  ),
                  const SizedBox(height: 24),

                  _ProTipCard(l10n: l10n),
                  const SizedBox(height: 96),
                ],
              ),
            ),
          ),

          // Find Solution button — slides up from bottom when both selected
          if (_bothSelected)
            Positioned(
              left:   AppSpacing.screenPadding,
              right:  AppSpacing.screenPadding,
              bottom: MediaQuery.of(context).padding.bottom + 20,
              child:  _FindSolutionButton(l10n: l10n, onTap: _onFindSolution),
            ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// _VideoThumbnail
// ─────────────────────────────────────────────────────────────────────────────
class _VideoThumbnail extends StatelessWidget {
  final VideoPlayerController controller;
  final VoidCallback           onRemove;
  const _VideoThumbnail({required this.controller, required this.onRemove});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(AppSpacing.uploadCardRadius),
        child: Stack(
          children: [
            AspectRatio(
              aspectRatio: 16 / 9,
              child: ClipRect(
                child: FittedBox(
                  fit: BoxFit.cover,
                  alignment: Alignment.center,
                  child: SizedBox(
                    width: controller.value.size.width,
                    height: controller.value.size.height,
                    child: VideoPlayer(controller),
                  ),
                ),
              ),
            ),
            // Gradient scrim
            Positioned(
              left: 0, right: 0, bottom: 0,
              child: Container(
                height: 56,
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topCenter, end: Alignment.bottomCenter,
                    colors: [Colors.transparent, Color(0x99000000)],
                  ),
                ),
              ),
            ),
            // "Video selected" badge
            Positioned(
              left: 12, bottom: 10,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: AppColors.uploadGreen,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.check_rounded, color: Colors.white, size: 12),
                    const SizedBox(width: 4),
                    Text('Video selected',
                        style: GoogleFonts.inter(
                            fontSize: 11, fontWeight: FontWeight.w600, color: Colors.white)),
                  ],
                ),
              ),
            ),
            // × remove
            Positioned(
              top: 10, right: 10,
              child: GestureDetector(
                onTap: onRemove,
                child: Container(
                  width: 30, height: 30,
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.55),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(Icons.close_rounded, color: Colors.white, size: 16),
                ),
              ),
            ),
          ],
        ),
      ),
    )
    .animate()
    .fadeIn(duration: 280.ms, curve: Curves.easeOut)
    .slideY(begin: -0.06, end: 0, duration: 280.ms, curve: Curves.easeOutCubic);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// _LiveVideoPanel
//
// Shows a bottom-sheet countdown while image_picker camera is recording.
// • Circular arc sweeps from 0 → full in kVideoMaxSec seconds.
// • Turns red in the last 5 s.
// • When the timer hits 0 the picker is still running (maxDuration handles OS
//   stop) — we dismiss and accept whatever file was returned.
// • Tapping × cancels and discards the recording.
// ─────────────────────────────────────────────────────────────────────────────
class _LiveVideoPanel extends StatefulWidget {
  final void Function(String path) onSave;
  final VoidCallback               onCancel;
  const _LiveVideoPanel({required this.onSave, required this.onCancel});

  @override
  State<_LiveVideoPanel> createState() => _LiveVideoPanelState();
}

class _LiveVideoPanelState extends State<_LiveVideoPanel>
    with SingleTickerProviderStateMixin {

  final _picker   = ImagePicker();
  int     _elapsed = 0;          // seconds elapsed
  Timer?  _timer;
  bool    _launched = false;     // picker already opened
  bool    _done     = false;     // guard against double-calls

  @override
  void initState() {
    super.initState();
    // Launch picker + start our UI timer after first frame
    WidgetsBinding.instance.addPostFrameCallback((_) => _launch());
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  // ── Computed helpers ───────────────────────────────────────────────────

  int    get _remaining => (kVideoMaxSec - _elapsed).clamp(0, kVideoMaxSec);
  double get _fraction  => (_elapsed / kVideoMaxSec).clamp(0.0, 1.0);
  Color  get _arcColor  => _remaining <= 5
      ? Colors.redAccent
      : AppColors.uploadGreen;

  String get _timerLabel {
    final m = (_elapsed ~/ 60).toString().padLeft(2, '0');
    final s = (_elapsed  % 60).toString().padLeft(2, '0');
    return '$m:$s';
  }

  // ── Logic ──────────────────────────────────────────────────────────────

  Future<void> _launch() async {
    if (_launched || !mounted) return;
    _launched = true;

    // Kick off the UI countdown
    _timer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (!mounted) return;
      setState(() => _elapsed++);
      // At limit: dismiss panel; picker's maxDuration will have stopped recording
      if (_elapsed >= kVideoMaxSec && !_done) {
        _timer?.cancel();
        // Give the picker a moment to finalise the file, then accept whatever comes
      }
    });

    try {
      // maxDuration tells the OS to stop recording after kVideoMaxSec
      final file = await _picker.pickVideo(
        source: ImageSource.camera,
        maxDuration: const Duration(seconds: kVideoMaxSec),
      );
      _finish(file?.path);
    } catch (_) {
      _finish(null);
    }
  }

  void _finish(String? path) {
    if (_done) return;
    _done = true;
    _timer?.cancel();
    if (!mounted) return;
    if (path != null) {
      widget.onSave(path);
    } else {
      widget.onCancel();
    }
  }

  void _cancel() {
    _done = true;
    _timer?.cancel();
    widget.onCancel();
  }

  // ── Build ──────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final screenH = MediaQuery.of(context).size.height;

    return Container(
      height: screenH * 0.52,
      decoration: const BoxDecoration(
        color: AppColors.cardWhite,
        borderRadius: BorderRadius.only(
          topLeft:  Radius.circular(28),
          topRight: Radius.circular(28),
        ),
      ),
      child: Column(
        children: [

          const SizedBox(height: 12),
          // Drag handle
          Center(child: Container(
            width: 36, height: 4,
            decoration: BoxDecoration(
                color: AppColors.divider,
                borderRadius: BorderRadius.circular(2)),
          )),
          const SizedBox(height: 8),

          // Header
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Row(children: [
              Expanded(
                child: Text('Recording Video',
                    style: AppTextStyles.uploadCardTitle.copyWith(fontSize: 17)),
              ),
              GestureDetector(
                onTap: _cancel,
                child: Container(
                  width: 30, height: 30,
                  decoration: const BoxDecoration(
                      color: AppColors.iconBg, shape: BoxShape.circle),
                  child: const Icon(Icons.close_rounded,
                      size: 16, color: AppColors.textSecondary),
                ),
              ),
            ]),
          ),

          const Spacer(),

          // ── Circular countdown arc ─────────────────────────────────────
          SizedBox(
            width: 168, height: 168,
            child: Stack(alignment: Alignment.center, children: [

              // Track
              SizedBox(
                width: 168, height: 168,
                child: CircularProgressIndicator(
                  value: 1.0,
                  strokeWidth: 8,
                  color: AppColors.iconBg,
                ),
              ),

              // Animated filled arc
              TweenAnimationBuilder<double>(
                tween: Tween(begin: 0.0, end: _fraction),
                duration: const Duration(milliseconds: 350),
                curve: Curves.easeOut,
                builder: (_, val, __) => SizedBox(
                  width: 168, height: 168,
                  child: CircularProgressIndicator(
                    value: val,
                    strokeWidth: 8,
                    strokeCap: StrokeCap.round,
                    color: _arcColor,
                  ),
                ),
              ),

              // Centre — camera icon + seconds remaining
              Column(mainAxisSize: MainAxisSize.min, children: [
                Icon(
                  Icons.videocam_rounded,
                  size: 28,
                  color: _launched ? _arcColor : AppColors.textHint,
                ),
                const SizedBox(height: 2),
                AnimatedDefaultTextStyle(
                  duration: const Duration(milliseconds: 200),
                  style: GoogleFonts.inter(
                    fontSize: 30,
                    fontWeight: FontWeight.w300,
                    letterSpacing: 0,
                    color: _remaining <= 5
                        ? Colors.redAccent
                        : AppColors.textPrimary,
                  ),
                  child: Text('$_remaining'),
                ),
                Text('sec left',
                    style: GoogleFonts.inter(
                        fontSize: 10, color: AppColors.textHint)),
              ]),
            ]),
          ),

          const SizedBox(height: 16),

          // Elapsed timer label
          AnimatedDefaultTextStyle(
            duration: const Duration(milliseconds: 200),
            style: GoogleFonts.inter(
              fontSize: 13,
              color: _remaining <= 5
                  ? Colors.redAccent
                  : AppColors.textSecondary,
            ),
            child: Text(_launched ? _timerLabel : '00:00'),
          ),

          const SizedBox(height: 4),

          Text(
            _remaining <= 5 && _remaining > 0
                ? 'Stopping soon…'
                : _remaining == 0
                    ? 'Saving…'
                    : 'Camera is recording',
            style: GoogleFonts.inter(
                fontSize: 12, color: AppColors.textSecondary),
          ),

          const SizedBox(height: 6),

          // Limit hint
          Text(
            'Max ${kVideoMaxSec}s · Max ${kVideoMaxMB} MB',
            style: GoogleFonts.inter(
                fontSize: 11, color: AppColors.textHint),
          ),

          const Spacer(),

          // Thin progress bar at bottom of sheet
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 32),
            child: TweenAnimationBuilder<double>(
              tween: Tween(begin: 0.0, end: _fraction),
              duration: const Duration(milliseconds: 350),
              curve: Curves.easeOut,
              builder: (_, val, __) => ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: val,
                  minHeight: 4,
                  backgroundColor: AppColors.iconBg,
                  valueColor: AlwaysStoppedAnimation(_arcColor),
                ),
              ),
            ),
          ),

          SizedBox(height: MediaQuery.of(context).padding.bottom + 28),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// _AudioRecorderPanel
//
// Real microphone recording using the `record` package.
// Controls: × close | Reset | Pause/Resume | Done
// Visual:   Pulsing concentric rings + countdown progress bar
//
// Auto-stop: when _elapsed reaches kAudioMaxSec the recorder stops and
// the file is automatically saved — exactly as if the farmer tapped Done.
// ─────────────────────────────────────────────────────────────────────────────
class _AudioRecorderPanel extends StatefulWidget {
  final void Function(String path) onSave;
  final VoidCallback               onCancel;
  const _AudioRecorderPanel({required this.onSave, required this.onCancel});

  @override
  State<_AudioRecorderPanel> createState() => _AudioRecorderPanelState();
}

class _AudioRecorderPanelState extends State<_AudioRecorderPanel>
    with TickerProviderStateMixin {

  final _recorder = AudioRecorder();

  RecordState _state    = RecordState.idle;
  String?     _savePath;
  int         _elapsed  = 0;     // seconds elapsed (counts up)
  Timer?      _timer;
  String?     _errorMsg;
  bool        _autoSaved = false; // true when limit triggered auto-save

  // Waveform pulse animation
  late AnimationController _waveCtrl;
  late Animation<double>   _waveAnim;

  @override
  void initState() {
    super.initState();
    _waveCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 1100))
      ..repeat(reverse: true);
    _waveAnim = CurvedAnimation(parent: _waveCtrl, curve: Curves.easeInOut);

    WidgetsBinding.instance
        .addPostFrameCallback((_) => _startRecording());
  }

  @override
  void dispose() {
    _timer?.cancel();
    _waveCtrl.dispose();
    _recorder.dispose();
    super.dispose();
  }

  // ── Computed helpers ───────────────────────────────────────────────────

  int    get _remaining => (kAudioMaxSec - _elapsed).clamp(0, kAudioMaxSec);
  double get _fraction  => (_elapsed / kAudioMaxSec).clamp(0.0, 1.0);

  // Bar turns red in the last 5 seconds
  Color  get _barColor  => _remaining <= 5
      ? Colors.redAccent
      : AppColors.uploadGreen;

  String get _timerLabel {
    final m = (_elapsed ~/ 60).toString().padLeft(2, '0');
    final s = (_elapsed  % 60).toString().padLeft(2, '0');
    return '$m:$s';
  }

  // ── Timer ──────────────────────────────────────────────────────────────

  void _startTimer() {
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (!mounted) return;
      setState(() => _elapsed++);

      // Auto-save when limit reached
      if (_elapsed >= kAudioMaxSec && !_autoSaved) {
        _autoSaved = true;
        _stopAndSave();
      }
    });
  }

  void _stopTimer() => _timer?.cancel();

  // ── Controls ───────────────────────────────────────────────────────────

  Future<void> _startRecording() async {
    final status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      if (mounted) {
        setState(() => _errorMsg =
            'Microphone permission denied.\nPlease allow access in Settings.');
      }
      return;
    }
    try {
      final dir  = await getTemporaryDirectory();
      final path = '${dir.path}/rec_${DateTime.now().millisecondsSinceEpoch}.m4a';
      await _recorder.start(
        const RecordConfig(encoder: AudioEncoder.aacLc, bitRate: 128000),
        path: path,
      );
      if (mounted) {
        setState(() { _state = RecordState.recording; _savePath = path; });
        _startTimer();
        _waveCtrl.repeat(reverse: true);
      }
    } catch (e) {
      if (mounted) setState(() => _errorMsg = 'Could not start recording.\n$e');
    }
  }

  Future<void> _pauseRecording() async {
    await _recorder.pause();
    _stopTimer();
    _waveCtrl.stop();
    if (mounted) setState(() => _state = RecordState.paused);
  }

  Future<void> _resumeRecording() async {
    await _recorder.resume();
    _startTimer();
    _waveCtrl.repeat(reverse: true);
    if (mounted) setState(() => _state = RecordState.recording);
  }

  Future<void> _resetRecording() async {
    await _recorder.stop();
    _stopTimer();
    _waveCtrl.stop();
    if (mounted) setState(() {
      _state     = RecordState.idle;
      _elapsed   = 0;
      _autoSaved = false;
    });
    await _startRecording();
  }

  Future<void> _stopAndSave() async {
    final path = await _recorder.stop();
    _stopTimer();
    _waveCtrl.stop();
    final finalPath = path ?? _savePath;
    if (finalPath != null) {
      widget.onSave(finalPath);
    } else {
      widget.onCancel();
    }
  }

  Future<void> _cancelAndClose() async {
    await _recorder.stop();
    _stopTimer();
    widget.onCancel();
  }

  // ── Build ──────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final screenH = MediaQuery.of(context).size.height;

    return Container(
      height: screenH * 0.64,
      decoration: const BoxDecoration(
        color: AppColors.cardWhite,
        borderRadius: BorderRadius.only(
          topLeft:  Radius.circular(28),
          topRight: Radius.circular(28),
        ),
      ),
      child: Column(
        children: [
          const SizedBox(height: 12),

          // Drag handle
          Center(child: Container(
            width: 36, height: 4,
            decoration: BoxDecoration(
                color: AppColors.divider,
                borderRadius: BorderRadius.circular(2)),
          )),
          const SizedBox(height: 8),

          // Header: title + × close
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Row(children: [
              Expanded(
                child: Text('Record Audio',
                    style: AppTextStyles.uploadCardTitle.copyWith(fontSize: 17)),
              ),
              GestureDetector(
                onTap: _cancelAndClose,
                child: Container(
                  width: 30, height: 30,
                  decoration: const BoxDecoration(
                      color: AppColors.iconBg, shape: BoxShape.circle),
                  child: const Icon(Icons.close_rounded,
                      size: 16, color: AppColors.textSecondary),
                ),
              ),
            ]),
          ),

          const Spacer(),

          // ── Error state ────────────────────────────────────────────────
          if (_errorMsg != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Text(_errorMsg!,
                  textAlign: TextAlign.center,
                  style: GoogleFonts.inter(
                      fontSize: 14, color: Colors.redAccent, height: 1.5)),
            )
          else ...[

            // Mic pulsing rings
            _MicWaveWidget(
              waveAnim: _waveAnim,
              isActive: _state == RecordState.recording,
              isPaused: _state == RecordState.paused,
            ),

            const SizedBox(height: 18),

            // ── Elapsed timer (large) ──────────────────────────────────
            AnimatedDefaultTextStyle(
              duration: const Duration(milliseconds: 200),
              style: GoogleFonts.inter(
                fontSize: 38,
                fontWeight: FontWeight.w200,
                letterSpacing: 3,
                color: _state == RecordState.recording
                    ? (_remaining <= 5 ? Colors.redAccent : AppColors.uploadGreen)
                    : AppColors.textSecondary,
              ),
              child: Text(_state == RecordState.idle ? '00:00' : _timerLabel),
            ),

            const SizedBox(height: 5),

            // Status / auto-saved label
            Text(
              _autoSaved
                  ? '✓ Auto-saved at ${kAudioMaxSec}s'
                  : _state == RecordState.recording
                      ? 'Recording in progress…'
                      : _state == RecordState.paused
                          ? 'Paused — tap mic to resume'
                          : 'Starting…',
              style: GoogleFonts.inter(
                fontSize: 13,
                color: _autoSaved ? AppColors.uploadGreen : AppColors.textSecondary,
              ),
            ),

            const SizedBox(height: 16),

            // ── Countdown progress bar (visible once recording starts) ──
            if (_state == RecordState.recording || _state == RecordState.paused)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 36),
                child: Column(
                  children: [

                    // Labels row: "Max 20s"  ·····  "Xs left"
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          'Max ${kAudioMaxSec}s',
                          style: GoogleFonts.inter(
                              fontSize: 11, color: AppColors.textHint),
                        ),
                        AnimatedDefaultTextStyle(
                          duration: const Duration(milliseconds: 200),
                          style: GoogleFonts.inter(
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                            color: _remaining <= 5
                                ? Colors.redAccent
                                : AppColors.textSecondary,
                          ),
                          child: Text('${_remaining}s left'),
                        ),
                      ],
                    ),
                    const SizedBox(height: 7),

                    // Animated filled bar
                    TweenAnimationBuilder<double>(
                      tween: Tween(begin: 0.0, end: _fraction),
                      duration: const Duration(milliseconds: 350),
                      curve: Curves.easeOut,
                      builder: (_, val, __) => ClipRRect(
                        borderRadius: BorderRadius.circular(4),
                        child: LinearProgressIndicator(
                          value: val,
                          minHeight: 5,
                          backgroundColor: AppColors.iconBg,
                          valueColor: AlwaysStoppedAnimation(_barColor),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
          ],

          const Spacer(),

          // ── Control row ─────────────────────────────────────────────────
          if (_errorMsg == null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 40),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [

                  // Reset
                  _ControlBtn(
                    icon:  Icons.replay_rounded,
                    label: 'Reset',
                    color: AppColors.textSecondary,
                    onTap: _state != RecordState.idle ? _resetRecording : null,
                  ),

                  // Pause / Resume central button
                  GestureDetector(
                    onTap: _state == RecordState.recording
                        ? _pauseRecording
                        : _state == RecordState.paused
                            ? _resumeRecording
                            : null,
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 200),
                      width: 64, height: 64,
                      decoration: BoxDecoration(
                        color: _state == RecordState.recording
                            ? AppColors.uploadGreen
                            : AppColors.iconBg,
                        shape: BoxShape.circle,
                        boxShadow: _state == RecordState.recording
                            ? AppShadows.findSolutionButton
                            : [],
                      ),
                      child: Icon(
                        _state == RecordState.recording
                            ? Icons.pause_rounded
                            : Icons.mic_rounded,
                        color: _state == RecordState.recording
                            ? Colors.white
                            : AppColors.uploadGreen,
                        size: 30,
                      ),
                    ),
                  ),

                  // Done
                  _ControlBtn(
                    icon:  Icons.check_rounded,
                    label: 'Done',
                    color: _state != RecordState.idle
                        ? AppColors.uploadGreen
                        : AppColors.textHint,
                    onTap: _state != RecordState.idle ? _stopAndSave : null,
                  ),
                ],
              ),
            ),

          SizedBox(height: MediaQuery.of(context).padding.bottom + 28),
        ],
      ),
    );
  }
}

// ─── Small labeled icon button ────────────────────────────────────────────────
class _ControlBtn extends StatelessWidget {
  final IconData      icon;
  final String        label;
  final Color         color;
  final VoidCallback? onTap;

  const _ControlBtn({
    required this.icon,
    required this.label,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Opacity(
        opacity: onTap == null ? 0.30 : 1.0,
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Container(
            width: 46, height: 46,
            decoration: BoxDecoration(
                color: color.withValues(alpha: 0.10), shape: BoxShape.circle),
            child: Icon(icon, color: color, size: 22),
          ),
          const SizedBox(height: 4),
          Text(label,
              style: GoogleFonts.inter(
                  fontSize: 11, fontWeight: FontWeight.w500, color: color)),
        ]),
      ),
    );
  }
}

// ─── Mic with concentric pulsing rings ────────────────────────────────────────
class _MicWaveWidget extends StatelessWidget {
  final Animation<double> waveAnim;
  final bool isActive;
  final bool isPaused;

  const _MicWaveWidget({
    required this.waveAnim,
    required this.isActive,
    required this.isPaused,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 170, height: 170,
      child: Stack(
        alignment: Alignment.center,
        children: [

          if (isActive)
            AnimatedBuilder(
              animation: waveAnim,
              builder: (_, __) => _ring(
                  size: 148 + 14 * waveAnim.value,
                  alpha: 0.08 + 0.04 * waveAnim.value),
            ),

          if (isActive)
            AnimatedBuilder(
              animation: waveAnim,
              builder: (_, __) {
                final t = math.sin(waveAnim.value * math.pi);
                return _ring(size: 120 + 10 * t, alpha: 0.12 + 0.06 * t);
              },
            ),

          if (isActive)
            AnimatedBuilder(
              animation: waveAnim,
              builder: (_, __) => _ring(
                  size: 96 + 8 * waveAnim.value,
                  alpha: 0.18 + 0.08 * waveAnim.value),
            ),

          // Core mic circle
          AnimatedContainer(
            duration: const Duration(milliseconds: 250),
            width: 74, height: 74,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: isActive
                  ? AppColors.uploadGreen
                  : isPaused
                      ? AppColors.uploadGreen.withValues(alpha: 0.45)
                      : AppColors.iconBg,
              boxShadow: isActive
                  ? [BoxShadow(
                      color: AppColors.uploadGreen.withValues(alpha: 0.35),
                      blurRadius: 22, offset: const Offset(0, 8))]
                  : [],
            ),
            child: Icon(
              isPaused ? Icons.pause_rounded : Icons.mic_rounded,
              color: (isActive || isPaused) ? Colors.white : AppColors.uploadGreen,
              size: 32,
            ),
          ),
        ],
      ),
    );
  }

  Widget _ring({required double size, required double alpha}) => Container(
    width: size, height: size,
    decoration: BoxDecoration(
      shape: BoxShape.circle,
      color: AppColors.uploadGreen.withValues(alpha: alpha),
    ),
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// _Header
// ─────────────────────────────────────────────────────────────────────────────
class _Header extends StatelessWidget {
  final AppLocalizations l10n;
  const _Header({required this.l10n});

  @override
  Widget build(BuildContext context) {
    return Row(children: [
      GestureDetector(
        onTap: () => context.pop(),
        child: const Icon(Icons.arrow_back_ios_new_rounded,
            size: 22, color: AppColors.textPrimary),
      ),
      const SizedBox(width: 12),
      Expanded(child: Text(l10n.uploadScreenTitle,
          style: AppTextStyles.uploadHeader)),
      GestureDetector(
        onTap: () {},
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Container(
            width: 28, height: 28,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(
                  color: AppColors.textDark.withValues(alpha: 0.4), width: 1.5)),
            child: const Icon(Icons.question_mark_rounded,
                size: 14, color: AppColors.textDark),
          ),
          const SizedBox(width: 6),
          Text(l10n.uploadHelp, style: GoogleFonts.inter(
              fontSize: 14, fontWeight: FontWeight.w500, color: AppColors.textDark)),
        ]),
      ),
    ]);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// _StepChip
// ─────────────────────────────────────────────────────────────────────────────
class _StepChip extends StatelessWidget {
  final AppLocalizations l10n;
  const _StepChip({required this.l10n});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 28,
      padding: const EdgeInsets.symmetric(horizontal: 12),
      decoration: BoxDecoration(
          color: AppColors.uploadGreenLight,
          borderRadius: BorderRadius.circular(16)),
      child: Center(child: Text(l10n.uploadStepChip,
          style: AppTextStyles.uploadChip)),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// _MediaCard
//
// The tappable card on the upload screen for video and audio.
// Now accepts a `limitHint` string shown as a small muted line below the
// subtitle — always visible so farmers know the limits before recording.
// ─────────────────────────────────────────────────────────────────────────────
class _MediaCard extends StatefulWidget {
  final IconData icon;
  final String   title;
  final String   subtitle;
  final String   limitHint;           // e.g. "Max 20s · Max 20 MB"
  final bool     isSelected;
  final Color    selectedBorderColor;
  final Color    selectedBg;
  final VoidCallback onTap;

  const _MediaCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.limitHint,
    required this.isSelected,
    required this.selectedBorderColor,
    required this.selectedBg,
    required this.onTap,
  });

  @override
  State<_MediaCard> createState() => _MediaCardState();
}

class _MediaCardState extends State<_MediaCard>
    with SingleTickerProviderStateMixin {

  late AnimationController _bCtrl;
  late Animation<double>   _bAnim;

  @override
  void initState() {
    super.initState();
    _bCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 150));
    _bAnim = TweenSequence<double>([
      TweenSequenceItem(tween: Tween(begin: 1.0, end: 1.04), weight: 1),
      TweenSequenceItem(tween: Tween(begin: 1.04, end: 1.0),  weight: 1),
    ]).animate(CurvedAnimation(parent: _bCtrl, curve: Curves.easeOutBack));
  }

  @override
  void didUpdateWidget(_MediaCard old) {
    super.didUpdateWidget(old);
    if (widget.isSelected && !old.isSelected) _bCtrl.forward(from: 0);
  }

  @override
  void dispose() { _bCtrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return ScaleTransition(
      scale: _bAnim,
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeOut,
          padding: const EdgeInsets.all(AppSpacing.screenPadding),
          decoration: BoxDecoration(
            color: widget.isSelected ? widget.selectedBg : AppColors.cardWhite,
            borderRadius: BorderRadius.circular(AppSpacing.uploadCardRadius),
            border: widget.isSelected
                ? Border.all(color: widget.selectedBorderColor, width: 2)
                : null,
            boxShadow: widget.isSelected ? [] : AppShadows.uploadCard,
          ),
          child: Row(children: [

            // Icon container
            AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              width: AppSpacing.iconContainerSize,
              height: AppSpacing.iconContainerSize,
              decoration: BoxDecoration(
                color: widget.isSelected
                    ? widget.selectedBorderColor.withValues(alpha: 0.12)
                    : AppColors.iconBg,
                borderRadius: BorderRadius.circular(AppSpacing.iconContainerRadius),
              ),
              child: widget.isSelected
                  ? Icon(Icons.check_rounded,
                      color: widget.selectedBorderColor, size: 26)
                  : Icon(widget.icon,
                      color: AppColors.textSecondary, size: 24),
            ),

            const SizedBox(width: 16),

            // Text column — title + subtitle + limitHint
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(widget.title, style: AppTextStyles.uploadCardTitle),
                  const SizedBox(height: 3),
                  Text(widget.subtitle,
                      style: AppTextStyles.uploadCardSub,
                      maxLines: 1, overflow: TextOverflow.ellipsis),
                  const SizedBox(height: 4),
                  // Limit hint — small, always visible, muted
                  Row(
                    children: [
                      Icon(
                        Icons.timer_outlined,
                        size: 11,
                        color: widget.isSelected
                            ? widget.selectedBorderColor.withValues(alpha: 0.65)
                            : AppColors.textHint,
                      ),
                      const SizedBox(width: 3),
                      Text(
                        widget.limitHint,
                        style: GoogleFonts.inter(
                          fontSize: 11,
                          fontWeight: FontWeight.w400,
                          color: widget.isSelected
                              ? widget.selectedBorderColor.withValues(alpha: 0.65)
                              : AppColors.textHint,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),

            // Chevron / check
            Icon(
              widget.isSelected
                  ? Icons.check_circle_rounded
                  : Icons.chevron_right_rounded,
              color: widget.isSelected
                  ? widget.selectedBorderColor
                  : AppColors.textHint,
              size: 22,
            ),
          ]),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// _ProTipCard
// ─────────────────────────────────────────────────────────────────────────────
class _ProTipCard extends StatelessWidget {
  final AppLocalizations l10n;
  const _ProTipCard({required this.l10n});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(AppSpacing.screenPadding),
      decoration: BoxDecoration(
        color: AppColors.cardWhite,
        borderRadius: BorderRadius.circular(AppSpacing.uploadCardRadius),
        boxShadow: [BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 16, offset: const Offset(0, 4))],
      ),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Container(
          width: 36, height: 36,
          decoration: const BoxDecoration(
              color: AppColors.uploadBlueBgIcon, shape: BoxShape.circle),
          child: const Icon(Icons.info_outline_rounded,
              color: AppColors.uploadBlue, size: 18),
        ),
        const SizedBox(width: 14),
        Expanded(child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(l10n.uploadProTipTitle, style: AppTextStyles.uploadProTipTitle),
            const SizedBox(height: 4),
            Text(l10n.uploadProTipBody, style: AppTextStyles.uploadProTipBody),
          ],
        )),
      ]),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// _FindSolutionButton
// ─────────────────────────────────────────────────────────────────────────────
class _FindSolutionButton extends StatefulWidget {
  final AppLocalizations l10n;
  final VoidCallback     onTap;
  const _FindSolutionButton({required this.l10n, required this.onTap});

  @override
  State<_FindSolutionButton> createState() => _FindSolutionButtonState();
}

class _FindSolutionButtonState extends State<_FindSolutionButton> {
  bool _pressed = false;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown:   (_) => setState(() => _pressed = true),
      onTapUp:     (_) => setState(() => _pressed = false),
      onTapCancel: ()  => setState(() => _pressed = false),
      onTap: widget.onTap,
      child: AnimatedScale(
        scale: _pressed ? 0.97 : 1.0,
        duration: const Duration(milliseconds: 100),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 150),
          width: double.infinity, height: AppSpacing.uploadButtonHeight,
          decoration: BoxDecoration(
            color: _pressed ? const Color(0xFF187A42) : AppColors.uploadGreen,
            borderRadius: BorderRadius.circular(AppSpacing.uploadButtonRadius),
            boxShadow: _pressed ? [] : AppShadows.findSolutionButton,
          ),
          child: Center(child: Text(widget.l10n.uploadFindSolution,
              style: AppTextStyles.buttonPrimary)),
        ),
      ),
    )
    .animate()
    .slideY(begin: 0.8, end: 0, duration: 280.ms, curve: Curves.easeOutCubic)
    .fadeIn(duration: 200.ms, curve: Curves.easeOut);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// _MediaBottomSheet
// ─────────────────────────────────────────────────────────────────────────────
class _MediaBottomSheet extends StatelessWidget {
  final String galleryLabel; final String liveLabel;
  final VoidCallback onGallery; final VoidCallback onLive;

  const _MediaBottomSheet({required this.galleryLabel, required this.liveLabel,
      required this.onGallery, required this.onLive});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 240,
      decoration: const BoxDecoration(
        color: AppColors.cardWhite,
        borderRadius: BorderRadius.only(
          topLeft: Radius.circular(28), topRight: Radius.circular(28)),
      ),
      child: Stack(children: [
        Column(children: [
          const SizedBox(height: 12),
          Center(child: Container(width: 36, height: 4,
              decoration: BoxDecoration(color: AppColors.divider,
                  borderRadius: BorderRadius.circular(2)))),
          const SizedBox(height: 20),
          _SheetTile(icon: Icons.photo_library_outlined,
              label: galleryLabel, onTap: onGallery),
          _SheetTile(icon: Icons.fiber_manual_record_rounded,
              label: liveLabel, iconColor: Colors.red, onTap: onLive),
        ]),
        Positioned(top: 14, right: 14,
          child: GestureDetector(
            onTap: () => Navigator.pop(context),
            child: Container(width: 28, height: 28,
              decoration: const BoxDecoration(
                  color: AppColors.iconBg, shape: BoxShape.circle),
              child: const Icon(Icons.close_rounded,
                  size: 16, color: AppColors.textSecondary)),
          )),
      ]),
    );
  }
}

class _SheetTile extends StatelessWidget {
  final IconData icon; final String label;
  final Color iconColor; final VoidCallback onTap;

  const _SheetTile({required this.icon, required this.label,
      this.iconColor = AppColors.textPrimary, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(28),
      splashColor: AppColors.uploadGreen.withValues(alpha: 0.08),
      highlightColor: Colors.transparent,
      child: SizedBox(height: 56, child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24),
        child: Row(children: [
          Icon(icon, size: 22, color: iconColor),
          const SizedBox(width: 18),
          Text(label, style: AppTextStyles.uploadSheetOption),
        ]),
      )),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// // _ProgressPanel + internals (unchanged from previous version)
// // ─────────────────────────────────────────────────────────────────────────────
// class _ProgressPanel extends StatelessWidget {
//   final AppLocalizations l10n; final int currentStep; final double progressValue;
//   final UploadState uploadState; final String errorTitle; final String errorSubtitle;
//   final VoidCallback onRetry;

//   const _ProgressPanel({required this.l10n, required this.currentStep,
//       required this.progressValue, required this.uploadState,
//       required this.errorTitle, required this.errorSubtitle, required this.onRetry});

//   @override
//   Widget build(BuildContext context) {
//     return Container(
//       height: MediaQuery.of(context).size.height * 0.62,
//       decoration: const BoxDecoration(
//         color: AppColors.cardWhite,
//         borderRadius: BorderRadius.only(
//           topLeft: Radius.circular(28), topRight: Radius.circular(28))),
//       padding: const EdgeInsets.fromLTRB(24, 0, 24, 32),
//       child: uploadState == UploadState.error
//           ? _ErrorContent(l10n: l10n, title: errorTitle,
//               subtitle: errorSubtitle, onRetry: onRetry)
//           : _ChecklistContent(l10n: l10n, currentStep: currentStep,
//               progressValue: progressValue),
//     );
//   }
// }

// class _ChecklistContent extends StatelessWidget {
//   final AppLocalizations l10n; final int currentStep; final double progressValue;
//   const _ChecklistContent({required this.l10n, required this.currentStep,
//       required this.progressValue});

//   @override
//   Widget build(BuildContext context) {
//     final steps = [l10n.uploadStep1, l10n.uploadStep2, l10n.uploadStep3,
//         l10n.uploadStep4, l10n.uploadStep5];
//     return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
//       const SizedBox(height: 12),
//       Center(child: Container(width: 36, height: 4,
//           decoration: BoxDecoration(color: AppColors.divider,
//               borderRadius: BorderRadius.circular(2)))),
//       const SizedBox(height: 24),
//       Text(l10n.uploadAnalyzing, style: AppTextStyles.progressTitle),
//       const SizedBox(height: 20),
//       Expanded(child: ListView.builder(
//         physics: const NeverScrollableScrollPhysics(),
//         itemCount: steps.length,
//         itemBuilder: (_, i) => _StepRow(
//           label: steps[i],
//           state: i < currentStep ? _StepState.done
//               : i == currentStep ? _StepState.active : _StepState.pending,
//         ),
//       )),
//       const SizedBox(height: 16),
//       ClipRRect(
//         borderRadius: BorderRadius.circular(3),
//         child: TweenAnimationBuilder<double>(
//           tween: Tween(begin: 0.0, end: progressValue),
//           duration: const Duration(milliseconds: 400),
//           curve: Curves.easeOut,
//           builder: (_, val, __) => LinearProgressIndicator(
//             value: val, minHeight: 6,
//             backgroundColor: const Color(0xFFEAEAEA),
//             valueColor: const AlwaysStoppedAnimation(AppColors.uploadGreen),
//           ),
//         ),
//       ),
//     ]);
//   }
// }

// enum _StepState { pending, active, done }

// class _StepRow extends StatelessWidget {
//   final String label; final _StepState state;
//   const _StepRow({required this.label, required this.state});

//   @override
//   Widget build(BuildContext context) {
//     return SizedBox(height: 60, child: Row(children: [
//       AnimatedSwitcher(
//           duration: const Duration(milliseconds: 200), child: _icon()),
//       const SizedBox(width: 16),
//       Expanded(child: Text(label, style: AppTextStyles.progressStep.copyWith(
//         color: state == _StepState.pending
//             ? AppColors.textHint : AppColors.textPrimary,
//       ))),
//     ]));
//   }

//   Widget _icon() {
//     switch (state) {
//       case _StepState.pending:
//         return Container(key: const ValueKey('p'), width: 32, height: 32,
//             decoration: BoxDecoration(shape: BoxShape.circle,
//                 border: Border.all(color: AppColors.divider, width: 2)));
//       case _StepState.active:
//         return const SizedBox(key: ValueKey('a'), width: 32, height: 32,
//             child: CircularProgressIndicator(strokeWidth: 2.5,
//                 valueColor: AlwaysStoppedAnimation(AppColors.uploadGreen)));
//       case _StepState.done:
//         return Container(key: const ValueKey('d'), width: 32, height: 32,
//             decoration: const BoxDecoration(
//                 color: AppColors.uploadGreen, shape: BoxShape.circle),
//             child: const Icon(Icons.check_rounded, color: Colors.white, size: 18))
//         .animate()
//         .scale(begin: const Offset(0.5, 0.5), duration: 200.ms,
//             curve: Curves.easeOutBack)
//         .fadeIn(duration: 150.ms);
//     }
//   }
// }

// class _ErrorContent extends StatelessWidget {
//   final AppLocalizations l10n; final String title; final String subtitle;
//   final VoidCallback onRetry;
//   const _ErrorContent({required this.l10n, required this.title,
//       required this.subtitle, required this.onRetry});

//   @override
//   Widget build(BuildContext context) {
//     return Column(mainAxisAlignment: MainAxisAlignment.center, children: [
//       const Icon(Icons.wifi_off_rounded, size: 52, color: Color(0xFFBBBBBB)),
//       const SizedBox(height: 20),
//       Text(title, style: AppTextStyles.progressTitle.copyWith(fontSize: 18),
//           textAlign: TextAlign.center),
//       const SizedBox(height: 8),
//       Text(subtitle, style: AppTextStyles.progressStep.copyWith(
//           color: AppColors.textSecondary), textAlign: TextAlign.center),
//       const SizedBox(height: 32),
//       GestureDetector(
//         onTap: onRetry,
//         child: Container(
//           width: double.infinity, height: AppSpacing.uploadButtonHeight,
//           decoration: BoxDecoration(
//             color: AppColors.uploadGreen,
//             borderRadius: BorderRadius.circular(AppSpacing.uploadButtonRadius),
//             boxShadow: AppShadows.findSolutionButton,
//           ),
//           child: Center(child: Text(l10n.uploadRetry,
//               style: AppTextStyles.buttonPrimary)),
//         ),
//       ),
//     ]);
//   }
// }