import 'package:flutter/material.dart';
import '../utils/step_formatter.dart';
// ── Step option for inspection / observation panels ──────────────────────────
class StepOption {
  final String id;
  final String labelEn;
  final String labelHi;
  final String nextStep;   // step_id to jump to, or '' for linear advance

  const StepOption({
    required this.id,
    required this.labelEn,
    required this.labelHi,
    required this.nextStep,
  });

  factory StepOption.fromJson(Map<String, dynamic> json) => StepOption(
    id:       json['id']       as String? ?? '',
    labelEn:  json['label_en'] as String? ?? json['label'] as String? ?? '',
    labelHi:  json['label_hi'] as String? ?? '',
    nextStep: json['next_step'] as String? ?? '',
  );

  String getLocalizedLabel(bool isHindi) =>
      isHindi && labelHi.isNotEmpty ? labelHi : labelEn;
}

// ── Step types matching the architecture document ─────────────────────────────
enum StepType {
  visual,      // Camera AR verification
  inspection,  // Physical check — decision panel with answer buttons
  action,      // Farmer performs a manual operation — Done button
  observation, // Subjective assessment — symptom buttons
}

StepType _parseStepType(String? raw) {
  switch ((raw ?? '').toLowerCase()) {
    case 'inspection':  return StepType.inspection;
    case 'action':      return StepType.action;
    case 'observation': return StepType.observation;
    default:            return StepType.visual;
  }
}

class StepData {
  final String stepId;        // unique step identifier for routing
  final StepType stepType;    // drives Flutter UI component selection
  final String text;
  final String textEn;
  final String textHi;
  final String action;
  final String stepTitleEn;
  final String stepTitleHi;
  final String? visualCue;
  final String? arModel;
  final String? safetyWarning;

  // 🟢 INJECTED FROM V1: Part identification fields for /verify_step
  final String requiredPart; 
  final String trackingScope;  
  final String areaHint;       

  // Inspection / observation fields
  final String? questionEn;
  final String? questionHi;
  final List<StepOption> options;   // non-empty for inspection + observation

  StepData({
    required this.stepId,
    required this.stepType,
    required this.text,
    required this.textEn,
    required this.textHi,
    required this.action,
    required this.stepTitleEn,
    required this.stepTitleHi,
    this.visualCue,
    this.arModel, 
    this.safetyWarning,
    this.questionEn,
    this.questionHi,
    this.options = const [],
    this.requiredPart = '',
    this.trackingScope = 'component',
    this.areaHint     = 'engine_compartment',
  });

  factory StepData.fromJson(Map<String, dynamic> json) => StepData(
        stepId:      json['step_id']     as String? ?? '',
        stepType:    _parseStepType(json['step_type'] as String?),
        text:        json['text']        as String? ?? '',
        textEn:      json['text_en']     as String? ?? json['text'] as String? ?? '',
        textHi:      json['text_hi']     as String? ?? json['text'] as String? ?? '',
        stepTitleEn: json['step_title_en'] as String? ?? json['step_title'] as String? ?? '',
        stepTitleHi: json['step_title_hi'] as String? ?? '',
        visualCue:   json['visual_cue']  as String?,
        arModel:     json['ar_model']    as String?,
        safetyWarning: json['safety_warning'] as String?,
        questionEn:  json['question_en'] as String? ?? json['question'] as String?,
        questionHi:  json['question_hi'] as String?,
        options: (json['options'] as List? ?? [])
            .map((o) => StepOption.fromJson(o as Map<String, dynamic>))
            .toList(),
        requiredPart:  json['required_part'] as String? ?? '',
        trackingScope: json['tracking_scope'] as String? ?? 'component',
        areaHint:      json['area_hint']     as String? ?? 'engine_compartment',
        action: json['action'] as String? ?? '',
      );

  String getLocalizedText(bool isHindi) =>
      isHindi && textHi.isNotEmpty ? textHi : textEn.isNotEmpty ? textEn : text;

  String getLocalizedTitle(bool isHindi) =>
      isHindi && stepTitleHi.isNotEmpty ? stepTitleHi : stepTitleEn;

  String? getLocalizedQuestion(bool isHindi) =>
      isHindi && (questionHi?.isNotEmpty ?? false) ? questionHi : questionEn;

  /// True when this step requires the inspection / observation decision panel
  bool get requiresDecisionPanel =>
      (stepType == StepType.inspection || stepType == StepType.observation) &&
      options.isNotEmpty;

  /// True when this step is a manual action (just a Done button)
  bool get isActionStep => stepType == StepType.action;

  /// Human-readable action label for UI display
  String get displayAction {
    if (action.isNotEmpty) return StepFormatter.title(action);
    if (visualCue != null && visualCue!.isNotEmpty) return StepFormatter.title(visualCue!);
    if (requiredPart.isNotEmpty) return StepFormatter.title(requiredPart);
    return '';
  }
  
  /// Human-readable part label for scanner display
  String get displayPart {
    if (visualCue != null && visualCue!.isNotEmpty) return StepFormatter.part(visualCue!);
    if (requiredPart.isNotEmpty) return StepFormatter.part(requiredPart);
    return '';
  }

}

class SolutionData {
  final String machineType;
  final double machineConfidence;    // from detection
  final bool   cacheHit;            // true when served from plan cache
  final String problemIdentifiedEn;
  final String problemIdentifiedHi;
  final List<StepData> steps;
  final List<String> safetyWarnings;
  final List<String> toolsNeeded;
  final String status;

  SolutionData({
    required this.machineType,
    this.machineConfidence = 0.0,
    this.cacheHit = false,
    required this.problemIdentifiedEn,
    required this.problemIdentifiedHi,
    required this.steps,
    required this.safetyWarnings,
    required this.toolsNeeded,
    this.status = 'success',
  });

  factory SolutionData.fromJson(Map<String, dynamic> json) => SolutionData(
        machineType:          json['machine_type'] as String? ?? 'Agricultural Machine',
        machineConfidence:    (json['machine_confidence'] as num?)?.toDouble() ?? 0.0,
        cacheHit:             json['cache_hit'] as bool? ?? false,
        problemIdentifiedEn:  json['problem_identified_en'] as String? ??
                              json['problem_identified']    as String? ?? '',
        problemIdentifiedHi:  json['problem_identified_hi'] as String? ?? '',
        steps: (json['steps'] as List? ?? [])
            .map((s) => StepData.fromJson(s as Map<String, dynamic>))
            .toList(),
        safetyWarnings: List<String>.from(json['safety_warnings_en'] as List? ?? []),
        toolsNeeded:    List<String>.from(json['tools_needed']        as List? ?? []),
        status: json['status'] as String? ?? 'success',
      );

  String getLocalizedProblem(bool isHindi) =>
      isHindi && problemIdentifiedHi.isNotEmpty ? problemIdentifiedHi : problemIdentifiedEn;

  /// Lookup a step by its step_id — used for non-linear inspection routing
  StepData? findStepById(String stepId) {
    try {
      return steps.firstWhere((s) => s.stepId == stepId);
    } catch (_) {
      return null;
    }
  }

  /// Index of a step by its step_id — used for inspection panel routing
  int indexOfStepId(String stepId) =>
      steps.indexWhere((s) => s.stepId == stepId);
}

class DiagnosisProvider extends ChangeNotifier {
  Map<String, dynamic>? diagnosis;
  SolutionData? _solution;
  String?       _problemDescription;
  bool          _isLoading = false;
  String?       _error;

  // ── Safety gate state ─────────────────────────────────────────────────────
  //
  // safetyConfirmed: true after the user has ticked all 4 safety checks.
  //   Checked by AR screen before allowing guidance to start.
  //   Reset on clear() so each new session requires a fresh confirmation.
  //
  // unsafeSceneSuspected: true when the backend pipeline detects evidence of
  //   an unsafe scene in the uploaded frames/transcript (no extra Gemini call —
  //   derived from already-computed diagnosis context in diagnosis_service.py).
  //   Shown as a non-blocking warning banner on the solution screen.
  bool _safetyConfirmed       = false;
  bool _unsafeSceneSuspected  = false;
  String _unsafeSceneMessage  = '';

  SolutionData? get solution              => _solution;
  String?       get problemDescription    => _problemDescription;
  bool          get isLoading             => _isLoading;
  String?       get error                 => _error;
  bool          get hasSolution           => _solution != null;

  // Safety getters
  bool   get safetyConfirmed      => _safetyConfirmed;
  bool   get unsafeSceneSuspected => _unsafeSceneSuspected;
  String get unsafeSceneMessage   => _unsafeSceneMessage;

  void setLoading(bool loading) {
    _isLoading = loading;
    notifyListeners();
  }

  // Called by safety gate widget after user confirms all 4 checks
  void confirmSafety() {
    _safetyConfirmed = true;
    notifyListeners();
  }

  // Called if user goes back through the flow without re-confirming
  // (e.g., uploads new media after a prior session)
  void resetSafety() {
    _safetyConfirmed = false;
    notifyListeners();
  }

  void setDiagnosis(Map<String, dynamic> json) {
    final solutionJson = Map<String, dynamic>.from(json['solution'] ?? json);
    _solution = SolutionData.fromJson(solutionJson);
    _problemDescription  = json['problem_description'];
    _error               = null;
    _isLoading           = false;

    final raw = json['unsafe_scene_suspected'];
    _unsafeSceneSuspected = raw == true;
    _unsafeSceneMessage   = json['unsafe_scene_message'] as String? ?? '';

    notifyListeners();
  }

  void setError(String error) {
    _error     = error;
    _isLoading = false;
    notifyListeners();
  }

  void clear() {
    _solution            = null;
    _error               = null;
    // Reset safety on clear so each upload session re-confirms
    _safetyConfirmed     = false;
    _unsafeSceneSuspected = false;
    _unsafeSceneMessage   = '';
    notifyListeners();
  }
}