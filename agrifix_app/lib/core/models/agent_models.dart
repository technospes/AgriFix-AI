import '../utils/step_formatter.dart';
class InteractionOption {
  final String id;
  final String label;
  final String nextState;
  InteractionOption({required this.id, required this.label, this.nextState = ''});
  factory InteractionOption.fromJson(Map<String, dynamic> j) => InteractionOption(
        id: j['id'] ?? '',
        label: j['label'] ?? '',
        nextState: j['next_state'] ?? '',
      );
}

enum InteractionType { choice, camera, boolean, text, number, none }

InteractionType _parseInteractionType(String? raw) {
  switch (raw) {
    case 'choice':  return InteractionType.choice;
    case 'camera':  return InteractionType.camera;
    case 'boolean': return InteractionType.boolean;
    case 'text':    return InteractionType.text;
    case 'number':  return InteractionType.number;
    default:        return InteractionType.none;
  }
}

class Interaction {
  final InteractionType type;
  final String question;
  final List<InteractionOption> options;
  final bool required;
  Interaction({
    required this.type,
    this.question = '',
    this.options = const [],
    this.required = true,
  });
  factory Interaction.fromJson(Map<String, dynamic> j) => Interaction(
        type: _parseInteractionType(j['type'] as String?),
        question: j['question'] ?? '',
        options: (j['options'] as List? ?? [])
            .map((o) => InteractionOption.fromJson(o as Map<String, dynamic>))
            .toList(),
        required: j['required'] ?? true,
      );
}

class NextStepDetail {
  final String text, textEn, textHi;
  final String visualCue, arModel, requiredPart, areaHint;
  final String trackingScope;
  final String? safetyWarning;
  final String expectedResult, expectedResultHi;
  final String ifFailed, ifFailedHi;
  final String escalateIf, escalateIfHi;
  final String? requiredTool;
  final Interaction? interaction;

  NextStepDetail({
    required this.text, required this.textEn, required this.textHi,
    required this.visualCue, required this.arModel,
    required this.requiredPart, required this.areaHint, required this.trackingScope,
    this.safetyWarning,
    this.expectedResult = '', this.expectedResultHi = '',
    this.ifFailed = '', this.ifFailedHi = '',
    this.escalateIf = '', this.escalateIfHi = '',
    this.requiredTool,
    this.interaction,
  });

  factory NextStepDetail.fromJson(Map<String, dynamic> j) => NextStepDetail(
        text: j['text'] ?? '', textEn: j['text_en'] ?? '', textHi: j['text_hi'] ?? '',
        visualCue: j['visual_cue'] ?? '', arModel: j['ar_model'] ?? 'none',
        requiredPart: j['required_part'] ?? '', areaHint: j['area_hint'] ?? '',
        trackingScope: j['tracking_scope'] ?? 'component',
        safetyWarning: j['safety_warning'] as String?,
        expectedResult: j['expected_result'] ?? '', expectedResultHi: j['expected_result_hi'] ?? '',
        ifFailed: j['if_failed'] ?? '', ifFailedHi: j['if_failed_hi'] ?? '',
        escalateIf: j['escalate_if'] ?? '', escalateIfHi: j['escalate_if_hi'] ?? '',
        requiredTool: j['required_tool'] as String?,
        interaction: j['interaction'] != null
            ? Interaction.fromJson(j['interaction'] as Map<String, dynamic>)
            : null,
      );

  String localizedText(bool isHindi) => isHindi && textHi.isNotEmpty ? textHi : (textEn.isNotEmpty ? textEn : text);

  /// Human-readable action label for UI display.
  /// Falls back: visualCue → requiredPart
  String get displayAction {
    if (visualCue.isNotEmpty) return StepFormatter.title(visualCue);
    if (requiredPart.isNotEmpty) return StepFormatter.title(requiredPart);
    return '';
  }
  
  /// Human-readable part label for scanner/target display
  String get displayPart {
    if (visualCue.isNotEmpty) return StepFormatter.part(visualCue);
    if (requiredPart.isNotEmpty) return StepFormatter.part(requiredPart);
    return '';
  }
}

enum AgentStatus { continueFlow, resolved, escalate, unsafe }

AgentStatus _parseStatus(String? s) {
  switch (s) {
    case 'resolved':  return AgentStatus.resolved;
    case 'escalate':  return AgentStatus.escalate;
    case 'unsafe':    return AgentStatus.unsafe;
    default:          return AgentStatus.continueFlow;
  }
}

class AgentNextResponse {
  final AgentStatus status;
  final String reasoningSummary;
  final NextStepDetail nextStep;
  final Map<String, dynamic> updatedMemory; // {verified_parts, diagnostic_path}

  AgentNextResponse({
    required this.status,
    required this.reasoningSummary,
    required this.nextStep,
    required this.updatedMemory,
  });

  factory AgentNextResponse.fromJson(Map<String, dynamic> j) => AgentNextResponse(
        status: _parseStatus(j['status'] as String?),
        reasoningSummary: j['reasoning_summary'] ?? '',
        nextStep: NextStepDetail.fromJson(j['next_step'] as Map<String, dynamic>),
        updatedMemory: (j['updated_memory'] as Map<String, dynamic>?) ?? {},
      );
}