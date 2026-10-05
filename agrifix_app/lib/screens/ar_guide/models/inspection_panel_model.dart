import '../../../core/models/agent_models.dart';
import '../../../core/providers/diagnosis_provider.dart';

/// Lightweight UI model for InspectionPanel.
/// Built from either StepData (solution screen) or NextStepDetail (AR agent).
class InspectionPanelModel {
  final String title;
  final String description;
  final String? question;
  final String? safetyWarning;
  final List<StepOption> options;
  final bool showDoneButton;
  final StepType stepType;

  InspectionPanelModel({
    required this.title,
    required this.description,
    this.question,
    this.safetyWarning,
    this.options = const [],
    this.showDoneButton = false,
    this.stepType = StepType.inspection,
  });

  /// Build from agent's NextStepDetail
  factory InspectionPanelModel.fromAgentStep(NextStepDetail step) {
    final interaction = step.interaction;
    final isNone = interaction?.type == InteractionType.none;
    
    return InspectionPanelModel(
      title: step.displayAction,
      description: step.textEn.isNotEmpty ? step.textEn : step.text,
      question: interaction?.question,
      safetyWarning: step.safetyWarning,
      options: (interaction?.options ?? []).map((opt) => StepOption(
        id: opt.id,
        labelEn: opt.label,
        labelHi: opt.label,
        nextStep: opt.nextState,
      )).toList(),
      showDoneButton: isNone,
      stepType: isNone ? StepType.action : StepType.inspection,
    );
  }

  /// Build from legacy StepData
  factory InspectionPanelModel.fromStepData(StepData step) {
    return InspectionPanelModel(
      title: step.displayAction,
      description: step.getLocalizedText(false),
      question: step.questionEn,
      safetyWarning: step.safetyWarning,
      options: step.options,
      showDoneButton: step.isActionStep,
      stepType: step.stepType,
    );
  }
}