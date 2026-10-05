enum InspectionOutcome {
  healthy,
  damaged,
  unclear,
  wrongTarget,
  hidden,
  occluded,
  dirty,
}

enum Severity {
  none,
  minor,
  moderate,
  severe,
  critical,
}

enum Repairability {
  diy,
  mechanicRequired,
  replacePart,
  monitor,
  unknown,
}

class InspectionSnapshot {
  final InspectionOutcome outcome;
  final Severity severity;
  final String damageDescription;
  final String damageDescriptionHi;
  final Repairability repairability;
  final List<String> observations;
  final bool safetyConcern;
  final String safetyNote;
  final double confidence;
  final List<int> imageBytes; // The frozen frame
  final DateTime timestamp;

  InspectionSnapshot({
    required this.outcome,
    required this.severity,
    required this.damageDescription,
    required this.damageDescriptionHi,
    required this.repairability,
    required this.observations,
    required this.safetyConcern,
    required this.safetyNote,
    required this.confidence,
    required this.imageBytes,
    required this.timestamp,
  });

  factory InspectionSnapshot.fromJson(Map<String, dynamic> json, List<int> imageBytes) {
    return InspectionSnapshot(
      outcome: _parseOutcome(json['outcome'] as String? ?? 'unclear'),
      severity: _parseSeverity(json['severity'] as String? ?? 'none'),
      damageDescription: json['damage_description'] as String? ?? '',
      damageDescriptionHi: json['damage_description_hi'] as String? ?? '',
      repairability: _parseRepairability(json['repairability'] as String? ?? 'unknown'),
      observations: List<String>.from(json['observations'] as List? ?? []),
      safetyConcern: json['safety_concern'] == true,
      safetyNote: json['safety_note'] as String? ?? '',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      imageBytes: imageBytes,
      timestamp: DateTime.now(),
    );
  }

  static InspectionOutcome _parseOutcome(String s) {
    switch (s) {
      case 'healthy': return InspectionOutcome.healthy;
      case 'damaged': return InspectionOutcome.damaged;
      case 'wrong_target': return InspectionOutcome.wrongTarget;
      case 'hidden': return InspectionOutcome.hidden;
      case 'occluded': return InspectionOutcome.occluded;
      case 'dirty': return InspectionOutcome.dirty;
      default: return InspectionOutcome.unclear;
    }
  }

  static Severity _parseSeverity(String s) {
    switch (s) {
      case 'minor': return Severity.minor;
      case 'moderate': return Severity.moderate;
      case 'severe': return Severity.severe;
      case 'critical': return Severity.critical;
      default: return Severity.none;
    }
  }

  static Repairability _parseRepairability(String s) {
    switch (s) {
      case 'diy': return Repairability.diy;
      case 'mechanic_required': return Repairability.mechanicRequired;
      case 'replace_part': return Repairability.replacePart;
      case 'monitor': return Repairability.monitor;
      default: return Repairability.unknown;
    }
  }

  String getLocalizedDescription(bool isHindi) =>
      isHindi && damageDescriptionHi.isNotEmpty ? damageDescriptionHi : damageDescription;
}