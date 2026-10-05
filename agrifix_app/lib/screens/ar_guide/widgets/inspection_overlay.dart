import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../../../core/models/inspection_snapshot.dart';

class InspectionOverlay extends StatelessWidget {
  final InspectionSnapshot snapshot;
  final AnimationController pulseCtrl;
  final bool isHindi;
  final VoidCallback onContinue;
  final VoidCallback onRetake;

  const InspectionOverlay({
    super.key,
    required this.snapshot,
    required this.pulseCtrl,
    required this.isHindi,
    required this.onContinue,
    required this.onRetake,
  });

  @override
  Widget build(BuildContext context) {
    final outcome = snapshot.outcome;
    final isDamaged = outcome == InspectionOutcome.damaged;
    final borderColor = isDamaged ? const Color(0xFFFF4B4B) : const Color(0xFF22C55E);
    final label = _outcomeLabel(outcome, isHindi);
    final description = snapshot.getLocalizedDescription(isHindi);

    return Stack(
      fit: StackFit.expand,
      children: [
        Image.memory(
          Uint8List.fromList(snapshot.imageBytes),
          fit: BoxFit.cover,
        ),
        Container(color: Colors.black.withValues(alpha: 0.3)),
        Positioned(
          left: 20, right: 20, bottom: 40,
          child: Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: const Color(0xF01E1E1E),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: borderColor.withValues(alpha: 0.5), width: 1.5),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Row(
                  children: [
                    Icon(
                      isDamaged ? Icons.warning_rounded : Icons.check_circle_rounded,
                      color: borderColor,
                      size: 28,
                    ),
                    const SizedBox(width: 10),
                    Text(label,
                      style: GoogleFonts.inter(
                        fontSize: 18, fontWeight: FontWeight.w700, color: borderColor)),
                    const Spacer(),
                    if (snapshot.severity != Severity.none)
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: _severityColor(snapshot.severity).withValues(alpha: 0.2),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          snapshot.severity.name.toUpperCase(),
                          style: GoogleFonts.inter(
                            fontSize: 11, fontWeight: FontWeight.w700,
                            color: _severityColor(snapshot.severity)),
                        ),
                      ),
                  ],
                ),
                const SizedBox(height: 12),
                if (description.isNotEmpty) ...[
                  Text(description,
                    style: GoogleFonts.inter(
                      fontSize: 14, color: const Color(0xFFEAEAEA), height: 1.5)),
                  const SizedBox(height: 16),
                ],
                if (snapshot.observations.isNotEmpty) ...[
                  ...snapshot.observations.map((obs) => Padding(
                    padding: const EdgeInsets.only(bottom: 6),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Icon(Icons.circle, size: 6, color: Color(0xFF9CA3AF)),
                        const SizedBox(width: 8),
                        Expanded(child: Text(obs,
                          style: GoogleFonts.inter(
                            fontSize: 13, color: const Color(0xFF9CA3AF)))),
                      ],
                    ),
                  )),
                  const SizedBox(height: 16),
                ],
                if (snapshot.repairability != Repairability.unknown)
                  _RepairabilityBadge(
                    repairability: snapshot.repairability, isHindi: isHindi),
                const SizedBox(height: 20),
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton(
                        onPressed: onRetake,
                        style: OutlinedButton.styleFrom(
                          foregroundColor: Colors.white,
                          side: const BorderSide(color: Color(0xFF6B7280)),
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(14)),
                        ),
                        child: Text(isHindi ? 'दोबारा लें' : 'Retake'),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      flex: 2,
                      child: ElevatedButton(
                        onPressed: onContinue,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: isDamaged
                              ? const Color(0xFF22C55E)
                              : const Color(0xFF3B82F6),
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(14)),
                        ),
                        child: Text(
                          isDamaged
                              ? (isHindi ? 'मरम्मत जारी रखें' : 'Continue to Repair')
                              : (isHindi ? 'अगला कदम' : 'Next Step'),
                          style: GoogleFonts.inter(
                            fontSize: 15, fontWeight: FontWeight.w600, color: Colors.white),
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  String _outcomeLabel(InspectionOutcome outcome, bool isHindi) {
    switch (outcome) {
      case InspectionOutcome.healthy: return isHindi ? 'सही है — कोई क्षति नहीं' : 'Part OK — No Damage';
      case InspectionOutcome.damaged: return isHindi ? 'क्षति का पता चला' : 'Damage Detected';
      case InspectionOutcome.unclear: return isHindi ? 'स्पष्ट नहीं' : 'Unclear Image';
      case InspectionOutcome.hidden: return isHindi ? 'भाग छिपा हुआ' : 'Part Hidden';
      case InspectionOutcome.occluded: return isHindi ? 'भाग ढका हुआ' : 'Part Blocked';
      case InspectionOutcome.dirty: return isHindi ? 'भाग गंदा है' : 'Part Dirty';
      case InspectionOutcome.wrongTarget: return isHindi ? 'गलत भाग' : 'Wrong Part';
    }
  }

  Color _severityColor(Severity severity) {
    switch (severity) {
      case Severity.minor: return const Color(0xFFF59E0B);
      case Severity.moderate: return const Color(0xFFF97316);
      case Severity.severe: return const Color(0xFFEF4444);
      case Severity.critical: return const Color(0xFFDC2626);
      default: return const Color(0xFF9CA3AF);
    }
  }
}

class _RepairabilityBadge extends StatelessWidget {
  final Repairability repairability;
  final bool isHindi;
  const _RepairabilityBadge({required this.repairability, required this.isHindi});

  @override
  Widget build(BuildContext context) {
    final (icon, label, color) = switch (repairability) {
      Repairability.diy => (Icons.build_rounded,
          isHindi ? 'खुद ठीक कर सकते हैं' : 'DIY Repairable', const Color(0xFF22C55E)),
      Repairability.mechanicRequired => (Icons.engineering_rounded,
          isHindi ? 'मैकेनिक आवश्यक' : 'Mechanic Required', const Color(0xFFF59E0B)),
      Repairability.replacePart => (Icons.swap_horiz_rounded,
          isHindi ? 'भाग बदलें' : 'Replace Part', const Color(0xFFF97316)),
      Repairability.monitor => (Icons.visibility_rounded,
          isHindi ? 'निगरानी रखें' : 'Monitor', const Color(0xFF3B82F6)),
      Repairability.unknown => (Icons.help_outline_rounded,
          isHindi ? 'अज्ञात' : 'Unknown', const Color(0xFF9CA3AF)),
    };

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: color),
          const SizedBox(width: 6),
          Text(label, style: GoogleFonts.inter(
            fontSize: 13, fontWeight: FontWeight.w600, color: color)),
        ],
      ),
    );
  }
}