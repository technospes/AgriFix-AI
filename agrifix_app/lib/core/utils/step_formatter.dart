/// Centralized formatter for all step-related display text.
/// Single source of truth for converting backend identifiers
/// into human-readable labels across the entire app.
///
/// When the backend eventually adds display fields (e.g., `action_display`),
/// update the relevant method here — zero UI changes needed.
class StepFormatter {
  StepFormatter._();

  /// Converts a backend identifier to display-friendly title case.
  ///
  /// - snake_case: "engine_off" → "Engine Off"
  /// - space-separated: "inspect clutch cable" → "Inspect Clutch Cable"
  /// - single word: "clutch" → "Clutch"
  /// - already formatted: "Check the Cable" → "Check the Cable"
  static String title(String text) {
    if (text.trim().isEmpty) return '';

    final cleaned = text.trim();

    // Split on underscores OR spaces
    final words = cleaned.contains('_')
        ? cleaned.split('_')
        : cleaned.split(RegExp(r'\s+'));

    return words
        .where((w) => w.isNotEmpty)
        .map(_capitalize)
        .join(' ');
  }

  /// Formats a part identifier for scanner/target display
  static String part(String partId) {
    return title(partId.replaceAll('_', ' '));
  }

  /// Formats an area identifier for display
  static String area(String areaId) {
    return title(areaId.replaceAll('_', ' '));
  }

  static String _capitalize(String word) {
    if (word.isEmpty) return '';
    // Don't re-capitalize already capitalized words
    if (word[0] == word[0].toUpperCase() && word.length > 1) return word;
    return '${word[0].toUpperCase()}${word.substring(1).toLowerCase()}';
  }
}