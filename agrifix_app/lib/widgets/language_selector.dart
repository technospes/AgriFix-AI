import 'package:flutter/material.dart';
import 'dart:ui';
import '../core/theme.dart';

class LanguageSelectorPanel extends StatelessWidget {
  final Locale currentLocale;
  final Function(Locale) onLocaleSelected;

  const LanguageSelectorPanel({
    super.key,
    required this.currentLocale,
    required this.onLocaleSelected,
  });

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
        child: Container(
          width: 220,
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.88),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
                color: Colors.white.withValues(alpha: 0.5), width: 1),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.15),
                blurRadius: 24,
                offset: const Offset(0, 8),
              ),
            ],
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
                child: Row(
                  children: [
                    const Icon(Icons.language_rounded,
                        color: AppColors.primaryGreen, size: 20),
                    const SizedBox(width: 8),
                    const Text('Select Language',
                        style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                            color: AppColors.textPrimary)),
                  ],
                ),
              ),
              Container(
                  height: 1, color: Colors.grey.withValues(alpha: 0.2)),
              _option(const Locale('en'), '🇬🇧', 'English'),
              _option(const Locale('hi'), '🇮🇳', 'हिन्दी'),
              _option(const Locale('pa'), '🌾', 'ਪੰਜਾਬੀ'),
              const SizedBox(height: 8),
            ],
          ),
        ),
      ),
    );
  }

  Widget _option(Locale locale, String flag, String label) {
    final isSelected = currentLocale.languageCode == locale.languageCode;
    return GestureDetector(
      onTap: () => onLocaleSelected(locale),
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 13),
        decoration: BoxDecoration(
          color: isSelected
              ? AppColors.primaryGreen.withValues(alpha: 0.10)
              : Colors.transparent,
        ),
        child: Row(
          children: [
            Text(flag, style: const TextStyle(fontSize: 20)),
            const SizedBox(width: 12),
            Expanded(
              child: Text(label,
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight:
                        isSelected ? FontWeight.w600 : FontWeight.w400,
                    color: isSelected
                        ? AppColors.primaryGreen
                        : AppColors.textPrimary,
                  )),
            ),
            if (isSelected)
              const Icon(Icons.check_rounded,
                  color: AppColors.primaryGreen, size: 18),
          ],
        ),
      ),
    );
  }
}
