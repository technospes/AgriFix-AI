import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

// ─────────────────────────────────────────────────────────────────────────────
// AppColors — covers Home + Upload + Solution + AR screens
// ─────────────────────────────────────────────────────────────────────────────
class AppColors {
  // ── Home screen brand ────────────────────────────────────────────────────
  static const primaryGreen     = Color(0xFF22C55E); // Home CTA button
  static const primaryDarkGreen = Color(0xFF16A34A); // pressed / shadow
  static const lightGreenBg     = Color(0xFFC8E6C9);

  // ── Upload screen accent (slightly different green from spec) ────────────
  static const uploadGreen      = Color(0xFF1E9E55); // Upload primary green
  static const uploadGreenLight = Color(0xFFDFF4E8); // Step chip background
  static const uploadGreenBg    = Color(0xFFF6FFFA); // Video card selected bg
  static const uploadBlueBg     = Color(0xFFF3F7FF); // Audio card selected bg
  static const uploadBlue       = Color(0xFF4C7DFF); // Audio border / info icon
  static const uploadBlueBgIcon = Color(0xFFEAF1FF); // Info icon container

  // ── Text ─────────────────────────────────────────────────────────────────
  static const textPrimary   = Color(0xFF111111);
  static const textSecondary = Color(0xFF6E6E6E);
  static const textWhite     = Color(0xFFFFFFFF);
  static const textHint      = Color(0xFFA0A0A0);
  static const textDark      = Color(0xFF4A4A4A); // Help text

  // ── Backgrounds ──────────────────────────────────────────────────────────
  static const pageBg         = Color(0xFFF6F6F6); // Upload screen bg
  static const cardWhite      = Color(0xFFFFFFFF);
  static const iconBg         = Color(0xFFF2F2F2); // Icon containers in cards
  static const divider        = Color(0xFFDADADA); // Drag handle, step pending

  // ── Home screen glass / AR ───────────────────────────────────────────────
  static const glassDark        = Color(0xE50D0F14);
  static const glassLight       = Color(0xCCFFFFFF);
  static const scrim            = Color(0x8C000000);
  static const glassWhiteBorder = Color(0x40FFFFFF);

  // ── AR status rows ───────────────────────────────────────────────────────
  static const statusBlue   = Color(0xB41E7BC8);
  static const statusGreen  = Color(0xB4147844);
  static const statusYellow = Color(0xB4A07800);
  static const statusRed    = Color(0xB4A01E1E);
}

// ─────────────────────────────────────────────────────────────────────────────
// AppTextStyles
// ─────────────────────────────────────────────────────────────────────────────
class AppTextStyles {
  // ── Home screen ──────────────────────────────────────────────────────────
  static TextStyle cardTitle = GoogleFonts.inter(
    fontSize: 24, fontWeight: FontWeight.w700,
    color: AppColors.textPrimary, height: 1.25,
  );
  static TextStyle cardSubtitle = GoogleFonts.inter(
    fontSize: 14, fontWeight: FontWeight.w400,
    color: AppColors.textSecondary, height: 1.43,
  );
  static TextStyle headline = GoogleFonts.inter(
    fontSize: 22, fontWeight: FontWeight.w600,
    color: AppColors.textWhite, height: 1.36,
  );
  static TextStyle buttonPrimary = GoogleFonts.inter(
    fontSize: 16, fontWeight: FontWeight.w600,
    color: AppColors.textWhite, letterSpacing: 0.2,
  );
  static TextStyle languageBtn = GoogleFonts.inter(
    fontSize: 15, fontWeight: FontWeight.w500,
    color: AppColors.textWhite,
  );
  static TextStyle stepLabel = GoogleFonts.inter(
    fontSize: 14, fontWeight: FontWeight.w600,
    color: AppColors.primaryGreen, letterSpacing: 0.5,
  );
  static TextStyle arInstruction = GoogleFonts.inter(
    fontSize: 22, fontWeight: FontWeight.w700,
    color: AppColors.textWhite, height: 1.4,
  );
  static TextStyle statusText = GoogleFonts.inter(
    fontSize: 15, fontWeight: FontWeight.w500,
    color: AppColors.textWhite,
  );

  // ── Upload screen ─────────────────────────────────────────────────────────
  static TextStyle uploadHeader = GoogleFonts.inter(
    fontSize: 18, fontWeight: FontWeight.w600,
    color: AppColors.textPrimary,
  );
  static TextStyle uploadHeading = GoogleFonts.inter(
    fontSize: 26, fontWeight: FontWeight.w700,
    color: AppColors.textPrimary, height: 1.23,
  );
  static TextStyle uploadSubtext = GoogleFonts.inter(
    fontSize: 15, fontWeight: FontWeight.w400,
    color: AppColors.textSecondary, height: 1.47,
  );
  static TextStyle uploadChip = GoogleFonts.inter(
    fontSize: 13, fontWeight: FontWeight.w600,
    color: AppColors.uploadGreen,
  );
  static TextStyle uploadCardTitle = GoogleFonts.inter(
    fontSize: 17, fontWeight: FontWeight.w600,
    color: AppColors.textPrimary,
  );
  static TextStyle uploadCardSub = GoogleFonts.inter(
    fontSize: 14, fontWeight: FontWeight.w400,
    color: AppColors.textSecondary,
  );
  static TextStyle uploadProTipTitle = GoogleFonts.inter(
    fontSize: 15, fontWeight: FontWeight.w600,
    color: AppColors.textPrimary,
  );
  static TextStyle uploadProTipBody = GoogleFonts.inter(
    fontSize: 14, fontWeight: FontWeight.w400,
    color: AppColors.textSecondary, height: 1.43,
  );
  static TextStyle uploadSheetOption = GoogleFonts.inter(
    fontSize: 16, fontWeight: FontWeight.w500,
    color: AppColors.textPrimary,
  );
  static TextStyle progressTitle = GoogleFonts.inter(
    fontSize: 20, fontWeight: FontWeight.w700,
    color: AppColors.textPrimary,
  );
  static TextStyle progressStep = GoogleFonts.inter(
    fontSize: 15, fontWeight: FontWeight.w500,
    color: AppColors.textPrimary,
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// AppSpacing
// ─────────────────────────────────────────────────────────────────────────────
class AppSpacing {
  static const double xs  = 8.0;
  static const double sm  = 16.0;
  static const double md  = 24.0;
  static const double lg  = 32.0;
  static const double xl  = 40.0;
  static const double xxl = 48.0;

  // Home
  static const double screenPadding = 20.0;
  static const double cardPadding   = 24.0;
  static const double buttonHeight  = 56.0;
  static const double buttonRadius  = 20.0;
  static const double cardRadius    = 28.0;
  static const double logoSize      = 64.0;
  static const double logoRadius    = 16.0;

  // Upload
  static const double uploadCardRadius   = 20.0;
  static const double uploadButtonHeight = 54.0;
  static const double uploadButtonRadius = 16.0;
  static const double iconContainerSize  = 56.0;
  static const double iconContainerRadius= 14.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// AppShadows
// ─────────────────────────────────────────────────────────────────────────────
class AppShadows {
  // Home green button shadow
  static List<BoxShadow> greenButton = [
    BoxShadow(
      color: AppColors.primaryDarkGreen.withValues(alpha: 0.35),
      blurRadius: 20,
      offset: const Offset(0, 8),
    ),
  ];

  // Home bottom card shadow
  static List<BoxShadow> card = [
    BoxShadow(
      color: Colors.black.withValues(alpha: 0.08),
      blurRadius: 20,
      offset: const Offset(0, -4),
    ),
  ];

  // Upload screen cards (spec: 6% opacity, blur 20, offsetY 6)
  static List<BoxShadow> uploadCard = [
    BoxShadow(
      color: Colors.black.withValues(alpha: 0.06),
      blurRadius: 20,
      spreadRadius: 0,
      offset: const Offset(0, 6),
    ),
  ];

  // Upload "Find Solution" button shadow (green glow)
  static List<BoxShadow> findSolutionButton = [
    BoxShadow(
      color: AppColors.uploadGreen.withValues(alpha: 0.25),
      blurRadius: 16,
      offset: const Offset(0, 6),
    ),
  ];

  // Floating card (generic)
  static List<BoxShadow> floatingCard = [
    BoxShadow(
      color: Colors.black.withValues(alpha: 0.12),
      blurRadius: 24,
      offset: const Offset(0, 8),
    ),
  ];
}