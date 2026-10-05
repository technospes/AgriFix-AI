// lib/screens/ar_guide/models/ar_state.dart
// ignore_for_file: deprecated_member_use
import 'package:flutter/material.dart';

// ── AR phase state machine ─────────────────────────────────────────────────
enum ARState {
  scanning,
  locating,
  guiding,
  verifying,    // NEW: Part verification in progress
  inspecting,   // NEW: Frozen-frame damage inspection
  repairing,    // NEW: AR-guided repair steps
  analyzing,
  unclear,
  verified,
  danger,
}
enum ToastKind { analyzing, sent, analyzed, resultOk, resultWarn, error, verifying, inspecting }

// ── Layout dimensions ──────────────────────────────────────────────────────
const double kPanelHeight = 240.0;
const double kCaptureSize = 72.0;
const double kBoxSize     = 300.0;

// ── Colour palette ─────────────────────────────────────────────────────────
abstract class C {
  static const bg          = Color(0xFF1A1A1A);
  static const primary     = Color(0xFF34D399);
  static const gold        = Color(0xFFFFCC15);
  static const danger      = Color(0xFFFF4B4B);
  static const warning     = Color(0xFFF59E0B);
  static const panelBg1    = Color(0xFF1B1B1B);
  static const panelBg2    = Color(0xFF111111);
  static const toastBg     = Color(0xF01E1E1E);
  static const goldBadgeBg = Color(0xFF5A4A2A);
  static const textPrimary = Color(0xFFFFFFFF);
  static const textSoft    = Color(0xFFEAEAEA);
  static const textMuted   = Color(0xFFCFCFCF);
  static const arrowGreen  = Color(0xFF22C55E);
  static const arrowBlue   = Color(0xFF38BDF8);
}
