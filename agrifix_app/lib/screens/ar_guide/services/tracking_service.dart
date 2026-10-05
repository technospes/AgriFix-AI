// lib/screens/ar_guide/services/tracking_service.dart
//
// Bounding-box smoothing, Kalman filter, velocity estimation, drift detection.
// Pure math/logic — no Flutter widgets, no timers, no setState.
// The ARController owns timers and calls into this service each tick.

import 'dart:math' as math;
import '../models/bbox.dart';

class TrackingService {
  // ── Kalman filter constants ───────────────────────────────────────────────
  static const _kKalmanQ = 0.001;   // process noise  — part moves slowly
  static const _kKalmanR = 0.015;   // measurement noise — ~1.5% bbox error

  // ── Tracking thresholds (identical to monolith) ───────────────────────────
  static const kStabilityThreshold  = 0.04;   // max delta for stable frame
  // Lowered 2→1: show arrow on first valid detection.
  // One Gemini confirmation + Kalman hold is enough for immediate feedback.
  // Re-acquire logic dismisses false positives within <1s.
  static const kStableFramesNeeded  = 2;
  static const kLockFramesNeeded    = 4;
  static const kConfThreshold       = 0.82;
  static const kLockConfThreshold   = 0.85;
  static const kJumpThreshold       = 0.25;
  static const kVelocityResetDist   = 0.15;
  static const kMaxVelocity         = 0.5;
  static const kDriftThreshold      = 0.08;
  static const kMaxCorrectionMs     = 4000;

  // ── Kalman state ──────────────────────────────────────────────────────────
  double _kfCxEst = 0.0, _kfCyEst = 0.0;
  double _kfPCx   = 1.0, _kfPCy   = 1.0;
  bool   _kfInitialised = false;

  // ── Velocity state ────────────────────────────────────────────────────────
  double    velCx = 0.0, velCy = 0.0;
  DateTime? lastBboxTime;

  // ── Public state exposed to controller ───────────────────────────────────
  int      stableFrameCount = 0;
  NormBbox? smoothBbox;
  NormBbox? prevBbox;
  NormBbox? lastAiBbox;

  // ── Reset ─────────────────────────────────────────────────────────────────
  void reset() {
    _kfCxEst = 0.0; _kfCyEst = 0.0;
    _kfPCx   = 1.0; _kfPCy   = 1.0;
    _kfInitialised = false;
    velCx = 0.0; velCy = 0.0;
    stableFrameCount = 0;
    smoothBbox = null;
    prevBbox   = null;
    lastAiBbox = null;
    lastBboxTime = null;
  }

  void resetVelocity() {
    velCx = 0.0; velCy = 0.0;
  }

  void resetKalman() {
    _kfInitialised = false;
    _kfPCx = 1.0; _kfPCy = 1.0;
  }

  // ── Kalman PREDICT (called at 30fps in guiding state) ─────────────────────
  // Returns updated smoothBbox if prediction was applied, null if skipped.
  NormBbox? predictTick(double clampedDt) {
    if (smoothBbox == null || !_kfInitialised) return null;
    final speed = math.sqrt(velCx * velCx + velCy * velCy);
    if (speed < 0.001) return null;  // stationary — skip

    final predCx = (_kfCxEst + velCx * clampedDt).clamp(0.01, 0.99);
    final predCy = (_kfCyEst + velCy * clampedDt).clamp(0.01, 0.99);
    _kfPCx += _kKalmanQ;
    _kfPCy += _kKalmanQ;
    _kfCxEst = predCx;
    _kfCyEst = predCy;
    final predicted = NormBbox(predCx, predCy, smoothBbox!.w, smoothBbox!.h);
    smoothBbox = predicted;
    return predicted;
  }

  // ── Kalman UPDATE (called on each Gemini measurement) ─────────────────────
  // Returns the validated+smoothed bbox, or null if the raw bbox failed sanity.
  NormBbox? update({
    required double cx, required double cy,
    required double bw, required double bh,
    required double confidence,
  }) {
    // ── Coordinate validation ────────────────────────────────────────────────
    if (cx <= 0 || cx >= 1 || cy <= 0 || cy >= 1 ||
        bw <= 0 || bw >= 1 || bh <= 0 || bh >= 1) return null;
    // Strict 0/1 bounds — no edge margin.
    // A bbox touching the frame edge is still a valid detection.
    if (cx - bw / 2 < 0.0 || cx + bw / 2 > 1.0 ||
        cy - bh / 2 < 0.0 || cy + bh / 2 > 1.0) return null;

    // ── Size sanity ──────────────────────────────────────────────────────────
    final bboxArea = bw * bh;
    // Per-axis upper limit removed — any single-axis size is valid.
    // Large parts (air_filter h=0.90, clutch_pedal w=0.65) are real detections.
    // Area max raised 0.60 → 0.70 to match backend (covers area=0.63 cases).
    // Full-image hallucinations (area > 0.70) are still blocked.
    if (bw < 0.03 || bh < 0.03 ||
        bboxArea < 0.009 || bboxArea > 0.70) return null;

    // ── Kalman UPDATE ────────────────────────────────────────────────────────
    final double newCx, newCy;
    if (!_kfInitialised) {
      _kfCxEst = cx; _kfCyEst = cy;
      _kfPCx   = _kKalmanR; _kfPCy = _kKalmanR;
      _kfInitialised = true;
      newCx = cx; newCy = cy;
    } else {
      final kGainCx = _kfPCx / (_kfPCx + _kKalmanR);
      final kGainCy = _kfPCy / (_kfPCy + _kKalmanR);
      newCx    = _kfCxEst + kGainCx * (cx - _kfCxEst);
      newCy    = _kfCyEst + kGainCy * (cy - _kfCyEst);
      _kfCxEst = newCx; _kfCyEst = newCy;
      _kfPCx   = (1 - kGainCx) * _kfPCx;
      _kfPCy   = (1 - kGainCy) * _kfPCy;
    }

    // ── Jump detection ───────────────────────────────────────────────────────
    final jumpDist = smoothBbox == null ? 1.0 : math.sqrt(
      math.pow(newCx - smoothBbox!.cx, 2) +
      math.pow(newCy - smoothBbox!.cy, 2),
    );
    final didJump = jumpDist > kJumpThreshold;
    if (didJump) {
      _kfCxEst = cx; _kfCyEst = cy;
      _kfPCx   = _kKalmanR; _kfPCy = _kKalmanR;
    }

    final newSmooth = NormBbox(
        newCx.clamp(0.01, 0.99), newCy.clamp(0.01, 0.99), bw, bh);
    if (didJump) stableFrameCount = 0;

    // ── Stability counting ───────────────────────────────────────────────────
    final delta = prevBbox == null ? 0.0 : newSmooth.distanceTo(prevBbox!);
    if (delta < kStabilityThreshold) { stableFrameCount++; }
    else                             { stableFrameCount = 0; }

    // ── Velocity ─────────────────────────────────────────────────────────────
    if (prevBbox != null && lastBboxTime != null) {
      final jumpDist2 = math.sqrt(
        math.pow(newSmooth.cx - prevBbox!.cx, 2) +
        math.pow(newSmooth.cy - prevBbox!.cy, 2),
      );
      if (jumpDist2 > kVelocityResetDist) {
        velCx = 0.0; velCy = 0.0;
      } else {
        final dtSec = DateTime.now().difference(lastBboxTime!)
            .inMilliseconds / 1000.0;
        if (dtSec > 0.05) {
          velCx = (newSmooth.cx - prevBbox!.cx) / dtSec;
          velCy = (newSmooth.cy - prevBbox!.cy) / dtSec;
          velCx = velCx.clamp(-kMaxVelocity, kMaxVelocity);
          velCy = velCy.clamp(-kMaxVelocity, kMaxVelocity);
        }
      }
    }
    lastBboxTime = DateTime.now();
    smoothBbox   = newSmooth;
    prevBbox     = newSmooth;
    lastAiBbox   = newSmooth;
    return newSmooth;
  }

  // ── Drift check ──────────────────────────────────────────────────────────
  bool needsCorrection(DateTime? lastCorrectionSent) {
    if (lastAiBbox == null || smoothBbox == null) return true;
    final msSince = lastCorrectionSent == null
        ? 9999
        : DateTime.now().difference(lastCorrectionSent).inMilliseconds;
    final drift   = smoothBbox!.distanceTo(lastAiBbox!);
    return drift >= kDriftThreshold || msSince >= kMaxCorrectionMs;
  }

  // ── Distance hint ─────────────────────────────────────────────────────────
  String distanceHint(double bboxArea, bool isHindi) {
    if (bboxArea > 0.65) {  // raised 0.55→0.65 to match new area max of 0.70
      return isHindi
          ? 'थोड़ा पीछे हटें — भाग बहुत करीब है'
          : 'Move phone back slightly — part fills too much of frame';
    }
    if (bboxArea < 0.009) {
      return isHindi ? 'मशीन के और करीब जाएं' : 'Move closer to the machine';
    }
    return '';
  }
}