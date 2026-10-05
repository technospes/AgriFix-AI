// lib/screens/ar_guide/models/bbox.dart
import 'dart:math' as math;
import 'package:flutter/material.dart';

/// Normalised bounding box — all coords in 0.0–1.0 image space.
/// Pure data model: no Flutter widgets, no side effects.
class NormBbox {
  final double cx, cy, w, h;
  const NormBbox(this.cx, this.cy, this.w, this.h);

  NormBbox lerp(NormBbox other, double t) => NormBbox(
    cx + (other.cx - cx) * t,
    cy + (other.cy - cy) * t,
    w  + (other.w  - w ) * t,
    h  + (other.h  - h ) * t,
  );

  double distanceTo(NormBbox other) {
    final dx = cx - other.cx, dy = cy - other.cy;
    final dw = w  - other.w,  dh = h  - other.h;
    return math.sqrt(dx*dx + dy*dy + dw*dw + dh*dh);
  }

  Rect toScreenRect(double previewW, double previewH) => Rect.fromCenter(
    center: Offset(cx * previewW, cy * previewH),
    width:  w * previewW,
    height: h * previewH,
  );
}
