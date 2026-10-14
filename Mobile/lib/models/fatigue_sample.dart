enum AlertTier { normal, mild, moderate, severe }

extension AlertTierX on AlertTier {
  String get label {
    switch (this) {
      case AlertTier.normal:
        return 'Normal';
      case AlertTier.mild:
        return 'Mild Fatigue';
      case AlertTier.moderate:
        return 'Moderate Fatigue';
      case AlertTier.severe:
        return 'Severe Fatigue';
    }
  }
}

class FatigueSample {
  final DateTime timestamp;
  final double faceScore; // 0-100
  final double poseScore; // 0-100
  final double postureScore; // 0-100
  final double msfi; // 0-100 fused
  final AlertTier tier;

  FatigueSample({
    required this.timestamp,
    required this.faceScore,
    required this.poseScore,
    required this.postureScore,
    required this.msfi,
    required this.tier,
  });

  static AlertTier tierFor(double msfi) {
    if (msfi >= 80) return AlertTier.severe;
    if (msfi >= 60) return AlertTier.moderate;
    if (msfi >= 40) return AlertTier.mild;
    return AlertTier.normal;
  }
}
