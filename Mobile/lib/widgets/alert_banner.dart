import 'package:flutter/material.dart';
import '../models/fatigue_sample.dart';
import 'msfi_gauge.dart';

class AlertBanner extends StatelessWidget {
  const AlertBanner({super.key, required this.tier});
  final AlertTier tier;

  @override
  Widget build(BuildContext context) {
    if (tier == AlertTier.normal) return const SizedBox.shrink();

    final color = tierColor(tier);
    final icon = switch (tier) {
      AlertTier.mild => Icons.visibility_outlined,
      AlertTier.moderate => Icons.warning_amber_rounded,
      AlertTier.severe => Icons.report_gmailerrorred_rounded,
      AlertTier.normal => Icons.check_circle_outline,
    };

    return AnimatedContainer(
      duration: const Duration(milliseconds: 250),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        border: Border.all(color: color, width: 1.5),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Icon(icon, color: color),
          const SizedBox(width: 10),
          Expanded(
            child: Text(tier.label, style: TextStyle(color: color, fontWeight: FontWeight.bold, fontSize: 15)),
          ),
        ],
      ),
    );
  }
}
