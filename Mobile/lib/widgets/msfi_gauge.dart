import 'dart:math';
import 'package:flutter/material.dart';
import '../models/fatigue_sample.dart';

Color tierColor(AlertTier tier) {
  switch (tier) {
    case AlertTier.normal:
      return const Color(0xFF2ECC71);
    case AlertTier.mild:
      return const Color(0xFFF1C40F);
    case AlertTier.moderate:
      return const Color(0xFFE67E22);
    case AlertTier.severe:
      return const Color(0xFFE74C3C);
  }
}

class MsfiGauge extends StatelessWidget {
  const MsfiGauge({super.key, required this.msfi, required this.tier});
  final double msfi;
  final AlertTier tier;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 220,
      height: 130,
      child: CustomPaint(
        painter: _GaugePainter(msfi: msfi, color: tierColor(tier)),
        child: Padding(
          padding: const EdgeInsets.only(top: 60),
          child: FittedBox(
            fit: BoxFit.scaleDown,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  msfi.toStringAsFixed(0),
                  style: const TextStyle(fontSize: 40, fontWeight: FontWeight.bold, color: Colors.white),
                ),
                const Text('MSFI', style: TextStyle(fontSize: 12, letterSpacing: 2, color: Colors.white70)),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _GaugePainter extends CustomPainter {
  _GaugePainter({required this.msfi, required this.color});
  final double msfi;
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height);
    final radius = size.width / 2 - 10;

    final bgPaint = Paint()
      ..color = Colors.white12
      ..style = PaintingStyle.stroke
      ..strokeWidth = 14
      ..strokeCap = StrokeCap.round;

    final fgPaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 14
      ..strokeCap = StrokeCap.round;

    const startAngle = pi;
    const sweepAngle = pi;

    canvas.drawArc(Rect.fromCircle(center: center, radius: radius), startAngle, sweepAngle, false, bgPaint);

    final valueSweep = sweepAngle * (msfi.clamp(0, 100) / 100.0);
    canvas.drawArc(Rect.fromCircle(center: center, radius: radius), startAngle, valueSweep, false, fgPaint);
  }

  @override
  bool shouldRepaint(covariant _GaugePainter oldDelegate) =>
      oldDelegate.msfi != msfi || oldDelegate.color != color;
}