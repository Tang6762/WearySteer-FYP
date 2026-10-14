import 'package:flutter/material.dart';
import '../models/fatigue_sample.dart';

class ModalityBreakdown extends StatelessWidget {
  const ModalityBreakdown({super.key, required this.sample});
  final FatigueSample sample;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _bar('Facial dynamics (EAR / PERCLOS / MAR)', sample.faceScore, const Color(0xFF3498DB)),
        const SizedBox(height: 10),
        _bar('Head pose (pitch / yaw)', sample.poseScore, const Color(0xFF9B59B6)),
        const SizedBox(height: 10),
        _bar('Posture proxy', sample.postureScore, const Color(0xFF1ABC9C)),
      ],
    );
  }

  Widget _bar(String label, double value, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Expanded(child: Text(label, style: const TextStyle(color: Colors.white70, fontSize: 12))),
            Text('${value.toStringAsFixed(0)}%',
                style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold)),
          ],
        ),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(6),
          child: LinearProgressIndicator(
            value: value.clamp(0, 100) / 100.0,
            minHeight: 8,
            backgroundColor: Colors.white12,
            valueColor: AlwaysStoppedAnimation<Color>(color),
          ),
        ),
      ],
    );
  }
}
