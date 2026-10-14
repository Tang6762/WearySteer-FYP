import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import '../models/fatigue_sample.dart';
import '../widgets/msfi_gauge.dart';

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key, required this.history});
  final List<FatigueSample> history;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(backgroundColor: const Color(0xFF0D1117), title: const Text('Session Trend')),
      body: history.isEmpty
          ? const Center(
              child: Text('No data yet — go back and hit Start.', style: TextStyle(color: Colors.white70)))
          : Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _summaryRow(),
                  const SizedBox(height: 24),
                  const Text('MSFI over time', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 12),
                  Expanded(child: _chart()),
                  const SizedBox(height: 12),
                  _legend(),
                ],
              ),
            ),
    );
  }

  Widget _summaryRow() {
    final peak = history.map((s) => s.msfi).reduce((a, b) => a > b ? a : b);
    final avg = history.map((s) => s.msfi).reduce((a, b) => a + b) / history.length;
    final alerts = history.where((s) => s.tier != AlertTier.normal).length;

    Widget stat(String label, String value) => Expanded(
          child: Column(
            children: [
              Text(value, style: const TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold)),
              Text(label, style: const TextStyle(color: Colors.white54, fontSize: 12)),
            ],
          ),
        );

    return Row(
      children: [
        stat('Peak MSFI', peak.toStringAsFixed(0)),
        stat('Avg MSFI', avg.toStringAsFixed(0)),
        stat('Alert samples', alerts.toString()),
      ],
    );
  }

  Widget _chart() {
    final spots = <FlSpot>[for (var i = 0; i < history.length; i++) FlSpot(i.toDouble(), history[i].msfi)];

    return LineChart(
      LineChartData(
        minY: 0,
        maxY: 100,
        gridData: const FlGridData(show: true, drawVerticalLine: false),
        titlesData: const FlTitlesData(
          topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          bottomTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          leftTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, reservedSize: 32, interval: 20)),
        ),
        borderData: FlBorderData(show: false),
        rangeAnnotations: RangeAnnotations(
          horizontalRangeAnnotations: [
            HorizontalRangeAnnotation(y1: 80, y2: 100, color: const Color(0x33E74C3C)),
            HorizontalRangeAnnotation(y1: 60, y2: 80, color: const Color(0x33E67E22)),
            HorizontalRangeAnnotation(y1: 40, y2: 60, color: const Color(0x33F1C40F)),
          ],
        ),
        lineBarsData: [
          LineChartBarData(
            spots: spots,
            isCurved: true,
            color: const Color(0xFF58A6FF),
            barWidth: 2.5,
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(show: true, color: const Color(0x2258A6FF)),
          ),
        ],
      ),
    );
  }

  Widget _legend() {
    Widget dot(Color c, String label) => Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(width: 10, height: 10, decoration: BoxDecoration(color: c, shape: BoxShape.circle)),
            const SizedBox(width: 6),
            Text(label, style: const TextStyle(color: Colors.white60, fontSize: 11)),
          ],
        );
    return Wrap(
      spacing: 16,
      children: [
        dot(tierColor(AlertTier.mild), 'Mild band'),
        dot(tierColor(AlertTier.moderate), 'Moderate band'),
        dot(tierColor(AlertTier.severe), 'Severe band'),
      ],
    );
  }
}
