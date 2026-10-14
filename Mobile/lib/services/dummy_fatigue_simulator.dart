import 'dart:async';
import 'dart:math';
import '../models/fatigue_sample.dart';

/// Generates fake but plausible MSFI readings on a timer, so the UI has
/// live-looking data without needing camera/ML dependencies.
class DummyFatigueSimulator {
  final _rand = Random();
  final _controller = StreamController<FatigueSample>.broadcast();
  Timer? _timer;

  double _baseline = 15; // slowly drifting baseline MSFI
  double _spikeBoost = 0; // temporary boost from "Simulate Drowsy Event"

  Stream<FatigueSample> get stream => _controller.stream;

  void start() {
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(milliseconds: 800), (_) => _tick());
  }

  void stop() {
    _timer?.cancel();
  }

  void triggerDrowsyEvent() {
    _spikeBoost = 55 + _rand.nextDouble() * 20; // push into moderate/severe
  }

  void reset() {
    _baseline = 15;
    _spikeBoost = 0;
  }

  void _tick() {
    // random walk on the baseline
    _baseline += (_rand.nextDouble() - 0.5) * 6;
    _baseline = _baseline.clamp(5, 35);

    // spike decays back down over a few ticks
    if (_spikeBoost > 0) {
      _spikeBoost -= 6;
      if (_spikeBoost < 0) _spikeBoost = 0;
    }

    final msfi = (_baseline + _spikeBoost + _rand.nextDouble() * 5).clamp(0, 100).toDouble();

    final faceScore = (msfi * (0.9 + _rand.nextDouble() * 0.2)).clamp(0, 100).toDouble();
    final poseScore = (msfi * (0.6 + _rand.nextDouble() * 0.3)).clamp(0, 100).toDouble();
    final postureScore = (msfi * (0.4 + _rand.nextDouble() * 0.3)).clamp(0, 100).toDouble();

    _controller.add(FatigueSample(
      timestamp: DateTime.now(),
      faceScore: faceScore,
      poseScore: poseScore,
      postureScore: postureScore,
      msfi: msfi,
      tier: FatigueSample.tierFor(msfi),
    ));
  }

  void dispose() {
    _timer?.cancel();
    _controller.close();
  }
}
