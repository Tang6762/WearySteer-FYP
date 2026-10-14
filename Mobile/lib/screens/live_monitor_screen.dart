import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import '../models/fatigue_sample.dart';
import '../services/dummy_fatigue_simulator.dart';
import '../services/live_camera_service.dart';
import '../widgets/alert_banner.dart';
import '../widgets/modality_breakdown.dart';
import '../widgets/msfi_gauge.dart';
import 'dashboard_screen.dart';

class LiveMonitorScreen extends StatefulWidget {
  const LiveMonitorScreen({super.key});

  @override
  State<LiveMonitorScreen> createState() => _LiveMonitorScreenState();
}

class _LiveMonitorScreenState extends State<LiveMonitorScreen> {
  final DummyFatigueSimulator _sim = DummyFatigueSimulator();
  final LiveCameraService _camera = LiveCameraService();
  final List<FatigueSample> _history = [];

  FatigueSample _current = FatigueSample(
    timestamp: DateTime.now(),
    faceScore: 0,
    poseScore: 0,
    postureScore: 0,
    msfi: 0,
    tier: AlertTier.normal,
  );
  bool _running = false;
  bool _cameraReady = false;
  String? _cameraError;

  @override
  void initState() {
    super.initState();
    _sim.stream.listen((sample) {
      if (!mounted) return;
      setState(() {
        _current = sample;
        _history.add(sample);
        if (_history.length > 200) _history.removeAt(0);
      });
    });
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      await _camera.initialize();
      if (!mounted) return;
      setState(() => _cameraReady = true);
    } catch (e) {
      if (!mounted) return;
      setState(() => _cameraError =
      'Camera unavailable: $e\n\nGrant camera permission in device Settings and restart the app.');
    }
  }

  void _toggleRunning() {
    setState(() => _running = !_running);
    if (_running) {
      _sim.start();
    } else {
      _sim.stop();
    }
  }

  void _reset() {
    _sim.reset();
    setState(() => _history.clear());
  }

  @override
  void dispose() {
    _sim.dispose();
    _camera.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        backgroundColor: const Color(0xFF0D1117),
        elevation: 0,
        titleSpacing: 16,
        title: _buildBrandTitle(),
        actions: [
          IconButton(
            icon: const Icon(Icons.show_chart),
            tooltip: 'Trend dashboard',
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute(builder: (_) => DashboardScreen(history: _history)),
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            flex: 4,
            child: Stack(
              fit: StackFit.expand,
              children: [
                _buildCameraArea(),
                Positioned(top: 12, left: 12, right: 12, child: AlertBanner(tier: _current.tier)),
              ],
            ),
          ),
          Expanded(
            flex: 6,
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              decoration: const BoxDecoration(
                color: Color(0xFF161B22),
                borderRadius: BorderRadius.only(topLeft: Radius.circular(24), topRight: Radius.circular(24)),
              ),
              child: SingleChildScrollView(
                child: Column(
                  children: [
                    MsfiGauge(msfi: _current.msfi, tier: _current.tier),
                    const SizedBox(height: 4),
                    Text(
                      _current.tier.label,
                      style: TextStyle(color: tierColor(_current.tier), fontWeight: FontWeight.bold, fontSize: 16),
                    ),
                    const SizedBox(height: 4),
                    const Text(
                      'MSFI shown here is simulated — not yet driven by the camera feed above.',
                      textAlign: TextAlign.center,
                      style: TextStyle(color: Colors.white38, fontSize: 11),
                    ),
                    const SizedBox(height: 16),
                    const Align(
                      alignment: Alignment.centerLeft,
                      child: Text('Per-modality breakdown',
                          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                    ),
                    const SizedBox(height: 12),
                    ModalityBreakdown(sample: _current),
                    const SizedBox(height: 24),
                    Row(
                      children: [
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: _toggleRunning,
                            icon: Icon(_running ? Icons.pause : Icons.play_arrow),
                            label: Text(_running ? 'Pause' : 'Start'),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: _running ? Colors.orange : Colors.green,
                              foregroundColor: Colors.white,
                              padding: const EdgeInsets.symmetric(vertical: 14),
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: _running ? _sim.triggerDrowsyEvent : null,
                            icon: const Icon(Icons.bolt),
                            label: const Text('Simulate Drowsy'),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFFE74C3C),
                              foregroundColor: Colors.white,
                              padding: const EdgeInsets.symmetric(vertical: 14),
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    SizedBox(
                      width: double.infinity,
                      child: OutlinedButton.icon(
                        onPressed: _reset,
                        icon: const Icon(Icons.refresh),
                        label: const Text('Reset session'),
                        style: OutlinedButton.styleFrom(
                          foregroundColor: Colors.white70,
                          padding: const EdgeInsets.symmetric(vertical: 14),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  /// Styled "WearySteer" brand title: a small glowing steering-wheel icon
  /// next to a gold-gradient wordmark with tightened, weighted letterforms.
  Widget _buildBrandTitle() {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          padding: const EdgeInsets.all(6),
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            gradient: const LinearGradient(
              colors: [Color(0xFFF39C12), Color(0xFFE67E22)],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFFF39C12).withOpacity(0.45),
                blurRadius: 10,
                spreadRadius: 1,
              ),
            ],
          ),
          child: const Icon(Icons.remove_red_eye_rounded, size: 16, color: Color(0xFF0D1117)),
        ),
        const SizedBox(width: 10),
        ShaderMask(
          shaderCallback: (bounds) => const LinearGradient(
            colors: [Color(0xFFFFE0A3), Color(0xFFF39C12)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ).createShader(bounds),
          child: const Text(
            'WearySteer',
            style: TextStyle(
              color: Colors.white,
              fontSize: 22,
              fontWeight: FontWeight.w800,
              letterSpacing: 0.6,
              height: 1.0,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildCameraArea() {
    if (_cameraError != null) {
      return Container(
        color: const Color(0xFF161B22),
        padding: const EdgeInsets.all(24),
        child: Center(
          child: Text(_cameraError!, textAlign: TextAlign.center, style: const TextStyle(color: Colors.white70)),
        ),
      );
    }
    if (!_cameraReady || _camera.controller == null) {
      return const Center(child: CircularProgressIndicator());
    }
    return CameraPreview(_camera.controller!);
  }
}