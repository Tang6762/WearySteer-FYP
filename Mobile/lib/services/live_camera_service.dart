import 'package:camera/camera.dart';

/// Opens a live camera preview only — no frame processing yet.
/// This is the hook point for later: once you're ready to wire in real
/// video-based fatigue signals, process frames inside a
/// `controller.startImageStream(...)` call added to this class, and feed
/// the results into FatigueEngine/whatever replaces DummyFatigueSimulator.
class LiveCameraService {
  CameraController? controller;
  List<CameraDescription> _cameras = [];

  Future<void> initialize() async {
    _cameras = await availableCameras();
    if (_cameras.isEmpty) {
      throw CameraException('noCamerasFound', 'No cameras available on this device.');
    }
    final frontCamera = _cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => _cameras.first,
    );

    controller = CameraController(
      frontCamera,
      ResolutionPreset.medium,
      enableAudio: false,
    );
    await controller!.initialize();
  }

  Future<void> dispose() async {
    await controller?.dispose();
  }
}
