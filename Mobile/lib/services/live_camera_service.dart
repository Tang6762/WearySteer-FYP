import 'dart:async';
import 'dart:collection';
import 'dart:io' show Platform;
import 'dart:math';
import 'dart:ui' show Size;

import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_commons/google_mlkit_commons.dart';
import 'package:google_mlkit_face_mesh_detection/google_mlkit_face_mesh_detection.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

import '../models/fatigue_sample.dart';
import 'drowsiness_classifier.dart';

/// Opens the front camera, extracts EAR / MAR / shoulder-slump from each
/// frame (same formulas as wearysteer_common.py) and feeds them to the
/// TFLite model. Emits one FatigueSample per analysed frame on [stream].
class LiveCameraService {
  // Landmark indices — identical to wearysteer_common.py
  static const _leftEye = [362, 385, 387, 263, 373, 380];
  static const _rightEye = [33, 160, 158, 133, 153, 144];
  static const _mouth = [61, 81, 13, 312, 317, 14, 87, 178];

  // ~6-7 analyses per second (matches the 30fps / every-5th-frame training rate)
  static const _minGap = Duration(milliseconds: 150);
  static const _smoothing = 10; // average the last N probability vectors

  static const _orientations = {
    DeviceOrientation.portraitUp: 0,
    DeviceOrientation.landscapeLeft: 90,
    DeviceOrientation.portraitDown: 180,
    DeviceOrientation.landscapeRight: 270,
  };

  CameraController? controller;
  CameraDescription? _camera;

  final FaceMeshDetector _faceMesh =
  FaceMeshDetector(option: FaceMeshDetectorOptions.faceMesh);
  final PoseDetector _pose =
  PoseDetector(options: PoseDetectorOptions(mode: PoseDetectionMode.stream));
  final DrowsinessClassifier _classifier = DrowsinessClassifier();

  final StreamController<FatigueSample> _out =
  StreamController<FatigueSample>.broadcast();
  final Queue<List<double>> _recent = Queue();

  bool _busy = false;
  DateTime _last = DateTime.fromMillisecondsSinceEpoch(0);
  double _uprightHeight = 1;

  Stream<FatigueSample> get stream => _out.stream;

  Future<void> initialize() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      throw CameraException('noCamerasFound', 'No cameras available on this device.');
    }
    _camera = cameras.firstWhere(
          (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );

    controller = CameraController(
      _camera!,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.nv21, // required by ML Kit on Android
    );
    await controller!.initialize();
    await _classifier.load();
  }

  Future<void> start() async {
    final c = controller;
    if (c == null || c.value.isStreamingImages) return;
    await c.startImageStream(_onFrame);
  }

  Future<void> stop() async {
    final c = controller;
    if (c == null || !c.value.isStreamingImages) return;
    await c.stopImageStream();
  }

  void reset() => _recent.clear();

  // ---------------------------------------------------------------- frames

  Future<void> _onFrame(CameraImage image) async {
    if (_busy) return;
    final now = DateTime.now();
    if (now.difference(_last) < _minGap) return;
    _busy = true;
    _last = now;
    try {
      await _analyse(image);
    } catch (_) {
      // drop the frame on any ML Kit hiccup
    } finally {
      _busy = false;
    }
  }

  Future<void> _analyse(CameraImage image) async {
    final input = _toInputImage(image);
    if (input == null) return;

    // --- face: EAR + MAR
    final meshes = await _faceMesh.processImage(input);
    if (meshes.isEmpty) return; // no face this frame
    final pts = {for (final p in meshes.first.points) p.index: p};
    if (pts.length < 400) return;

    final ear = (_ear(pts, _leftEye) + _ear(pts, _rightEye)) / 2.0;
    final mar = _mar(pts, _mouth);

    // --- pose: shoulder slump = (mean shoulder y - nose y) / image height
    var slump = 0.0;
    final poses = await _pose.processImage(input);
    if (poses.isNotEmpty) {
      final lm = poses.first.landmarks;
      final nose = lm[PoseLandmarkType.nose];
      final ls = lm[PoseLandmarkType.leftShoulder];
      final rs = lm[PoseLandmarkType.rightShoulder];
      if (nose != null && ls != null && rs != null) {
        slump = (((ls.y + rs.y) / 2.0) - nose.y) / _uprightHeight;
      }
    }

    // --- model
    final probs = _classifier.predict(ear, mar, slump); // [alert, low, drowsy]
    _recent.addLast(probs);
    while (_recent.length > _smoothing) {
      _recent.removeFirst();
    }
    final avg = List<double>.filled(3, 0);
    for (final p in _recent) {
      for (var i = 0; i < 3; i++) {
        avg[i] += p[i] / _recent.length;
      }
    }

    // 0-100 fatigue index: low vigilance counts half, drowsy counts full
    final msfi = (100.0 * (0.5 * avg[1] + avg[2])).clamp(0.0, 100.0).toDouble();

    if (_out.isClosed) return;
    _out.add(FatigueSample(
      timestamp: DateTime.now(),
      faceScore: msfi,
      poseScore: msfi,
      postureScore: msfi,
      msfi: msfi,
      tier: FatigueSample.tierFor(msfi),
    ));
  }

  // -------------------------------------------------------------- features

  double _dist(FaceMeshPoint a, FaceMeshPoint b) =>
      sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));

  double _ear(Map<int, FaceMeshPoint> p, List<int> i) {
    final dV1 = _dist(p[i[1]]!, p[i[5]]!);
    final dV2 = _dist(p[i[2]]!, p[i[4]]!);
    final dH = _dist(p[i[0]]!, p[i[3]]!);
    return dH == 0 ? 0.0 : (dV1 + dV2) / (2.0 * dH);
  }

  double _mar(Map<int, FaceMeshPoint> p, List<int> i) {
    final dV = _dist(p[i[2]]!, p[i[5]]!);
    final dH = _dist(p[i[0]]!, p[i[3]]!);
    return dH == 0 ? 0.0 : dV / dH;
  }

  // ----------------------------------------------------------- image convert

  InputImage? _toInputImage(CameraImage image) {
    final cam = _camera;
    final c = controller;
    if (cam == null || c == null || !Platform.isAndroid) return null;

    final comp = _orientations[c.value.deviceOrientation];
    if (comp == null) return null;
    final rot = cam.lensDirection == CameraLensDirection.front
        ? (cam.sensorOrientation + comp) % 360
        : (cam.sensorOrientation - comp + 360) % 360;

    final rotation = InputImageRotationValue.fromRawValue(rot);
    final format = InputImageFormatValue.fromRawValue(image.format.raw);
    if (rotation == null || format != InputImageFormat.nv21 || image.planes.length != 1) {
      return null;
    }

    // ML Kit reports coordinates in the upright image, so height swaps at 90/270
    _uprightHeight = (rot == 90 || rot == 270 ? image.width : image.height).toDouble();

    final plane = image.planes.first;
    return InputImage.fromBytes(
      bytes: plane.bytes,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: format!,
        bytesPerRow: plane.bytesPerRow,
      ),
    );
  }

  Future<void> dispose() async {
    try {
      await stop();
    } catch (_) {}
    await controller?.dispose();
    await _faceMesh.close();
    await _pose.close();
    _classifier.dispose();
    await _out.close();
  }
}