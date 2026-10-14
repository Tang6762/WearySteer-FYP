import 'package:tflite_flutter/tflite_flutter.dart';

class DrowsinessClassifier {
  // Replace with the exact values from wearysteer_scaler.pkl (mean_ / scale_)
  static const _mean = [0.2687, 0.0128, 0.2780];
  static const _std  = [0.0507, 0.0437, 0.0724];

  late final Interpreter _dense;
  late final Interpreter _deep;

  Future<void> load() async {
    _dense = await Interpreter.fromAsset('assets/models/wearysteer_dense.tflite');
    _deep  = await Interpreter.fromAsset('assets/models/wearysteer_deep.tflite');
  }

  /// Returns [pAlert, pLowVigilance, pDrowsy]. Class order is assumed to match CLASS_NAMES in your Python.
  List<double> predict(double ear, double mar, double slump, {bool useDeep = true}) {
    final f = [ear, mar, slump];
    final input = [List<double>.generate(3, (i) => (f[i] - _mean[i]) / _std[i])];
    final output = [List<double>.filled(3, 0)];
    (useDeep ? _deep : _dense).run(input, output);
    return output[0];
  }

  void dispose() { _dense.close(); _deep.close(); }
}