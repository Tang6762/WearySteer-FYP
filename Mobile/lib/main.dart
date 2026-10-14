import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'screens/live_monitor_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Hide the status bar and navigation bar (taskbar). "immersiveSticky" means
  // they only reappear temporarily when the user swipes from the top/bottom
  // edge, then auto-hide again after a moment.
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
  runApp(const WearySteerApp());
}

class WearySteerApp extends StatefulWidget {
  const WearySteerApp({super.key});

  @override
  State<WearySteerApp> createState() => _WearySteerAppState();
}

class _WearySteerAppState extends State<WearySteerApp> with WidgetsBindingObserver {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Android can restore the system bars when the app comes back to the
    // foreground, so re-hide them each time.
    if (state == AppLifecycleState.resumed) {
      SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'WearySteer',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0D1117),
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFFF39C12), brightness: Brightness.dark),
      ),
      home: const LiveMonitorScreen(),
    );
  }
}