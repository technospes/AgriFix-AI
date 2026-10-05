// import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../screens/home/home_screen.dart';
import '../screens/upload/upload_screen.dart';
import '../../screens/solution/solution_screen.dart';
import '../../screens/ar_guide/screens/ar_guide_screen.dart';

class AppRoutes {
  static const home     = '/';
  static const upload   = '/upload';
  static const solution = '/solution';
  static const arGuide  = '/ar-guide';
}

final appRouter = GoRouter(
  initialLocation: AppRoutes.home,
  routes: [
    GoRoute(
      path: AppRoutes.home,
      builder: (context, state) => const HomeScreen(),
    ),
    GoRoute(
      path: AppRoutes.upload,
      builder: (context, state) => const UploadScreen(),
    ),
    GoRoute(
      path: AppRoutes.solution,
      builder: (context, state) => const SolutionScreen(),
    ),
    GoRoute(
      path: AppRoutes.arGuide,
      builder: (context, state) {
        final step = (state.extra as int?) ?? 0;
        return ARGuideScreen(initialStep: step);
      },
    ),
  ],
);