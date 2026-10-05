import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:go_router/go_router.dart';
import 'package:provider/provider.dart';
import 'package:video_player/video_player.dart';
import 'dart:ui';

import '../../core/theme.dart';
import '../../core/router.dart';
import '../../core/providers/language_provider.dart';
import '../../widgets/language_selector.dart';
import '../../l10n/app_localizations.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
  late VideoPlayerController _videoController;
  bool _videoReady        = false;
  bool _videoError        = false;
  bool _showLanguagePanel = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initVideo();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _videoController.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (!_videoReady) return;
    switch (state) {
      case AppLifecycleState.paused:
      case AppLifecycleState.inactive:
      case AppLifecycleState.hidden:
      case AppLifecycleState.detached:
        _videoController.pause();
        break;
      case AppLifecycleState.resumed:
        _videoController.play();
        break;
    }
  }

  Future<void> _initVideo() async {
    try {
      _videoController =
          VideoPlayerController.asset('assets/videos/farm_bg.mp4');
      await _videoController.initialize();
      _videoController
        ..setLooping(true)
        ..setVolume(0)
        ..play();
      if (mounted) setState(() => _videoReady = true);
    } catch (e) {
      debugPrint('HomeScreen: video init failed — $e');
      if (mounted) setState(() => _videoError = true);
    }
  }

  @override
  Widget build(BuildContext context) {
    final langProvider = context.watch<LanguageProvider>();
    // nullable-getter is false in l10n.yaml so of(context) is non-null
    final l10n = AppLocalizations.of(context);

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          _buildBackground(),
          _buildForeground(langProvider, l10n),
          _buildBottomCard(l10n),
          if (_showLanguagePanel) _buildLanguageOverlay(langProvider),
        ],
      ),
    );
  }

  // ── Layer 1: Background ───────────────────────────────────────────────────

// ── Layer 1: Background ───────────────────────────────────────────────────

  Widget _buildBackground() {
    if (_videoError || !_videoReady) {
      return Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFF1B5E20), Color(0xFF4E342E)],
          ),
        ),
      );
    }
    return SizedBox.expand(
      child: Stack(
        fit: StackFit.expand,
        children: [
          // The actual video
          FittedBox(
            fit: BoxFit.cover,
            child: SizedBox(
              width:  _videoController.value.size.width,
              height: _videoController.value.size.height,
              child: VideoPlayer(_videoController),
            ),
          ),
          // 👇 NEW: A subtle dark overlay over the video to make text readable
          Container(
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.25), // 25% overall darkness
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.center,
                colors: [
                  Colors.black.withValues(alpha: 0.65), // Darker at the top where the text is
                  Colors.transparent,
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ── Layer 2: Logo + headline + language pill ──────────────────────────────

  Widget _buildForeground(
      LanguageProvider langProvider, AppLocalizations? l10n) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.symmetric(
            horizontal: AppSpacing.screenPadding),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Align(
              alignment: Alignment.topRight,
              child: _LanguagePill(
                flag: langProvider.flag,
                onTap: () =>
                    setState(() => _showLanguagePanel = true),
              ),
            )
            .animate()
            .fadeIn(duration: 600.ms, delay: 200.ms)
            .slideY(begin: -0.3, end: 0, curve: Curves.easeOutCubic),

            const SizedBox(height: 24),

            _LogoAndHeadline(
              headline: l10n?.homeHeadline ??
                  'Professional grade machine\nrepair, powered by AI and AR.',
            )
            .animate()
            .fadeIn(duration: 800.ms, delay: 400.ms)
            .slideY(begin: 0.2, end: 0, curve: Curves.easeOutCubic),
          ],
        ),
      ),
    );
  }

  // ── Layer 3: Bottom white card ────────────────────────────────────────────

  Widget _buildBottomCard(AppLocalizations? l10n) {
    return Align(
      alignment: Alignment.bottomCenter,
      child: Container(
        width: double.infinity,
        decoration: BoxDecoration(
          color: AppColors.cardWhite,
          borderRadius: const BorderRadius.only(
            topLeft:  Radius.circular(AppSpacing.cardRadius),
            topRight: Radius.circular(AppSpacing.cardRadius),
          ),
          boxShadow: AppShadows.card,
        ),
        child: SafeArea(
          top: false,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(
              AppSpacing.cardPadding,
              AppSpacing.md,
              AppSpacing.cardPadding,
              AppSpacing.cardPadding,
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Drag handle
                Container(
                  width: 40, height: 4,
                  decoration: BoxDecoration(
                    color: const Color(0xFFDDDDDD),
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
                const SizedBox(height: AppSpacing.md),
                Text(
                  l10n?.homeCardTitle ?? 'Machine Issues?',
                  style: AppTextStyles.cardTitle,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 8),
                Text(
                  l10n?.homeCardSubtitle ??
                      'Get instant expert guidance for your equipment.',
                  style: AppTextStyles.cardSubtitle,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 28),
                _PrimaryButton(
                  label: l10n?.homeCtaButton ?? "Let's Fix Your Machine",
                  onTap: () => context.push(AppRoutes.upload),
                ),
              ],
            ),
          ),
        ),
      ),
    )
    .animate()
    .fadeIn(duration: 700.ms, delay: 100.ms)
    .slideY(begin: 0.12, end: 0, curve: Curves.easeOutCubic);
  }

  // ── Layer 4: Language overlay ─────────────────────────────────────────────

  Widget _buildLanguageOverlay(LanguageProvider langProvider) {
    return GestureDetector(
      onTap: () => setState(() => _showLanguagePanel = false),
      child: Stack(
        fit: StackFit.expand,
        children: [
          BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 14, sigmaY: 14),
            child: Container(
                color: Colors.black.withValues(alpha: 0.32)),
          ),
          SafeArea(
            child: Align(
              alignment: Alignment.topRight,
              child: Padding(
                padding: const EdgeInsets.only(
                    top: 58, right: AppSpacing.screenPadding),
                child: GestureDetector(
                  onTap: () {},
                  child: LanguageSelectorPanel(
                    currentLocale: langProvider.locale,
                    onLocaleSelected: (locale) {
                      langProvider.setLocale(locale);
                      setState(() => _showLanguagePanel = false);
                    },
                  )
                  .animate()
                  .fadeIn(duration: 180.ms)
                  .scale(
                    begin: const Offset(0.88, 0.88),
                    end:   const Offset(1.0, 1.0),
                    duration: 240.ms,
                    curve: Curves.easeOutBack,
                    alignment: Alignment.topRight,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ── Sub-widgets ───────────────────────────────────────────────────────────────

class _LanguagePill extends StatelessWidget {
  final String flag;
  final VoidCallback onTap;
  const _LanguagePill({required this.flag, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 44,
        height: 44,
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.18),
          shape: BoxShape.circle,
          border: Border.all(
              color: Colors.white.withValues(alpha: 0.28), width: 1),
        ),
        child: Center(
          child: Text(flag, style: const TextStyle(fontSize: 22)),
        ),
      ),
    );
  }
}

class _LogoAndHeadline extends StatelessWidget {
  final String headline;
  const _LogoAndHeadline({required this.headline});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // Removed the green background container entirely
        // SizedBox(
        //   width: AppSpacing.logoSize + 40, // Made slightly bigger for visibility
        //   height: AppSpacing.logoSize + 40,
        //   child: Image.asset(
        //     'assets/images/app_icon.png', 
        //     fit: BoxFit.contain, // Ensures the whole PNG fits without clipping
        //   ),
        // ),
        const SizedBox(height: AppSpacing.md),
        Text(
          headline,
          style: AppTextStyles.headline.copyWith(
            shadows: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.7),
                offset: const Offset(0, 2),
                blurRadius: 6,
              ),
            ],
          ),
          textAlign: TextAlign.center,
        ),
      ],
    );
  }
}

class _PrimaryButton extends StatefulWidget {
  final String label;
  final VoidCallback onTap;
  const _PrimaryButton({required this.label, required this.onTap});

  @override
  State<_PrimaryButton> createState() => _PrimaryButtonState();
}

class _PrimaryButtonState extends State<_PrimaryButton> {
  bool _pressed = false;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown:   (_) => setState(() => _pressed = true),
      onTapUp:     (_) => setState(() => _pressed = false),
      onTapCancel: ()  => setState(() => _pressed = false),
      onTap: widget.onTap,
      child: AnimatedScale(
        scale: _pressed ? 0.97 : 1.0,
        duration: const Duration(milliseconds: 100),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 150),
          width: double.infinity,
          height: AppSpacing.buttonHeight,
          decoration: BoxDecoration(
            color: _pressed
                ? AppColors.primaryDarkGreen
                : AppColors.primaryGreen,
            borderRadius:
                BorderRadius.circular(AppSpacing.buttonRadius),
            boxShadow: _pressed ? [] : AppShadows.greenButton,
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text('🔧', style: TextStyle(fontSize: 18)),
              const SizedBox(width: 8),
              Text(widget.label, style: AppTextStyles.buttonPrimary),
            ],
          ),
        ),
      ),
    );
  }
}