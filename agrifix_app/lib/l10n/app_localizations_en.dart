// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for English (`en`).
class AppLocalizationsEn extends AppLocalizations {
  AppLocalizationsEn([String locale = 'en']) : super(locale);

  @override
  String get homeHeadline =>
      'Professional grade machine\nrepair, powered by AI and AR.';

  @override
  String get homeCardTitle => 'Machine Issues?';

  @override
  String get homeCardSubtitle =>
      'Get instant expert guidance for your equipment.';

  @override
  String get homeCtaButton => 'Let\'s Fix Your Machine';

  @override
  String get languageSelectTitle => 'Select Language';

  @override
  String get langEnglish => 'English';

  @override
  String get langHindi => 'हिन्दी';

  @override
  String get langPunjabi => 'ਪੰਜਾਬੀ';

  @override
  String get uploadScreenTitle => 'Diagnose Issue';

  @override
  String get uploadHelp => 'Help';

  @override
  String get uploadStepChip => 'Step 1: Data Collection';

  @override
  String get uploadHeading => 'How does it sound and look?';

  @override
  String get uploadSubtext =>
      'Please provide both video and audio for the most accurate repair diagnosis.';

  @override
  String get uploadRecordVideo => 'Record Video';

  @override
  String get uploadRecordVideoSub => 'Show us the mechanical issue';

  @override
  String get uploadRecordAudio => 'Record Audio';

  @override
  String get uploadRecordAudioSub => 'Describe the engine sounds';

  @override
  String get uploadProTipTitle => 'Pro Tip';

  @override
  String get uploadProTipBody =>
      'Hold the phone steady and ensure there is plenty of sunlight for the best analysis.';

  @override
  String get uploadFromGallery => 'Upload from Gallery';

  @override
  String get uploadRecordLiveCamera => 'Record Live Through Camera';

  @override
  String get uploadRecordLiveMic => 'Record Live Through Mic';

  @override
  String get uploadFindSolution => 'Find Solution';

  @override
  String get uploadAnalyzing => 'Analyzing your machine...';

  @override
  String get uploadStep1 => 'Uploading your video & audio';

  @override
  String get uploadStep2 => 'Understanding the problem';

  @override
  String get uploadStep3 => 'Analyzing machine type';

  @override
  String get uploadStep4 => 'Generating repair steps';

  @override
  String get uploadStep5 => 'Solution ready';

  @override
  String get uploadErrorTitle => 'Something went wrong';

  @override
  String get uploadErrorSubtitle => 'Check your internet and try again';

  @override
  String get uploadRetry => 'Retry';

  @override
  String get aiRepairGuide => 'AI Repair Guide';

  @override
  String get solutionFound => 'SOLUTION FOUND';

  @override
  String stepNumber(int number) {
    return 'Step $number';
  }

  @override
  String get currentTask => 'Current Task';

  @override
  String get startArGuide => 'Start AR Guide';

  @override
  String get prev => 'Prev';

  @override
  String get next => 'Next';
}
