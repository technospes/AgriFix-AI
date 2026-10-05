import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:intl/intl.dart' as intl;

import 'app_localizations_en.dart';
import 'app_localizations_hi.dart';
import 'app_localizations_pa.dart';

// ignore_for_file: type=lint

/// Callers can lookup localized strings with an instance of AppLocalizations
/// returned by `AppLocalizations.of(context)`.
///
/// Applications need to include `AppLocalizations.delegate()` in their app's
/// `localizationDelegates` list, and the locales they support in the app's
/// `supportedLocales` list. For example:
///
/// ```dart
/// import 'l10n/app_localizations.dart';
///
/// return MaterialApp(
///   localizationsDelegates: AppLocalizations.localizationsDelegates,
///   supportedLocales: AppLocalizations.supportedLocales,
///   home: MyApplicationHome(),
/// );
/// ```
///
/// ## Update pubspec.yaml
///
/// Please make sure to update your pubspec.yaml to include the following
/// packages:
///
/// ```yaml
/// dependencies:
///   # Internationalization support.
///   flutter_localizations:
///     sdk: flutter
///   intl: any # Use the pinned version from flutter_localizations
///
///   # Rest of dependencies
/// ```
///
/// ## iOS Applications
///
/// iOS applications define key application metadata, including supported
/// locales, in an Info.plist file that is built into the application bundle.
/// To configure the locales supported by your app, you’ll need to edit this
/// file.
///
/// First, open your project’s ios/Runner.xcworkspace Xcode workspace file.
/// Then, in the Project Navigator, open the Info.plist file under the Runner
/// project’s Runner folder.
///
/// Next, select the Information Property List item, select Add Item from the
/// Editor menu, then select Localizations from the pop-up menu.
///
/// Select and expand the newly-created Localizations item then, for each
/// locale your application supports, add a new item and select the locale
/// you wish to add from the pop-up menu in the Value field. This list should
/// be consistent with the languages listed in the AppLocalizations.supportedLocales
/// property.
abstract class AppLocalizations {
  AppLocalizations(String locale)
      : localeName = intl.Intl.canonicalizedLocale(locale.toString());

  final String localeName;

  static AppLocalizations of(BuildContext context) {
    return Localizations.of<AppLocalizations>(context, AppLocalizations)!;
  }

  static const LocalizationsDelegate<AppLocalizations> delegate =
      _AppLocalizationsDelegate();

  /// A list of this localizations delegate along with the default localizations
  /// delegates.
  ///
  /// Returns a list of localizations delegates containing this delegate along with
  /// GlobalMaterialLocalizations.delegate, GlobalCupertinoLocalizations.delegate,
  /// and GlobalWidgetsLocalizations.delegate.
  ///
  /// Additional delegates can be added by appending to this list in
  /// MaterialApp. This list does not have to be used at all if a custom list
  /// of delegates is preferred or required.
  static const List<LocalizationsDelegate<dynamic>> localizationsDelegates =
      <LocalizationsDelegate<dynamic>>[
    delegate,
    GlobalMaterialLocalizations.delegate,
    GlobalCupertinoLocalizations.delegate,
    GlobalWidgetsLocalizations.delegate,
  ];

  /// A list of this localizations delegate's supported locales.
  static const List<Locale> supportedLocales = <Locale>[
    Locale('en'),
    Locale('hi'),
    Locale('pa')
  ];

  /// No description provided for @homeHeadline.
  ///
  /// In en, this message translates to:
  /// **'Professional grade machine\nrepair, powered by AI and AR.'**
  String get homeHeadline;

  /// No description provided for @homeCardTitle.
  ///
  /// In en, this message translates to:
  /// **'Machine Issues?'**
  String get homeCardTitle;

  /// No description provided for @homeCardSubtitle.
  ///
  /// In en, this message translates to:
  /// **'Get instant expert guidance for your equipment.'**
  String get homeCardSubtitle;

  /// No description provided for @homeCtaButton.
  ///
  /// In en, this message translates to:
  /// **'Let\'s Fix Your Machine'**
  String get homeCtaButton;

  /// No description provided for @languageSelectTitle.
  ///
  /// In en, this message translates to:
  /// **'Select Language'**
  String get languageSelectTitle;

  /// No description provided for @langEnglish.
  ///
  /// In en, this message translates to:
  /// **'English'**
  String get langEnglish;

  /// No description provided for @langHindi.
  ///
  /// In en, this message translates to:
  /// **'हिन्दी'**
  String get langHindi;

  /// No description provided for @langPunjabi.
  ///
  /// In en, this message translates to:
  /// **'ਪੰਜਾਬੀ'**
  String get langPunjabi;

  /// No description provided for @uploadScreenTitle.
  ///
  /// In en, this message translates to:
  /// **'Diagnose Issue'**
  String get uploadScreenTitle;

  /// No description provided for @uploadHelp.
  ///
  /// In en, this message translates to:
  /// **'Help'**
  String get uploadHelp;

  /// No description provided for @uploadStepChip.
  ///
  /// In en, this message translates to:
  /// **'Step 1: Data Collection'**
  String get uploadStepChip;

  /// No description provided for @uploadHeading.
  ///
  /// In en, this message translates to:
  /// **'How does it sound and look?'**
  String get uploadHeading;

  /// No description provided for @uploadSubtext.
  ///
  /// In en, this message translates to:
  /// **'Please provide both video and audio for the most accurate repair diagnosis.'**
  String get uploadSubtext;

  /// No description provided for @uploadRecordVideo.
  ///
  /// In en, this message translates to:
  /// **'Record Video'**
  String get uploadRecordVideo;

  /// No description provided for @uploadRecordVideoSub.
  ///
  /// In en, this message translates to:
  /// **'Show us the mechanical issue'**
  String get uploadRecordVideoSub;

  /// No description provided for @uploadRecordAudio.
  ///
  /// In en, this message translates to:
  /// **'Record Audio'**
  String get uploadRecordAudio;

  /// No description provided for @uploadRecordAudioSub.
  ///
  /// In en, this message translates to:
  /// **'Describe the engine sounds'**
  String get uploadRecordAudioSub;

  /// No description provided for @uploadProTipTitle.
  ///
  /// In en, this message translates to:
  /// **'Pro Tip'**
  String get uploadProTipTitle;

  /// No description provided for @uploadProTipBody.
  ///
  /// In en, this message translates to:
  /// **'Hold the phone steady and ensure there is plenty of sunlight for the best analysis.'**
  String get uploadProTipBody;

  /// No description provided for @uploadFromGallery.
  ///
  /// In en, this message translates to:
  /// **'Upload from Gallery'**
  String get uploadFromGallery;

  /// No description provided for @uploadRecordLiveCamera.
  ///
  /// In en, this message translates to:
  /// **'Record Live Through Camera'**
  String get uploadRecordLiveCamera;

  /// No description provided for @uploadRecordLiveMic.
  ///
  /// In en, this message translates to:
  /// **'Record Live Through Mic'**
  String get uploadRecordLiveMic;

  /// No description provided for @uploadFindSolution.
  ///
  /// In en, this message translates to:
  /// **'Find Solution'**
  String get uploadFindSolution;

  /// No description provided for @uploadAnalyzing.
  ///
  /// In en, this message translates to:
  /// **'Analyzing your machine...'**
  String get uploadAnalyzing;

  /// No description provided for @uploadStep1.
  ///
  /// In en, this message translates to:
  /// **'Uploading your video & audio'**
  String get uploadStep1;

  /// No description provided for @uploadStep2.
  ///
  /// In en, this message translates to:
  /// **'Understanding the problem'**
  String get uploadStep2;

  /// No description provided for @uploadStep3.
  ///
  /// In en, this message translates to:
  /// **'Analyzing machine type'**
  String get uploadStep3;

  /// No description provided for @uploadStep4.
  ///
  /// In en, this message translates to:
  /// **'Generating repair steps'**
  String get uploadStep4;

  /// No description provided for @uploadStep5.
  ///
  /// In en, this message translates to:
  /// **'Solution ready'**
  String get uploadStep5;

  /// No description provided for @uploadErrorTitle.
  ///
  /// In en, this message translates to:
  /// **'Something went wrong'**
  String get uploadErrorTitle;

  /// No description provided for @uploadErrorSubtitle.
  ///
  /// In en, this message translates to:
  /// **'Check your internet and try again'**
  String get uploadErrorSubtitle;

  /// No description provided for @uploadRetry.
  ///
  /// In en, this message translates to:
  /// **'Retry'**
  String get uploadRetry;

  /// No description provided for @aiRepairGuide.
  ///
  /// In en, this message translates to:
  /// **'AI Repair Guide'**
  String get aiRepairGuide;

  /// No description provided for @solutionFound.
  ///
  /// In en, this message translates to:
  /// **'SOLUTION FOUND'**
  String get solutionFound;

  /// No description provided for @stepNumber.
  ///
  /// In en, this message translates to:
  /// **'Step {number}'**
  String stepNumber(int number);

  /// No description provided for @currentTask.
  ///
  /// In en, this message translates to:
  /// **'Current Task'**
  String get currentTask;

  /// No description provided for @startArGuide.
  ///
  /// In en, this message translates to:
  /// **'Start AR Guide'**
  String get startArGuide;

  /// No description provided for @prev.
  ///
  /// In en, this message translates to:
  /// **'Prev'**
  String get prev;

  /// No description provided for @next.
  ///
  /// In en, this message translates to:
  /// **'Next'**
  String get next;
}

class _AppLocalizationsDelegate
    extends LocalizationsDelegate<AppLocalizations> {
  const _AppLocalizationsDelegate();

  @override
  Future<AppLocalizations> load(Locale locale) {
    return SynchronousFuture<AppLocalizations>(lookupAppLocalizations(locale));
  }

  @override
  bool isSupported(Locale locale) =>
      <String>['en', 'hi', 'pa'].contains(locale.languageCode);

  @override
  bool shouldReload(_AppLocalizationsDelegate old) => false;
}

AppLocalizations lookupAppLocalizations(Locale locale) {
  // Lookup logic when only language code is specified.
  switch (locale.languageCode) {
    case 'en':
      return AppLocalizationsEn();
    case 'hi':
      return AppLocalizationsHi();
    case 'pa':
      return AppLocalizationsPa();
  }

  throw FlutterError(
      'AppLocalizations.delegate failed to load unsupported locale "$locale". This is likely '
      'an issue with the localizations generation tool. Please file an issue '
      'on GitHub with a reproducible sample app and the gen-l10n configuration '
      'that was used.');
}
