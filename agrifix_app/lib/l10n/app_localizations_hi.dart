// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for Hindi (`hi`).
class AppLocalizationsHi extends AppLocalizations {
  AppLocalizationsHi([String locale = 'hi']) : super(locale);

  @override
  String get homeHeadline => 'पेशेवर मशीन मरम्मत,\nAI और AR द्वारा संचालित।';

  @override
  String get homeCardTitle => 'मशीन में समस्या?';

  @override
  String get homeCardSubtitle =>
      'अपने उपकरण के लिए तत्काल विशेषज्ञ मार्गदर्शन प्राप्त करें।';

  @override
  String get homeCtaButton => 'आइए ठीक करें';

  @override
  String get languageSelectTitle => 'भाषा चुनें';

  @override
  String get langEnglish => 'English';

  @override
  String get langHindi => 'हिन्दी';

  @override
  String get langPunjabi => 'ਪੰਜਾਬੀ';

  @override
  String get uploadScreenTitle => 'समस्या निदान';

  @override
  String get uploadHelp => 'सहायता';

  @override
  String get uploadStepChip => 'चरण 1: डेटा संग्रह';

  @override
  String get uploadHeading => 'यह कैसा दिखता और सुनाई देता है?';

  @override
  String get uploadSubtext =>
      'सबसे सटीक मरम्मत निदान के लिए कृपया वीडियो और ऑडियो दोनों प्रदान करें।';

  @override
  String get uploadRecordVideo => 'वीडियो रिकॉर्ड करें';

  @override
  String get uploadRecordVideoSub => 'हमें यांत्रिक समस्या दिखाएं';

  @override
  String get uploadRecordAudio => 'ऑडियो रिकॉर्ड करें';

  @override
  String get uploadRecordAudioSub => 'इंजन की आवाज़ें बताएं';

  @override
  String get uploadProTipTitle => 'प्रो टिप';

  @override
  String get uploadProTipBody =>
      'फोन स्थिर रखें और सर्वोत्तम विश्लेषण के लिए पर्याप्त धूप सुनिश्चित करें।';

  @override
  String get uploadFromGallery => 'गैलरी से अपलोड करें';

  @override
  String get uploadRecordLiveCamera => 'कैमरे से लाइव रिकॉर्ड करें';

  @override
  String get uploadRecordLiveMic => 'माइक से लाइव रिकॉर्ड करें';

  @override
  String get uploadFindSolution => 'समाधान खोजें';

  @override
  String get uploadAnalyzing => 'आपकी मशीन का विश्लेषण हो रहा है...';

  @override
  String get uploadStep1 => 'वीडियो और ऑडियो अपलोड हो रही है';

  @override
  String get uploadStep2 => 'समस्या को समझ रहे हैं';

  @override
  String get uploadStep3 => 'मशीन के प्रकार का विश्लेषण';

  @override
  String get uploadStep4 => 'मरम्मत के चरण तैयार हो रहे हैं';

  @override
  String get uploadStep5 => 'समाधान तैयार है';

  @override
  String get uploadErrorTitle => 'कुछ गलत हुआ';

  @override
  String get uploadErrorSubtitle => 'इंटरनेट जांचें और पुनः प्रयास करें';

  @override
  String get uploadRetry => 'पुनः प्रयास करें';

  @override
  String get aiRepairGuide => 'एआई मरम्मत गाइड';

  @override
  String get solutionFound => 'समाधान मिल गया';

  @override
  String stepNumber(int number) {
    return 'चरण $number';
  }

  @override
  String get currentTask => 'वर्तमान कार्य';

  @override
  String get startArGuide => 'एआर गाइड शुरू करें';

  @override
  String get prev => 'पिछला';

  @override
  String get next => 'अगला';
}
