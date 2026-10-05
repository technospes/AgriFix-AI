import 'package:flutter/foundation.dart';
import '../../services/api_service.dart';
import '../models/agent_models.dart';

class AgentSessionProvider extends ChangeNotifier {
  String? sessionId;
  AgentNextResponse? current;
  bool loading = false;
  String? error;
  int turnsSoFar = 0;

  Future<void> startSession({
    required String machineType,
    required String problemDescription,
    required String language,
    required List<Map<String, dynamic>> diagnosisSteps,
  }) async {
    loading = true;
    error = null;
    notifyListeners();
    try {
      final json = await ApiService.createAgentSession(
        machineType: machineType,
        problemDescription: problemDescription,
        language: language,
        diagnosisSteps: diagnosisSteps,
      );
      sessionId = json['session_id'] as String;
      
      // Auto-advance to get the first step
      await advance({'status': 'session_start'});
      
    } catch (e) {
      error = e.toString();
    } finally {
      loading = false;
      notifyListeners();
    }
  }

  Future<void> advance(Map<String, dynamic> resultPayload) async {
    if (sessionId == null) {
      error = 'No active session';
      notifyListeners();
      return;
    }
    loading = true;
    error = null;
    notifyListeners();
    try {
      final json = await ApiService.agentNext(
        sessionId: sessionId!,
        lastVerificationResult: resultPayload,
      );
      current = AgentNextResponse.fromJson(json);
      turnsSoFar++;
    } catch (e) {
      if (e is ServerException && e.statusCode == 404) {
        error = 'session_expired';
      } else {
        error = e.toString();
      }
    } finally {
      loading = false;
      notifyListeners();
    }
  }

  Future<void> endSession() async {
    if (sessionId != null) await ApiService.deleteAgentSession(sessionId!);
    sessionId = null;
    current = null;
    notifyListeners();
  }
}

class AgentResultBuilder {
  static Map<String, dynamic> fromVerifyStep(Map<String, dynamic> raw) => raw;

  static Map<String, dynamic> fromChoice(InteractionOption opt) => {
        'interaction_type': 'choice',
        'selected_option_id': opt.id,
        'selected_next_state': opt.nextState,
        'answer_text': opt.label,
      };

  static Map<String, dynamic> fromBoolean(bool value, String questionEcho) => {
        'interaction_type': 'boolean',
        'answer_bool': value,
        'question': questionEcho,
      };

  static Map<String, dynamic> fromText(String text) => {
        'interaction_type': 'text',
        'answer_text': text,
      };

  static Map<String, dynamic> fromNumber(num value) => {
        'interaction_type': 'number',
        'answer_number': value,
      };
}