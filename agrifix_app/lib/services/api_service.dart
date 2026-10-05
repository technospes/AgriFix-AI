import 'dart:async';
import 'dart:typed_data';
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

// ─────────────────────────────────────────────────────────────────────────────
// Exceptions
// ─────────────────────────────────────────────────────────────────────────────
class RequestTimeoutException implements Exception {
  final String message;
  const RequestTimeoutException([this.message = 'Request timed out']);
  @override
  String toString() => message;
}

class NetworkException implements Exception {
  final String message;
  const NetworkException(this.message);
  @override
  String toString() => message;
}

class ServerException implements Exception {
  final int statusCode;
  final String body;
  const ServerException(this.statusCode, this.body);
  @override
  String toString() => 'ServerException($statusCode): $body';
}
// ─────────────────────────────────────────────────────────────────────────────
// Result type passed to DiagnosisProvider
// ─────────────────────────────────────────────────────────────────────────────
class DiagnosisResult {
  final Map<String, dynamic> raw;
  DiagnosisResult(this.raw);
}

// ─────────────────────────────────────────────────────────────────────────────
// ApiService
// ─────────────────────────────────────────────────────────────────────────────
class ApiService {
  static const String _baseUrl = 'http://192.168.29.116:7860';
  
    //'https://technospes-agrifixar-backend-new.hf.space';
                                  
  static const String _appKey = '020b082f133f403abf8694e6144df1a79396b2706dd9de108bc54a05e891fc29';
  static const Duration _uploadTimeout = Duration(seconds: 90);
  static const Duration _streamTimeout = Duration(minutes: 3);
  static const int _maxRetries = 2;

  // ── Original batch endpoint ─────────────────────────────────────────────
  static Future<DiagnosisResult> uploadAndDiagnose({
    required String videoPath,
    required String audioPath,
    String machineType = 'tractor',
    String language = 'en',
    void Function(double progress)? onProgress,
    void Function(String status)? onStatus,
  }) async {
    if (!await File(videoPath).exists()) {
      throw NetworkException('Video file not found: $videoPath');
    }
    if (!await File(audioPath).exists()) {
      throw NetworkException('Audio file not found: $audioPath');
    }

    onStatus?.call('Preparing files…');
    onProgress?.call(0.05);

    final request = http.MultipartRequest('POST', Uri.parse('$_baseUrl/diagnose'));
    request.headers['X-App-Key'] = _appKey;
    request.fields['machine_type'] = machineType;
    request.fields['language'] = language;
    request.files.add(await http.MultipartFile.fromPath('video', videoPath));
    request.files.add(await http.MultipartFile.fromPath('audio', audioPath));

    onStatus?.call('Uploading to server…');
    onProgress?.call(0.15);

    return _sendWithRetry(
      request: request,
      timeout: _uploadTimeout,
      onRetry: (attempt) {
        onStatus?.call('Retrying… (attempt $attempt)');
        onProgress?.call(0.15);
      },
      onSuccess: (json) {
        onStatus?.call('Solution ready!');
        onProgress?.call(1.0);
        return DiagnosisResult(json);
      },
    );
  }

  // ── Streaming SSE endpoint with retry capability ─────────────────────────
  static Future<DiagnosisResult> uploadAndDiagnoseStreaming({
    required String videoPath,
    required String audioPath,
    required void Function(int stageIndex) onStageStart,
    required void Function(int stageIndex, Map<String, dynamic> data) onStageComplete,
    String machineType = 'tractor',
    String language = 'en',
    void Function(double progress)? onUploadProgress,
    void Function(String status)? onUploadStatus,
  }) async {
    if (!await File(videoPath).exists()) {
      throw NetworkException('Video file not found: $videoPath');
    }
    if (!await File(audioPath).exists()) {
      throw NetworkException('Audio file not found: $audioPath');
    }

    onUploadStatus?.call('Preparing files…');
    onUploadProgress?.call(0.05);

    final request = http.MultipartRequest(
        'POST', Uri.parse('$_baseUrl/diagnose/stream'));
    request.headers['X-App-Key'] = _appKey;
    request.fields['machine_type'] = machineType;
    request.fields['language'] = language;
    request.files.add(await http.MultipartFile.fromPath('video', videoPath));
    request.files.add(await http.MultipartFile.fromPath('audio', audioPath));

    onUploadStatus?.call('Uploading to server…');
    onUploadProgress?.call(0.15);

    return _sendStreamWithRetry(
      request: request,
      timeout: _streamTimeout,
      onRetry: (attempt) {
        onUploadStatus?.call('Retrying upload… (attempt $attempt)');
        onUploadProgress?.call(0.15);
      },
      onStageStart: onStageStart,
      onStageComplete: onStageComplete,
      onUploadProgress: onUploadProgress,
      onUploadStatus: onUploadStatus,
    );
  }

  // ── Fast streaming endpoint — 3 pre-extracted frames + audio (~350 KB) ───
  //
  // Flutter extracts 3 JPEG frames from the video locally BEFORE uploading.
  // Upload size: ~150 KB (frames) + ~200 KB (audio) = ~350 KB total,
  // vs old path: 20–35 MB video → reduces upload from ~45 s to ~0.5 s on
  // a typical Indian rural 4G connection (1–3 Mbps upload).
  //
  // The backend runs the IDENTICAL CLIP → fusion → Gemini pipeline on these
  // frames — accuracy, token usage, and all stage events are unchanged.
  // The video stub (1 byte) satisfies the backend's required 'video' field
  // but is ignored when frame0/frame1/frame2 are present.
  //
  // framePaths: [earlyFramePath, midFramePath, lateFramePath]
  //   extracted at 15% / 40% / 70% of video duration respectively.
  static Future<DiagnosisResult> uploadAndDiagnoseStreamingFast({
    required List<String> framePaths,   // exactly 3 JPEG paths [early, mid, late]
    required String audioPath,
    required void Function(int stageIndex) onStageStart,
    required void Function(int stageIndex, Map<String, dynamic> data) onStageComplete,
    String machineType = '',
    String language = 'en',
    void Function(double progress)? onUploadProgress,
    void Function(String status)? onUploadStatus,
  }) async {
    assert(framePaths.length == 3,
        'uploadAndDiagnoseStreamingFast requires exactly 3 frame paths');

    for (final p in framePaths) {
      if (!await File(p).exists()) {
        throw NetworkException('Frame file not found: $p');
      }
    }
    if (!await File(audioPath).exists()) {
      throw NetworkException('Audio file not found: $audioPath');
    }

    onUploadStatus?.call('Preparing frames…');
    onUploadProgress?.call(0.03);

    final request = http.MultipartRequest(
        'POST', Uri.parse('$_baseUrl/diagnose/stream'));
    request.headers['X-App-Key'] = _appKey;
    request.fields['machine_type'] = machineType;
    request.fields['language'] = language;

    // 1-byte stub for the required 'video' field — backend ignores video bytes
    // when frame0/frame1/frame2 are present (fast-path detection).
    request.files.add(http.MultipartFile.fromBytes(
      'video',
      [0],
      filename: 'stub.mp4',
    ));

    // Pre-extracted frames — field names frame0 / frame1 / frame2
    const frameFields  = ['frame0', 'frame1', 'frame2'];
    const frameLabels  = ['early',  'mid',    'late' ];
    for (int i = 0; i < 3; i++) {
      request.files.add(await http.MultipartFile.fromPath(
        frameFields[i],
        framePaths[i],
        filename: '${frameLabels[i]}.jpg',
      ));
    }

    request.files.add(await http.MultipartFile.fromPath('audio', audioPath));

    onUploadStatus?.call('Uploading frames…');
    onUploadProgress?.call(0.05);

    return _sendStreamWithRetry(
      request: request,
      timeout: _streamTimeout,
      onRetry: (attempt) {
        onUploadStatus?.call('Retrying… (attempt $attempt)');
        onUploadProgress?.call(0.05);
      },
      onStageStart: onStageStart,
      onStageComplete: onStageComplete,
      onUploadProgress: onUploadProgress,
      onUploadStatus: onUploadStatus,
    );
  }

  // ── Internal: send with timeout + exponential back-off retry ─────────────
  static Future<DiagnosisResult> _sendWithRetry({
    required http.MultipartRequest request,
    required Duration timeout,
    int maxRetries = _maxRetries,
    void Function(int attempt)? onRetry,
    required DiagnosisResult Function(Map<String, dynamic>) onSuccess,
  }) async {
    int attempt = 0;

    while (true) {
      try {
        final clone = _clone(request);
        final streamed = await clone.send().timeout(
          timeout,
          onTimeout: () { throw const RequestTimeoutException(); },
        );
        final body = await streamed.stream.bytesToString();

        if (streamed.statusCode == 200) {
          try {
            final json = jsonDecode(body) as Map<String, dynamic>;
            return onSuccess(json);
          } catch (_) {
            throw ServerException(200, 'Response was not valid JSON: $body');
          }
        }

        if (streamed.statusCode >= 500 && attempt < maxRetries) {
          attempt++;
          onRetry?.call(attempt);
          await Future.delayed(Duration(seconds: attempt * 2));
          continue;
        }

        throw ServerException(streamed.statusCode, body);

      } on RequestTimeoutException {
        rethrow;

      } on SocketException catch (e) {
        if (attempt < maxRetries) {
          attempt++;
          onRetry?.call(attempt);
          await Future.delayed(Duration(seconds: attempt));
          continue;
        }
        throw NetworkException('No internet connection: ${e.message}');

      } on http.ClientException catch (e) {
        throw NetworkException('Connection error: ${e.message}');

      } catch (e) {
        if (e is RequestTimeoutException ||
            e is NetworkException ||
            e is ServerException) rethrow;
        throw NetworkException('Unexpected error: $e');
      }
    }
  }

  // ── Internal: streaming version with retry capability ────────────────────
  static Future<DiagnosisResult> _sendStreamWithRetry({
    required http.MultipartRequest request,
    required Duration timeout,
    int maxRetries = _maxRetries,
    void Function(int attempt)? onRetry,
    required void Function(int stageIndex) onStageStart,
    required void Function(int stageIndex, Map<String, dynamic> data) onStageComplete,
    void Function(double progress)? onUploadProgress,
    void Function(String status)? onUploadStatus,
  }) async {
    int attempt = 0;

    while (true) {
      final client = http.Client();
      DiagnosisResult? finalResult;
      
      try {
        final clone = _clone(request);
        final streamed = await client.send(clone).timeout(
          timeout,
          onTimeout: () { throw const RequestTimeoutException(); },
        );

        if (streamed.statusCode != 200) {
          final body = await streamed.stream.bytesToString();
          
          if (streamed.statusCode >= 500 && attempt < maxRetries) {
            attempt++;
            onRetry?.call(attempt);
            client.close();
            await Future.delayed(Duration(seconds: attempt * 2));
            continue;
          }
          
          throw ServerException(streamed.statusCode, body);
        }

        onUploadStatus?.call('Processing diagnosis…');
        onUploadProgress?.call(0.3);

        final buffer = StringBuffer();
        await for (final chunk in streamed.stream.transform(utf8.decoder)) {
          buffer.write(chunk);
          final raw = buffer.toString();
          final events = raw.split('\n\n');
          buffer.clear();
          buffer.write(events.last);

          for (int i = 0; i < events.length - 1; i++) {
            final eventText = events[i].trim();
            if (eventText.isEmpty) continue;
            
            final jsonStr = eventText.startsWith('data: ') 
                ? eventText.substring(6).trim() 
                : eventText.trim();
            if (jsonStr.isEmpty) continue;

            final Map<String, dynamic> parsed;
            try { 
              parsed = json.decode(jsonStr) as Map<String, dynamic>; 
            } catch (_) { 
              continue; 
            }

            final eventType = parsed['event'] as String?;
            final stage = parsed['stage'] as int?;

            if (stage != null) {
              final progress = 0.3 + (stage * 0.2);
              onUploadProgress?.call(progress.clamp(0.3, 0.9));
            }

            switch (eventType) {
              case 'stage_start':
                if (stage != null) {
                  onUploadStatus?.call('Stage ${stage + 1}/4 starting…');
                  onStageStart(stage);
                }
                break;

              case 'stage_done':
                if (stage != null) {
                  onUploadStatus?.call('Stage ${stage + 1}/4 completed');
                  onStageComplete(stage, parsed);
                  if (stage == 3 && parsed.containsKey('result')) {
                    finalResult = DiagnosisResult(parsed['result'] as Map<String, dynamic>);
                    onUploadStatus?.call('Diagnosis complete!');
                    onUploadProgress?.call(1.0);
                  }
                }
                break;

              case 'error':
                final msg = parsed['message'] as String? ?? 'Unknown backend error';
                throw Exception(msg);
            }
          }
        }

        if (finalResult == null) {
          throw const NetworkException('Stream ended without a diagnosis result. Check backend logs.');
        }
        
        return finalResult;

      } on TimeoutException {
        rethrow;
        
      } on SocketException catch (e) {
        if (attempt < maxRetries) {
          attempt++;
          onRetry?.call(attempt);
          client.close();
          await Future.delayed(Duration(seconds: attempt));
          continue;
        }
        throw NetworkException('Connection lost during streaming: ${e.message}');
        
      } on http.ClientException catch (e) {
        if (attempt < maxRetries) {
          attempt++;
          onRetry?.call(attempt);
          client.close();
          await Future.delayed(Duration(seconds: attempt));
          continue;
        }
        throw NetworkException('Stream connection error: ${e.message}');
        
      } catch (e) {
        if (e is RequestTimeoutException ||
            e is NetworkException ||
            e is ServerException) rethrow;
        throw NetworkException('Unexpected streaming error: $e');
        
      } finally {
        client.close();
      }
    }
  }

  static http.MultipartRequest _clone(http.MultipartRequest src) {
    final dst = http.MultipartRequest(src.method, src.url);
    dst.headers.addAll(src.headers);
    dst.fields.addAll(src.fields);
    for (final f in src.files) dst.files.add(f);
    return dst;
  }

  // ── Verify a repair step photo ────────────────────────────────────────────
  static Future<Map<String, dynamic>> verifyStep({
    required File   imageFile,
    required String stepText,
    required String machineType,
    required String problemContext,
    required int    attemptCount,
    String requiredPart  = '',
    String areaHint      = '',
    String trackingScope = 'component',
    String previousSteps = '[]',
    Uint8List? imageCropBytes,
    void Function(double progress)? onProgress,
    void Function(String status)? onStatus,
    bool isBlindSearch = false, // ADDED: isBlindSearch parameter
  }) async {
    final request = http.MultipartRequest(
        'POST', Uri.parse('$_baseUrl/verify_step'))
      ..headers['X-App-Key'] = _appKey;
    // Use cropped PNG bytes when available (AR flow) — otherwise full JPEG file
    if (imageCropBytes != null) {
      request.files.add(http.MultipartFile.fromBytes(
        'image', imageCropBytes, filename: 'crop.png'));
    } else {
      request.files.add(
          await http.MultipartFile.fromPath('image', imageFile.path));
    }
    request
      ..fields.addAll({
        'step_text':       stepText,
        'machine_type':    machineType,
        'problem_context': problemContext,
        'attempt_count':   attemptCount.toString(),
        'previous_steps':  previousSteps,
        'is_blind_search': isBlindSearch.toString(), // ADDED: sending blind search flag to API
        // Send real step metadata — backend was falling back to
        // 'machine_part' / 'engine_compartment' defaults without these.
        if (requiredPart.isNotEmpty) 'required_part': requiredPart,
        if (areaHint.isNotEmpty)     'area_hint':     areaHint,
        'tracking_scope': trackingScope,
      });

    onStatus?.call('Verifying step…');
    onProgress?.call(0.5);

    final result = await _sendWithRetry(
      request: request,
      timeout: const Duration(seconds: 30),
      maxRetries: 1,
      onRetry: (attempt) {
        onStatus?.call('Retrying verification… (attempt $attempt)');
        onProgress?.call(0.5);
      },
      onSuccess: (json) => DiagnosisResult(json), 
    );

    onStatus?.call('Verification complete');
    onProgress?.call(1.0);
    return result.raw; 
  }

  /// Inspect a verified part for damage.
  /// Uses the NEW /inspect_part endpoint (separate from /verify_step).
  static Future<Map<String, dynamic>> inspectPart({
    required Uint8List imageBytes,
    required String machineType,
    required String requiredPart,
    required String areaHint,
    String language = 'en',
  }) async {
    final request = http.MultipartRequest(
        'POST', Uri.parse('$_baseUrl/inspect_part'))
      ..headers['X-App-Key'] = _appKey
      ..files.add(http.MultipartFile.fromBytes(
        'image', imageBytes, filename: 'inspect.jpg'))
      ..fields.addAll({
        'machine_type': machineType,
        'required_part': requiredPart,
        'area_hint': areaHint,
        'language': language,
      });

    final streamed = await request.send()
        .timeout(const Duration(seconds: 15));
    final response = await http.Response.fromStream(streamed);
    
    if (response.statusCode != 200) {
      throw ServerException(response.statusCode, response.body);
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  // ── AR Part Location ───────────────────────────────────────────────────────
  // Called by the AR guidance loop every ~3 seconds.
  // Returns { found, bbox, confidence, camera_guidance, part_description }.
  // FIX 6: 3 s timeout — AR loop fires every 1 s, so a slow response
  // is already stale before it arrives. Drop it and use the next frame.
  static Future<Map<String, dynamic>> locatePart({
    required File  imageFile,
    required String requiredPart,
    required String areaHint,
    required String machineType,
    required int    attemptCount,
    String language   = 'en',
    int    frameId    = 0,
    String searchRoi  = '',   // 'cx,cy,margin' from last known bbox, '' if none
  }) async {
    final request = http.MultipartRequest(
        'POST', Uri.parse('$_baseUrl/locate_part'))
      ..headers['X-App-Key'] = _appKey
      ..files.add(await http.MultipartFile.fromPath('image', imageFile.path))
      ..fields.addAll({
        'required_part': requiredPart,
        'area_hint':     areaHint,
        'machine_type':  machineType,
        'attempt_count': attemptCount.toString(),
        'language':      language,
        'frame_id':      frameId.toString(),
        if (searchRoi.isNotEmpty) 'search_roi': searchRoi,
      });

    try {
      final streamedResponse = await request.send()
          .timeout(const Duration(seconds: 3));  // FIX 6: short — stale frames replaced by next tick
      final response = await http.Response.fromStream(streamedResponse);
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
    } catch (_) {
      // Timeout or network error — return not-found so loop continues
    }
    return {
      'found': false, 'bbox': null, 'confidence': 0.0,
      'camera_guidance': 'Network error — retrying…',
      'part_description': null,
    };
  }
  // ── Agent session lifecycle ─────────────────────────────────────────────
static Future<Map<String, dynamic>> createAgentSession({
  required String machineType,
  required String problemDescription,
  required String language,
  required List<Map<String, dynamic>> diagnosisSteps,
}) async {
  final uri = Uri.parse('$_baseUrl/agent/session');
  final resp = await http.post(
    uri,
    headers: {'Content-Type': 'application/json', 'X-App-Key': _appKey},
    body: jsonEncode({
      'machine_type': machineType,
      'problem_description': problemDescription,
      'language': language,
      'diagnosis_steps': diagnosisSteps,
    }),
  ).timeout(const Duration(seconds: 20));

  if (resp.statusCode != 200) {
    throw ServerException(resp.statusCode, resp.body);
  }
  return jsonDecode(resp.body) as Map<String, dynamic>;
}

static Future<Map<String, dynamic>> agentNext({
  required String sessionId,
  required Map<String, dynamic> lastVerificationResult,
}) async {
  final uri = Uri.parse('$_baseUrl/agent/next');
  final resp = await http.post(
    uri,
    headers: {'Content-Type': 'application/json', 'X-App-Key': _appKey},
    body: jsonEncode({
      'session_id': sessionId,
      'last_verification_result': lastVerificationResult,
    }),
  ).timeout(const Duration(seconds: 30));

  if (resp.statusCode == 404) {
    throw const ServerException(404, 'Session expired or not found');
  }
  if (resp.statusCode != 200) {
    throw ServerException(resp.statusCode, resp.body);
  }
  return jsonDecode(resp.body) as Map<String, dynamic>;
}

static Future<void> deleteAgentSession(String sessionId) async {
  try {
    await http.delete(
      Uri.parse('$_baseUrl/agent/session/$sessionId'),
      headers: {'X-App-Key': _appKey},
    ).timeout(const Duration(seconds: 10));
  } catch (_) {
    // best-effort cleanup — server TTL will reclaim it anyway
  }
  }
}