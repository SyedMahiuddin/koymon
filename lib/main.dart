import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:google_fonts/google_fonts.dart';
// Optional: If you want to use TFLite for enhanced ML
// import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Cattle Measurements',
      theme: ThemeData(
        primarySwatch: Colors.indigo,
        brightness: Brightness.light,
        fontFamily: GoogleFonts.poppins().fontFamily,
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        primarySwatch: Colors.indigo,
        brightness: Brightness.dark,
        fontFamily: GoogleFonts.poppins().fontFamily,
        useMaterial3: true,
      ),
      themeMode: ThemeMode.system,
      home: const CattleMeasurementPage(),
    );
  }
}

class CattleMeasurementPage extends StatefulWidget {
  const CattleMeasurementPage({Key? key}) : super(key: key);

  @override
  State<CattleMeasurementPage> createState() => _CattleMeasurementPageState();
}

class MeasurementPoint {
  Offset position;

  MeasurementPoint(this.position);
}

class _CattleMeasurementPageState extends State<CattleMeasurementPage> {
  File? _imageFile;
  final ImagePicker _picker = ImagePicker();
  final ImageLabeler _imageLabeler = ImageLabeler(
    options: ImageLabelerOptions(confidenceThreshold: 0.7),
  );

  // Optional: Enhanced ML detector
  final PoseDetector _poseDetector = PoseDetector(options: PoseDetectorOptions(
    model: PoseDetectionModel.accurate,
    mode: PoseDetectionMode.single,
  ));

  bool _isProcessing = false;
  String _resultText = 'No image selected';
  bool _cattleDetected = false;
  bool _useEnhancedML = true; // Set to false if not using enhanced ML

  // Points for manual measurement
  MeasurementPoint? _bellyPoint;
  MeasurementPoint? _spinePoint;
  MeasurementPoint? _neckPoint;
  MeasurementPoint? _rearPoint;
  MeasurementPoint? _heartGirthLeftPoint;
  MeasurementPoint? _heartGirthRightPoint;

  // Active point being dragged
  MeasurementPoint? _activePoint;

  // Calibration value (cm per pixel)
  double _cmPerPixel = 0.2; // Default
  double _calibrationLength = 50.0; // Default reference length in cm

  // Image dimensions for scaling
  Size _imageSize = Size.zero;
  Size _displaySize = Size.zero;

  // Measurement mode (calibration or measurement)
  bool _isCalibrationMode = false;
  MeasurementPoint? _calibrationStart;
  MeasurementPoint? _calibrationEnd;

  // Expanded results view
  bool _showExpandedResults = false;

  // Breed and condition selection for more accurate measurements
  String _selectedBreed = 'Angus';
  String _selectedCondition = 'Average';

  // Breed and condition options
  final List<String> _breeds = [
    'Angus', 'Hereford', 'Charolais', 'Limousin', 'Simmental',
    'Brahman', 'Holstein', 'Jersey', 'Other'
  ];

  final List<String> _conditions = [
    'Thin', 'Average', 'Good', 'Excellent'
  ];

  @override
  void initState() {
    super.initState();

    // If you're using TFLite models:
    // _loadCattleSegmentationModel();
  }

  @override
  void dispose() {
    _imageLabeler.close();
    _poseDetector.close();
    super.dispose();
  }

// Image capture and selection methods
  void _pickImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _imageFile = File(image.path);
        _isProcessing = true;
        _resultText = 'Processing image...';
        _cattleDetected = false;
        _showExpandedResults = false;

        // Reset measurement points
        _bellyPoint = null;
        _spinePoint = null;
        _neckPoint = null;
        _rearPoint = null;
        _heartGirthLeftPoint = null;
        _heartGirthRightPoint = null;
        _activePoint = null;

        // Reset calibration points
        _calibrationStart = null;
        _calibrationEnd = null;
      });
      _processImageWithML();
    }
  }

  void _captureImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.camera);
    if (image != null) {
      setState(() {
        _imageFile = File(image.path);
        _isProcessing = true;
        _resultText = 'Processing image...';
        _cattleDetected = false;
        _showExpandedResults = false;

        // Reset measurement points
        _bellyPoint = null;
        _spinePoint = null;
        _neckPoint = null;
        _rearPoint = null;
        _heartGirthLeftPoint = null;
        _heartGirthRightPoint = null;
        _activePoint = null;

        // Reset calibration points
        _calibrationStart = null;
        _calibrationEnd = null;
      });
      _processImageWithML();
    }
  }

  // Enhanced image processing with ML
  Future<void> _processImageWithML() async {
    if (_imageFile == null) return;

    try {
      // Get image dimensions
      final image = await decodeImageFromList(_imageFile!.readAsBytesSync());
      setState(() {
        _imageSize = Size(image.width.toDouble(), image.height.toDouble());
        _isProcessing = true;
      });

      // Use ML Kit for initial detection
      final inputImage = InputImage.fromFile(_imageFile!);
      final labels = await _imageLabeler.processImage(inputImage);

      bool cattleFound = false;
      for (final label in labels) {
        if (label.label.toLowerCase() == 'cow' ||
            label.label.toLowerCase() == 'cattle' ||
            label.label.toLowerCase() == 'ox' ||
            label.label.toLowerCase() == 'animal') {
          cattleFound = true;
          break;
        }
      }

      // If cattle detected and enhanced ML is enabled, use pose detection
      if (cattleFound && _useEnhancedML) {
        try {
          final poses = await _poseDetector.processImage(inputImage);

          if (poses.isNotEmpty) {
            // Attempt to identify key points on the cattle body
            final pose = poses.first;

            // Map human pose points to cattle anatomy (rough approximation)
            final landmarks = pose.landmarks;

            // Look for points that could represent the body outline
            if (landmarks.containsKey(PoseLandmarkType.leftHip) &&
                landmarks.containsKey(PoseLandmarkType.rightHip) &&
                landmarks.containsKey(PoseLandmarkType.leftShoulder) &&
                landmarks.containsKey(PoseLandmarkType.rightShoulder)) {

              // Map pose points to cattle measurements
              final leftHip = landmarks[PoseLandmarkType.leftHip]!;
              final rightHip = landmarks[PoseLandmarkType.rightHip]!;
              final leftShoulder = landmarks[PoseLandmarkType.leftShoulder]!;
              final rightShoulder = landmarks[PoseLandmarkType.rightShoulder]!;

              // Place points more intelligently
              setState(() {
                // Neck to rear (body length) using shoulders and hips
                _neckPoint = MeasurementPoint(Offset(
                    (leftShoulder.x + rightShoulder.x) / 2,
                    (leftShoulder.y + rightShoulder.y) / 2
                ));

                _rearPoint = MeasurementPoint(Offset(
                    (leftHip.x + rightHip.x) / 2,
                    (leftHip.y + rightHip.y) / 2
                ));

                // Try to estimate heart girth position
                final midX = (_neckPoint!.position.dx + _rearPoint!.position.dx) / 2;
                final midY = (_neckPoint!.position.dy + _rearPoint!.position.dy) / 2;

                // Heart girth is typically right behind the front legs
                final heartGirthX = _neckPoint!.position.dx +
                    (_rearPoint!.position.dx - _neckPoint!.position.dx) * 0.3;

                // Place points for heart girth measurement
                _heartGirthLeftPoint = MeasurementPoint(Offset(
                    heartGirthX - _imageSize.width * 0.15, midY
                ));

                _heartGirthRightPoint = MeasurementPoint(Offset(
                    heartGirthX + _imageSize.width * 0.15, midY
                ));

                // Estimate belly and spine positions
                _bellyPoint = MeasurementPoint(Offset(
                    heartGirthX, midY + _imageSize.height * 0.15
                ));

                _spinePoint = MeasurementPoint(Offset(
                    heartGirthX, midY - _imageSize.height * 0.15
                ));

                _cattleDetected = true;
                _resultText = 'Cattle detected with enhanced ML!\n\n'
                    'Points have been placed based on detected body positions.\n'
                    'Please adjust them as needed for accurate measurements.';
              });

              _isProcessing = false;
              return;
            }
          }
        } catch (e) {
          print('Error in pose detection: $e');
          // Fall back to basic detection if pose detection fails
        }
      }

      // If enhanced ML failed or is disabled, fall back to basic initialization
      setState(() {
        _cattleDetected = cattleFound;

        // Set more anatomically correct default points for a side view of cattle
        final centerX = _imageSize.width / 2;
        final centerY = _imageSize.height / 2;

        // Vertical measurements (belly to spine)
        // Belly point should be at the bottom of the belly
        _bellyPoint = MeasurementPoint(Offset(centerX, centerY + _imageSize.height * 0.15));
        // Spine point should be directly above belly point at the top of the body
        _spinePoint = MeasurementPoint(Offset(centerX, centerY - _imageSize.height * 0.15));

        // Heart girth points should be at the sides of the body, centered vertically
        _heartGirthLeftPoint = MeasurementPoint(Offset(centerX - _imageSize.width * 0.2, centerY));
        _heartGirthRightPoint = MeasurementPoint(Offset(centerX + _imageSize.width * 0.2, centerY));

        // Body length points
        // Neck point should be at the front of the body near the shoulder
        _neckPoint = MeasurementPoint(Offset(centerX - _imageSize.width * 0.3, centerY - _imageSize.height * 0.05));
        // Rear point should be at the pin bones/rear of the animal
        _rearPoint = MeasurementPoint(Offset(centerX + _imageSize.width * 0.3, centerY - _imageSize.height * 0.05));

        if (cattleFound) {
          _resultText = 'Cattle detected!\n\n'
              'Drag the measurement points to position them correctly:\n\n'
              'Blue: Belly to spine height\n'
              'Purple: Heart girth (chest circumference)\n'
              'Green: Body length (neck to rear)\n\n'
              'Use calibration mode to set accurate measurements.';
        } else {
          _resultText = 'No cattle detected in the image. You can still manually place measurement points.';
        }

        _isProcessing = false;
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _resultText = 'Error processing image: $e';
      });
    }
  }

  // Fallback to basic image processing if enhanced ML is not available
  void _processImage() async {
    if (_imageFile == null) return;

    try {
      // Get image dimensions
      final image = await decodeImageFromList(_imageFile!.readAsBytesSync());
      setState(() {
        _imageSize = Size(image.width.toDouble(), image.height.toDouble());
      });

      // Use ML Kit for initial detection
      final inputImage = InputImage.fromFile(_imageFile!);
      final labels = await _imageLabeler.processImage(inputImage);

      bool cattleFound = false;
      for (final label in labels) {
        if (label.label.toLowerCase() == 'cow' ||
            label.label.toLowerCase() == 'cattle' ||
            label.label.toLowerCase() == 'ox' ||
            label.label.toLowerCase() == 'animal') {
          cattleFound = true;
          break;
        }
      }

      // Even if we don't detect cattle, we'll still set up the points
      // Initialize measurement points at default positions
      final centerX = _imageSize.width / 2;
      final centerY = _imageSize.height / 2;

      setState(() {
        _cattleDetected = cattleFound;

        // Set more reasonable default points
        // Belly point should be at bottom of belly
        _bellyPoint = MeasurementPoint(Offset(centerX, centerY + _imageSize.height * 0.15));
        // Spine point should be directly above belly point
        _spinePoint = MeasurementPoint(Offset(centerX, centerY - _imageSize.height * 0.15));
        // Neck and rear for body length
        _neckPoint = MeasurementPoint(Offset(centerX - _imageSize.width * 0.3, centerY - _imageSize.height * 0.05));
        _rearPoint = MeasurementPoint(Offset(centerX + _imageSize.width * 0.3, centerY - _imageSize.height * 0.05));
        // Heart girth points (behind front legs)
        _heartGirthLeftPoint = MeasurementPoint(Offset(centerX - _imageSize.width * 0.2, centerY));
        _heartGirthRightPoint = MeasurementPoint(Offset(centerX + _imageSize.width * 0.2, centerY));

        if (cattleFound) {
          _resultText = 'Cattle detected!\n\n'
              'Drag the measurement points to position them correctly:\n\n'
              'Blue: Belly to spine height\n'
              'Purple: Heart girth (chest circumference)\n'
              'Green: Body length (neck to rear)\n\n'
              'Use calibration mode to set accurate measurements.';
        } else {
          _resultText = 'No cattle detected in the image. You can still manually place measurement points.';
        }

        _isProcessing = false;
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _resultText = 'Error processing image: $e';
      });
    }
  }

  // Calibration mode toggle
  void _toggleCalibrationMode() {
    setState(() {
      _isCalibrationMode = !_isCalibrationMode;

      if (_isCalibrationMode) {
        // Initialize calibration points if needed
        if (_calibrationStart == null || _calibrationEnd == null) {
          final centerX = _imageSize.width / 2;
          final bottomY = _imageSize.height * 0.8;

          _calibrationStart = MeasurementPoint(Offset(centerX - 50, bottomY));
          _calibrationEnd = MeasurementPoint(Offset(centerX + 50, bottomY));
        }
      }
    });
  }

  void _updateCalibrationLength(double newLength) {
    setState(() {
      _calibrationLength = newLength;
      _updateCmPerPixel();
    });
  }

  void _updateCmPerPixel() {
    if (_calibrationStart != null && _calibrationEnd != null) {
      final dx = _calibrationEnd!.position.dx - _calibrationStart!.position.dx;
      final dy = _calibrationEnd!.position.dy - _calibrationStart!.position.dy;
      final pixelDistance = math.sqrt(dx * dx + dy * dy);

      if (pixelDistance > 0) {
        _cmPerPixel = _calibrationLength / pixelDistance;
      }
    }
  }

  // Convert screen position to image position
  Offset _screenToImagePosition(Offset screenPosition) {
    final double scaleX = _imageSize.width / _displaySize.width;
    final double scaleY = _imageSize.height / _displaySize.height;
    final double scale = math.min(scaleX, scaleY);

    // Calculate offsets to center the image
    final double offsetX = (_displaySize.width - _imageSize.width / scale) / 2;
    final double offsetY = (_displaySize.height - _imageSize.height / scale) / 2;

    return Offset(
      (screenPosition.dx - offsetX) * scale,
      (screenPosition.dy - offsetY) * scale,
    );
  }

  // Convert image position to screen position
  Offset _imageToScreenPosition(Offset imagePosition) {
    final double scaleX = _displaySize.width / _imageSize.width;
    final double scaleY = _displaySize.height / _imageSize.height;
    final double scale = math.min(scaleX, scaleY);

    // Calculate offsets to center the image
    final double offsetX = (_displaySize.width - _imageSize.width * scale) / 2;
    final double offsetY = (_displaySize.height - _imageSize.height * scale) / 2;

    return Offset(
      imagePosition.dx * scale + offsetX,
      imagePosition.dy * scale + offsetY,
    );
  }// Calculate the belly to spine height in cm
  double get _bellyToSpineHeightCm {
    if (_bellyPoint == null || _spinePoint == null) return 0;

    final dx = _spinePoint!.position.dx - _bellyPoint!.position.dx;
    final dy = _spinePoint!.position.dy - _bellyPoint!.position.dy;
    final pixelDistance = math.sqrt(dx * dx + dy * dy);

    return pixelDistance * _cmPerPixel;
  }

  // Calculate the body length in cm
  double get _bodyLengthCm {
    if (_neckPoint == null || _rearPoint == null) return 0;

    final dx = _rearPoint!.position.dx - _neckPoint!.position.dx;
    final dy = _rearPoint!.position.dy - _neckPoint!.position.dy;
    final pixelDistance = math.sqrt(dx * dx + dy * dy);

    return pixelDistance * _cmPerPixel;
  }

  // Calculate the heart girth (chest circumference) in cm using a proper 3D estimation
  double get _heartGirthCm {
    if (_heartGirthLeftPoint == null || _heartGirthRightPoint == null ||
        _bellyPoint == null || _spinePoint == null) return 0;

    // Get vertical height (belly to spine)
    final dyVertical = (_spinePoint!.position.dy - _bellyPoint!.position.dy).abs();

    // Get horizontal width (side to side at heart position)
    final dxHorizontal = (_heartGirthRightPoint!.position.dx - _heartGirthLeftPoint!.position.dx).abs();

    // Use a 3D approximation for cattle body shape
    // A more accurate formula for cattle girth using the Perimeter of Ellipse approximation
    // P ≈ 2π × √[(a² + b²) / 2] where a and b are semi-major and semi-minor axes

    // Semi-major and semi-minor axes of the ellipse
    final double a = dxHorizontal / 2 * _cmPerPixel; // half-width in cm
    final double b = dyVertical / 2 * _cmPerPixel;   // half-height in cm

    // Better ellipse circumference calculation (Ramanujan's approximation)
    // with cattle-specific adjustment factor
    final double h = math.pow(a - b, 2) / math.pow(a + b, 2);
    final ellipseCircumference = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)));

    // Add 8% to account for the non-elliptical shape of cattle chest
    final adjustedCircumference = ellipseCircumference * 1.08;

    return adjustedCircumference;
  }

  // Calculate dressing percentage based on breed and condition
  double get _dressingPercentage {
    // Base dressing percentage
    double percentage = 0.58; // Default 58%

    // Adjust for breed
    switch (_selectedBreed) {
      case 'Angus':
      case 'Hereford':
      case 'Charolais':
      case 'Limousin':
      case 'Simmental':
        percentage += 0.02; // Add 2% for beef breeds
        break;
      case 'Holstein':
      case 'Jersey':
        percentage -= 0.03; // Subtract 3% for dairy breeds
        break;
      default:
      // No adjustment for other breeds
        break;
    }

    // Adjust for condition
    switch (_selectedCondition) {
      case 'Thin':
        percentage -= 0.04; // Subtract 4% for thin cattle
        break;
      case 'Good':
        percentage += 0.02; // Add 2% for good condition
        break;
      case 'Excellent':
        percentage += 0.04; // Add 4% for excellent condition
        break;
      default:
      // No adjustment for average condition
        break;
    }

    return percentage;
  }

  // Calculate estimated live weight using improved formulas
  double get _estimatedLiveWeightKg {
    final heartGirth = _heartGirthCm;
    final bodyLength = _bodyLengthCm;

    if (heartGirth <= 0) return 0;

    // Based on multiple scientific studies for beef cattle weight estimation
    // We'll implement multiple formulas and take the average for better accuracy

    double weight1 = 0.0;
    double weight2 = 0.0;
    double weight3 = 0.0;

    // Formula 1: Schaeffer formula
    // LW (kg) = (HG^2 × BL × 0.000078) + 40
    if (bodyLength > 0) {
      weight1 = (heartGirth * heartGirth * bodyLength * 0.000078) + 40;
    }

    // Formula 2: USDA formula (adjusted)
    // LW (kg) = (HG^2 × 0.00065) × 0.454
    weight2 = (heartGirth * heartGirth * 0.00065) * 0.454;

    // Formula 3: Cook's formula
    // LW (kg) = (HG (in) + 18)^2 / 300 × 0.45359
    double heartGirthInches = heartGirth / 2.54;
    weight3 = ((heartGirthInches + 18) * (heartGirthInches + 18) / 300) * 0.45359;

    // If we have body length, average all three formulas
    if (bodyLength > 0) {
      return (weight1 + weight2 + weight3) / 3;
    }
    // Otherwise average just the two that don't need body length
    else {
      return (weight2 + weight3) / 2;
    }
  }

  // Calculate estimated meat yield
  double get _estimatedMeatYieldKg {
    return _estimatedLiveWeightKg * _dressingPercentage;
  }

  // Find the closest measurement point to the given position
  MeasurementPoint? _findClosestPoint(Offset position) {
    final points = [
      _bellyPoint,
      _spinePoint,
      _neckPoint,
      _rearPoint,
      _heartGirthLeftPoint,
      _heartGirthRightPoint
    ];

    if (_isCalibrationMode) {
      points.addAll([_calibrationStart, _calibrationEnd]);
    }

    MeasurementPoint? closest;
    double minDistance = double.infinity;

    for (final point in points) {
      if (point != null) {
        final screenPos = _imageToScreenPosition(point.position);
        final dx = screenPos.dx - position.dx;
        final dy = screenPos.dy - position.dy;
        final distance = math.sqrt(dx * dx + dy * dy);

        if (distance < minDistance && distance < 30) { // 30px touch radius
          minDistance = distance;
          closest = point;
        }
      }
    }

    return closest;
  }

  void _toggleExpandedResults() {
    setState(() {
      _showExpandedResults = !_showExpandedResults;
    });
  }

  // UI Helper methods
  Widget _buildMeasurementItem(BuildContext context, String label, String value, Color color) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Row(
          children: [
            Container(
              width: 12,
              height: 12,
              decoration: BoxDecoration(
                color: color,
                shape: BoxShape.circle,
              ),
            ),
            const SizedBox(width: 8),
            Text(
              label,
              style: Theme.of(context).textTheme.bodyMedium,
            ),
          ],
        ),
        Text(
          value,
          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }

  Widget _buildEstimationItem(BuildContext context, String label, String value, {bool isHighlighted = false}) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: Theme.of(context).textTheme.bodyMedium,
        ),
        Text(
          value,
          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
            fontWeight: FontWeight.bold,
            color: isHighlighted ? Theme.of(context).colorScheme.primary : null,
          ),
        ),
      ],
    );
  }

  Widget _buildBreedConditionSelectors() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Divider(height: 16),
        Text(
          'Cattle Information:',
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Breed:',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                  const SizedBox(height: 4),
                  DropdownButtonFormField<String>(
                    value: _selectedBreed,
                    decoration: InputDecoration(
                      contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                      isDense: true,
                    ),
                    items: _breeds.map((String breed) {
                      return DropdownMenuItem<String>(
                        value: breed,
                        child: Text(breed),
                      );
                    }).toList(),
                    onChanged: (String? newValue) {
                      if (newValue != null) {
                        setState(() {
                          _selectedBreed = newValue;
                        });
                      }
                    },
                  ),
                ],
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Condition:',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                  const SizedBox(height: 4),
                  DropdownButtonFormField<String>(
                    value: _selectedCondition,
                    decoration: InputDecoration(
                      contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                      isDense: true,
                    ),
                    items: _conditions.map((String condition) {
                      return DropdownMenuItem<String>(
                        value: condition,
                        child: Text(condition),
                      );
                    }).toList(),
                    onChanged: (String? newValue) {
                      if (newValue != null) {
                        setState(() {
                          _selectedCondition = newValue;
                        });
                      }
                    },
                  ),
                ],
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              'Dressing Percentage:',
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            Text(
              '${(_dressingPercentage * 100).toStringAsFixed(1)}%',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ],
    );
  }// Main build method for the UI
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cattle Measurements'),
        centerTitle: true,
        elevation: 0,
        actions: [
          if (_imageFile != null)
            IconButton(
              onPressed: _toggleCalibrationMode,
              icon: Icon(
                _isCalibrationMode ? Icons.straighten : Icons.straighten_outlined,
              ),
              tooltip: _isCalibrationMode ? 'Exit Calibration' : 'Calibrate',
            ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: Center(
              child: _imageFile == null
                  ? Container(
                color: Theme.of(context).colorScheme.surfaceVariant,
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.photo_outlined,
                        size: 72,
                        color: Theme.of(context).colorScheme.primary.withOpacity(0.5),
                      ),
                      const SizedBox(height: 16),
                      Text(
                        'No image selected',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Upload or capture an image to begin measurements',
                        style: Theme.of(context).textTheme.bodyMedium,
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              )
                  : LayoutBuilder(
                builder: (context, constraints) {
                  _displaySize = Size(constraints.maxWidth, constraints.maxHeight);

                  return GestureDetector(
                    onPanStart: (details) {
                      if (_isProcessing) return;

                      final point = _findClosestPoint(details.localPosition);
                      if (point != null) {
                        setState(() {
                          _activePoint = point;
                        });
                      }
                    },
                    onPanUpdate: (details) {
                      if (_isProcessing || _activePoint == null) return;

                      setState(() {
                        final imagePos = _screenToImagePosition(details.localPosition);
                        _activePoint!.position = imagePos;

                        if (_isCalibrationMode) {
                          _updateCmPerPixel();
                        }
                      });
                    },
                    onPanEnd: (details) {
                      setState(() {
                        _activePoint = null;
                      });
                    },
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        Image.file(
                          _imageFile!,
                          fit: BoxFit.contain,
                        ),
                        CustomPaint(
                          painter: MeasurementPainter(
                            imageSize: _imageSize,
                            displaySize: _displaySize,
                            bellyPoint: _bellyPoint,
                            spinePoint: _spinePoint,
                            neckPoint: _neckPoint,
                            rearPoint: _rearPoint,
                            heartGirthLeftPoint: _heartGirthLeftPoint,
                            heartGirthRightPoint: _heartGirthRightPoint,
                            calibrationStart: _isCalibrationMode ? _calibrationStart : null,
                            calibrationEnd: _isCalibrationMode ? _calibrationEnd : null,
                            bellyToSpineHeightCm: _bellyToSpineHeightCm,
                            bodyLengthCm: _bodyLengthCm,
                            heartGirthCm: _heartGirthCm,
                            isCalibrationMode: _isCalibrationMode,
                            calibrationLengthCm: _calibrationLength,
                          ),
                        ),
                      ],
                    ),
                  );
                },
              ),
            ),
          ),
          AnimatedContainer(
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeInOut,
            padding: const EdgeInsets.all(16),
            height: _showExpandedResults ? 500 : 200,
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surface,
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.05),
                  blurRadius: 10,
                  offset: const Offset(0, -3),
                ),
              ],
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(24),
                topRight: Radius.circular(24),
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                if (_isCalibrationMode)
                  Column(
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            'Calibration Mode',
                            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          Text(
                            '${_calibrationLength.toStringAsFixed(1)} cm',
                            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                              fontWeight: FontWeight.bold,
                              color: Theme.of(context).colorScheme.primary,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Drag the yellow points to a known length on the cattle',
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                      const SizedBox(height: 16),
                      Text(
                        'Reference length (cm):',
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                      Slider(
                        value: _calibrationLength,
                        min: 10,
                        max: 200,
                        divisions: 190,
                        label: '${_calibrationLength.toStringAsFixed(1)} cm',
                        onChanged: _updateCalibrationLength,
                      ),
                    ],
                  )
                else if (_cattleDetected &&
                    _bellyPoint != null && _spinePoint != null &&
                    _heartGirthLeftPoint != null && _heartGirthRightPoint != null)
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            'Measurement Results:',
                            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          GestureDetector(
                            onTap: _toggleExpandedResults,
                            child: Row(
                              children: [
                                Text(
                                  _showExpandedResults ? 'Simple View' : 'Detailed View',
                                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                    color: Theme.of(context).colorScheme.primary,
                                  ),
                                ),
                                Icon(
                                  _showExpandedResults ? Icons.arrow_drop_up : Icons.arrow_drop_down,
                                  color: Theme.of(context).colorScheme.primary,
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      _buildMeasurementItem(
                        context,
                        'Belly to spine height:',
                        '${_bellyToSpineHeightCm.toStringAsFixed(1)} cm',
                        Colors.blue,
                      ),
                      const SizedBox(height: 8),
                      _buildMeasurementItem(
                        context,
                        'Heart girth (circumference):',
                        '${_heartGirthCm.toStringAsFixed(1)} cm',
                        Colors.purple,
                      ),
                      const SizedBox(height: 8),
                      _buildMeasurementItem(
                        context,
                        'Body length (neck to rear):',
                        '${_bodyLengthCm.toStringAsFixed(1)} cm',
                        Colors.green,
                      ),
                      if (_showExpandedResults) ...[
                        // Breed and condition selectors
                        _buildBreedConditionSelectors(),

                        const Divider(height: 24),

                        Text(
                          'Weight Estimation:',
                          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 12),
                        _buildEstimationItem(
                          context,
                          'Estimated Live Weight:',
                          '${_estimatedLiveWeightKg.toStringAsFixed(1)} kg',
                        ),
                        const SizedBox(height: 8),
                        _buildEstimationItem(
                          context,
                          'Estimated Meat Yield:',
                          '${_estimatedMeatYieldKg.toStringAsFixed(1)} kg (${(_dressingPercentage * 100).round()}%)',
                          isHighlighted: true,
                        ),

                        const SizedBox(height: 12),
                        Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: Theme.of(context).colorScheme.primary.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Row(
                            children: [
                              Icon(
                                Icons.info_outline,
                                color: Theme.of(context).colorScheme.primary,
                                size: 20,
                              ),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  'Estimated weights are based on scientific formulas for cattle. For more accurate results, ensure calibration is correct and points are properly positioned.',
                                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                    color: Theme.of(context).colorScheme.primary,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ],
                  )
                else
                  Text(_resultText),
                const Spacer(),
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: _isProcessing ? null : _captureImage,
                        icon: Icon(Icons.camera_alt),
                        label: Text('Camera'),
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 12),
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: _isProcessing ? null : _pickImage,
                        icon: Icon(Icons.photo_library),
                        label: Text('Gallery'),
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 12),
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// Measurement Painter Class to draw the measurement points and lines
class MeasurementPainter extends CustomPainter {
  final Size imageSize;
  final Size displaySize;
  final MeasurementPoint? bellyPoint;
  final MeasurementPoint? spinePoint;
  final MeasurementPoint? neckPoint;
  final MeasurementPoint? rearPoint;
  final MeasurementPoint? heartGirthLeftPoint;
  final MeasurementPoint? heartGirthRightPoint;
  final MeasurementPoint? calibrationStart;
  final MeasurementPoint? calibrationEnd;
  final double bellyToSpineHeightCm;
  final double bodyLengthCm;
  final double heartGirthCm;
  final bool isCalibrationMode;
  final double calibrationLengthCm;

  MeasurementPainter({
    required this.imageSize,
    required this.displaySize,
    this.bellyPoint,
    this.spinePoint,
    this.neckPoint,
    this.rearPoint,
    this.heartGirthLeftPoint,
    this.heartGirthRightPoint,
    this.calibrationStart,
    this.calibrationEnd,
    required this.bellyToSpineHeightCm,
    required this.bodyLengthCm,
    required this.heartGirthCm,
    required this.isCalibrationMode,
    required this.calibrationLengthCm,
  });

  // Helper method to convert image position to screen position
  Offset _imageToScreenPosition(Offset imagePosition) {
    final double scaleX = displaySize.width / imageSize.width;
    final double scaleY = displaySize.height / imageSize.height;
    final double scale = math.min(scaleX, scaleY);

    // Calculate offsets to center the image
    final double offsetX = (displaySize.width - imageSize.width * scale) / 2;
    final double offsetY = (displaySize.height - imageSize.height * scale) / 2;

    return Offset(
      imagePosition.dx * scale + offsetX,
      imagePosition.dy * scale + offsetY,
    );
  }

  // Helper method to draw measurement labels
  void _drawLabel(Canvas canvas, String text, Offset position, Color color) {
    final textPainter = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: color,
          fontSize: 14,
          fontWeight: FontWeight.bold,
          shadows: [
            Shadow(
              blurRadius: 3,
              color: Colors.black.withOpacity(0.3),
              offset: const Offset(1, 1),
            ),
          ],
        ),
      ),
      textDirection: TextDirection.ltr,
    );

    textPainter.layout();
    textPainter.paint(
      canvas,
      Offset(position.dx - textPainter.width / 2,
          position.dy - textPainter.height / 2),
    );
  }

  @override
  void paint(Canvas canvas, Size size) {
    final Paint pointPaint = Paint()
      ..style = PaintingStyle.fill
      ..strokeWidth = 3.0;

    final Paint linePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    // Draw measurements if not in calibration mode
    if (!isCalibrationMode) {
      // Draw belly to spine height (vertical) line
      if (bellyPoint != null && spinePoint != null) {
        final bellyScreenPos = _imageToScreenPosition(bellyPoint!.position);
        final spineScreenPos = _imageToScreenPosition(spinePoint!.position);

        linePaint.color = Colors.blue;
        canvas.drawLine(bellyScreenPos, spineScreenPos, linePaint);

        // Draw belly point
        pointPaint.color = Colors.blue;
        canvas.drawCircle(bellyScreenPos, 8, pointPaint);

        // Draw spine point
        canvas.drawCircle(spineScreenPos, 8, pointPaint);

        // Draw height measurement label
        _drawLabel(
          canvas,
          '${bellyToSpineHeightCm.toStringAsFixed(1)} cm',
          Offset((bellyScreenPos.dx + spineScreenPos.dx) / 2 + 15,
              (bellyScreenPos.dy + spineScreenPos.dy) / 2),
          Colors.blue,
        );
      }

      // Draw heart girth circumference visualization (ellipse)
      if (bellyPoint != null && spinePoint != null &&
          heartGirthLeftPoint != null && heartGirthRightPoint != null) {
        final bellyScreenPos = _imageToScreenPosition(bellyPoint!.position);
        final spineScreenPos = _imageToScreenPosition(spinePoint!.position);
        final leftScreenPos = _imageToScreenPosition(heartGirthLeftPoint!.position);
        final rightScreenPos = _imageToScreenPosition(heartGirthRightPoint!.position);

        // Draw side points
        pointPaint.color = Colors.purple;
        canvas.drawCircle(leftScreenPos, 8, pointPaint);
        canvas.drawCircle(rightScreenPos, 8, pointPaint);

        // Calculate ellipse center and dimensions
        final centerX = (leftScreenPos.dx + rightScreenPos.dx) / 2;
        final centerY = (bellyScreenPos.dy + spineScreenPos.dy) / 2;
        final width = (rightScreenPos.dx - leftScreenPos.dx).abs();
        final height = (spineScreenPos.dy - bellyScreenPos.dy).abs();

        // Draw ellipse for heart girth
        linePaint.color = Colors.purple;
        final ellipseRect = Rect.fromCenter(
          center: Offset(centerX, centerY),
          width: width,
          height: height,
        );
        canvas.drawOval(ellipseRect, linePaint);

        // Draw auxiliary lines for better visualization
        linePaint.color = Colors.purple.withOpacity(0.6);

        // Draw center vertical line
        canvas.drawLine(
            Offset(centerX, bellyScreenPos.dy),
            Offset(centerX, spineScreenPos.dy),
            linePaint
        );

        // Draw heart girth measurement label
        _drawLabel(
          canvas,
          '${heartGirthCm.toStringAsFixed(1)} cm',
          Offset(centerX + width/2 + 25, centerY),
          Colors.purple,
        );
      }

      // Draw body length (neck to rear) line
      if (neckPoint != null && rearPoint != null) {
        final neckScreenPos = _imageToScreenPosition(neckPoint!.position);
        final rearScreenPos = _imageToScreenPosition(rearPoint!.position);

        linePaint.color = Colors.green;
        canvas.drawLine(neckScreenPos, rearScreenPos, linePaint);

        // Draw neck point
        pointPaint.color = Colors.green;
        canvas.drawCircle(neckScreenPos, 8, pointPaint);

        // Draw rear point
        canvas.drawCircle(rearScreenPos, 8, pointPaint);

        // Draw length measurement label
        _drawLabel(
          canvas,
          '${bodyLengthCm.toStringAsFixed(1)} cm',
          Offset((neckScreenPos.dx + rearScreenPos.dx) / 2,
              (neckScreenPos.dy + rearScreenPos.dy) / 2 - 15),
          Colors.green,
        );
      }
    } else {
      // Draw calibration line and points
      if (calibrationStart != null && calibrationEnd != null) {
        final startScreenPos = _imageToScreenPosition(calibrationStart!.position);
        final endScreenPos = _imageToScreenPosition(calibrationEnd!.position);

        // Draw calibration line
        linePaint.color = Colors.amber;
        canvas.drawLine(startScreenPos, endScreenPos, linePaint);

        // Draw calibration points
        pointPaint.color = Colors.amber;
        canvas.drawCircle(startScreenPos, 8, pointPaint);
        canvas.drawCircle(endScreenPos, 8, pointPaint);

        // Draw calibration length label
        _drawLabel(
          canvas,
          '${calibrationLengthCm.toStringAsFixed(1)} cm',
          Offset((startScreenPos.dx + endScreenPos.dx) / 2,
              (startScreenPos.dy + endScreenPos.dy) / 2 - 15),
          Colors.amber,
        );
      }
    }
  }

  @override
  bool shouldRepaint(MeasurementPainter oldDelegate) {
    return true;
  }
}