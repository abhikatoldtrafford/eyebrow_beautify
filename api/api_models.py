"""
Pydantic models for API request/response schemas.
"""

from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class BeautifyConfig(BaseModel):
    """Configuration for beautification pipeline."""

    # Detection thresholds
    yolo_conf_threshold: float = Field(0.25, ge=0.1, le=0.9, description="YOLO confidence threshold")
    mediapipe_conf_threshold: float = Field(0.5, ge=0.1, le=0.9, description="MediaPipe confidence threshold")

    # Face alignment
    straightening_threshold: float = Field(5.0, ge=0.0, le=15.0, description="Face straightening threshold (degrees)")

    # Validation thresholds
    min_mp_coverage: float = Field(80.0, ge=50.0, le=100.0, description="Minimum MediaPipe coverage (%)")
    eye_dist_range: Tuple[float, float] = Field((4.0, 8.0), description="Eye distance range (% of height)")
    aspect_ratio_range: Tuple[float, float] = Field((3.0, 10.0), description="Aspect ratio range")
    expansion_range: Tuple[float, float] = Field((0.9, 2.0), description="Expansion ratio range")

    # Extension parameters
    min_arch_thickness_pct: float = Field(0.015, ge=0.01, le=0.03, description="Minimum arch thickness (% of height)")
    connection_thickness_pct: float = Field(0.01, ge=0.005, le=0.02, description="Connection thickness (% of height)")

    # Eye exclusion
    eye_buffer_kernel: Tuple[int, int] = Field((15, 15), description="Eye buffer kernel size")
    eye_buffer_iterations: int = Field(2, ge=1, le=5, description="Eye buffer iterations")

    # Hair filtering
    hair_overlap_threshold: float = Field(0.15, ge=0.0, le=0.5, description="Hair overlap threshold")
    hair_distance_threshold: float = Field(0.3, ge=0.1, le=0.5, description="Hair distance threshold")

    # Beautification
    close_kernel: Tuple[int, int] = Field((7, 7), description="Close kernel size")
    open_kernel: Tuple[int, int] = Field((5, 5), description="Open kernel size")
    gaussian_kernel: Tuple[int, int] = Field((9, 9), description="Gaussian kernel size")
    gaussian_sigma: float = Field(2.0, ge=0.5, le=5.0, description="Gaussian sigma")


# =============================================================================
# VALIDATION MODELS
# =============================================================================

class ValidationMetrics(BaseModel):
    """Validation metrics for a single eyebrow."""

    mp_coverage: float = Field(..., description="MediaPipe coverage (%)")
    mp_coverage_pass: bool = Field(..., description="MediaPipe coverage check passed")
    mp_available: bool = Field(..., description="MediaPipe landmarks available (False = graceful degradation)")

    eye_distance_pct: float = Field(..., description="Eye distance (% of image height)")
    eye_distance_pass: bool = Field(..., description="Eye distance check passed")
    eye_available: bool = Field(..., description="Eye detected by YOLO (False = cannot validate distance)")

    aspect_ratio: float = Field(..., description="Eyebrow aspect ratio")
    aspect_ratio_pass: bool = Field(..., description="Aspect ratio check passed")

    eye_overlap: int = Field(..., description="Eye overlap (pixels)")
    eye_overlap_pass: bool = Field(..., description="Eye overlap check passed")

    expansion_ratio: float = Field(..., description="Expansion ratio (final / original)")
    expansion_ratio_pass: bool = Field(..., description="Expansion ratio check passed")

    thickness_ratio: float = Field(..., description="Thickness ratio (final / original)")
    thickness_ratio_pass: bool = Field(..., description="Thickness ratio check passed")

    overall_pass: bool = Field(..., description="Overall validation passed")


class EyebrowMetadata(BaseModel):
    """Metadata for a single eyebrow."""

    yolo_confidence: float = Field(..., description="YOLO detection confidence")
    yolo_area: int = Field(..., description="Original YOLO mask area (pixels)")
    final_area: int = Field(..., description="Final mask area (pixels)")
    has_eye: bool = Field(..., description="Associated eye detected")
    has_eye_box: bool = Field(..., description="Associated eye_box detected")
    hair_regions: int = Field(..., description="Number of overlapping hair regions")
    has_mediapipe: bool = Field(..., description="MediaPipe landmarks detected")


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class PreprocessingSummary(BaseModel):
    """Summary of preprocessing results included in beautify response."""

    valid: bool = Field(..., description="Face passed validation")
    rotation_angle: Optional[float] = Field(None, description="Detected rotation angle (degrees)")
    rotation_corrected: bool = Field(False, description="Whether rotation correction was applied")
    eye_validation_status: str = Field(..., description="Eye validation status")
    eyebrow_validation_status: str = Field(..., description="Eyebrow validation status")
    has_asymmetry: bool = Field(False, description="Asymmetries detected")
    warnings: List[str] = Field(default_factory=list, description="Warnings")


class EyebrowResult(BaseModel):
    """Result for a single eyebrow."""

    side: str = Field(..., description="Eyebrow side ('left' or 'right')")
    validation: ValidationMetrics
    metadata: EyebrowMetadata
    original_mask_base64: Optional[str] = Field(None, description="Base64 encoded original YOLO mask (PNG)")
    final_mask_base64: Optional[str] = Field(None, description="Base64 encoded final beautified mask (PNG)")
    preprocessing: Optional[PreprocessingSummary] = Field(None, description="Preprocessing summary (if enabled)")

    model_config = ConfigDict(protected_namespaces=())


class BeautifyResponse(BaseModel):
    """Response from /beautify endpoint."""

    success: bool = Field(..., description="Operation succeeded")
    message: str = Field(..., description="Status message")
    eyebrows: List[EyebrowResult] = Field(..., description="List of eyebrow results")
    processing_time_ms: float = Field(..., description="Processing time (milliseconds)")
    image_shape: Tuple[int, int] = Field(..., description="Input image shape (height, width)")
    preprocessing: Optional[PreprocessingSummary] = Field(None, description="Global preprocessing summary (if enabled)")


# =============================================================================
# YOLO DETECTION MODELS
# =============================================================================

class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float


class YOLODetection(BaseModel):
    """Single YOLO detection."""

    class_id: int = Field(..., description="Class ID")
    class_name: str = Field(..., description="Class name")
    confidence: float = Field(..., description="Detection confidence")
    box: BoundingBox = Field(..., description="Bounding box")
    box_width: float = Field(..., description="Box width")
    box_height: float = Field(..., description="Box height")
    center: Tuple[int, int] = Field(..., description="Box center (x, y)")
    mask_area: int = Field(..., description="Mask area (pixels)")
    mask_centroid: Tuple[int, int] = Field(..., description="Mask centroid (x, y)")
    mask_base64: Optional[str] = Field(None, description="Base64 encoded mask (PNG)")


class YOLOResponse(BaseModel):
    """Response from /detect/yolo endpoint."""

    success: bool
    message: str
    detections: Dict[str, List[YOLODetection]] = Field(..., description="Detections by class")
    processing_time_ms: float
    image_shape: Tuple[int, int]


# =============================================================================
# MEDIAPIPE DETECTION MODELS
# =============================================================================

class LandmarkGroup(BaseModel):
    """Group of landmarks (e.g., eyebrow, eye)."""

    points: List[Tuple[int, int]] = Field(..., description="List of (x, y) coordinates")
    indices: List[int] = Field(..., description="Landmark indices in MediaPipe mesh")
    center: Tuple[int, int] = Field(..., description="Center point")
    bbox: BoundingBox = Field(..., description="Bounding box")


class MediaPipeResponse(BaseModel):
    """Response from /detect/mediapipe endpoint."""

    success: bool
    message: str
    landmarks: Optional[Dict[str, LandmarkGroup]] = Field(None, description="Landmark groups")
    processing_time_ms: float
    image_shape: Tuple[int, int]


# =============================================================================
# REQUEST MODELS
# =============================================================================

class BeautifyRequest(BaseModel):
    """Request for /beautify endpoint (with base64 image)."""

    image_base64: str = Field(..., description="Base64 encoded image")
    config: Optional[BeautifyConfig] = Field(None, description="Optional custom configuration")
    return_masks: bool = Field(True, description="Include base64 encoded masks in response")
    return_visualizations: bool = Field(False, description="Include base64 encoded visualizations")


class DetectionRequest(BaseModel):
    """Request for detection endpoints (with base64 image)."""

    image_base64: str = Field(..., description="Base64 encoded image")
    conf_threshold: Optional[float] = Field(None, ge=0.1, le=0.9, description="Confidence threshold")
    return_masks: bool = Field(True, description="Include base64 encoded masks in response")


# =============================================================================
# ERROR MODELS
# =============================================================================

class SubmitEditedMaskRequest(BaseModel):
    """Request for /beautify/submit-edit endpoint."""

    image_base64: str = Field(..., description="Original image (base64 encoded)")
    edited_mask_base64: str = Field(..., description="User-edited mask (base64 encoded)")
    side: str = Field(..., description="Eyebrow side ('left' or 'right')")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class SubmitEditedMaskResponse(BaseModel):
    """Response from /beautify/submit-edit endpoint."""

    success: bool = Field(..., description="Operation succeeded")
    message: str = Field(..., description="Status message")
    side: str = Field(..., description="Eyebrow side")
    final_mask_base64: str = Field(..., description="Final mask (base64 encoded)")
    mask_area: int = Field(..., description="Final mask area (pixels)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata")


class SDBeautifyRequest(BaseModel):
    """Request for /generate/sd-beautify endpoint (Stable Diffusion integration)."""

    image_base64: str = Field(..., description="Original face image (base64 encoded)")
    left_eyebrow_mask_base64: Optional[str] = Field(None, description="Left eyebrow mask (base64 encoded)")
    right_eyebrow_mask_base64: Optional[str] = Field(None, description="Right eyebrow mask (base64 encoded)")
    prompt: str = Field("natural, well-groomed eyebrows", description="SD generation prompt")
    negative_prompt: str = Field("blurry, distorted, unnatural", description="SD negative prompt")
    strength: float = Field(0.7, ge=0.0, le=1.0, description="Inpainting strength (0.0-1.0)")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    num_inference_steps: int = Field(50, ge=10, le=150, description="Number of inference steps")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class SDBeautifyResponse(BaseModel):
    """Response from /generate/sd-beautify endpoint."""

    success: bool = Field(..., description="Operation succeeded")
    message: str = Field(..., description="Status message")
    result_image_base64: str = Field(..., description="Generated image with beautified eyebrows (base64)")
    processing_time_ms: float = Field(..., description="Processing time (milliseconds)")
    seed_used: int = Field(..., description="Seed used for generation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ErrorResponse(BaseModel):
    """Error response."""

    success: bool = Field(False, description="Always False for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


# =============================================================================
# HEALTH CHECK MODEL
# =============================================================================

class HealthResponse(BaseModel):
    """Response from /health endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="Service status ('healthy' or 'unhealthy')")
    model_loaded: bool = Field(..., description="YOLO model loaded successfully")
    mediapipe_available: bool = Field(..., description="MediaPipe available")
    version: str = Field(..., description="API version")


# =============================================================================
# EYEBROW ADJUSTMENT MODELS (NEW)
# =============================================================================

class AdjustEyebrowRequest(BaseModel):
    """Request for eyebrow adjustment endpoints."""

    mask_base64: str = Field(..., description="Base64 encoded eyebrow mask (PNG)")
    side: str = Field('unknown', description="Eyebrow side: 'left', 'right', or 'unknown'")
    increment: float = Field(0.05, ge=0.01, le=0.20, description="Adjustment increment (default: 0.05 = 5%)")
    num_clicks: int = Field(1, ge=1, le=10, description="Number of adjustment clicks (default: 1)")

    model_config = ConfigDict(protected_namespaces=())


class AdjustEyebrowResponse(BaseModel):
    """Response from eyebrow adjustment endpoints."""

    success: bool = Field(..., description="Operation succeeded")
    message: str = Field(..., description="Status message")
    adjusted_mask_base64: str = Field(..., description="Base64 encoded adjusted mask (PNG)")
    adjustment_type: str = Field(..., description="Type of adjustment ('thickness' or 'span')")
    direction: str = Field(..., description="Direction of adjustment ('increase' or 'decrease')")
    increment_applied: float = Field(..., description="Increment applied per click")
    num_clicks_applied: int = Field(..., description="Number of clicks applied")
    total_change_pct: float = Field(..., description="Total change percentage (e.g., 5.0 for +5%)")
    original_area: int = Field(..., description="Original mask area (pixels)")
    adjusted_area: int = Field(..., description="Adjusted mask area (pixels)")
    area_change_pct: float = Field(..., description="Actual area change (%)")
    processing_time_ms: float = Field(..., description="Processing time (milliseconds)")

    model_config = ConfigDict(protected_namespaces=())


# =============================================================================
# PREPROCESSING MODELS (NEW)
# =============================================================================

class PreprocessRequest(BaseModel):
    """Request model for preprocessing endpoint."""

    image_base64: str = Field(..., description="Base64 encoded image")
    config: Optional[Dict[str, Any]] = Field(None, description="Custom preprocessing configuration")

    model_config = ConfigDict(protected_namespaces=())


class EyeValidation(BaseModel):
    """Eye validation results."""

    is_valid: bool = Field(..., description="Eyes valid")
    status: str = Field(..., description="Validation status")
    mediapipe_has_eyes: bool = Field(False, description="MediaPipe detected eyes")
    yolo_has_eyes: bool = Field(False, description="YOLO detected eyes")
    yolo_eye_count: int = Field(0, description="Number of eyes detected by YOLO")
    eye_source: Optional[str] = Field(None, description="Source used for validation (mediapipe/yolo)")
    eye_distance_px: Optional[float] = Field(None, description="Eye distance (pixels)")
    eye_distance_pct: Optional[float] = Field(None, description="Eye distance (% of width)")
    vertical_diff_px: Optional[float] = Field(None, description="Vertical difference (pixels)")
    vertical_diff_pct: Optional[float] = Field(None, description="Vertical difference (% of height)")


class EyebrowValidation(BaseModel):
    """Eyebrow validation results."""

    is_valid: bool = Field(..., description="Eyebrows valid")
    status: str = Field(..., description="Validation status")
    yolo_eyebrow_count: int = Field(0, description="Number of eyebrows detected by YOLO")
    mediapipe_has_eyebrows: bool = Field(False, description="MediaPipe detected eyebrows")
    eyebrow_eye_overlaps: Optional[List[float]] = Field(None, description="Eyebrow-eye overlap IoU values")
    eyebrows_above_eyes: Optional[List[bool]] = Field(None, description="Whether eyebrows are above eyes")
    yolo_mediapipe_overlaps: Optional[List[float]] = Field(None, description="YOLO-MediaPipe overlap IoU values")


class AngleMetadata(BaseModel):
    """Rotation angle metadata."""

    final_angle: Optional[float] = Field(None, description="Final rotation angle (degrees)")
    status: str = Field(..., description="Angle estimation status")
    num_sources: int = Field(0, description="Number of angle sources")
    sources: List[str] = Field(default_factory=list, description="Angle sources used")
    raw_angles: List[float] = Field(default_factory=list, description="Raw angle values")
    filtered_angles: List[float] = Field(default_factory=list, description="Filtered angles (outliers removed)")
    outliers_removed: int = Field(0, description="Number of outliers removed")
    angle_std: float = Field(0.0, description="Standard deviation of filtered angles")
    source_agreement: Optional[float] = Field(None, description="Max difference between sources (degrees)")


class AsymmetryDetection(BaseModel):
    """Asymmetry detection results."""

    has_asymmetry: bool = Field(False, description="Asymmetries detected")
    angle_asymmetry: bool = Field(False, description="Angle asymmetry detected")
    position_asymmetry: bool = Field(False, description="Position asymmetry detected")
    span_asymmetry: bool = Field(False, description="Span asymmetry detected")
    status: str = Field(..., description="Detection status")
    left_angle: Optional[float] = Field(None, description="Left eyebrow angle (degrees)")
    right_angle: Optional[float] = Field(None, description="Right eyebrow angle (degrees)")
    angle_difference: Optional[float] = Field(None, description="Angle difference (degrees)")
    left_y_position: Optional[float] = Field(None, description="Left eyebrow Y position")
    right_y_position: Optional[float] = Field(None, description="Right eyebrow Y position")
    position_difference_pct: Optional[float] = Field(None, description="Position difference (%)")
    left_span: Optional[float] = Field(None, description="Left eyebrow span (pixels)")
    right_span: Optional[float] = Field(None, description="Right eyebrow span (pixels)")
    span_difference_pct: Optional[float] = Field(None, description="Span difference (%)")


class QualityValidation(BaseModel):
    """Face quality validation results."""

    is_valid: bool = Field(..., description="Quality valid")
    status: str = Field(..., description="Validation status")
    image_size: Optional[Tuple[int, int]] = Field(None, description="Image size (height, width)")
    min_dimension: Optional[int] = Field(None, description="Minimum dimension (pixels)")
    mediapipe_detected: bool = Field(False, description="MediaPipe detected face")
    mediapipe_confidence: float = Field(0.0, description="MediaPipe confidence")
    yolo_avg_confidence: float = Field(0.0, description="YOLO average confidence")


class PreprocessResponse(BaseModel):
    """Response model for preprocessing endpoint."""

    success: bool = Field(..., description="Preprocessing succeeded")
    valid: bool = Field(..., description="Face is valid")
    rejection_reason: Optional[str] = Field(None, description="Reason for rejection (if invalid)")
    image_shape: Tuple[int, int] = Field(..., description="Image shape (height, width)")
    rotation_angle: Optional[float] = Field(None, description="Detected rotation angle (degrees)")
    rotation_corrected: bool = Field(False, description="Rotation was corrected")
    eye_validation: EyeValidation = Field(..., description="Eye validation results")
    eyebrow_validation: EyebrowValidation = Field(..., description="Eyebrow validation results")
    quality_validation: QualityValidation = Field(..., description="Quality validation results")
    angle_metadata: AngleMetadata = Field(..., description="Angle calculation metadata")
    asymmetry_detection: AsymmetryDetection = Field(..., description="Asymmetry detection results")
    warnings: List[str] = Field(default_factory=list, description="Warnings during preprocessing")
    processing_time_ms: float = Field(..., description="Processing time (milliseconds)")
    report: str = Field(..., description="Human-readable preprocessing report")

    model_config = ConfigDict(protected_namespaces=())
