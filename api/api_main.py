"""
Eyebrow Beautification FastAPI Application

REST API for eyebrow detection and beautification using YOLO + MediaPipe.

Run with: uvicorn api.api_main:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import time
from pathlib import Path
from typing import Optional
import io

# Add parent directory to path to import core modules
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2

# Import core modules
import yolo_pred
import mediapipe_pred
import preprocess
from api import api_models, api_utils


# =============================================================================
# APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="Eyebrow Beautification API",
    description="REST API for detecting and beautifying eyebrows using YOLO + MediaPipe",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# GLOBAL STATE (Model Singleton)
# =============================================================================

class AppState:
    """Application state to store loaded models."""
    yolo_model = None
    mediapipe_available = True
    current_config = {
        'yolo_conf_threshold': 0.25,
        'yolo_simplify_epsilon': 0.005,
        'alignment_iou_threshold': 0.3,
        'alignment_distance_threshold': 20
    }


@app.on_event("startup")
async def startup_event():
    """Load YOLO model on startup."""
    try:
        print("Loading YOLO model...")
        AppState.yolo_model = yolo_pred.load_yolo_model()
        print("✓ YOLO model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        AppState.yolo_model = None

    # Check MediaPipe availability
    try:
        import mediapipe as mp
        AppState.mediapipe_available = True
        print("✓ MediaPipe available")
    except ImportError:
        AppState.mediapipe_available = False
        print("⚠ MediaPipe not available")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Shutting down API...")


# =============================================================================
# HEALTH CHECK ENDPOINT
# =============================================================================

@app.get("/health", response_model=api_models.HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns service status and model availability.
    """
    return api_models.HealthResponse(
        status="healthy" if AppState.yolo_model is not None else "unhealthy",
        model_loaded=AppState.yolo_model is not None,
        mediapipe_available=AppState.mediapipe_available,
        version="1.0.0"
    )


# =============================================================================
# CONFIGURATION ENDPOINTS
# =============================================================================

@app.get("/config", response_model=api_models.BeautifyConfig, tags=["Configuration"])
async def get_config():
    """
    Get current beautification configuration.

    Returns the current configuration parameters.
    """
    return api_models.BeautifyConfig(**AppState.current_config)


@app.post("/config", response_model=api_models.BeautifyConfig, tags=["Configuration"])
async def update_config(config: api_models.BeautifyConfig):
    """
    Update beautification configuration.

    Updates the global configuration parameters. These will be used as defaults
    for subsequent requests unless overridden.
    """
    AppState.current_config = api_utils.config_to_dict(config)
    return config


# =============================================================================
# PREPROCESSING ENDPOINT
# =============================================================================

@app.post("/preprocess", response_model=api_models.PreprocessResponse, tags=["Preprocessing"])
async def preprocess_face_endpoint(request: api_models.PreprocessRequest):
    """
    Comprehensive face preprocessing endpoint.

    Validates face quality, detects rotation angle, checks for asymmetries,
    and performs face sanity checks. Returns detailed validation report
    without performing beautification.

    **Features:**
    - Multi-source rotation angle detection (MediaPipe, YOLO eyes, eye_box)
    - Face sanity checks (eyes, eyebrows, quality validation)
    - Asymmetry detection (angle, position, span)
    - Robust outlier removal using IQR statistics

    **Use Cases:**
    - Validate face before beautification
    - Get rotation angle for manual correction
    - Check for asymmetries
    - Quality control for batch processing
    """
    start_time = time.time()

    # Check model availability
    if AppState.yolo_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="YOLO model not loaded"
        )

    try:
        # Decode base64 image
        img = api_utils.base64_to_image(request.image_base64)

        # Save to temp file (preprocessing needs file path)
        import tempfile
        import os
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(temp_fd)
        cv2.imwrite(temp_path, img)

        # Prepare preprocessing config
        preprocess_config = preprocess.DEFAULT_PREPROCESS_CONFIG.copy()
        if request.config:
            # Merge custom config
            preprocess_config.update(request.config)

        # Run preprocessing
        result = preprocess.preprocess_face(temp_path, AppState.yolo_model, preprocess_config)

        # Generate human-readable report
        report = preprocess.generate_preprocessing_report(result)

        # Clean up temp file
        os.unlink(temp_path)

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        # Build response using Pydantic models
        return api_models.PreprocessResponse(
            success=True,
            valid=result['valid'],
            rejection_reason=result.get('rejection_reason'),
            image_shape=result['image_shape'],
            rotation_angle=result.get('rotation_angle'),
            rotation_corrected=False,  # This endpoint doesn't apply correction
            eye_validation=api_models.EyeValidation(**result['eye_validation']),
            eyebrow_validation=api_models.EyebrowValidation(**result['eyebrow_validation']),
            quality_validation=api_models.QualityValidation(**result['quality_validation']),
            angle_metadata=api_models.AngleMetadata(**result['angle_metadata']),
            asymmetry_detection=api_models.AsymmetryDetection(**result['asymmetry_detection']),
            warnings=result.get('warnings', []),
            processing_time_ms=processing_time,
            report=report
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing failed: {str(e)}"
        )



@app.post("/beautify/base64", response_model=api_models.StencilExtractionResponse, tags=["Stencil Extraction"])
async def beautify_eyebrows_base64(request: api_models.BeautifyRequest):
    """
    Extract eyebrow stencil polygons from photo (v6.0).

    **New behavior (v6.0):** Returns polygon boundaries, NOT filled masks.
    **Use case:** Generate stencil for eyebrow grooming guide.

    **Process:**
    1. Detect eyebrows with YOLO (dense regions)
    2. Detect landmarks with MediaPipe (precise boundaries)
    3. Extract polygon contours from YOLO masks
    4. Ground polygons against MediaPipe landmarks (with 10% buffer)
    5. Return simplified polygon vertices

    **Response:** Polygon coordinates [[x, y], ...] for React curve editor
    """
    start_time = time.time()

    # Check model loaded
    if AppState.yolo_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="YOLO model not loaded. Service unavailable."
        )

    try:
        # Decode base64 image
        img = api_utils.base64_to_image(request.image_base64)

        # Validate image
        is_valid, error_msg = api_utils.validate_image_format(img)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image: {error_msg}"
            )

        # Save to temp file for processing
        import uuid
        from pathlib import Path

        # Ensure temp directory exists
        Path("temp").mkdir(exist_ok=True)

        temp_path = f"temp/{uuid.uuid4()}.jpg"
        cv2.imwrite(temp_path, img)

        # Get config
        use_config = request.config if request.config else AppState.current_config.copy()
        if isinstance(use_config, api_models.BeautifyConfig):
            use_config = api_utils.config_to_dict(use_config)

        # NEW: Run stencil extraction (polygon-based, not mask-based)
        stencil_results = api_utils.extract_stencils_from_image(
            temp_path,
            AppState.yolo_model,
            config=use_config
        )

        img_shape = img.shape[:2]

        # Convert to API response format
        stencil_polygons = [
            api_utils.convert_stencil_result(r, img_shape)
            for r in stencil_results
        ]

        # Cleanup
        api_utils.cleanup_temp_file(temp_path)

        processing_time = (time.time() - start_time) * 1000

        return api_models.StencilExtractionResponse(
            success=True,
            message=f"Successfully extracted {len(stencil_polygons)} stencil polygon(s)",
            stencils=stencil_polygons,
            processing_time_ms=processing_time,
            image_shape=img_shape,
            preprocessing=None  # TODO: Add if preprocessing enabled
        )

    except HTTPException:
        # Re-raise HTTP exceptions (like 400 from validation) without modification
        raise
    except ValueError as e:
        # User/data error - validation failed, no eyebrows detected, etc.
        if 'temp_path' in locals():
            api_utils.cleanup_temp_file(temp_path)

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Processing failed: {str(e)}"
        )
    except Exception as e:
        # Actual server error - model crash, OOM, etc.
        import traceback
        print(f"\n❌ EXCEPTION in beautify endpoint:")
        traceback.print_exc()

        if 'temp_path' in locals():
            api_utils.cleanup_temp_file(temp_path)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# =============================================================================
# YOLO DETECTION ENDPOINT
# =============================================================================

@app.post("/detect/yolo", response_model=api_models.YOLOResponse, tags=["Detection"])
async def detect_yolo_endpoint(
    file: UploadFile = File(..., description="Image file (JPG, PNG)"),
    conf_threshold: float = 0.25
):
    """
    YOLO detection only.

    Detects eyebrows, eyes, eye_box, and hair using YOLO segmentation model.

    **Returns:**
    - Detections organized by class
    - Bounding boxes
    - Segmentation masks (optional, base64 encoded)
    """
    start_time = time.time()

    if AppState.yolo_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="YOLO model not loaded"
        )

    try:
        # Save uploaded file
        file_content = await file.read()
        temp_path = api_utils.save_uploaded_file(file_content, file.filename)

        # Run YOLO detection
        detections = yolo_pred.detect_yolo(
            AppState.yolo_model,
            temp_path,
            conf_threshold=conf_threshold
        )

        # Load image for shape
        img = cv2.imread(temp_path)
        img_shape = img.shape[:2]

        # Convert to API format
        api_detections = {}
        for class_name, det_list in detections.items():
            api_detections[class_name] = [
                api_utils.convert_yolo_detection(det, include_mask=True)
                for det in det_list
            ]

        # Cleanup
        api_utils.cleanup_temp_file(temp_path)

        processing_time = (time.time() - start_time) * 1000

        return api_models.YOLOResponse(
            success=True,
            message="Detection successful",
            detections=api_detections,
            processing_time_ms=processing_time,
            image_shape=img_shape
        )

    except Exception as e:
        if 'temp_path' in locals():
            api_utils.cleanup_temp_file(temp_path)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}"
        )


# =============================================================================
# MEDIAPIPE DETECTION ENDPOINT
# =============================================================================

@app.post("/detect/mediapipe", response_model=api_models.MediaPipeResponse, tags=["Detection"])
async def detect_mediapipe_endpoint(
    file: UploadFile = File(..., description="Image file (JPG, PNG)"),
    conf_threshold: float = 0.5
):
    """
    MediaPipe face landmark detection only.

    Detects 468 face landmarks including eyebrows and eyes.

    **Returns:**
    - Landmark coordinates organized by facial feature
    - Bounding boxes for each feature
    """
    start_time = time.time()

    if not AppState.mediapipe_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MediaPipe not available"
        )

    try:
        # Save and load image
        file_content = await file.read()
        temp_path = api_utils.save_uploaded_file(file_content, file.filename)

        img = cv2.imread(temp_path)
        img_shape = img.shape[:2]

        # Run MediaPipe detection
        landmarks = mediapipe_pred.detect_mediapipe(img, conf_threshold=conf_threshold)

        # Convert to API format
        api_landmarks = None
        if landmarks:
            api_landmarks = {}
            for feature_name in ['left_eyebrow', 'right_eyebrow', 'left_eye', 'right_eye', 'face_oval']:
                if feature_name in landmarks and landmarks[feature_name]:
                    api_landmarks[feature_name] = api_utils.convert_landmark_group(landmarks[feature_name])

        # Cleanup
        api_utils.cleanup_temp_file(temp_path)

        processing_time = (time.time() - start_time) * 1000

        return api_models.MediaPipeResponse(
            success=True,
            message="Detection successful" if landmarks else "No face detected",
            landmarks=api_landmarks,
            processing_time_ms=processing_time,
            image_shape=img_shape
        )

    except Exception as e:
        if 'temp_path' in locals():
            api_utils.cleanup_temp_file(temp_path)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}"
        )


@app.post("/detect/yolo/base64", response_model=api_models.YOLOResponse, tags=["Detection"])
async def detect_yolo_base64(request: api_models.DetectionRequest):
    """
    YOLO detection only (base64 input).

    Alternative endpoint that accepts base64 encoded images in JSON request body.
    """
    start_time = time.time()

    if AppState.yolo_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="YOLO model not loaded"
        )

    try:
        # Decode base64 image
        img = api_utils.base64_to_image(request.image_base64)

        # Validate image
        is_valid, error_msg = api_utils.validate_image_format(img)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image: {error_msg}"
            )

        # Save to temp file
        import uuid
        temp_path = f"temp/{uuid.uuid4()}.jpg"
        cv2.imwrite(temp_path, img)

        # Run YOLO detection
        conf_threshold = request.conf_threshold if request.conf_threshold else 0.25
        detections = yolo_pred.detect_yolo(
            AppState.yolo_model,
            temp_path,
            conf_threshold=conf_threshold
        )

        img_shape = img.shape[:2]

        # Convert to API format
        api_detections = {}
        for class_name, det_list in detections.items():
            api_detections[class_name] = [
                api_utils.convert_yolo_detection(det, include_mask=request.return_masks)
                for det in det_list
            ]

        # Cleanup
        api_utils.cleanup_temp_file(temp_path)

        processing_time = (time.time() - start_time) * 1000

        return api_models.YOLOResponse(
            success=True,
            message="Detection successful",
            detections=api_detections,
            processing_time_ms=processing_time,
            image_shape=img_shape
        )

    except HTTPException:
        raise
    except ValueError as e:
        if 'temp_path' in locals():
            api_utils.cleanup_temp_file(temp_path)

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Detection failed: {str(e)}"
        )
    except Exception as e:
        if 'temp_path' in locals():
            api_utils.cleanup_temp_file(temp_path)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/detect/mediapipe/base64", response_model=api_models.MediaPipeResponse, tags=["Detection"])
async def detect_mediapipe_base64(request: api_models.DetectionRequest):
    """
    MediaPipe face landmark detection only (base64 input).

    Alternative endpoint that accepts base64 encoded images in JSON request body.
    """
    start_time = time.time()

    if not AppState.mediapipe_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MediaPipe not available"
        )

    try:
        # Decode base64 image
        img = api_utils.base64_to_image(request.image_base64)

        # Validate image
        is_valid, error_msg = api_utils.validate_image_format(img)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image: {error_msg}"
            )

        img_shape = img.shape[:2]

        # Run MediaPipe detection
        conf_threshold = request.conf_threshold if request.conf_threshold else 0.5
        landmarks = mediapipe_pred.detect_mediapipe(img, conf_threshold=conf_threshold)

        # Convert to API format
        api_landmarks = None
        if landmarks:
            api_landmarks = {}
            for feature_name in ['left_eyebrow', 'right_eyebrow', 'left_eye', 'right_eye', 'face_oval']:
                if feature_name in landmarks and landmarks[feature_name]:
                    api_landmarks[feature_name] = api_utils.convert_landmark_group(landmarks[feature_name])

        processing_time = (time.time() - start_time) * 1000

        return api_models.MediaPipeResponse(
            success=True,
            message="Detection successful" if landmarks else "No face detected",
            landmarks=api_landmarks,
            processing_time_ms=processing_time,
            image_shape=img_shape
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Detection failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# =============================================================================
# SUBMIT EDITED MASK ENDPOINT
# =============================================================================

@app.post("/beautify/submit-edit", response_model=api_models.SubmitEditedMaskResponse, tags=["Beautification"])
async def submit_edited_mask(request: api_models.SubmitEditedMaskRequest):
    """
    Submit user-edited eyebrow mask.

    This endpoint allows users to submit their manually edited eyebrow mask
    after performing edits in the UI (Streamlit, etc.).

    **Use case:**
    After calling /beautify, user edits the mask in UI (move, resize, redraw),
    then submits the final edited version here for saving/further processing.

    **Input:**
    - Original image (base64)
    - User-edited mask (base64 PNG, binary mask)
    - Side ('left' or 'right')
    - Optional metadata

    **Returns:**
    - Confirmation with final mask details
    - Mask statistics
    """
    try:
        # Decode edited mask
        edited_mask = api_utils.base64_to_image(request.edited_mask_base64)

        # Convert to binary mask if not already
        if len(edited_mask.shape) == 3:
            edited_mask = cv2.cvtColor(edited_mask, cv2.COLOR_BGR2GRAY)
        _, edited_mask = cv2.threshold(edited_mask, 127, 1, cv2.THRESH_BINARY)

        # Calculate mask statistics
        mask_area = int(cv2.countNonZero(edited_mask))

        # Convert back to base64
        final_mask_b64 = api_utils.mask_to_base64(edited_mask)

        return api_models.SubmitEditedMaskResponse(
            success=True,
            message=f"Successfully received edited {request.side} eyebrow mask",
            side=request.side,
            final_mask_base64=final_mask_b64,
            mask_area=mask_area,
            metadata=request.metadata
        )

    except (ValueError, cv2.error) as e:
        # User error - invalid mask format or data
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mask format: {str(e)}"
        )
    except Exception as e:
        # Server error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# =============================================================================
# STABLE DIFFUSION BEAUTIFY ENDPOINT (FUTURE IMPLEMENTATION)
# =============================================================================

@app.post("/generate/sd-beautify", response_model=api_models.SDBeautifyResponse, tags=["Generation"])
async def sd_beautify_eyebrows(request: api_models.SDBeautifyRequest):
    """
    Generate beautified eyebrows using Stable Diffusion inpainting.

    **FUTURE IMPLEMENTATION** - This endpoint is a provision for SD integration.

    **Workflow:**
    1. User calls /beautify to get initial masks
    2. User edits masks in UI (optional)
    3. User calls this endpoint with final masks
    4. SD model inpaints eyebrow regions with natural, beautified eyebrows
    5. Returns full image with generated eyebrows

    **Input:**
    - Original face image (base64)
    - Left/right eyebrow masks (base64, user-edited or from /beautify)
    - SD parameters (prompt, strength, steps, etc.)

    **Returns:**
    - Full image with SD-generated beautified eyebrows
    - Processing time and seed for reproducibility

    **TODO:**
    - Integrate Stable Diffusion model (e.g., sd-inpainting-v1-5)
    - Implement mask preprocessing (dilation, feathering for natural blending)
    - Add batch processing capability
    - GPU acceleration
    """
    try:
        # Decode inputs
        img = api_utils.base64_to_image(request.image_base64)

        left_mask = None
        right_mask = None

        if request.left_eyebrow_mask_base64:
            left_mask = api_utils.base64_to_image(request.left_eyebrow_mask_base64)
            if len(left_mask.shape) == 3:
                left_mask = cv2.cvtColor(left_mask, cv2.COLOR_BGR2GRAY)

        if request.right_eyebrow_mask_base64:
            right_mask = api_utils.base64_to_image(request.right_eyebrow_mask_base64)
            if len(right_mask.shape) == 3:
                right_mask = cv2.cvtColor(right_mask, cv2.COLOR_BGR2GRAY)

        # Combine masks
        import numpy as np
        combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if left_mask is not None:
            combined_mask = cv2.bitwise_or(combined_mask, left_mask)
        if right_mask is not None:
            combined_mask = cv2.bitwise_or(combined_mask, right_mask)

        # TODO: Implement SD inpainting here
        # For now, return placeholder response

        # Placeholder: just return original image
        result_img_b64 = api_utils.image_to_base64(img)

        # Use provided seed or generate random
        import random
        seed_used = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)

        return api_models.SDBeautifyResponse(
            success=False,  # Set to False until SD is implemented
            message="SD inpainting not yet implemented. This is a placeholder endpoint.",
            result_image_base64=result_img_b64,
            processing_time_ms=0.0,
            seed_used=seed_used,
            metadata={
                "status": "placeholder",
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "strength": request.strength,
                "guidance_scale": request.guidance_scale,
                "num_inference_steps": request.num_inference_steps,
                "left_mask_provided": request.left_eyebrow_mask_base64 is not None,
                "right_mask_provided": request.right_eyebrow_mask_base64 is not None,
                "combined_mask_area": int(cv2.countNonZero(combined_mask))
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SD beautify failed: {str(e)}"
        )


# =============================================================================
# ROOT ENDPOINT
# =============================================================================

# =============================================================================
# EYEBROW ADJUSTMENT ENDPOINTS (NEW)
# =============================================================================

@app.post("/adjust/thickness/increase", response_model=api_models.AdjustEyebrowResponse, tags=["Adjustments"])
async def increase_thickness(request: api_models.AdjustEyebrowRequest):
    """
    Increase eyebrow thickness by specified increment (default: +5% per click).

    Uses morphological dilation perpendicular to contours (along normal vectors).
    Maintains natural eyebrow arch curvature.
    """
    import utils

    start_time = time.time()

    try:
        # Decode mask from base64
        mask = api_utils.decode_mask_from_base64(request.mask_base64)
        original_area = int(mask.sum())

        # Apply adjustment
        adjusted_mask = utils.adjust_eyebrow_multiple_times(
            mask=mask,
            adjustment_type='thickness',
            direction='increase',
            num_clicks=request.num_clicks,
            increment=request.increment,
            side=request.side
        )

        # Calculate metrics
        adjusted_area = int(adjusted_mask.sum())
        area_change_pct = ((adjusted_area - original_area) / original_area * 100) if original_area > 0 else 0
        total_change_pct = request.increment * request.num_clicks * 100

        # Encode result
        adjusted_mask_base64 = api_utils.encode_mask_to_base64(adjusted_mask)

        processing_time = (time.time() - start_time) * 1000

        return api_models.AdjustEyebrowResponse(
            success=True,
            message=f"Thickness increased by {total_change_pct:.1f}%",
            adjusted_mask_base64=adjusted_mask_base64,
            adjustment_type='thickness',
            direction='increase',
            increment_applied=request.increment,
            num_clicks_applied=request.num_clicks,
            total_change_pct=total_change_pct,
            original_area=original_area,
            adjusted_area=adjusted_area,
            area_change_pct=area_change_pct,
            processing_time_ms=processing_time
        )

    except (ValueError, cv2.error) as e:
        # User error - invalid mask format or data
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mask: {str(e)}"
        )
    except Exception as e:
        # Server error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/adjust/thickness/decrease", response_model=api_models.AdjustEyebrowResponse, tags=["Adjustments"])
async def decrease_thickness(request: api_models.AdjustEyebrowRequest):
    """
    Decrease eyebrow thickness by specified increment (default: -5% per click).

    Uses morphological erosion perpendicular to contours (along normal vectors).
    Maintains natural eyebrow arch curvature.
    """
    import utils

    start_time = time.time()

    try:
        # Decode mask from base64
        mask = api_utils.decode_mask_from_base64(request.mask_base64)
        original_area = int(mask.sum())

        # Apply adjustment
        adjusted_mask = utils.adjust_eyebrow_multiple_times(
            mask=mask,
            adjustment_type='thickness',
            direction='decrease',
            num_clicks=request.num_clicks,
            increment=request.increment,
            side=request.side
        )

        # Calculate metrics
        adjusted_area = int(adjusted_mask.sum())
        area_change_pct = ((adjusted_area - original_area) / original_area * 100) if original_area > 0 else 0
        total_change_pct = -request.increment * request.num_clicks * 100

        # Encode result
        adjusted_mask_base64 = api_utils.encode_mask_to_base64(adjusted_mask)

        processing_time = (time.time() - start_time) * 1000

        return api_models.AdjustEyebrowResponse(
            success=True,
            message=f"Thickness decreased by {abs(total_change_pct):.1f}%",
            adjusted_mask_base64=adjusted_mask_base64,
            adjustment_type='thickness',
            direction='decrease',
            increment_applied=request.increment,
            num_clicks_applied=request.num_clicks,
            total_change_pct=total_change_pct,
            original_area=original_area,
            adjusted_area=adjusted_area,
            area_change_pct=area_change_pct,
            processing_time_ms=processing_time
        )

    except (ValueError, cv2.error) as e:
        # User error - invalid mask format or data
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mask: {str(e)}"
        )
    except Exception as e:
        # Server error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/adjust/span/increase", response_model=api_models.AdjustEyebrowResponse, tags=["Adjustments"])
async def increase_span(request: api_models.AdjustEyebrowRequest):
    """
    Increase eyebrow span/length by specified increment (default: +5% per click).

    DIRECTIONAL EXPANSION: Extends TAIL side only (temple side), protects center (nose side).
    - Left eyebrow: extends right
    - Right eyebrow: extends left

    Creates natural tail lengthening effect.
    """
    import utils

    start_time = time.time()

    try:
        # Decode mask from base64
        mask = api_utils.decode_mask_from_base64(request.mask_base64)
        original_area = int(mask.sum())

        # Apply adjustment
        adjusted_mask = utils.adjust_eyebrow_multiple_times(
            mask=mask,
            adjustment_type='span',
            direction='increase',
            num_clicks=request.num_clicks,
            increment=request.increment,
            side=request.side
        )

        # Calculate metrics
        adjusted_area = int(adjusted_mask.sum())
        area_change_pct = ((adjusted_area - original_area) / original_area * 100) if original_area > 0 else 0
        total_change_pct = request.increment * request.num_clicks * 100

        # Encode result
        adjusted_mask_base64 = api_utils.encode_mask_to_base64(adjusted_mask)

        processing_time = (time.time() - start_time) * 1000

        return api_models.AdjustEyebrowResponse(
            success=True,
            message=f"Span increased by {total_change_pct:.1f}% (tail side)",
            adjusted_mask_base64=adjusted_mask_base64,
            adjustment_type='span',
            direction='increase',
            increment_applied=request.increment,
            num_clicks_applied=request.num_clicks,
            total_change_pct=total_change_pct,
            original_area=original_area,
            adjusted_area=adjusted_area,
            area_change_pct=area_change_pct,
            processing_time_ms=processing_time
        )

    except (ValueError, cv2.error) as e:
        # User error - invalid mask format or data
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mask: {str(e)}"
        )
    except Exception as e:
        # Server error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/adjust/span/decrease", response_model=api_models.AdjustEyebrowResponse, tags=["Adjustments"])
async def decrease_span(request: api_models.AdjustEyebrowRequest):
    """
    Decrease eyebrow span/length by specified increment (default: -5% per click).

    DIRECTIONAL CONTRACTION: Contracts TAIL side only (temple side), protects center (nose side).
    - Left eyebrow: contracts right
    - Right eyebrow: contracts left

    Creates natural tail shortening effect.
    """
    import utils

    start_time = time.time()

    try:
        # Decode mask from base64
        mask = api_utils.decode_mask_from_base64(request.mask_base64)
        original_area = int(mask.sum())

        # Apply adjustment
        adjusted_mask = utils.adjust_eyebrow_multiple_times(
            mask=mask,
            adjustment_type='span',
            direction='decrease',
            num_clicks=request.num_clicks,
            increment=request.increment,
            side=request.side
        )

        # Calculate metrics
        adjusted_area = int(adjusted_mask.sum())
        area_change_pct = ((adjusted_area - original_area) / original_area * 100) if original_area > 0 else 0
        total_change_pct = -request.increment * request.num_clicks * 100

        # Encode result
        adjusted_mask_base64 = api_utils.encode_mask_to_base64(adjusted_mask)

        processing_time = (time.time() - start_time) * 1000

        return api_models.AdjustEyebrowResponse(
            success=True,
            message=f"Span decreased by {abs(total_change_pct):.1f}% (tail side)",
            adjusted_mask_base64=adjusted_mask_base64,
            adjustment_type='span',
            direction='decrease',
            increment_applied=request.increment,
            num_clicks_applied=request.num_clicks,
            total_change_pct=total_change_pct,
            original_area=original_area,
            adjusted_area=adjusted_area,
            area_change_pct=area_change_pct,
            processing_time_ms=processing_time
        )

    except (ValueError, cv2.error) as e:
        # User error - invalid mask format or data
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mask: {str(e)}"
        )
    except Exception as e:
        # Server error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/", tags=["Info"])
async def root():
    """
    API information endpoint.
    """
    return {
        "name": "Eyebrow Beautification API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "adjustments": [
                "/adjust/thickness/increase",
                "/adjust/thickness/decrease",
                "/adjust/span/increase",
                "/adjust/span/decrease"
            ]
        }
    }


# =============================================================================
# STENCIL LIBRARY ENDPOINTS (v6.0)
# =============================================================================

@app.post("/stencils/save", response_model=api_models.StencilSaveResponse, tags=["Stencil Library"])
async def save_stencil(request: api_models.StencilSaveRequest):
    """
    Save edited stencil to library.

    **Use case:** After user finishes editing polygon in React, save to persistent storage.

    **Storage:** File-based JSON in `stencil_data/` directory (no database required).
    """
    try:
        import stencil_storage

        storage = stencil_storage.StencilStorage()

        # Prepare polygon dict (as expected by storage layer)
        polygon_dict = {
            'points': request.polygon,
            'num_points': len(request.polygon),
            'source': 'user_edited'  # Since this is a user-edited stencil
        }

        # Prepare metadata
        metadata = {
            'side': request.side,
            'name': request.name,
            'tags': request.tags,
            'notes': request.notes
        }

        # Convert base64 image to bytes if provided
        image_data = None
        if request.image_base64:
            import base64
            image_data = base64.b64decode(request.image_base64)

        # Save stencil
        stencil_id = storage.save_stencil(
            polygon_dict,
            metadata,
            image_data=image_data
        )

        return api_models.StencilSaveResponse(
            success=True,
            message="Stencil saved successfully",
            stencil_id=stencil_id,
            file_path=f"stencil_data/stencil_{stencil_id}.json"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save stencil: {str(e)}"
        )


@app.get("/stencils/list", response_model=api_models.StencilListResponse, tags=["Stencil Library"])
async def list_stencils(
    side: Optional[str] = None,
    limit: int = 50
):
    """
    List all saved stencils with optional filtering.

    **Query params:**
    - `side`: Filter by side ('left' or 'right')
    - `limit`: Maximum number of results (default: 50)

    **Returns:** List of stencils with metadata (no full polygon data)
    """
    try:
        import stencil_storage

        storage = stencil_storage.StencilStorage()

        # Get stencils
        stencils = storage.list_stencils(side=side, limit=limit)

        # Convert to API format
        stencil_items = [
            api_models.StencilListItem(
                stencil_id=s['id'],  # Storage layer uses 'id', API uses 'stencil_id'
                side=s['side'],
                name=s.get('name'),
                tags=s.get('tags', []),
                created_at=s['created_at'],
                num_points=s['num_points'],
                bbox=s.get('bbox', [0, 0, 0, 0])
            )
            for s in stencils
        ]

        return api_models.StencilListResponse(
            success=True,
            message=f"Found {len(stencil_items)} stencils",
            stencils=stencil_items,
            total_count=len(stencil_items),  # TODO: Get actual total from storage
            filtered_count=len(stencil_items)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list stencils: {str(e)}"
        )


@app.get("/stencils/{stencil_id}", response_model=api_models.StencilGetResponse, tags=["Stencil Library"])
async def get_stencil(stencil_id: str):
    """
    Get full stencil data by ID.

    **Returns:** Complete stencil polygon with all metadata and optional original photo.

    **Use case:** Load stencil for re-editing or export.
    """
    try:
        import stencil_storage

        storage = stencil_storage.StencilStorage()

        # Get stencil
        stencil_data = storage.get_stencil(stencil_id)

        if not stencil_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stencil not found: {stencil_id}"
            )

        # Convert to API format (simplified - full conversion would need more fields)
        # Storage uses 'id' and 'polygon.points', API uses 'stencil_id' and 'polygon'
        stencil_polygon = api_models.StencilPolygon(
            stencil_id=stencil_data['id'],
            side=stencil_data['side'],
            polygon=stencil_data['polygon']['points'],
            num_points=stencil_data['polygon']['num_points'],
            source=stencil_data['polygon'].get('source', 'user_edited'),
            bbox=stencil_data.get('metadata', {}).get('bbox', [0, 0, 0, 0]),
            alignment=api_models.PolygonAlignment(
                aligned=True,
                iou=1.0,
                avg_distance=0.0,
                mp_inside_count=0,
                mp_inside_ratio=1.0,
                all_mp_inside=True,
                buffer_distance=0.0,
                mp_inside_with_buffer_count=0,
                mp_inside_with_buffer_ratio=1.0,
                all_mp_inside_with_buffer=True
            ),
            validation=api_models.PolygonValidation(valid=True, checks={}),
            metadata=api_models.StencilMetadata(
                yolo_vertices=0,
                mp_landmarks=0,
                final_vertices=stencil_data['polygon']['num_points'],
                yolo_confidence=0.0,
                source=stencil_data['polygon'].get('source', 'user_edited'),
                merged=False
            ),
            created_at=stencil_data.get('metadata', {}).get('created_at', ''),
            image_shape=None
        )

        return api_models.StencilGetResponse(
            success=True,
            message="Stencil retrieved successfully",
            stencil=stencil_polygon,
            image_base64=stencil_data.get('image_base64')
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stencil: {str(e)}"
        )


@app.delete("/stencils/{stencil_id}", response_model=api_models.StencilDeleteResponse, tags=["Stencil Library"])
async def delete_stencil(stencil_id: str):
    """
    Delete stencil from library.

    **WARNING:** This action is permanent! Deletes both JSON file and any exported files.
    """
    try:
        import stencil_storage

        storage = stencil_storage.StencilStorage()

        # Delete stencil
        success = storage.delete_stencil(stencil_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stencil not found: {stencil_id}"
            )

        return api_models.StencilDeleteResponse(
            success=True,
            message="Stencil deleted successfully",
            stencil_id=stencil_id
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete stencil: {str(e)}"
        )


@app.get("/stencils/{stencil_id}/export", response_model=api_models.StencilExportResponse, tags=["Stencil Library"])
async def export_stencil(
    stencil_id: str,
    format: str = "svg"
):
    """
    Export stencil as SVG, JSON, or PNG.

    **Query params:**
    - `format`: Export format ('svg', 'json', 'png')

    **SVG:** For laser cutting / physical stencils
    **JSON:** For data interchange
    **PNG:** For visual preview

    **Returns:** File path and optional download content
    """
    try:
        import stencil_storage

        storage = stencil_storage.StencilStorage()

        # Export stencil (storage methods return content directly, not file paths)
        if format == "svg":
            content = storage.export_svg(stencil_id)
            file_path = f"stencil_data/exports/stencil_{stencil_id[:8]}.svg"
        elif format == "json":
            import json
            json_data = storage.export_json(stencil_id)
            if json_data:
                content = json.dumps(json_data, indent=2)
            else:
                content = None
            file_path = f"stencil_data/exports/stencil_{stencil_id[:8]}.json"
        elif format == "png":
            # TODO: Implement PNG export
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="PNG export not yet implemented"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid format: {format}. Must be 'svg', 'json', or 'png'"
            )

        if content is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stencil not found: {stencil_id}"
            )

        return api_models.StencilExportResponse(
            success=True,
            message=f"Stencil exported as {format.upper()}",
            format=format,
            file_path=file_path,
            download_url=None,  # Could add CDN URL here
            content=content
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export stencil: {str(e)}"
        )


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("="*60)
    print("Eyebrow Beautification API")
    print("="*60)
    print("Starting server...")
    print("Docs: http://localhost:8000/docs")
    print("="*60)

    uvicorn.run(
        "api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
