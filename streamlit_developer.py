"""
Streamlit Developer Corner

Comprehensive developer testing and debugging interface:
- API endpoint testing with live request/response
- Test suite execution and results
- Log viewing (real-time API logs)
- Mask visualization (all intermediate outputs)
- Pipeline debugging (step-by-step)
- Validation metrics dashboard
- Configuration playground
"""

import streamlit as st
import requests
import json
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np
from streamlit_config import API_BASE_URL
from streamlit_utils import (
    image_to_base64, base64_to_image, base64_to_mask,
    overlay_mask_on_image, display_validation_metrics,
    pil_to_cv2, cv2_to_pil
)
from streamlit_api_client import APIClient


def render_developer_corner():
    """Main developer corner interface."""
    st.title("🛠️ Developer Corner")
    st.markdown("**Comprehensive testing, debugging, and visualization hub**")

    tabs = st.tabs([
        "🌐 API Tester",
        "🧪 Test Runner",
        "📊 Visualizer",
        "🔍 Preprocessing",
        "📝 Logs",
        "⚙️ Config Playground"
    ])

    with tabs[0]:
        render_api_tester()

    with tabs[1]:
        render_test_runner()

    with tabs[2]:
        render_visualizer()

    with tabs[3]:
        render_preprocessing_tab()

    with tabs[4]:
        render_log_viewer()

    with tabs[5]:
        render_config_playground()


# =============================================================================
# API ENDPOINT TESTER
# =============================================================================

def render_api_tester():
    """Interactive API endpoint tester."""
    st.header("🌐 API Endpoint Tester")
    st.markdown("Test all API endpoints interactively with live request/response visualization")

    # Health check first
    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("🏥 Health Check", use_container_width=True):
            with st.spinner("Checking API health..."):
                try:
                    response = requests.get(f"{API_BASE_URL}/health", timeout=5)
                    st.session_state['health_response'] = response.json()
                    st.session_state['health_status'] = response.status_code
                except Exception as e:
                    st.session_state['health_error'] = str(e)

    with col2:
        if 'health_status' in st.session_state:
            if st.session_state['health_status'] == 200:
                health = st.session_state['health_response']
                st.success(f"✓ API Healthy | Model: {health['model_loaded']} | MediaPipe: {health['mediapipe_available']}")
            else:
                st.error(f"✗ API Error: {st.session_state['health_status']}")
        elif 'health_error' in st.session_state:
            st.error(f"✗ Connection Failed: {st.session_state['health_error']}")

    st.divider()

    # Endpoint selector
    endpoint = st.selectbox(
        "Select Endpoint",
        [
            "POST /beautify/base64",
            "POST /detect/yolo/base64",
            "POST /detect/mediapipe/base64",
            "POST /adjust/thickness/increase",
            "POST /adjust/thickness/decrease",
            "POST /adjust/span/increase",
            "POST /adjust/span/decrease",
            "POST /beautify/submit-edit",
            "GET /config",
            "POST /config"
        ]
    )

    # Image upload for endpoints that need it
    if endpoint.startswith("POST /beautify") or endpoint.startswith("POST /detect"):
        uploaded_file = st.file_uploader("Upload Test Image", type=['jpg', 'png', 'jpeg'])

        if uploaded_file:
            # Show preview
            col1, col2 = st.columns(2)

            with col1:
                st.image(uploaded_file, caption="Test Image", use_container_width=True)

            with col2:
                if st.button("🚀 Execute Request", use_container_width=True, type="primary"):
                    execute_api_request(endpoint, uploaded_file)

    elif endpoint.startswith("POST /adjust"):
        st.info("💡 Tip: First run /beautify to get a mask, then test adjustment endpoints")

        if 'last_beautify_result' in st.session_state:
            result = st.session_state['last_beautify_result']

            if result.get('eyebrows'):
                selected_eyebrow = st.selectbox(
                    "Select Eyebrow to Adjust",
                    [f"{eb['side'].title()} Eyebrow" for eb in result['eyebrows']]
                )

                idx = 0 if "Left" in selected_eyebrow else 1
                mask_b64 = result['eyebrows'][idx]['final_mask_base64']
                side = result['eyebrows'][idx]['side']

                col1, col2 = st.columns(2)

                with col1:
                    increment = st.slider("Adjustment Increment", 0.01, 0.20, 0.05, 0.01)
                    num_clicks = st.slider("Number of Clicks", 1, 10, 1)

                with col2:
                    if st.button("🚀 Execute Adjustment", use_container_width=True, type="primary"):
                        execute_adjustment_request(endpoint, mask_b64, side, increment, num_clicks)
        else:
            st.warning("⚠️ No beautify results available. Run /beautify first.")

    elif endpoint == "GET /config":
        if st.button("🚀 Get Config", use_container_width=True, type="primary"):
            try:
                response = requests.get(f"{API_BASE_URL}/config")
                st.session_state['config_response'] = response.json()
                st.session_state['config_status'] = response.status_code
            except Exception as e:
                st.error(f"Error: {e}")

    elif endpoint == "POST /config":
        st.markdown("**Update Configuration**")

        # Get current config first
        try:
            current_config = requests.get(f"{API_BASE_URL}/config").json()

            with st.expander("🔧 Edit Configuration", expanded=True):
                new_config = {}

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Detection")
                    new_config['yolo_conf_threshold'] = st.slider(
                        "YOLO Confidence", 0.1, 0.9,
                        current_config.get('yolo_conf_threshold', 0.25), 0.05
                    )
                    new_config['mediapipe_conf_threshold'] = st.slider(
                        "MediaPipe Confidence", 0.1, 0.9,
                        current_config.get('mediapipe_conf_threshold', 0.3), 0.05
                    )

                with col2:
                    st.subheader("Validation")
                    new_config['min_mp_coverage'] = st.slider(
                        "Min MP Coverage (%)", 50.0, 100.0,
                        current_config.get('min_mp_coverage', 80.0), 5.0
                    )

                if st.button("📤 Update Config", type="primary"):
                    try:
                        response = requests.post(f"{API_BASE_URL}/config", json=new_config)
                        st.success(f"✓ Config updated: {response.status_code}")
                        st.session_state['config_response'] = response.json()
                    except Exception as e:
                        st.error(f"Error: {e}")

        except Exception as e:
            st.error(f"Failed to get current config: {e}")

    # Show response
    if 'api_response' in st.session_state:
        st.divider()
        st.subheader("📥 Response")

        col1, col2 = st.columns([1, 3])

        with col1:
            status = st.session_state.get('api_status', 0)
            if status == 200:
                st.success(f"Status: {status} OK")
            elif status in [400, 422]:
                st.warning(f"Status: {status}")
            else:
                st.error(f"Status: {status}")

        with col2:
            st.caption(f"Time: {st.session_state.get('api_time', 0):.1f}ms")

        # Show JSON response
        with st.expander("📄 Raw JSON Response", expanded=False):
            st.json(st.session_state['api_response'])

        # Show visualizations based on endpoint type
        if status == 200:
            if endpoint.startswith("POST /beautify"):
                show_beautify_visualization(st.session_state['api_response'])
            elif endpoint == "POST /detect/yolo/base64":
                show_yolo_visualization(st.session_state['api_response'])
            elif endpoint == "POST /detect/mediapipe/base64":
                show_mediapipe_visualization(st.session_state['api_response'])


def execute_api_request(endpoint, uploaded_file):
    """Execute API request and store results."""
    try:
        # Convert image to base64
        file_bytes = uploaded_file.read()
        img_b64 = image_to_base64(cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR))

        start_time = time.time()

        if endpoint == "POST /beautify/base64":
            payload = {
                'image_base64': img_b64,
                'return_masks': True
            }
            response = requests.post(f"{API_BASE_URL}/beautify/base64", json=payload)

        elif endpoint == "POST /detect/yolo/base64":
            payload = {
                'image_base64': img_b64,
                'return_masks': True
            }
            response = requests.post(f"{API_BASE_URL}/detect/yolo/base64", json=payload)

        elif endpoint == "POST /detect/mediapipe/base64":
            payload = {
                'image_base64': img_b64
            }
            response = requests.post(f"{API_BASE_URL}/detect/mediapipe/base64", json=payload)

        elif endpoint == "POST /beautify/submit-edit":
            # For submit-edit, we need the edited mask as well
            # This would typically come from a drawing tool, but for testing we can use the uploaded image
            payload = {
                'image_base64': img_b64,
                'edited_mask_base64': img_b64,  # Placeholder - in real use, this comes from editor
                'side': 'left',  # Default for testing
                'metadata': {'editor': 'developer_corner', 'timestamp': time.time()}
            }
            response = requests.post(f"{API_BASE_URL}/beautify/submit-edit", json=payload)

        elapsed = (time.time() - start_time) * 1000

        st.session_state['api_response'] = response.json()
        st.session_state['api_status'] = response.status_code
        st.session_state['api_time'] = elapsed

        if endpoint == "POST /beautify/base64" and response.status_code == 200:
            st.session_state['last_beautify_result'] = response.json()

        st.rerun()

    except Exception as e:
        st.error(f"Request failed: {e}")


def execute_adjustment_request(endpoint, mask_b64, side, increment, num_clicks):
    """Execute adjustment request."""
    try:
        # Determine endpoint
        if "thickness/increase" in endpoint:
            url = f"{API_BASE_URL}/adjust/thickness/increase"
        elif "thickness/decrease" in endpoint:
            url = f"{API_BASE_URL}/adjust/thickness/decrease"
        elif "span/increase" in endpoint:
            url = f"{API_BASE_URL}/adjust/span/increase"
        elif "span/decrease" in endpoint:
            url = f"{API_BASE_URL}/adjust/span/decrease"

        payload = {
            'mask_base64': mask_b64,
            'side': side,
            'increment': increment,
            'num_clicks': num_clicks
        }

        start_time = time.time()
        response = requests.post(url, json=payload)
        elapsed = (time.time() - start_time) * 1000

        st.session_state['api_response'] = response.json()
        st.session_state['api_status'] = response.status_code
        st.session_state['api_time'] = elapsed

        st.rerun()

    except Exception as e:
        st.error(f"Request failed: {e}")


def show_beautify_visualization(result):
    """Show visualization of beautify results."""
    st.subheader("🎨 Mask Visualization")

    for eyebrow in result.get('eyebrows', []):
        st.markdown(f"### {eyebrow['side'].title()} Eyebrow")

        col1, col2, col3 = st.columns(3)

        with col1:
            if eyebrow.get('original_mask_base64'):
                original_mask = base64_to_mask(eyebrow['original_mask_base64'])
                st.image(original_mask * 255, caption="Original YOLO Mask", use_container_width=True)

        with col2:
            if eyebrow.get('final_mask_base64'):
                final_mask = base64_to_mask(eyebrow['final_mask_base64'])
                st.image(final_mask * 255, caption="Final Beautified Mask", use_container_width=True)

        with col3:
            # Show difference
            if eyebrow.get('original_mask_base64') and eyebrow.get('final_mask_base64'):
                original = base64_to_mask(eyebrow['original_mask_base64'])
                final = base64_to_mask(eyebrow['final_mask_base64'])

                diff = np.zeros((original.shape[0], original.shape[1], 3), dtype=np.uint8)
                diff[:, :, 1] = final * 255  # Green = added
                diff[:, :, 0] = original * 255  # Blue = original

                st.image(diff, caption="Difference (Green=Added, Blue=Original)", use_container_width=True)

        # Show validation
        with st.expander(f"📊 Validation Metrics - {eyebrow['side'].title()}", expanded=False):
            display_validation_metrics(eyebrow['validation'], eyebrow['side'])


def show_yolo_visualization(result):
    """Show visualization of YOLO detection results."""
    st.subheader("🎯 YOLO Detections")

    detections = result.get('detections', {})

    if not detections:
        st.info("No detections found")
        return

    # Show statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Eyebrows", len(detections.get('eyebrows', [])))
    with col2:
        st.metric("Eyes", len(detections.get('eye', [])))
    with col3:
        st.metric("Eye Boxes", len(detections.get('eye_box', [])))
    with col4:
        st.metric("Hair", len(detections.get('hair', [])))

    # Show masks for each class
    for class_name, class_detections in detections.items():
        if class_detections:
            with st.expander(f"📦 {class_name.title()} ({len(class_detections)} detected)", expanded=(class_name == 'eyebrows')):
                for i, detection in enumerate(class_detections):
                    st.markdown(f"**Detection {i+1}** - Confidence: {detection['confidence']:.3f}")

                    col1, col2 = st.columns(2)

                    with col1:
                        if detection.get('mask_base64'):
                            mask = base64_to_mask(detection['mask_base64'])
                            st.image(mask * 255, caption=f"{class_name} mask", use_container_width=True)

                    with col2:
                        st.json({
                            'class': detection['class_name'],
                            'confidence': detection['confidence'],
                            'box': detection['box'],
                            'area': detection['mask_area'],
                            'centroid': detection['mask_centroid']
                        })


def show_mediapipe_visualization(result):
    """Show visualization of MediaPipe landmark results."""
    st.subheader("🗺️ MediaPipe Landmarks")

    landmarks = result.get('landmarks')

    if not landmarks:
        st.info("No face landmarks detected")
        return

    # Show statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Left Eyebrow", len(landmarks.get('left_eyebrow', {}).get('points', [])) if 'left_eyebrow' in landmarks else 0)
    with col2:
        st.metric("Right Eyebrow", len(landmarks.get('right_eyebrow', {}).get('points', [])) if 'right_eyebrow' in landmarks else 0)
    with col3:
        st.metric("Left Eye", len(landmarks.get('left_eye', {}).get('points', [])) if 'left_eye' in landmarks else 0)
    with col4:
        st.metric("Right Eye", len(landmarks.get('right_eye', {}).get('points', [])) if 'right_eye' in landmarks else 0)

    # Show landmark details
    for feature_name, feature_data in landmarks.items():
        with st.expander(f"📍 {feature_name.replace('_', ' ').title()}", expanded=(feature_name in ['left_eyebrow', 'right_eyebrow'])):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Points**")
                points_df = {
                    'Index': feature_data.get('indices', []),
                    'X': [p[0] for p in feature_data.get('points', [])],
                    'Y': [p[1] for p in feature_data.get('points', [])]
                }
                st.dataframe(points_df, hide_index=True)

            with col2:
                st.json({
                    'center': feature_data.get('center'),
                    'bbox': feature_data.get('bbox'),
                    'num_points': len(feature_data.get('points', []))
                })


# =============================================================================
# TEST RUNNER
# =============================================================================

def render_test_runner():
    """Run automated test suites."""
    st.header("🧪 Test Suite Runner")
    st.markdown("Execute automated tests and view results")

    # List available tests (all 13 test files)
    test_files = [
        "tests/run_all_tests.py",  # Run all tests at once
        "tests/test_developer_corner_e2e.py",  # Developer Corner E2E test
        "tests/test_critical_fixes.py",  # Critical system fixes
        "tests/test_api_endpoints.py",  # API endpoint tests
        "tests/test_integration.py",  # Integration tests
        "tests/test_adjustment_api.py",  # Adjustment API tests
        "tests/test_adjustments.py",  # Adjustment logic tests
        "tests/test_model_loading.py",  # Model loading tests
        "tests/test_config.py",  # Configuration tests
        "tests/test_smooth_normal.py",  # Smoothing algorithm tests
        "tests/test_statistical.py",  # Statistical validation tests
        "tests/test_visual.py"  # Visual output tests
    ]

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_test = st.selectbox(
            "Select Test Suite",
            test_files,
            format_func=lambda x: x.split('/')[-1]
        )

    with col2:
        run_test = st.button("▶️ Run Test", use_container_width=True, type="primary")

    if run_test:
        run_test_suite(selected_test)

    # Show last test results
    if 'test_output' in st.session_state:
        st.divider()
        st.subheader("📋 Test Results")

        output = st.session_state['test_output']
        returncode = st.session_state.get('test_returncode', 0)

        if returncode == 0:
            st.success("✓ All tests passed!")
        else:
            st.error("✗ Some tests failed")

        with st.expander("📄 Full Output", expanded=True):
            st.code(output, language="text")


def run_test_suite(test_file):
    """Execute test suite and capture output."""
    with st.spinner(f"Running {test_file}..."):
        try:
            result = subprocess.run(
                ['python3', test_file],
                capture_output=True,
                text=True,
                timeout=120
            )

            st.session_state['test_output'] = result.stdout + result.stderr
            st.session_state['test_returncode'] = result.returncode
            st.session_state['test_file'] = test_file

            st.rerun()

        except subprocess.TimeoutExpired:
            st.error("Test execution timeout (120s)")
        except Exception as e:
            st.error(f"Test execution failed: {e}")


# =============================================================================
# MASK VISUALIZER
# =============================================================================

def render_visualizer():
    """Comprehensive mask and pipeline visualization."""
    st.header("📊 Pipeline Visualizer")
    st.markdown("Step-by-step visualization of the beautification pipeline")

    uploaded_file = st.file_uploader("Upload Image for Visualization", type=['jpg', 'png', 'jpeg'], key="viz_upload")

    if uploaded_file:
        # Process image
        file_bytes = uploaded_file.read()
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)

        with col1:
            st.image(cv2_to_pil(img), caption="Original Image", use_container_width=True)

        with col2:
            if st.button("🔍 Analyze Pipeline", use_container_width=True, type="primary"):
                analyze_pipeline(img)

        # Show pipeline steps
        if 'pipeline_analysis' in st.session_state:
            show_pipeline_steps(st.session_state['pipeline_analysis'])


def analyze_pipeline(img):
    """Analyze full pipeline and store intermediate results."""
    client = APIClient(API_BASE_URL)

    with st.spinner("Analyzing pipeline..."):
        # Convert to base64
        img_b64 = image_to_base64(img)

        # Call beautify API
        try:
            result = client.beautify_image(img_b64)

            st.session_state['pipeline_analysis'] = {
                'success': result.get('success'),
                'eyebrows': result.get('eyebrows', []),
                'processing_time': result.get('processing_time_ms'),
                'image': img
            }

            st.rerun()

        except Exception as e:
            st.error(f"Pipeline analysis failed: {e}")


def show_pipeline_steps(analysis):
    """Display pipeline steps with visualizations."""
    st.divider()
    st.subheader("📈 Pipeline Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Status", "✓ Success" if analysis['success'] else "✗ Failed")

    with col2:
        st.metric("Eyebrows Detected", len(analysis['eyebrows']))

    with col3:
        st.metric("Processing Time", f"{analysis['processing_time']:.0f}ms")

    # Show each eyebrow
    for eyebrow in analysis['eyebrows']:
        st.markdown(f"### {eyebrow['side'].title()} Eyebrow")

        # Metadata
        with st.expander("📋 Metadata", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("YOLO Confidence", f"{eyebrow['metadata']['yolo_confidence']:.3f}")
                st.metric("YOLO Area", f"{eyebrow['metadata']['yolo_area']:,} px")

            with col2:
                st.metric("Final Area", f"{eyebrow['metadata']['final_area']:,} px")
                st.metric("Has Eye", "✓" if eyebrow['metadata']['has_eye'] else "✗")

            with col3:
                st.metric("Has MediaPipe", "✓" if eyebrow['metadata']['has_mediapipe'] else "✗")
                st.metric("Hair Regions", eyebrow['metadata']['hair_regions'])

        # Validation
        with st.expander("✅ Validation", expanded=True):
            display_validation_metrics(eyebrow['validation'], eyebrow['side'])

        # Masks
        st.markdown("**Mask Evolution**")

        col1, col2 = st.columns(2)

        with col1:
            if eyebrow.get('original_mask_base64'):
                original = base64_to_mask(eyebrow['original_mask_base64'])
                st.image(original * 255, caption="Phase 2: YOLO Detection", use_container_width=True)

        with col2:
            if eyebrow.get('final_mask_base64'):
                final = base64_to_mask(eyebrow['final_mask_base64'])
                st.image(final * 255, caption="Phase 7: Final Beautified", use_container_width=True)

        # Overlay on original
        if eyebrow.get('final_mask_base64'):
            final_mask = base64_to_mask(eyebrow['final_mask_base64'])
            color = (255, 0, 0) if eyebrow['side'] == 'left' else (0, 0, 255)
            overlay = overlay_mask_on_image(analysis['image'], final_mask, color, alpha=0.5)

            st.image(cv2_to_pil(overlay), caption=f"Final Result - {eyebrow['side'].title()}", use_container_width=True)


# =============================================================================
# LOG VIEWER
# =============================================================================

def render_log_viewer():
    """Real-time API log viewer."""
    st.header("📝 API Logs")
    st.markdown("View real-time API server logs")

    col1, col2 = st.columns([3, 1])

    with col1:
        log_lines = st.slider("Number of log lines", 10, 500, 100, 10)

    with col2:
        refresh = st.button("🔄 Refresh Logs", use_container_width=True)

    if refresh or 'api_logs' not in st.session_state:
        fetch_api_logs(log_lines)

    if 'api_logs' in st.session_state:
        with st.expander("📄 API Server Logs", expanded=True):
            st.code(st.session_state['api_logs'], language="log")


def fetch_api_logs(num_lines):
    """Fetch API logs (simulated - would tail actual log file in production)."""
    try:
        # Try to get recent uvicorn logs
        result = subprocess.run(
            ['tail', '-n', str(num_lines), 'api_logs.txt'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            st.session_state['api_logs'] = result.stdout
        else:
            st.session_state['api_logs'] = "No logs available (api_logs.txt not found)\nNote: Logs are printed to console when running API"

    except Exception as e:
        st.session_state['api_logs'] = f"Unable to fetch logs: {e}\n\nRun API with: python3 -m uvicorn api.api_main:app --reload --host 0.0.0.0 --port 8000 2>&1 | tee api_logs.txt"


# =============================================================================
# CONFIG PLAYGROUND
# =============================================================================

def render_config_playground():
    """Interactive configuration playground."""
    st.header("⚙️ Configuration Playground")
    st.markdown("Test different configurations and compare results")

    st.info("💡 Upload an image, adjust config parameters, and see how they affect the pipeline")

    uploaded_file = st.file_uploader("Upload Test Image", type=['jpg', 'png', 'jpeg'], key="config_upload")

    if uploaded_file:
        file_bytes = uploaded_file.read()
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

        st.image(cv2_to_pil(img), caption="Test Image", width=400)

        # Configuration sliders
        with st.expander("🔧 Configuration Parameters", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Detection")
                yolo_conf = st.slider("YOLO Confidence", 0.1, 0.9, 0.25, 0.05)
                mp_conf = st.slider("MediaPipe Confidence", 0.1, 0.9, 0.3, 0.05)

                st.subheader("Validation")
                min_mp_cov = st.slider("Min MP Coverage (%)", 50.0, 100.0, 80.0, 5.0)

            with col2:
                st.subheader("Eye Exclusion")
                eye_buffer_size = st.slider("Eye Buffer Kernel", 5, 25, 15, 2)
                eye_buffer_iter = st.slider("Eye Buffer Iterations", 1, 5, 2, 1)

                st.subheader("Smoothing")
                gaussian_sigma = st.slider("Gaussian Sigma", 0.5, 5.0, 2.0, 0.5)

        if st.button("🚀 Run with Custom Config", type="primary"):
            run_with_custom_config(img, {
                'yolo_conf_threshold': yolo_conf,
                'mediapipe_conf_threshold': mp_conf,
                'min_mp_coverage': min_mp_cov,
                'eye_buffer_kernel': (eye_buffer_size, eye_buffer_size),
                'eye_buffer_iterations': eye_buffer_iter,
                'gaussian_sigma': gaussian_sigma
            })

        # Show comparison
        if 'config_results' in st.session_state:
            show_config_comparison(st.session_state['config_results'])


def run_with_custom_config(img, config):
    """Run pipeline with custom configuration."""
    client = APIClient(API_BASE_URL)

    with st.spinner("Running with custom config..."):
        try:
            img_b64 = image_to_base64(img)

            # Update config first
            requests.post(f"{API_BASE_URL}/config", json=config)

            # Run beautify
            result = client.beautify_image(img_b64)

            st.session_state['config_results'] = {
                'config': config,
                'result': result,
                'image': img
            }

            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")


def show_config_comparison(results):
    """Show results with custom config."""
    st.divider()
    st.subheader("📊 Results with Custom Config")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Configuration Used:**")
        st.json(results['config'])

    with col2:
        st.markdown("**Results:**")
        st.metric("Success", "✓" if results['result'].get('success') else "✗")
        st.metric("Eyebrows", len(results['result'].get('eyebrows', [])))
        st.metric("Time", f"{results['result'].get('processing_time_ms', 0):.0f}ms")

    # Show visualizations
    for eyebrow in results['result'].get('eyebrows', []):
        with st.expander(f"{eyebrow['side'].title()} Eyebrow", expanded=True):
            display_validation_metrics(eyebrow['validation'], eyebrow['side'])


# =============================================================================
# PREPROCESSING TAB
# =============================================================================

def render_preprocessing_tab():
    """
    Face Preprocessing Analyzer.
    
    Interactive testing of the preprocessing pipeline:
    - Face validation (eyes, eyebrows, quality)
    - Rotation angle detection (multi-source)
    - Asymmetry detection (angle, position, span)
    - Rotation correction visualization
    """
    st.header("🔍 Face Preprocessing Analyzer")
    st.markdown("**Comprehensive face validation, rotation detection, and asymmetry analysis**")
    
    # Image upload
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['png', 'jpg', 'jpeg'],
        key="preprocess_uploader"
    )
    
    if not uploaded_file:
        st.info("👆 Upload an image to analyze face preprocessing")
        return
    
    # Convert to base64
    image_b64 = image_to_base64(uploaded_file)
    
    # Display original image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Original Image**")
        st.image(uploaded_file, use_container_width=True)
    
    with col2:
        st.markdown("**Preprocessing Settings**")
        min_rotation_threshold = st.slider(
            "Minimum Rotation Threshold (degrees)",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.5,
            help="Only correct rotation if angle exceeds this threshold"
        )
        
        reject_invalid = st.checkbox(
            "Reject invalid faces",
            value=True,
            help="Fail preprocessing if validation checks fail"
        )
    
    # Run preprocessing button
    if st.button("🔍 Analyze Preprocessing", use_container_width=True, type="primary"):
        with st.spinner("Running face preprocessing..."):
            try:
                # Call preprocessing endpoint
                api_client = APIClient()
                config = {
                    'min_rotation_threshold': min_rotation_threshold,
                    'reject_invalid_faces': reject_invalid
                }
                
                response = requests.post(
                    f"{API_BASE_URL}/preprocess",
                    json={"image_base64": image_b64, "config": config},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state['preprocess_result'] = result
                    st.success("✓ Preprocessing completed!")
                else:
                    st.error(f"✗ Error: {response.status_code} - {response.text}")
                    return
                    
            except Exception as e:
                st.error(f"✗ Error: {str(e)}")
                return
    
    # Display results
    if 'preprocess_result' not in st.session_state:
        return
    
    result = st.session_state['preprocess_result']
    
    st.divider()
    
    # Overall status
    st.subheader("📊 Preprocessing Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_icon = "✅" if result['valid'] else "❌"
        st.metric("Face Valid", status_icon, delta=None)
    
    with col2:
        angle_text = f"{result['rotation_angle']:.2f}°" if result['rotation_angle'] else "N/A"
        st.metric("Rotation Angle", angle_text)
    
    with col3:
        corrected_icon = "✅" if result.get('rotation_corrected') else "❌"
        st.metric("Rotation Corrected", corrected_icon)
    
    with col4:
        asymmetry = result['asymmetry_detection'].get('has_asymmetry', False)
        asym_icon = "⚠️" if asymmetry else "✅"
        st.metric("Asymmetry", asym_icon)
    
    st.divider()
    
    # Validation details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👁️ Eye Validation")
        eye_val = result['eye_validation']
        st.write(f"**Status**: {eye_val['status']}")
        st.write(f"**MediaPipe**: {'✓' if eye_val.get('mediapipe_has_eyes') else '✗'}")
        st.write(f"**YOLO**: {'✓' if eye_val.get('yolo_has_eyes') else '✗'} ({eye_val.get('yolo_eye_count', 0)} eyes)")
        if eye_val.get('eye_distance_pct'):
            st.write(f"**Distance**: {eye_val['eye_distance_pct']*100:.1f}% of width")
        if eye_val.get('vertical_diff_pct') is not None:
            st.write(f"**Vertical Diff**: {eye_val['vertical_diff_pct']*100:.1f}% of height")
    
    with col2:
        st.markdown("### ✨ Eyebrow Validation")
        eb_val = result['eyebrow_validation']
        st.write(f"**Status**: {eb_val['status']}")
        st.write(f"**YOLO Count**: {eb_val.get('yolo_eyebrow_count', 0)}")
        st.write(f"**MediaPipe**: {'✓' if eb_val.get('mediapipe_has_eyebrows') else '✗'}")
        if eb_val.get('eyebrow_eye_overlaps'):
            overlaps = eb_val['eyebrow_eye_overlaps']
            st.write(f"**Eye Overlap**: {max(overlaps)*100:.1f}% max")
    
    with col3:
        st.markdown("### 📐 Quality Validation")
        qual_val = result['quality_validation']
        st.write(f"**Status**: {qual_val['status']}")
        if qual_val.get('image_size'):
            st.write(f"**Size**: {qual_val['image_size'][0]}x{qual_val['image_size'][1]}")
        st.write(f"**MediaPipe**: {'✓' if qual_val.get('mediapipe_detected') else '✗'}")
        if qual_val.get('yolo_avg_confidence'):
            st.write(f"**YOLO Conf**: {qual_val['yolo_avg_confidence']*100:.0f}%")
    
    st.divider()
    
    # Rotation angle metadata
    st.markdown("### 🔄 Rotation Angle Detection")
    angle_meta = result['angle_metadata']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Final Angle**: {angle_meta['final_angle']:.2f}°" if angle_meta['final_angle'] else "**Final Angle**: N/A")
        st.write(f"**Status**: {angle_meta['status']}")
        st.write(f"**Sources Used**: {angle_meta['num_sources']}")
        
    with col2:
        if angle_meta.get('all_sources'):
            st.write("**All Sources**:")
            for i, source in enumerate(angle_meta['all_sources']):
                angle = angle_meta['all_angles'][i] if i < len(angle_meta['all_angles']) else 0
                st.write(f"  - {source}: {angle:.2f}°")
        
        st.write(f"**Outliers Removed**: {angle_meta.get('outliers_removed', 0)}")
        if angle_meta.get('angle_std'):
            st.write(f"**Std Dev**: {angle_meta['angle_std']:.2f}°")
    
    # Asymmetry detection
    if result['asymmetry_detection'].get('has_asymmetry'):
        st.divider()
        st.markdown("### ⚠️ Asymmetry Detection")
        
        asymm = result['asymmetry_detection']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if asymm.get('angle_asymmetry'):
                st.warning("**Angle Asymmetry**")
                st.write(f"Left: {asymm.get('left_angle', 0):.1f}°")
                st.write(f"Right: {asymm.get('right_angle', 0):.1f}°")
                st.write(f"Diff: {asymm.get('angle_difference', 0):.1f}°")
        
        with col2:
            if asymm.get('position_asymmetry'):
                st.warning("**Position Asymmetry**")
                st.write(f"Diff: {asymm.get('position_difference_pct', 0)*100:.1f}% of height")
        
        with col3:
            if asymm.get('span_asymmetry'):
                st.warning("**Span Asymmetry**")
                st.write(f"Diff: {asymm.get('span_difference_pct', 0)*100:.1f}% of width")
    
    # Warnings
    if result.get('warnings'):
        st.divider()
        st.markdown("### ⚠️ Warnings")
        for warning in result['warnings']:
            st.warning(warning)
    
    # Processing time
    st.divider()
    st.metric("⏱️ Processing Time", f"{result['processing_time_ms']:.2f}ms")
    
    # Full report
    with st.expander("📄 Full Preprocessing Report", expanded=False):
        st.code(result['report'], language=None)

