"""
Eyebrow Beautification Streamlit App

User-friendly web interface for eyebrow detection, beautification, and editing.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Optional, Tuple
import time

# Import custom modules
from streamlit_api_client import get_api_client
from streamlit_utils import (
    pil_to_cv2, cv2_to_pil, image_to_base64, base64_to_image,
    mask_to_base64, base64_to_mask, overlay_mask_on_image,
    draw_mediapipe_points, create_comparison_view,
    display_validation_metrics, display_statistics,
    resize_image_if_needed, apply_rotation_to_mask,
    apply_scale_to_mask, apply_translation_to_mask,
    create_download_data, show_error, show_success,
    show_info, show_warning
)
from streamlit_config import (
    COLORS, SESSION_KEYS, MESSAGES, FEATURES,
    MAX_IMAGE_SIZE, THICKNESS_INCREMENT, SPAN_INCREMENT,
    ROTATION_RANGE, SCALE_RANGE, SD_DEFAULTS
)
from streamlit_developer import render_developer_corner


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Eyebrow Beautification",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state with default values."""
    for key, default_value in SESSION_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


init_session_state()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def reset_session():
    """Reset all session state to defaults."""
    for key, default_value in SESSION_KEYS.items():
        st.session_state[key] = default_value
    st.rerun()


def check_api_connection():
    """Check if API is healthy."""
    try:
        client = get_api_client()
        health = client.check_health()
        st.session_state.api_healthy = health.get('model_loaded', False)
        return st.session_state.api_healthy
    except Exception as e:
        st.session_state.api_healthy = False
        show_error(f"API connection failed: {str(e)}")
        return False


def process_uploaded_image(uploaded_file):
    """Process uploaded image and run beautification."""
    try:
        # Load image
        pil_image = Image.open(uploaded_file)
        cv2_image = pil_to_cv2(pil_image)

        # Resize if needed
        cv2_image = resize_image_if_needed(cv2_image, MAX_IMAGE_SIZE)

        # Store in session state
        st.session_state.original_image = cv2_image
        st.session_state.original_image_b64 = image_to_base64(cv2_image)

        # Get API client
        client = get_api_client()

        # Run YOLO detection
        with st.spinner("Running YOLO detection..."):
            yolo_result = client.detect_yolo(st.session_state.original_image_b64)
            st.session_state.yolo_detections = yolo_result.get('detections', {})

        # Run MediaPipe detection
        with st.spinner("Running MediaPipe detection..."):
            mp_result = client.detect_mediapipe(st.session_state.original_image_b64)
            st.session_state.mediapipe_landmarks = mp_result.get('landmarks', {})

        # Run beautification
        with st.spinner("Beautifying eyebrows..."):
            beautify_result = client.beautify_image(
                st.session_state.original_image_b64,
                return_masks=True
            )

            if beautify_result.get('success') and beautify_result.get('eyebrows'):
                st.session_state.eyebrows = beautify_result['eyebrows']

                # Initialize current masks
                for eyebrow in st.session_state.eyebrows:
                    side = eyebrow['side']
                    st.session_state.current_masks[side] = {
                        'mask_b64': eyebrow['final_mask_base64'],
                        'transform': {'rotation': 0, 'scale': 1.0, 'dx': 0, 'dy': 0}
                    }

                show_success(MESSAGES['success'])
            else:
                show_warning(MESSAGES['no_eyebrows'])

    except Exception as e:
        show_error(f"Processing failed: {str(e)}")


def get_current_mask(side: str) -> Optional[np.ndarray]:
    """Get current mask for a side as numpy array."""
    if st.session_state.current_masks.get(side):
        mask_b64 = st.session_state.current_masks[side]['mask_b64']
        return base64_to_mask(mask_b64)
    return None


def update_current_mask(side: str, new_mask: np.ndarray):
    """Update current mask for a side."""
    if st.session_state.current_masks.get(side):
        st.session_state.current_masks[side]['mask_b64'] = mask_to_base64(new_mask)
        # Add to history for undo
        st.session_state.edit_history.append({
            'side': side,
            'mask_b64': mask_to_base64(new_mask),
            'timestamp': time.time()
        })


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main Streamlit app."""

    # Header
    st.title("✨ Eyebrow Beautification System")
    st.markdown("**Multi-Source Fusion Algorithm with Interactive Editing**")

    # Mode selector (User vs Developer)
    mode = st.radio(
        "Select Mode",
        ["👤 User Mode", "🛠️ Developer Corner"],
        horizontal=True,
        help="User Mode: Beautify your eyebrows | Developer Corner: Test APIs, view logs, debug pipeline"
    )

    # If developer mode, render developer corner and exit
    if mode == "🛠️ Developer Corner":
        render_developer_corner()
        return

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # API connection status
        st.subheader("Connection Status")
        if st.button("Check API Connection"):
            if check_api_connection():
                show_success("API is healthy!")
            else:
                show_error(MESSAGES['api_error'])

        st.markdown(f"**Status:** {'✓ Connected' if st.session_state.api_healthy else '✗ Disconnected'}")

        st.divider()

        # Reset button
        if st.button("🔄 Reset All", type="secondary"):
            reset_session()

        st.divider()

        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Eyebrow Beautification System**

            This app uses:
            - YOLO for eyebrow detection
            - MediaPipe for facial landmarks
            - Multi-source fusion algorithm
            - Stable Diffusion for enhancement (Phase 2)

            **Workflow:**
            1. Upload image
            2. View detection results
            3. Edit eyebrows (auto or manual)
            4. Finalize and enhance with SD
            5. Download result
            """)

    # Main content area
    if not st.session_state.api_healthy:
        show_warning("⚠️ Please check API connection before proceeding.")
        if st.button("Connect to API"):
            check_api_connection()

    # Step 1: Upload Image (Full Width)
    st.header("📤 Upload Image")

    uploaded_file = st.file_uploader(
        MESSAGES['upload_prompt'],
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=False
    )

    if uploaded_file is not None and st.session_state.original_image is None:
        process_uploaded_image(uploaded_file)

    # Main Editing Interface (Side-by-Side Layout)
    if st.session_state.original_image is not None and st.session_state.eyebrows:

        st.divider()

        # Create side-by-side layout (2:1 ratio)
        col_preview, col_controls = st.columns([2, 1])

        # LEFT COLUMN: Live Preview
        with col_preview:
            render_live_preview()

        # RIGHT COLUMN: Edit Controls
        with col_controls:
            st.subheader("🎨 Edit Controls")

            # Tab-based edit modes
            tab_auto, tab_manual, tab_brush = st.tabs([
                "⚡ Auto Adjust",
                "🔧 Transform",
                "🎨 Brush & Eraser"
            ])

            with tab_auto:
                render_auto_edit_mode()

            with tab_manual:
                render_manual_edit_mode()

            with tab_brush:
                st.info("Brush & Eraser mode coming in Phase 3!")
                st.caption("Will allow pixel-level mask editing with brush/eraser tools.")

        # Bottom Section: Finalize & Download (Full Width)
        st.divider()

        col_bottom1, col_bottom2 = st.columns([1, 1])

        with col_bottom1:
            st.subheader("🚀 Finalize")

            col_btn1, col_btn2 = st.columns(2)

            with col_btn1:
                if st.button("✓ Finalize Masks", type="primary", use_container_width=True):
                    finalize_masks()

            with col_btn2:
                if st.button("✨ Enhance with AI", type="primary", use_container_width=True,
                            disabled=not st.session_state.finalized):
                    render_sd_enhancement()

            # Show finalized status
            if st.session_state.finalized:
                show_success(MESSAGES['finalized'])

        with col_bottom2:
            st.subheader("💾 Download")
            render_download_section()

        # Display SD result if available
        if st.session_state.sd_result:
            st.divider()
            st.subheader("🎯 SD Enhanced Result")
            sd_image = base64_to_image(st.session_state.sd_result)
            st.image(cv2_to_pil(sd_image), use_container_width=True)


# =============================================================================
# AUTO EDIT MODE
# =============================================================================

def render_auto_edit_mode():
    """Render auto edit mode with +/− buttons (vertical layout - no nested columns)."""
    st.caption("Adjust thickness and span with +/− buttons. Each click = 5% change.")

    client = get_api_client()

    # Select which eyebrow to edit (no nested columns!)
    edit_side = st.radio("Select Eyebrow:", ["left", "right"], horizontal=True, key="auto_edit_side")

    if get_current_mask(edit_side) is None:
        st.info(f"No {edit_side} eyebrow detected")
        return

    st.divider()

    # Thickness controls (simple horizontal buttons)
    st.write("**Thickness**")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("−", key=f"{edit_side}_thick_dec", use_container_width=True):
            adjust_eyebrow(client, edit_side, 'thickness', 'decrease')

    with col2:
        clicks = st.session_state.clicks[edit_side]['thickness']
        change = clicks * THICKNESS_INCREMENT * 100
        st.metric("", f"{change:+.0f}%", label_visibility="collapsed")

    with col3:
        if st.button("+", key=f"{edit_side}_thick_inc", use_container_width=True):
            adjust_eyebrow(client, edit_side, 'thickness', 'increase')

    # Span controls (simple horizontal buttons)
    st.write("**Span (Length)**")
    col4, col5, col6 = st.columns([1, 2, 1])

    with col4:
        if st.button("−", key=f"{edit_side}_span_dec", use_container_width=True):
            adjust_eyebrow(client, edit_side, 'span', 'decrease')

    with col5:
        clicks = st.session_state.clicks[edit_side]['span']
        change = clicks * SPAN_INCREMENT * 100
        st.metric("", f"{change:+.0f}%", label_visibility="collapsed")

    with col6:
        if st.button("+", key=f"{edit_side}_span_inc", use_container_width=True):
            adjust_eyebrow(client, edit_side, 'span', 'increase')

    st.divider()

    # Reset button
    if st.button(f"🔄 Reset {edit_side.title()}", key=f"{edit_side}_reset_auto", use_container_width=True):
        reset_eyebrow(edit_side)

    # Show validation metrics
    eyebrow_data = next((eb for eb in st.session_state.eyebrows if eb['side'] == edit_side), None)
    if eyebrow_data:
        with st.expander(f"📊 Validation Metrics"):
            display_validation_metrics(eyebrow_data.get('validation', {}), edit_side)


def adjust_eyebrow(client, side: str, adjustment_type: str, direction: str):
    """Adjust eyebrow thickness or span."""
    try:
        current_mask_b64 = st.session_state.current_masks[side]['mask_b64']

        with st.spinner(f"Adjusting {adjustment_type}..."):
            if adjustment_type == 'thickness':
                result = client.adjust_thickness(
                    mask_base64=current_mask_b64,
                    side=side,
                    direction=direction,
                    increment=THICKNESS_INCREMENT,
                    num_clicks=1
                )
            else:  # span
                result = client.adjust_span(
                    mask_base64=current_mask_b64,
                    side=side,
                    direction=direction,
                    increment=SPAN_INCREMENT,
                    num_clicks=1
                )

            if result.get('success'):
                # Update mask
                new_mask = base64_to_mask(result['adjusted_mask_base64'])
                update_current_mask(side, new_mask)

                # Update click counter
                if direction == 'increase':
                    st.session_state.clicks[side][adjustment_type] += 1
                else:
                    st.session_state.clicks[side][adjustment_type] -= 1

                st.rerun()
            else:
                show_error(f"Adjustment failed: {result.get('message', 'Unknown error')}")

    except Exception as e:
        show_error(f"Adjustment failed: {str(e)}")


def reset_eyebrow(side: str):
    """Reset eyebrow to original beautified version."""
    eyebrow_data = next((eb for eb in st.session_state.eyebrows if eb['side'] == side), None)
    if eyebrow_data:
        st.session_state.current_masks[side]['mask_b64'] = eyebrow_data['final_mask_base64']
        st.session_state.current_masks[side]['transform'] = {'rotation': 0, 'scale': 1.0, 'dx': 0, 'dy': 0}
        st.session_state.clicks[side] = {'thickness': 0, 'span': 0}
        st.rerun()


# =============================================================================
# LIVE PREVIEW PANEL
# =============================================================================

def render_live_preview():
    """Render live preview panel with multiple display modes."""

    st.subheader("👁️ Live Preview")

    # Preview controls
    col_mode, col_opacity = st.columns([3, 2])

    with col_mode:
        preview_mode = st.selectbox(
            "View Mode",
            ["Overlay", "Side-by-Side", "Difference"],
            key="preview_mode",
            help="Overlay: Current masks on image | Side-by-Side: Before/After | Difference: Changed pixels"
        )

    with col_opacity:
        opacity = st.slider(
            "Opacity",
            0.0, 1.0,
            st.session_state.preview_opacity,
            0.1,
            key="opacity_slider",
            help="Adjust mask overlay transparency"
        )
        st.session_state.preview_opacity = opacity

    # Get current masks
    left_mask = get_current_mask('left')
    right_mask = get_current_mask('right')
    original = st.session_state.original_image

    # Generate preview based on mode
    if preview_mode == "Overlay":
        # Show original image with current masks overlaid
        preview = original.copy()

        if left_mask is not None:
            preview = overlay_mask_on_image(preview, left_mask, COLORS['left_eyebrow'], opacity)

        if right_mask is not None:
            preview = overlay_mask_on_image(preview, right_mask, COLORS['right_eyebrow'], opacity)

        st.image(cv2_to_pil(preview), use_container_width=True, caption="Current Eyebrows (Red=Left, Blue=Right)")

    elif preview_mode == "Side-by-Side":
        # Show before (YOLO) vs after (current) comparison
        yolo_masks = {'left': None, 'right': None}
        current_masks = {'left': left_mask, 'right': right_mask}

        for eyebrow in st.session_state.eyebrows:
            side = eyebrow['side']
            if eyebrow.get('original_mask_base64'):
                yolo_masks[side] = base64_to_mask(eyebrow['original_mask_base64'])

        before_view, after_view = create_comparison_view(
            original,
            yolo_masks,
            current_masks,
            st.session_state.mediapipe_landmarks,
            alpha=opacity
        )

        # Show both views stacked vertically
        col_before, col_after = st.columns(2)

        with col_before:
            st.caption("🔍 Before (YOLO)")
            st.image(cv2_to_pil(before_view), use_container_width=True)

        with col_after:
            st.caption("✨ After (Current)")
            st.image(cv2_to_pil(after_view), use_container_width=True)

    elif preview_mode == "Difference":
        # Show difference map (added/removed pixels)
        diff_image = create_difference_map(left_mask, right_mask)
        st.image(cv2_to_pil(diff_image), use_container_width=True,
                caption="Difference Map (Green=Added, Red=Removed)")

    # Zoom controls
    st.caption("🔍 Zoom")
    zoom_col1, zoom_col2, zoom_col3 = st.columns([1, 2, 1])

    with zoom_col1:
        if st.button("➖", key="zoom_out", help="Zoom out", use_container_width=True):
            from streamlit_config import ZOOM_CONFIG
            st.session_state.zoom_level = max(
                ZOOM_CONFIG['min_zoom'],
                st.session_state.zoom_level - ZOOM_CONFIG['zoom_step']
            )
            st.rerun()

    with zoom_col2:
        zoom_pct = st.session_state.zoom_level * 100
        st.caption(f"**{zoom_pct:.0f}%**", help="Current zoom level")

    with zoom_col3:
        if st.button("➕", key="zoom_in", help="Zoom in", use_container_width=True):
            from streamlit_config import ZOOM_CONFIG
            st.session_state.zoom_level = min(
                ZOOM_CONFIG['max_zoom'],
                st.session_state.zoom_level + ZOOM_CONFIG['zoom_step']
            )
            st.rerun()


def create_difference_map(left_mask: Optional[np.ndarray], right_mask: Optional[np.ndarray]) -> np.ndarray:
    """
    Create difference map showing changes from original YOLO masks.

    Green = pixels added
    Red = pixels removed
    """
    # Get YOLO original masks
    yolo_left = None
    yolo_right = None

    for eyebrow in st.session_state.eyebrows:
        side = eyebrow['side']
        if eyebrow.get('original_mask_base64'):
            if side == 'left':
                yolo_left = base64_to_mask(eyebrow['original_mask_base64'])
            else:
                yolo_right = base64_to_mask(eyebrow['original_mask_base64'])

    # Create difference visualization
    diff_image = st.session_state.original_image.copy()

    # Left eyebrow difference
    if left_mask is not None and yolo_left is not None:
        # Added pixels (in current but not in YOLO)
        added = (left_mask > 0) & (yolo_left == 0)
        # Removed pixels (in YOLO but not in current)
        removed = (left_mask == 0) & (yolo_left > 0)

        diff_image[added] = (0, 255, 0)  # Green = added
        diff_image[removed] = (0, 0, 255)  # Red = removed

    # Right eyebrow difference
    if right_mask is not None and yolo_right is not None:
        # Added pixels (in current but not in YOLO)
        added = (right_mask > 0) & (yolo_right == 0)
        # Removed pixels (in YOLO but not in current)
        removed = (right_mask == 0) & (yolo_right > 0)

        diff_image[added] = (0, 255, 0)  # Green = added
        diff_image[removed] = (0, 0, 255)  # Red = removed

    return diff_image


# =============================================================================
# MANUAL EDIT MODE
# =============================================================================

def render_manual_edit_mode():
    """Render manual edit mode with live transform preview."""
    st.caption("Apply geometric transformations to eyebrow masks. Changes appear instantly in the live preview.")

    # Select eyebrow to edit
    edit_side = st.radio("Select Eyebrow:", ["left", "right"], horizontal=True, key="manual_edit_side")

    if get_current_mask(edit_side) is None:
        st.info(f"No {edit_side} eyebrow detected")
        return

    st.divider()

    # Get current transform
    current_transform = st.session_state.current_masks[edit_side]['transform']

    # Rotation
    st.write("**Rotation**")
    rotation = st.slider(
        "Angle (degrees)",
        min_value=ROTATION_RANGE[0],
        max_value=ROTATION_RANGE[1],
        value=current_transform['rotation'],
        step=1,
        key=f"{edit_side}_rotation_slider",
        help="Rotate the eyebrow mask",
        label_visibility="collapsed"
    )

    # Scale
    st.write("**Scale**")
    scale = st.slider(
        "Size",
        min_value=SCALE_RANGE[0],
        max_value=SCALE_RANGE[1],
        value=current_transform['scale'],
        step=0.05,
        key=f"{edit_side}_scale_slider",
        help="Resize the eyebrow mask",
        label_visibility="collapsed"
    )

    # Translation
    st.write("**Position**")
    col_dx, col_dy = st.columns(2)

    with col_dx:
        dx = st.number_input(
            "Horizontal",
            min_value=-100,
            max_value=100,
            value=current_transform['dx'],
            step=1,
            key=f"{edit_side}_dx_input"
        )

    with col_dy:
        dy = st.number_input(
            "Vertical",
            min_value=-100,
            max_value=100,
            value=current_transform['dy'],
            step=1,
            key=f"{edit_side}_dy_input"
        )

    # Check if transforms changed and apply immediately
    transform_changed = (
        rotation != current_transform['rotation'] or
        scale != current_transform['scale'] or
        dx != current_transform['dx'] or
        dy != current_transform['dy']
    )

    if transform_changed:
        # Apply transforms in real-time (no button needed!)
        apply_manual_transforms(edit_side, rotation, scale, dx, dy)

    st.divider()

    # Action buttons
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button(f"🔄 Reset", key=f"{edit_side}_reset_transform", use_container_width=True):
            reset_eyebrow(edit_side)

    with col_btn2:
        if st.button(f"⬅️ Center", key=f"{edit_side}_center_transform", use_container_width=True):
            # Reset translation only
            apply_manual_transforms(edit_side, rotation, scale, 0, 0)


def apply_manual_transforms(side: str, rotation: float, scale: float, dx: int, dy: int):
    """Apply manual transformations to mask."""
    try:
        # Get original beautified mask (before any transforms)
        eyebrow_data = next((eb for eb in st.session_state.eyebrows if eb['side'] == side), None)
        if not eyebrow_data:
            show_error("No eyebrow data found")
            return

        original_mask = base64_to_mask(eyebrow_data['final_mask_base64'])

        # Apply transforms in order: scale -> rotate -> translate
        transformed = original_mask.copy()

        if scale != 1.0:
            transformed = apply_scale_to_mask(transformed, scale)

        if rotation != 0:
            transformed = apply_rotation_to_mask(transformed, rotation)

        if dx != 0 or dy != 0:
            transformed = apply_translation_to_mask(transformed, dx, dy)

        # Update mask and transform state
        update_current_mask(side, transformed)
        st.session_state.current_masks[side]['transform'] = {
            'rotation': rotation,
            'scale': scale,
            'dx': dx,
            'dy': dy
        }

        show_success(f"Transformations applied to {side} eyebrow")
        st.rerun()

    except Exception as e:
        show_error(f"Transform failed: {str(e)}")


# =============================================================================
# FINALIZE & SD ENHANCEMENT
# =============================================================================

def finalize_masks():
    """Finalize masks and submit to API."""
    try:
        client = get_api_client()

        with st.spinner("Finalizing masks..."):
            for side in ['left', 'right']:
                if get_current_mask(side) is not None:
                    result = client.submit_edited_mask(
                        image_base64=st.session_state.original_image_b64,
                        edited_mask_base64=st.session_state.current_masks[side]['mask_b64'],
                        side=side,
                        metadata={
                            'transform': st.session_state.current_masks[side]['transform'],
                            'clicks': st.session_state.clicks[side]
                        }
                    )

                    if not result.get('success'):
                        show_error(f"Failed to finalize {side} mask")
                        return

        st.session_state.finalized = True
        show_success("All masks finalized successfully!")
        st.rerun()

    except Exception as e:
        show_error(f"Finalization failed: {str(e)}")


def render_sd_enhancement():
    """Render SD enhancement section."""
    st.subheader("Stable Diffusion Enhancement")

    # SD parameters
    with st.expander("⚙️ Advanced Settings"):
        prompt = st.text_area("Prompt", value=SD_DEFAULTS['prompt'])
        negative_prompt = st.text_area("Negative Prompt", value=SD_DEFAULTS['negative_prompt'])

        col1, col2 = st.columns(2)
        with col1:
            strength = st.slider("Strength", 0.0, 1.0, SD_DEFAULTS['strength'], 0.05)
            guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, SD_DEFAULTS['guidance_scale'], 0.5)

        with col2:
            num_steps = st.slider("Inference Steps", 10, 100, SD_DEFAULTS['num_inference_steps'], 5)
            seed = st.number_input("Seed (optional)", min_value=0, value=None, step=1)

    # Run SD
    try:
        client = get_api_client()

        with st.spinner(MESSAGES['sd_processing']):
            left_mask = st.session_state.current_masks['left']['mask_b64'] if st.session_state.current_masks['left'] else None
            right_mask = st.session_state.current_masks['right']['mask_b64'] if st.session_state.current_masks['right'] else None

            result = client.sd_beautify(
                image_base64=st.session_state.original_image_b64,
                left_mask_base64=left_mask,
                right_mask_base64=right_mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                seed=seed
            )

            if result.get('success'):
                st.session_state.sd_result = result['result_image_base64']
                show_success("SD enhancement complete!")
            else:
                show_warning(f"SD enhancement: {result.get('message', 'Not implemented yet')}")

            st.rerun()

    except Exception as e:
        show_error(f"SD enhancement failed: {str(e)}")


# =============================================================================
# DOWNLOAD SECTION
# =============================================================================

def render_download_section():
    """Render download options."""
    col1, col2, col3 = st.columns(3)

    with col1:
        # Download current view
        beautified_masks = {
            'left': get_current_mask('left'),
            'right': get_current_mask('right')
        }

        _, beautified_view = create_comparison_view(
            st.session_state.original_image,
            {'left': None, 'right': None},
            beautified_masks,
            alpha=0.5
        )

        download_data = create_download_data(beautified_view, "beautified_eyebrows.png")

        st.download_button(
            label="📥 Download Beautified View",
            data=download_data,
            file_name="beautified_eyebrows.png",
            mime="image/png",
            use_container_width=True
        )

    with col2:
        # Download original
        original_data = create_download_data(st.session_state.original_image, "original.png")

        st.download_button(
            label="📥 Download Original",
            data=original_data,
            file_name="original.png",
            mime="image/png",
            use_container_width=True
        )

    with col3:
        # Download SD result if available
        if st.session_state.sd_result:
            sd_image = base64_to_image(st.session_state.sd_result)
            sd_data = create_download_data(sd_image, "sd_enhanced.png")

            st.download_button(
                label="📥 Download SD Enhanced",
                data=sd_data,
                file_name="sd_enhanced.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.button(
                label="📥 Download SD Enhanced",
                disabled=True,
                use_container_width=True
            )


# =============================================================================
# RUN APP
# =============================================================================

if __name__ == "__main__":
    main()
