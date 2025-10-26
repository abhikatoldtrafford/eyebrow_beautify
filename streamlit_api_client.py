"""
Streamlit API Client
Wrapper functions for all API endpoints
"""

import requests
from typing import Dict, Optional, List, Tuple
import streamlit as st
from streamlit_config import API_BASE_URL, API_TIMEOUT


class APIClient:
    """Client for interacting with Eyebrow Beautification API."""

    def __init__(self, base_url: str = API_BASE_URL, timeout: int = API_TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout

    def _handle_response(self, response: requests.Response) -> Dict:
        """Handle API response and errors."""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_detail = response.json().get('detail', str(e)) if response.text else str(e)
            raise Exception(f"API Error: {error_detail}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Connection Error: {str(e)}")

    # =============================================================================
    # HEALTH & CONFIG
    # =============================================================================

    def check_health(self) -> Dict:
        """
        Check API health status.

        Returns:
            {
                'status': 'healthy',
                'model_loaded': True,
                'mediapipe_available': True,
                'version': '1.0.0'
            }
        """
        url = f"{self.base_url}/health"
        response = requests.get(url, timeout=self.timeout)
        return self._handle_response(response)

    def get_config(self) -> Dict:
        """Get current beautification configuration."""
        url = f"{self.base_url}/config"
        response = requests.get(url, timeout=self.timeout)
        return self._handle_response(response)

    def update_config(self, config: Dict) -> Dict:
        """Update beautification configuration."""
        url = f"{self.base_url}/config"
        response = requests.post(url, json=config, timeout=self.timeout)
        return self._handle_response(response)

    # =============================================================================
    # DETECTION ENDPOINTS
    # =============================================================================

    def detect_yolo(self, image_base64: str, conf_threshold: float = 0.25) -> Dict:
        """
        YOLO detection only.

        Args:
            image_base64: Base64 encoded image
            conf_threshold: Confidence threshold (0-1)

        Returns:
            {
                'success': True,
                'detections': {
                    'eyebrows': [...],
                    'eye': [...],
                    'eye_box': [...],
                    'hair': [...]
                },
                'image_shape': (H, W)
            }
        """
        url = f"{self.base_url}/detect/yolo"

        # For file upload endpoint, we need to send as multipart
        # But we have base64, so we'll use the /beautify/base64 pattern
        # Actually, /detect/yolo expects file upload, not base64
        # We need to convert base64 to file-like object

        import base64
        import io

        img_data = base64.b64decode(image_base64)
        files = {'file': ('image.png', io.BytesIO(img_data), 'image/png')}
        data = {'conf_threshold': conf_threshold}

        response = requests.post(url, files=files, data=data, timeout=self.timeout)
        return self._handle_response(response)

    def detect_mediapipe(self, image_base64: str, conf_threshold: float = 0.5) -> Dict:
        """
        MediaPipe face landmark detection only.

        Args:
            image_base64: Base64 encoded image
            conf_threshold: Confidence threshold (0-1)

        Returns:
            {
                'success': True,
                'landmarks': {
                    'left_eyebrow': {'points': [...], 'bbox': [...]},
                    'right_eyebrow': {...},
                    'left_eye': {...},
                    'right_eye': {...}
                },
                'image_shape': (H, W)
            }
        """
        url = f"{self.base_url}/detect/mediapipe"

        import base64
        import io

        img_data = base64.b64decode(image_base64)
        files = {'file': ('image.png', io.BytesIO(img_data), 'image/png')}
        data = {'conf_threshold': conf_threshold}

        response = requests.post(url, files=files, data=data, timeout=self.timeout)
        return self._handle_response(response)

    # =============================================================================
    # BEAUTIFICATION ENDPOINTS
    # =============================================================================

    def beautify_image(
        self,
        image_base64: str,
        config: Optional[Dict] = None,
        return_masks: bool = True
    ) -> Dict:
        """
        Complete eyebrow beautification pipeline (Base64 input).

        Args:
            image_base64: Base64 encoded image
            config: Optional custom configuration
            return_masks: Whether to return mask data

        Returns:
            {
                'success': True,
                'eyebrows': [
                    {
                        'side': 'left',
                        'validation': {...},
                        'metadata': {...},
                        'original_mask_base64': '...',
                        'final_mask_base64': '...'
                    },
                    {...}  # right eyebrow
                ],
                'processing_time_ms': 250.5,
                'image_shape': (H, W)
            }
        """
        url = f"{self.base_url}/beautify/base64"

        payload = {
            'image_base64': image_base64,
            'return_masks': return_masks
        }

        if config:
            payload['config'] = config

        response = requests.post(url, json=payload, timeout=self.timeout)
        return self._handle_response(response)

    def submit_edited_mask(
        self,
        image_base64: str,
        edited_mask_base64: str,
        side: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Submit user-edited eyebrow mask.

        Args:
            image_base64: Original image (base64)
            edited_mask_base64: User-edited mask (base64)
            side: 'left' or 'right'
            metadata: Optional metadata

        Returns:
            {
                'success': True,
                'side': 'left',
                'final_mask_base64': '...',
                'mask_area': 10500,
                'metadata': {...}
            }
        """
        url = f"{self.base_url}/beautify/submit-edit"

        payload = {
            'image_base64': image_base64,
            'edited_mask_base64': edited_mask_base64,
            'side': side,
            'metadata': metadata or {}
        }

        response = requests.post(url, json=payload, timeout=self.timeout)
        return self._handle_response(response)

    # =============================================================================
    # ADJUSTMENT ENDPOINTS
    # =============================================================================

    def adjust_thickness(
        self,
        mask_base64: str,
        side: str,
        direction: str,
        increment: float = 0.05,
        num_clicks: int = 1
    ) -> Dict:
        """
        Adjust eyebrow thickness.

        Args:
            mask_base64: Current mask (base64)
            side: 'left' or 'right'
            direction: 'increase' or 'decrease'
            increment: Amount per click (default: 0.05 = 5%)
            num_clicks: Number of clicks to apply

        Returns:
            {
                'success': True,
                'adjusted_mask_base64': '...',
                'original_area': 10000,
                'adjusted_area': 10500,
                'area_change_pct': 5.0,
                'total_change_pct': 5.0
            }
        """
        url = f"{self.base_url}/adjust/thickness/{direction}"

        payload = {
            'mask_base64': mask_base64,
            'side': side,
            'increment': increment,
            'num_clicks': num_clicks
        }

        response = requests.post(url, json=payload, timeout=self.timeout)
        return self._handle_response(response)

    def adjust_span(
        self,
        mask_base64: str,
        side: str,
        direction: str,
        increment: float = 0.05,
        num_clicks: int = 1
    ) -> Dict:
        """
        Adjust eyebrow span/length (directional: tail-only).

        Args:
            mask_base64: Current mask (base64)
            side: 'left' or 'right'
            direction: 'increase' or 'decrease'
            increment: Amount per click (default: 0.05 = 5%)
            num_clicks: Number of clicks to apply

        Returns:
            {
                'success': True,
                'adjusted_mask_base64': '...',
                'original_area': 10000,
                'adjusted_area': 10500,
                'area_change_pct': 5.0,
                'total_change_pct': 5.0
            }
        """
        url = f"{self.base_url}/adjust/span/{direction}"

        payload = {
            'mask_base64': mask_base64,
            'side': side,
            'increment': increment,
            'num_clicks': num_clicks
        }

        response = requests.post(url, json=payload, timeout=self.timeout)
        return self._handle_response(response)

    # =============================================================================
    # SD ENHANCEMENT (PHASE 2)
    # =============================================================================

    def sd_beautify(
        self,
        image_base64: str,
        left_mask_base64: Optional[str] = None,
        right_mask_base64: Optional[str] = None,
        prompt: str = "natural, well-groomed eyebrows, high detail, photorealistic",
        negative_prompt: str = "blurry, distorted, unnatural",
        strength: float = 0.7,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Generate beautified eyebrows using Stable Diffusion inpainting.

        NOTE: Currently returns placeholder response (SD not implemented yet).

        Args:
            image_base64: Original image
            left_mask_base64: Left eyebrow mask
            right_mask_base64: Right eyebrow mask
            prompt: SD prompt
            negative_prompt: SD negative prompt
            strength: Denoising strength (0-1)
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility

        Returns:
            {
                'success': False,  # True when SD implemented
                'message': '...',
                'result_image_base64': '...',
                'processing_time_ms': 0.0,
                'seed_used': 12345,
                'metadata': {...}
            }
        """
        url = f"{self.base_url}/generate/sd-beautify"

        payload = {
            'image_base64': image_base64,
            'left_eyebrow_mask_base64': left_mask_base64,
            'right_eyebrow_mask_base64': right_mask_base64,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'strength': strength,
            'guidance_scale': guidance_scale,
            'num_inference_steps': num_inference_steps,
            'seed': seed
        }

        response = requests.post(url, json=payload, timeout=60)  # Longer timeout for SD
        return self._handle_response(response)


# Cached client instance
@st.cache_resource
def get_api_client() -> APIClient:
    """Get cached API client instance."""
    return APIClient()
