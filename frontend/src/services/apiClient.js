import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class APIClient {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json'
      },
      timeout: 60000 // 60 second timeout for image processing
    });
  }

  /**
   * Convert File to base64 string
   */
  fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = error => reject(error);
    });
  }

  /**
   * Health check
   */
  async checkHealth() {
    const response = await this.client.get('/health');
    return response.data;
  }

  /**
   * Preprocess face (validation)
   */
  async preprocess(imageBase64) {
    const response = await this.client.post('/preprocess', {
      image_base64: imageBase64,
      config: {
        min_rotation_threshold: 1.0,
        max_rotation_angle: 30.0
      }
    });
    return response.data;
  }

  /**
   * Extract stencil polygons from image (main endpoint)
   */
  async extractStencil(imageBase64, config = {}) {
    const response = await this.client.post('/beautify/base64', {
      image_base64: imageBase64,
      config: {
        yolo_conf_threshold: 0.25,
        yolo_simplify_epsilon: 0.005,
        alignment_iou_threshold: 0.3,
        alignment_distance_threshold: 20,
        ...config
      },
      return_masks: false
    });
    return response.data;
  }

  /**
   * Save edited stencil to library
   */
  async saveStencil(data) {
    const response = await this.client.post('/stencils/save', data);
    return response.data;
  }

  /**
   * List all stencils
   */
  async listStencils(params = {}) {
    const response = await this.client.get('/stencils/list', {
      params: {
        limit: 50,
        ...params
      }
    });
    return response.data;
  }

  /**
   * Get specific stencil by ID
   */
  async getStencil(id) {
    const response = await this.client.get(`/stencils/${id}`);
    return response.data;
  }

  /**
   * Delete stencil
   */
  async deleteStencil(id) {
    const response = await this.client.delete(`/stencils/${id}`);
    return response.data;
  }

  /**
   * Export stencil (SVG, JSON, PNG)
   */
  async exportStencil(id, format = 'json') {
    const response = await this.client.get(`/stencils/${id}/export`, {
      params: { format }
    });
    return response.data;
  }

  /**
   * Get current configuration
   */
  async getConfig() {
    const response = await this.client.get('/config');
    return response.data;
  }

  /**
   * Update configuration
   */
  async updateConfig(config) {
    const response = await this.client.post('/config', config);
    return response.data;
  }
}

// Export singleton instance
const apiClient = new APIClient();
export default apiClient;
