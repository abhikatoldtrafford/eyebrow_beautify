"""
Stencil Storage Module (v6.0)

File-based JSON storage system for managing saved eyebrow stencils.

This module provides CRUD (Create, Read, Update, Delete) operations for stencil
polygons, with support for metadata, tags, and export to various formats (SVG, JSON, PNG).

Storage Structure:
    stencil_data/
    ├── stencils.json              # Master index
    ├── stencil_{id}.json          # Individual stencil files
    └── exports/
        ├── stencil_{id}.svg       # SVG exports
        └── stencil_{id}.png       # PNG previews

Author: Brow Stencil System
Date: 2025-01-13
"""

import json
import uuid
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StencilStorage:
    """
    File-based JSON storage for eyebrow stencils.

    Features:
        - Save/load stencils with metadata
        - List stencils with filtering (side, date, tags)
        - Delete stencils
        - Export to SVG/JSON/PNG
        - Automatic ID generation (UUID)
        - Index management for fast lookups

    Example:
        >>> storage = StencilStorage('stencil_data')
        >>> stencil_id = storage.save_stencil(polygon, metadata)
        >>> stencils = storage.list_stencils(side='left')
        >>> stencil = storage.get_stencil(stencil_id)
        >>> storage.delete_stencil(stencil_id)
    """

    def __init__(self, data_dir: str = 'stencil_data'):
        """
        Initialize storage system.

        Parameters:
            data_dir: Directory path for storing stencil data

        Creates directory structure if it doesn't exist:
            data_dir/
            ├── stencils.json
            └── exports/
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.index_file = self.data_dir / 'stencils.json'
        self.exports_dir = self.data_dir / 'exports'
        self.exports_dir.mkdir(exist_ok=True)

        self._init_index()

        logger.info(f"StencilStorage initialized at: {self.data_dir}")

    def _init_index(self):
        """Initialize index file if it doesn't exist."""
        if not self.index_file.exists():
            initial_index = {
                'stencils': [],
                'version': '6.0',
                'created_at': datetime.utcnow().isoformat(),
                'last_updated': datetime.utcnow().isoformat()
            }
            self.index_file.write_text(json.dumps(initial_index, indent=2))
            logger.info("Created new stencils index")

    def _load_index(self) -> Dict:
        """Load index file."""
        try:
            return json.loads(self.index_file.read_text())
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return {'stencils': [], 'version': '6.0'}

    def _save_index(self, index: Dict):
        """Save index file with updated timestamp."""
        index['last_updated'] = datetime.utcnow().isoformat()
        self.index_file.write_text(json.dumps(index, indent=2))

    def _generate_image_hash(self, image_data: Optional[bytes] = None) -> str:
        """Generate SHA256 hash of source image (for tracking)."""
        if image_data is None:
            return 'no_image'
        return hashlib.sha256(image_data).hexdigest()

    def save_stencil(
        self,
        polygon: Dict,
        metadata: Optional[Dict] = None,
        image_data: Optional[bytes] = None
    ) -> str:
        """
        Save new stencil to storage.

        Parameters:
            polygon: Polygon data dict with 'points', 'source', etc.
            metadata: Optional metadata (side, tags, confidence, etc.)
            image_data: Optional source image bytes (for hashing)

        Returns:
            stencil_id (UUID string)

        Example:
            >>> polygon = {
            ...     'points': [[120, 85], [135, 82], ...],
            ...     'num_points': 28,
            ...     'source': 'merged'
            ... }
            >>> metadata = {
            ...     'side': 'left',
            ...     'tags': ['thick', 'arched'],
            ...     'yolo_confidence': 0.87
            ... }
            >>> stencil_id = storage.save_stencil(polygon, metadata)
        """
        # Generate unique ID
        stencil_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        # Prepare metadata
        if metadata is None:
            metadata = {}

        metadata.update({
            'created_at': timestamp,
            'source_image_hash': self._generate_image_hash(image_data)
        })

        # Create stencil data
        stencil_data = {
            'id': stencil_id,
            'side': metadata.get('side', 'unknown'),
            'polygon': polygon,
            'metadata': metadata,
            'version': '6.0'
        }

        # Save individual stencil file
        stencil_filename = f'stencil_{stencil_id[:8]}.json'
        stencil_file = self.data_dir / stencil_filename
        stencil_file.write_text(json.dumps(stencil_data, indent=2))

        # Update index
        index = self._load_index()
        index['stencils'].append({
            'id': stencil_id,
            'side': stencil_data['side'],
            'created_at': timestamp,
            'num_points': polygon.get('num_points', len(polygon.get('points', []))),
            'tags': metadata.get('tags', []),
            'file_path': str(stencil_file),
            'source': polygon.get('source', 'unknown')
        })
        self._save_index(index)

        logger.info(f"Saved stencil: {stencil_id[:8]} ({stencil_data['side']})")
        return stencil_id

    def list_stencils(
        self,
        limit: int = 50,
        side: Optional[str] = None,
        sort_by: str = 'created_at',
        reverse: bool = True
    ) -> List[Dict]:
        """
        List stencils with optional filtering and sorting.

        Parameters:
            limit: Maximum number of results
            side: Filter by side ('left', 'right', or None for all)
            sort_by: Sort field ('created_at', 'num_points', 'side')
            reverse: Sort descending if True

        Returns:
            List of stencil summary dicts

        Example:
            >>> # Get all left eyebrows, newest first
            >>> stencils = storage.list_stencils(side='left')
            >>> for s in stencils:
            ...     print(f"{s['id']}: {s['num_points']} points")
        """
        index = self._load_index()
        stencils = index.get('stencils', [])

        # Filter by side
        if side and side != 'all':
            stencils = [s for s in stencils if s.get('side') == side]

        # Sort
        if sort_by in stencils[0] if stencils else {}:
            stencils = sorted(stencils, key=lambda x: x.get(sort_by, ''), reverse=reverse)

        # Limit
        return stencils[:limit]

    def get_stencil(self, stencil_id: str) -> Optional[Dict]:
        """
        Get specific stencil by ID.

        Parameters:
            stencil_id: UUID string

        Returns:
            Stencil dict or None if not found

        Example:
            >>> stencil = storage.get_stencil('550e8400-...')
            >>> if stencil:
            ...     print(stencil['polygon']['points'])
        """
        index = self._load_index()

        for entry in index.get('stencils', []):
            if entry['id'] == stencil_id:
                stencil_file = Path(entry['file_path'])
                if stencil_file.exists():
                    try:
                        return json.loads(stencil_file.read_text())
                    except Exception as e:
                        logger.error(f"Error loading stencil {stencil_id}: {e}")
                        return None
                else:
                    logger.warning(f"Stencil file not found: {stencil_file}")
                    return None

        logger.warning(f"Stencil not found in index: {stencil_id}")
        return None

    def delete_stencil(self, stencil_id: str) -> bool:
        """
        Delete stencil and its associated files.

        Parameters:
            stencil_id: UUID string

        Returns:
            True if deleted, False if not found

        Example:
            >>> success = storage.delete_stencil('550e8400-...')
        """
        index = self._load_index()

        for i, entry in enumerate(index.get('stencils', [])):
            if entry['id'] == stencil_id:
                # Delete stencil file
                stencil_file = Path(entry['file_path'])
                if stencil_file.exists():
                    stencil_file.unlink()
                    logger.info(f"Deleted stencil file: {stencil_file}")

                # Delete exports (if any)
                self._delete_exports(stencil_id)

                # Remove from index
                index['stencils'].pop(i)
                self._save_index(index)

                logger.info(f"Deleted stencil: {stencil_id[:8]}")
                return True

        logger.warning(f"Cannot delete - stencil not found: {stencil_id}")
        return False

    def _delete_exports(self, stencil_id: str):
        """Delete all export files for a stencil."""
        prefix = f'stencil_{stencil_id[:8]}'
        for export_file in self.exports_dir.glob(f'{prefix}.*'):
            export_file.unlink()
            logger.info(f"Deleted export: {export_file}")

    def export_svg(self, stencil_id: str, width: int = 800, height: int = 600) -> Optional[str]:
        """
        Export stencil as SVG file.

        Parameters:
            stencil_id: UUID string
            width: SVG canvas width
            height: SVG canvas height

        Returns:
            SVG content string or None if stencil not found

        Example:
            >>> svg_content = storage.export_svg('550e8400-...')
            >>> with open('stencil.svg', 'w') as f:
            ...     f.write(svg_content)
        """
        stencil = self.get_stencil(stencil_id)
        if not stencil:
            return None

        polygon = stencil['polygon']
        points = polygon['points']

        # Convert points to SVG path format
        points_str = ' '.join([f"{x},{y}" for x, y in points])

        # Calculate viewBox from polygon bbox
        bbox = polygon.get('bbox', self._calculate_bbox(points))
        viewbox_x = bbox[0] - 10  # Add 10px padding
        viewbox_y = bbox[1] - 10
        viewbox_w = bbox[2] - bbox[0] + 20
        viewbox_h = bbox[3] - bbox[1] + 20

        # Generate SVG
        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}"
     viewBox="{viewbox_x} {viewbox_y} {viewbox_w} {viewbox_h}"
     xmlns="http://www.w3.org/2000/svg">

  <!-- Eyebrow stencil boundary -->
  <polygon points="{points_str}"
           fill="none"
           stroke="black"
           stroke-width="2"
           stroke-linecap="round"
           stroke-linejoin="round"/>

  <!-- Metadata -->
  <metadata>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns:dc="http://purl.org/dc/elements/1.1/">
      <rdf:Description>
        <dc:title>Eyebrow Stencil - {stencil['side'].capitalize()}</dc:title>
        <dc:creator>Brow Stencil System v6.0</dc:creator>
        <dc:date>{stencil['metadata'].get('created_at', '')}</dc:date>
        <dc:description>
          Stencil ID: {stencil_id}
          Side: {stencil['side']}
          Points: {polygon['num_points']}
          Source: {polygon.get('source', 'unknown')}
        </dc:description>
      </rdf:Description>
    </rdf:RDF>
  </metadata>

</svg>'''

        # Save to exports directory
        export_filename = f"stencil_{stencil_id[:8]}.svg"
        export_path = self.exports_dir / export_filename
        export_path.write_text(svg)

        logger.info(f"Exported SVG: {export_path}")
        return svg

    def export_json(self, stencil_id: str) -> Optional[Dict]:
        """
        Export stencil as JSON (same as get_stencil, but saves to exports).

        Parameters:
            stencil_id: UUID string

        Returns:
            Stencil dict or None

        Example:
            >>> json_data = storage.export_json('550e8400-...')
        """
        stencil = self.get_stencil(stencil_id)
        if not stencil:
            return None

        # Save to exports directory
        export_filename = f"stencil_{stencil_id[:8]}.json"
        export_path = self.exports_dir / export_filename
        export_path.write_text(json.dumps(stencil, indent=2))

        logger.info(f"Exported JSON: {export_path}")
        return stencil

    def export_dxf(self, stencil_id: str) -> Optional[str]:
        """
        Export stencil as DXF file (AutoCAD format for laser cutting).

        NOTE: This is a simplified DXF export. For production use, consider
        using a library like ezdxf for full DXF support.

        Parameters:
            stencil_id: UUID string

        Returns:
            DXF content string or None

        Example:
            >>> dxf_content = storage.export_dxf('550e8400-...')
        """
        stencil = self.get_stencil(stencil_id)
        if not stencil:
            return None

        polygon = stencil['polygon']
        points = polygon['points']

        # Simple DXF format (R12)
        dxf_lines = [
            "0",
            "SECTION",
            "2",
            "ENTITIES",
        ]

        # Add POLYLINE
        dxf_lines.extend([
            "0",
            "POLYLINE",
            "8",
            "0",  # Layer 0
            "66",
            "1",  # Vertices follow
            "70",
            "1",  # Closed polyline
        ])

        # Add vertices
        for x, y in points:
            dxf_lines.extend([
                "0",
                "VERTEX",
                "8",
                "0",
                "10",
                str(float(x)),
                "20",
                str(float(y)),
            ])

        # End sequence
        dxf_lines.extend([
            "0",
            "SEQEND",
            "0",
            "ENDSEC",
            "0",
            "EOF",
        ])

        dxf_content = "\n".join(dxf_lines)

        # Save to exports directory
        export_filename = f"stencil_{stencil_id[:8]}.dxf"
        export_path = self.exports_dir / export_filename
        export_path.write_text(dxf_content)

        logger.info(f"Exported DXF: {export_path}")
        return dxf_content

    def _calculate_bbox(self, points: List[List[int]]) -> List[int]:
        """Calculate bounding box from points."""
        if not points:
            return [0, 0, 0, 0]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return [min(xs), min(ys), max(xs), max(ys)]

    def search_by_tags(self, tags: List[str], match_all: bool = False) -> List[Dict]:
        """
        Search stencils by tags.

        Parameters:
            tags: List of tag strings to search for
            match_all: If True, require all tags to match. If False, match any tag.

        Returns:
            List of matching stencils

        Example:
            >>> # Find stencils tagged as "thick" OR "arched"
            >>> results = storage.search_by_tags(['thick', 'arched'])
            >>>
            >>> # Find stencils tagged as "thick" AND "arched"
            >>> results = storage.search_by_tags(['thick', 'arched'], match_all=True)
        """
        index = self._load_index()
        stencils = index.get('stencils', [])

        results = []
        for stencil in stencils:
            stencil_tags = stencil.get('tags', [])

            if match_all:
                # Require all tags to be present
                if all(tag in stencil_tags for tag in tags):
                    results.append(stencil)
            else:
                # Match any tag
                if any(tag in stencil_tags for tag in tags):
                    results.append(stencil)

        return results

    def get_statistics(self) -> Dict:
        """
        Get storage statistics.

        Returns:
            {
                'total_stencils': int,
                'left_count': int,
                'right_count': int,
                'merged_count': int,
                'mediapipe_only_count': int,
                'avg_points': float,
                'storage_size_mb': float
            }

        Example:
            >>> stats = storage.get_statistics()
            >>> print(f"Total stencils: {stats['total_stencils']}")
        """
        index = self._load_index()
        stencils = index.get('stencils', [])

        stats = {
            'total_stencils': len(stencils),
            'left_count': sum(1 for s in stencils if s.get('side') == 'left'),
            'right_count': sum(1 for s in stencils if s.get('side') == 'right'),
            'merged_count': sum(1 for s in stencils if s.get('source') == 'merged'),
            'mediapipe_only_count': sum(1 for s in stencils if s.get('source') == 'mediapipe_only'),
            'avg_points': sum(s.get('num_points', 0) for s in stencils) / len(stencils) if stencils else 0,
            'storage_size_mb': sum(f.stat().st_size for f in self.data_dir.rglob('*.json')) / (1024 * 1024)
        }

        return stats


# =============================================================================
# MAIN ENTRY POINT (for testing/CLI)
# =============================================================================

if __name__ == '__main__':
    """
    Test the stencil storage system.

    Usage:
        python stencil_storage.py
    """
    print("=" * 70)
    print("Stencil Storage Test")
    print("=" * 70)

    # Initialize storage
    storage = StencilStorage('test_stencil_data')

    # Test polygon
    test_polygon = {
        'points': [[120, 85], [135, 82], [150, 80], [165, 82], [180, 85]],
        'num_points': 5,
        'source': 'merged',
        'bbox': [120, 80, 180, 85]
    }

    test_metadata = {
        'side': 'left',
        'tags': ['test', 'demo'],
        'yolo_confidence': 0.87,
        'alignment_iou': 0.65
    }

    # Save stencil
    print("\n1. Saving test stencil...")
    stencil_id = storage.save_stencil(test_polygon, test_metadata)
    print(f"   Saved with ID: {stencil_id[:8]}")

    # List stencils
    print("\n2. Listing all stencils...")
    stencils = storage.list_stencils()
    print(f"   Found {len(stencils)} stencils")
    for s in stencils:
        print(f"   - {s['id'][:8]}: {s['side']}, {s['num_points']} points")

    # Get specific stencil
    print("\n3. Retrieving stencil...")
    retrieved = storage.get_stencil(stencil_id)
    if retrieved:
        print(f"   Retrieved: {retrieved['side']}, {retrieved['polygon']['num_points']} points")

    # Export SVG
    print("\n4. Exporting to SVG...")
    svg = storage.export_svg(stencil_id)
    if svg:
        print(f"   SVG exported ({len(svg)} bytes)")

    # Export JSON
    print("\n5. Exporting to JSON...")
    json_data = storage.export_json(stencil_id)
    if json_data:
        print(f"   JSON exported")

    # Get statistics
    print("\n6. Storage statistics...")
    stats = storage.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Delete stencil
    print("\n7. Deleting test stencil...")
    deleted = storage.delete_stencil(stencil_id)
    print(f"   Deleted: {deleted}")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
