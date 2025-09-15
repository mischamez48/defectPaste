"""
Interactive Copy-Paste Tool for MVTec AD Dataset
A user-friendly GUI for manually placing defects on objects with drag-and-drop functionality
"""

import sys
import os
import json
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QListWidget, QGroupBox,
    QSplitter, QFileDialog, QMessageBox, QSpinBox, QCheckBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem,
    QGraphicsRectItem, QGraphicsPathItem, QListWidgetItem, QToolBar, QStatusBar, QDockWidget, QColorDialog,
    QScrollArea
)
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QBrush, QColor, QPen, QTransform, QCursor, QPainterPath
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from typing import List, Tuple, Optional, Dict
import random

# Note: These modules are not used in the current implementation
# from dataset import MVTecDataset
# from augmentation import CopyPasteAugmentation


class DefectItem(QGraphicsPixmapItem):
    """Draggable defect item on the canvas"""
    
    def __init__(self, pixmap, mask_pixmap, defect_data, parent=None):
        super().__init__(pixmap)
        self.mask_pixmap = mask_pixmap
        self.defect_data = defect_data
        self.setFlags(
            QGraphicsPixmapItem.ItemIsMovable |
            QGraphicsPixmapItem.ItemIsSelectable |
            QGraphicsPixmapItem.ItemSendsGeometryChanges
        )
        self.setZValue(1)  # Above background
        self.original_pixmap = pixmap
        self.original_mask = mask_pixmap
        self.scale_factor = 1.0
        self.rotation_angle = 0
        self.opacity = 0.7
        
    def update_transform(self, scale, rotation, opacity):
        """Update defect transformation"""
        self.scale_factor = scale
        self.rotation_angle = rotation
        self.opacity = opacity
        
        # Apply transformations
        transform = QTransform()
        transform.scale(scale, scale)
        transform.rotate(rotation)
        
        # Apply to pixmap (preserve RGBA format)
        transformed_pixmap = self.original_pixmap.transformed(transform, Qt.SmoothTransformation)
        transformed_mask = self.original_mask.transformed(transform, Qt.SmoothTransformation)
        
        self.setPixmap(transformed_pixmap)
        self.mask_pixmap = transformed_mask
        self.setOpacity(opacity)
        
    def get_position(self):
        """Get current position"""
        return self.pos().x(), self.pos().y()


class SelectedRegionItem(QGraphicsPixmapItem):
    """Draggable selected region item on the canvas"""
    
    def __init__(self, pixmap, mask_pixmap, region_data, parent=None):
        super().__init__(pixmap)
        self.mask_pixmap = mask_pixmap
        self.region_data = region_data
        self.setFlags(
            QGraphicsPixmapItem.ItemIsMovable |
            QGraphicsPixmapItem.ItemIsSelectable |
            QGraphicsPixmapItem.ItemSendsGeometryChanges
        )
        self.setZValue(2)  # Above defects
        self.original_pixmap = pixmap
        self.original_mask = mask_pixmap
        self.scale_factor = 1.0
        self.rotation_angle = 0
        self.opacity = 0.8
        
    def update_transform(self, scale, rotation, opacity):
        """Update region transformation"""
        self.scale_factor = scale
        self.rotation_angle = rotation
        self.opacity = opacity
        
        # Apply transformations
        transform = QTransform()
        transform.scale(scale, scale)
        transform.rotate(rotation)
        
        # Apply to pixmap (preserve RGBA format)
        transformed_pixmap = self.original_pixmap.transformed(transform, Qt.SmoothTransformation)
        transformed_mask = self.original_mask.transformed(transform, Qt.SmoothTransformation)
        
        self.setPixmap(transformed_pixmap)
        self.mask_pixmap = transformed_mask
        self.setOpacity(opacity)
        
    def get_position(self):
        """Get current position"""
        return self.pos().x(), self.pos().y()


class InteractiveCanvas(QGraphicsView):
    """Main canvas for placing defects"""
    
    defect_placed = pyqtSignal(dict)
    paint_changed = pyqtSignal()
    region_placed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.scene.selectionChanged.connect(self.on_selection_changed)
        
        # Background image
        self.background_item = None
        self.background_tensor = None
        
        # Defect items
        self.defect_items = []
        self.selected_defect = None
        
        # Selected region items
        self.region_items = []
        self.selected_region = None
        
        # Selection tool settings
        self.selection_enabled = False
        self.selection_mode = "Rectangle"  # "Rectangle" or "Freehand"
        self.is_selecting = False
        self.selection_start = None
        self.selection_rect = None
        self.selection_item = None
        self.current_selection = None  # Store the current selection data
        
        # Freehand selection
        self.freehand_path = None
        self.freehand_points = []
        self.freehand_item = None
        
        # Paint brush settings
        self.brush_enabled = False
        self.brush_mode = "Paint"  # "Paint" or "Erase"
        self.brush_size = 10
        self.brush_opacity = 100
        self.brush_color = QColor(0, 0, 0)  # Black by default
        self.is_painting = False
        self.last_paint_point = None
        
        # Paint layer for brush strokes
        self.paint_layer = None
        self.paint_layer_item = None
        
        # Settings
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        
        # [Removed] Object mask functionality
        
    def set_background_image(self, image_tensor):
        """Set the background image"""
        self.background_tensor = image_tensor
        
        # Convert tensor to QPixmap
        image_np = image_tensor.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
        h, w, c = image_np.shape
        
        qimage = QImage(image_np.tobytes(), w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        
        # Clear scene
        self.scene.clear()
        self.defect_items.clear()
        self.region_items.clear()
        self.background_item = None
        self.selected_defect = None
        self.selected_region = None
        self.current_selection = None
        
        # Add background
        self.background_item = self.scene.addPixmap(pixmap)
        self.background_item.setZValue(0)
        
        # Create paint layer
        self.paint_layer = QPixmap(pixmap.size())
        self.paint_layer.fill(Qt.transparent)
        self.paint_layer_item = self.scene.addPixmap(self.paint_layer)
        self.paint_layer_item.setZValue(1)  # Above background, below defects
        
        # Fit in view
        self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        
    # [Removed] Object mask overlay methods
            
    def add_defect(self, defect_tensor, mask_tensor, defect_info, position: Optional[Tuple[int, int]] = None, opacity_override: Optional[float] = None):
        """Add a defect to the canvas.
        Optionally provide a top-left position (x,y) and opacity to place without recentering.
        """
        # Convert defect tensor to QPixmap with transparency
        defect_np = defect_tensor.permute(1, 2, 0).numpy()
        defect_np = (defect_np * 255).astype(np.uint8)
        mask_np = mask_tensor.squeeze(0).numpy()
        mask_np = (mask_np * 255).astype(np.uint8)
        
        h, w, c = defect_np.shape
        
        # Create RGBA image with transparency
        rgba_np = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_np[:, :, :3] = defect_np  # RGB channels
        rgba_np[:, :, 3] = mask_np     # Alpha channel from mask
        
        qimage = QImage(rgba_np.tobytes(), w, h, w * 4, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage)
        
        # Convert mask to pixmap (for internal use)
        mask_qimage = QImage(mask_np.tobytes(), w, h, w, QImage.Format_Grayscale8)
        mask_pixmap = QPixmap.fromImage(mask_qimage)
        
        # Create defect item
        defect_item = DefectItem(pixmap, mask_pixmap, defect_info)
        
        # Position
        if position is not None:
            defect_item.setPos(float(position[0]), float(position[1]))
        else:
            if self.background_item:
                bg_rect = self.background_item.boundingRect()
                defect_item.setPos(
                    bg_rect.width() / 2 - w / 2,
                    bg_rect.height() / 2 - h / 2
                )
        if opacity_override is not None:
            defect_item.setOpacity(float(opacity_override))
        
        self.scene.addItem(defect_item)
        self.defect_items.append(defect_item)
        self.selected_defect = defect_item
        
        
        # Emit signal
        self.defect_placed.emit({
            'type': defect_info.get('type', 'unknown'),
            'position': (defect_item.x(), defect_item.y())
        })
        
    def remove_selected_defect(self):
        """Remove the selected defect"""
        try:
            if not self.scene:
                return
            # Gather selected items from the scene
            selected_items = [item for item in self.scene.selectedItems() if isinstance(item, DefectItem)]
            
            # Fallback to single tracked selection
            if not selected_items and self.selected_defect and self.selected_defect in self.defect_items:
                selected_items = [self.selected_defect]
            
            if not selected_items:
                return
            
            # Remove all selected defect items
            for item in selected_items:
                if item in self.defect_items:
                    self.scene.removeItem(item)
                    self.defect_items.remove(item)
            
            # Update current selection reference
            remaining_selected = [item for item in self.scene.selectedItems() if isinstance(item, DefectItem)]
            self.selected_defect = remaining_selected[-1] if remaining_selected else None
        except RuntimeError:
            # Scene has been deleted, ignore
            pass
            
    def clear_defects(self):
        """Clear all defects from canvas"""
        for item in list(self.defect_items):
            self.scene.removeItem(item)
        self.defect_items.clear()
        self.selected_defect = None
        
    
    def on_selection_changed(self):
        """Sync tracked selected defect and region with scene selection"""
        try:
            if not self.scene:
                return
            selected_defects = [item for item in self.scene.selectedItems() if isinstance(item, DefectItem)]
            selected_regions = [item for item in self.scene.selectedItems() if isinstance(item, SelectedRegionItem)]
            
            self.selected_defect = selected_defects[-1] if selected_defects else None
            self.selected_region = selected_regions[-1] if selected_regions else None
        except RuntimeError:
            # Scene has been deleted, ignore
            pass
    
    def mousePressEvent(self, event):
        """Handle mouse press events for painting and selection"""
        if self.selection_enabled and event.button() == Qt.LeftButton:
            self.is_selecting = True
            self.selection_start = self.mapToScene(event.pos())
            if self.selection_mode == "Rectangle":
                self.start_rectangle_selection()
            elif self.selection_mode == "Freehand":
                self.start_freehand_selection()
        elif self.brush_enabled and event.button() == Qt.LeftButton:
            self.is_painting = True
            self.last_paint_point = self.mapToScene(event.pos())
            self.paint_at_point(self.last_paint_point)
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for painting and selection"""
        if self.selection_enabled and self.is_selecting and event.buttons() & Qt.LeftButton:
            current_point = self.mapToScene(event.pos())
            if self.selection_mode == "Rectangle":
                self.update_rectangle_selection(current_point)
            elif self.selection_mode == "Freehand":
                self.update_freehand_selection(current_point)
        elif self.brush_enabled and self.is_painting and event.buttons() & Qt.LeftButton:
            current_point = self.mapToScene(event.pos())
            self.paint_line(self.last_paint_point, current_point)
            self.last_paint_point = current_point
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if self.selection_enabled and event.button() == Qt.LeftButton:
            self.is_selecting = False
            if self.selection_mode == "Rectangle":
                self.finish_rectangle_selection()
            elif self.selection_mode == "Freehand":
                self.finish_freehand_selection()
        elif self.brush_enabled and event.button() == Qt.LeftButton:
            self.is_painting = False
            self.last_paint_point = None
        else:
            super().mouseReleaseEvent(event)
    
    def keyPressEvent(self, event):
        """Handle keyboard events"""
        if self.selection_enabled and event.key() == Qt.Key_Return:
            # Enter key to confirm selection
            if self.selection_item:
                self.finish_rectangle_selection()
            elif self.freehand_item:
                self.finish_freehand_selection()
        elif event.key() == Qt.Key_Escape:
            # Escape key to cancel selection
            if self.selection_enabled and (self.selection_item or self.freehand_item):
                self.clear_selection()
        else:
            super().keyPressEvent(event)
    
    def paint_at_point(self, point):
        """Paint at a specific point"""
        if not self.paint_layer or not self.background_item:
            return
            
        painter = QPainter(self.paint_layer)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set brush properties
        brush = QBrush(self.brush_color)
        painter.setBrush(brush)
        painter.setPen(Qt.NoPen)
        
        # Set opacity
        painter.setOpacity(self.brush_opacity / 100.0)
        
        # Convert scene coordinates to pixmap coordinates
        pixmap_rect = self.background_item.boundingRect()
        x = int(point.x() - pixmap_rect.x())
        y = int(point.y() - pixmap_rect.y())
        
        # Draw circle
        radius = self.brush_size // 2
        if self.brush_mode == "Paint":
            painter.drawEllipse(x - radius, y - radius, self.brush_size, self.brush_size)
        elif self.brush_mode == "Erase":
            # For erase mode, we'll use a different approach
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.drawEllipse(x - radius, y - radius, self.brush_size, self.brush_size)
        
        painter.end()
        
        # Update the paint layer item
        self.paint_layer_item.setPixmap(self.paint_layer)
        
        # Emit signal to mark changes as unsaved
        self.paint_changed.emit()
    
    def paint_line(self, start_point, end_point):
        """Paint a line between two points"""
        if not self.paint_layer or not self.background_item:
            return
            
        painter = QPainter(self.paint_layer)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set brush properties
        pen = QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        
        # Set opacity
        painter.setOpacity(self.brush_opacity / 100.0)
        
        # Convert scene coordinates to pixmap coordinates
        pixmap_rect = self.background_item.boundingRect()
        start_x = int(start_point.x() - pixmap_rect.x())
        start_y = int(start_point.y() - pixmap_rect.y())
        end_x = int(end_point.x() - pixmap_rect.x())
        end_y = int(end_point.y() - pixmap_rect.y())
        
        if self.brush_mode == "Paint":
            painter.drawLine(start_x, start_y, end_x, end_y)
        elif self.brush_mode == "Erase":
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.drawLine(start_x, start_y, end_x, end_y)
        
        painter.end()
        
        # Update the paint layer item
        self.paint_layer_item.setPixmap(self.paint_layer)
        
        # Emit signal to mark changes as unsaved
        self.paint_changed.emit()
    
    def clear_paint_layer(self):
        """Clear the paint layer"""
        if self.paint_layer:
            self.paint_layer.fill(Qt.transparent)
            self.paint_layer_item.setPixmap(self.paint_layer)
    
    def start_rectangle_selection(self):
        """Start rectangle selection"""
        if self.selection_item:
            self.scene.removeItem(self.selection_item)
        self.selection_item = None
    
    def start_freehand_selection(self):
        """Start freehand selection"""
        if self.freehand_item:
            self.scene.removeItem(self.freehand_item)
        self.freehand_item = None
        self.freehand_points = []
        self.freehand_path = QPainterPath()
        self.freehand_path.moveTo(self.selection_start)
        self.freehand_points.append(self.selection_start)
    
    def update_rectangle_selection(self, current_point):
        """Update rectangle selection during drag"""
        if not self.selection_start:
            return
            
        # Create rectangle from start to current point
        rect = QRectF(self.selection_start, current_point).normalized()
        
        # Remove previous selection rectangle
        if self.selection_item:
            self.scene.removeItem(self.selection_item)
        
        # Create new selection rectangle with better visual feedback
        self.selection_item = QGraphicsRectItem(rect)
        self.selection_item.setPen(QPen(QColor(0, 255, 0), 1, Qt.SolidLine))  # Green solid line - thinner
        self.selection_item.setBrush(QBrush(QColor(0, 255, 0, 30)))  # Light green fill
        self.selection_item.setZValue(10)  # Above everything
        self.scene.addItem(self.selection_item)
    
    def update_freehand_selection(self, current_point):
        """Update freehand selection during drag"""
        if not self.freehand_path:
            return
            
        # Add line to current point
        self.freehand_path.lineTo(current_point)
        self.freehand_points.append(current_point)
        
        # Remove previous freehand path
        if self.freehand_item:
            self.scene.removeItem(self.freehand_item)
        
        # Create new freehand path item
        self.freehand_item = QGraphicsPathItem(self.freehand_path)
        self.freehand_item.setPen(QPen(QColor(0, 255, 0), 1, Qt.SolidLine))  # Green solid line - thinner
        self.freehand_item.setBrush(QBrush(QColor(0, 255, 0, 30)))  # Light green fill
        self.freehand_item.setZValue(10)  # Above everything
        self.scene.addItem(self.freehand_item)
    
    def finish_rectangle_selection(self):
        """Finish rectangle selection and automatically create draggable region"""
        if not self.selection_item or not self.background_item:
            return
            
        # Get the selection rectangle
        selection_rect = self.selection_item.rect()
        
        # Convert to background image coordinates
        bg_rect = self.background_item.boundingRect()
        x = int(selection_rect.x() - bg_rect.x())
        y = int(selection_rect.y() - bg_rect.y())
        w = int(selection_rect.width())
        h = int(selection_rect.height())
        
        # Ensure selection is within bounds
        x = max(0, min(x, int(bg_rect.width()) - 1))
        y = max(0, min(y, int(bg_rect.height()) - 1))
        w = min(w, int(bg_rect.width()) - x)
        h = min(h, int(bg_rect.height()) - y)
        
        if w > 5 and h > 5:  # Minimum size check
            # Remove selection rectangle
            self.scene.removeItem(self.selection_item)
            self.selection_item = None
            
            # Automatically create draggable region
            self.create_region_from_selection(x, y, w, h, bg_rect)
        else:
            # Remove selection rectangle if too small
            self.scene.removeItem(self.selection_item)
            self.selection_item = None
            self.region_placed.emit({'has_selection': False})
    
    def finish_freehand_selection(self):
        """Finish freehand selection and automatically create draggable region"""
        if not self.freehand_item or not self.background_item or len(self.freehand_points) < 3:
            # Clean up if selection is too small
            if self.freehand_item:
                self.scene.removeItem(self.freehand_item)
                self.freehand_item = None
            self.region_placed.emit({'has_selection': False})
            return
            
        # Get the bounding rect of the freehand path
        path_rect = self.freehand_path.boundingRect()
        
        # Convert to background image coordinates
        bg_rect = self.background_item.boundingRect()
        x = int(path_rect.x() - bg_rect.x())
        y = int(path_rect.y() - bg_rect.y())
        w = int(path_rect.width())
        h = int(path_rect.height())
        
        # Ensure selection is within bounds
        x = max(0, min(x, int(bg_rect.width()) - 1))
        y = max(0, min(y, int(bg_rect.height()) - 1))
        w = min(w, int(bg_rect.width()) - x)
        h = min(h, int(bg_rect.height()) - y)
        
        if w > 5 and h > 5:  # Minimum size check
            # Remove freehand path
            self.scene.removeItem(self.freehand_item)
            self.freehand_item = None
            
            # Create region from freehand selection
            self.create_region_from_freehand_selection(x, y, w, h, bg_rect)
        else:
            # Remove freehand path if too small
            self.scene.removeItem(self.freehand_item)
            self.freehand_item = None
            self.region_placed.emit({'has_selection': False})
    
    def create_region_from_selection(self, x, y, w, h, bg_rect):
        """Create a draggable region from selection coordinates"""
        # Extract the selected region from the background image
        region_pixmap = self.background_item.pixmap().copy(x, y, w, h)
        
        # Create a mask for the region (fully opaque)
        mask_pixmap = QPixmap(w, h)
        mask_pixmap.fill(Qt.white)
        
        # Create region data
        region_data = {
            'type': 'selected_region',
            'source': f'region_{len(self.region_items)}',
            'original_rect': (x, y, w, h)
        }
        
        # Create region item
        region_item = SelectedRegionItem(region_pixmap, mask_pixmap, region_data)
        
        # Position at center of canvas
        canvas_center_x = bg_rect.width() / 2 - w / 2
        canvas_center_y = bg_rect.height() / 2 - h / 2
        region_item.setPos(canvas_center_x, canvas_center_y)
        
        # Add to scene
        self.scene.addItem(region_item)
        self.region_items.append(region_item)
        self.selected_region = region_item
        
        # Emit signal
        self.region_placed.emit({
            'type': 'selected_region',
            'position': (region_item.x(), region_item.y())
        })
    
    def create_region_from_freehand_selection(self, x, y, w, h, bg_rect):
        """Create a draggable region from freehand selection"""
        # Extract the background region
        background_region = self.background_item.pixmap().copy(x, y, w, h)
        
        # Create a mask based on the freehand path
        mask_pixmap = QPixmap(w, h)
        mask_pixmap.fill(Qt.transparent)
        
        # Create painter for mask
        mask_painter = QPainter(mask_pixmap)
        mask_painter.setRenderHint(QPainter.Antialiasing)
        
        # Create a path relative to the region coordinates
        relative_path = QPainterPath()
        if self.freehand_points:
            # Convert absolute coordinates to relative coordinates
            first_point = self.freehand_points[0]
            relative_x = first_point.x() - bg_rect.x() - x
            relative_y = first_point.y() - bg_rect.y() - y
            relative_path.moveTo(relative_x, relative_y)
            
            for point in self.freehand_points[1:]:
                rel_x = point.x() - bg_rect.x() - x
                rel_y = point.y() - bg_rect.y() - y
                relative_path.lineTo(rel_x, rel_y)
        
        # Close the path to create a filled shape
        relative_path.closeSubpath()
        
        # Fill the mask path with white (opaque)
        mask_painter.fillPath(relative_path, QColor(255, 255, 255))
        mask_painter.end()
        
        # Convert mask to numpy array for processing
        mask_image = mask_pixmap.toImage()
        mask_np = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                pixel = mask_image.pixel(x, y)
                mask_np[y, x] = (pixel >> 24) & 0xFF  # Get alpha channel
        
        # Convert background to numpy array
        bg_image = background_region.toImage()
        bg_np = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                pixel = bg_image.pixel(x, y)
                bg_np[y, x, 0] = (pixel >> 16) & 0xFF  # Red
                bg_np[y, x, 1] = (pixel >> 8) & 0xFF   # Green
                bg_np[y, x, 2] = pixel & 0xFF          # Blue
        
        # Create RGBA image with transparency
        rgba_np = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_np[:, :, :3] = bg_np  # RGB channels
        rgba_np[:, :, 3] = mask_np  # Alpha channel from mask
        
        # Convert back to QPixmap
        qimage = QImage(rgba_np.tobytes(), w, h, w * 4, QImage.Format_RGBA8888)
        region_pixmap = QPixmap.fromImage(qimage)
        
        # Create region data
        region_data = {
            'type': 'freehand_region',
            'source': f'freehand_region_{len(self.region_items)}',
            'original_rect': (x, y, w, h),
            'freehand_points': [(p.x() - bg_rect.x(), p.y() - bg_rect.y()) for p in self.freehand_points]
        }
        
        # Create region item
        region_item = SelectedRegionItem(region_pixmap, mask_pixmap, region_data)
        
        # Position at center of canvas
        canvas_center_x = bg_rect.width() / 2 - w / 2
        canvas_center_y = bg_rect.height() / 2 - h / 2
        region_item.setPos(canvas_center_x, canvas_center_y)
        
        # Add to scene
        self.scene.addItem(region_item)
        self.region_items.append(region_item)
        self.selected_region = region_item
        
        # Clear freehand data
        self.freehand_path = None
        self.freehand_points = []
        
        # Emit signal
        self.region_placed.emit({
            'type': 'freehand_region',
            'position': (region_item.x(), region_item.y())
        })
    
    def copy_selection(self):
        """Copy the current selection as a draggable region (legacy method)"""
        if not self.current_selection or not self.background_item:
            return
            
        x, y, w, h = self.current_selection['rect']
        bg_rect = self.current_selection['background_rect']
        
        # Create region from selection
        self.create_region_from_selection(x, y, w, h, bg_rect)
        
        # Clear current selection
        self.current_selection = None
    
    def clear_selection(self):
        """Clear the current selection"""
        if self.selection_item:
            self.scene.removeItem(self.selection_item)
            self.selection_item = None
        if self.freehand_item:
            self.scene.removeItem(self.freehand_item)
            self.freehand_item = None
        self.current_selection = None
        self.freehand_path = None
        self.freehand_points = []
        self.region_placed.emit({'has_selection': False})
    
    def remove_selected_region(self):
        """Remove the selected region"""
        try:
            if not self.scene:
                return
            selected_items = [item for item in self.scene.selectedItems() if isinstance(item, SelectedRegionItem)]
            
            if not selected_items and self.selected_region and self.selected_region in self.region_items:
                selected_items = [self.selected_region]
            
            if not selected_items:
                return
            
            # Remove all selected region items
            for item in selected_items:
                if item in self.region_items:
                    self.scene.removeItem(item)
                    self.region_items.remove(item)
            
            # Update current selection reference
            remaining_selected = [item for item in self.scene.selectedItems() if isinstance(item, SelectedRegionItem)]
            self.selected_region = remaining_selected[-1] if remaining_selected else None
        except RuntimeError:
            # Scene has been deleted, ignore
            pass
    
    def clear_regions(self):
        """Clear all regions from canvas"""
        for item in list(self.region_items):
            self.scene.removeItem(item)
        self.region_items.clear()
        self.selected_region = None
    
    def create_eraser_cursor(self, size):
        """Create a custom eraser cursor"""
        # Create a pixmap for the eraser cursor
        pixmap = QPixmap(size + 4, size + 4)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw eraser shape (rounded rectangle)
        pen = QPen(QColor(100, 100, 100), 2)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(200, 200, 200, 180)))
        
        # Draw the eraser body
        painter.drawRoundedRect(2, 2, size, size, 3, 3)
        
        # Draw a small highlight
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawLine(4, 4, size - 2, 4)
        
        painter.end()
        
        # Create cursor with hotspot at center
        return QCursor(pixmap, size // 2, size // 2)
    
    def set_brush_settings(self, enabled, mode, size, opacity, color):
        """Update brush settings"""
        self.brush_enabled = enabled
        self.brush_mode = mode
        self.brush_size = size
        self.brush_opacity = opacity
        self.brush_color = color
        
        # Update cursor based on mode
        if enabled:
            if mode == "Erase":
                # Use custom eraser cursor
                eraser_cursor = self.create_eraser_cursor(min(max(size, 16), 32))
                self.setCursor(eraser_cursor)
            else:
                # Use crosshair for paint mode
                self.setCursor(Qt.CrossCursor)
        elif self.selection_enabled:
            # Use crosshair for selection mode
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        
    def get_augmented_image(self):
        """Generate the final augmented image by painting the scene (WYSIWYG)."""
        if self.background_tensor is None:
            return None, None
        # Prepare base sizes
        base_h, base_w = self.background_tensor.shape[1], self.background_tensor.shape[2]
        
        # Render color image
        color_img = QImage(base_w, base_h, QImage.Format_RGB888)
        color_img.fill(QColor(0, 0, 0))
        painter = QPainter(color_img)
        # Draw background
        if self.background_item:
            painter.setOpacity(1.0)
            painter.drawPixmap(0, 0, self.background_item.pixmap())
        
        # Draw paint layer
        if self.paint_layer_item:
            painter.setOpacity(1.0)
            painter.drawPixmap(0, 0, self.paint_layer_item.pixmap())
        
        # Draw defects with their current opacity
        for item in self.defect_items:
            painter.setOpacity(float(item.opacity))
            painter.drawPixmap(int(item.x()), int(item.y()), item.pixmap())
        
        # Draw regions with their current opacity
        for item in self.region_items:
            painter.setOpacity(float(item.opacity))
            painter.drawPixmap(int(item.x()), int(item.y()), item.pixmap())
        painter.end()
        
        # Render mask (grayscale): draw transformed masks in white
        mask_img = QImage(base_w, base_h, QImage.Format_Grayscale8)
        mask_img.fill(0)
        mp = QPainter(mask_img)
        for item in self.defect_items:
            mp.setOpacity(1.0)
            mp.drawPixmap(int(item.x()), int(item.y()), item.mask_pixmap)
        for item in self.region_items:
            mp.setOpacity(1.0)
            mp.drawPixmap(int(item.x()), int(item.y()), item.mask_pixmap)
        mp.end()
        
        # Convert to tensors
        color_bytes = color_img.bits().asstring(base_w * base_h * 3)
        color_np = np.frombuffer(color_bytes, dtype=np.uint8).reshape((base_h, base_w, 3)).copy()
        mask_bytes = mask_img.bits().asstring(base_w * base_h)
        mask_np = np.frombuffer(mask_bytes, dtype=np.uint8).reshape((base_h, base_w)).copy()
        
        color_tensor = torch.from_numpy(color_np).float().permute(2, 0, 1) / 255.0
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0) / 255.0
        
        return color_tensor, mask_tensor
        
    def pixmap_to_numpy(self, pixmap, grayscale=False):
        """Convert QPixmap to numpy array"""
        qimage = pixmap.toImage()
        width, height = qimage.width(), qimage.height()
        
        if grayscale:
            qimage = qimage.convertToFormat(QImage.Format_Grayscale8)
            ptr = qimage.bits()
            ptr.setsize(height * width)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width))
        else:
            qimage = qimage.convertToFormat(QImage.Format_RGB888)
            ptr = qimage.bits()
            ptr.setsize(height * width * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
            
        return arr
        
    def apply_defect_to_image(self, image, mask, defect, defect_mask, x, y, opacity):
        """Apply defect to image at position"""
        _, h_img, w_img = image.shape
        h_def, w_def = defect_mask.shape if len(defect_mask.shape) == 2 else defect_mask.shape[:2]
        
        # Calculate valid region
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(w_img, x + w_def)
        y_end = min(h_img, y + h_def)
        
        # Calculate defect region
        def_x_start = max(0, -x)
        def_y_start = max(0, -y)
        def_x_end = def_x_start + (x_end - x_start)
        def_y_end = def_y_start + (y_end - y_start)
        
        if x_end > x_start and y_end > y_start:
            # Extract regions
            img_region = image[:, y_start:y_end, x_start:x_end]
            def_region = defect[def_y_start:def_y_end, def_x_start:def_x_end]
            mask_region = defect_mask[def_y_start:def_y_end, def_x_start:def_x_end]
            
            # Convert to tensors
            def_tensor = torch.from_numpy(def_region).float() / 255.0
            if len(def_tensor.shape) == 3:
                def_tensor = def_tensor.permute(2, 0, 1)
            
            mask_tensor = torch.from_numpy(mask_region).float() / 255.0
            if len(mask_tensor.shape) == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
                
            # Simple alpha blend
            mask_3ch = mask_tensor.repeat(3, 1, 1)
            blend_mask = mask_3ch * opacity
            img_region[:] = img_region * (1 - blend_mask) + def_tensor * blend_mask
            
            # Update mask
            mask[0, y_start:y_end, x_start:x_end] = torch.max(
                mask[0, y_start:y_end, x_start:x_end],
                mask_tensor[0]
            )

    


class DefectPlacementTool(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.target_images_dir = None
        self.defect_images_dir = None
        self.defect_masks_dir = None
        self.current_image_path = None
        self.target_images = []  # List of image paths
        self.defect_images = []  # List of defect image paths
        self.defect_masks = []   # List of defect mask paths
        # Per-image augmentation state cache
        self.augmentation_states: Dict[str, Dict] = {}
        self.has_unsaved_changes = False
        
        # Paint layer cache
        self.paint_layer_cache: Dict[str, QPixmap] = {}
        
        # Track which images have been saved
        self.saved_images: set = set()
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("DefectPaste - Interactive Defect Placement Tool")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls (with scroll area)
        left_panel_content = self.create_left_panel()
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_panel_content)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMaximumWidth(450)
        left_scroll.setMinimumWidth(350)
        
        # Center - Canvas
        self.canvas = InteractiveCanvas()
        self.canvas.defect_placed.connect(self.on_defect_placed)
        self.canvas.region_placed.connect(self.on_region_placed)
        self.canvas.paint_changed.connect(self._mark_unsaved)
        # Track changes for unsaved prompt
        self.canvas.scene.selectionChanged.connect(self._mark_unsaved)
        
        # Right panel - Defect library
        right_panel = self.create_right_panel()
        
        # Add to layout with splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self.canvas)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800, 400])
        
        main_layout.addWidget(splitter)
        
        # Create toolbar
        self.create_toolbar()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load target images and defect masks to begin")
        
    def create_toolbar(self):
        """Create main toolbar"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Load target images directory
        load_targets_action = toolbar.addAction("Load Target Images")
        load_targets_action.triggered.connect(self.load_target_images)
        
        # Load defect images directory
        load_defect_images_action = toolbar.addAction("Load Defect Images Folder")
        load_defect_images_action.triggered.connect(self.load_defect_images)
        
        # Load defect masks directory
        load_defect_masks_action = toolbar.addAction("Load Defect Masks Folder")
        load_defect_masks_action.triggered.connect(self.load_defect_masks)
        
        toolbar.addSeparator()
        
        # Save augmented image
        save_action = toolbar.addAction("Save Augmented")
        save_action.triggered.connect(self.save_augmented_image)
        
        toolbar.addSeparator()
        
        # Save all augmentations
        save_all_action = toolbar.addAction("Save All Augmented…")
        save_all_action.triggered.connect(self.save_all_augmentations)
        
        # Clear all
        clear_action = toolbar.addAction("Clear All")
        clear_action.triggered.connect(self.clear_all)
        
    def create_left_panel(self):
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)  # Add spacing between group boxes
        layout.setContentsMargins(5, 5, 5, 5)  # Add margins
        
        # Target image selection
        target_group = QGroupBox("Target Image")
        target_layout = QVBoxLayout()
        
        self.target_list = QListWidget()
        self.target_list.itemClicked.connect(self.on_target_selected)
        target_layout.addWidget(QLabel("Select target image:"))
        target_layout.addWidget(self.target_list)
        
        # [Removed] Object mask checkbox
        
        target_group.setLayout(target_layout)
        layout.addWidget(target_group)
        
        # Transformation controls
        transform_group = QGroupBox("Defect Transformation")
        transform_layout = QVBoxLayout()
        
        # Scale slider
        transform_layout.addWidget(QLabel("Scale:"))
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(25, 200)
        self.scale_slider.setValue(100)
        self.scale_slider.valueChanged.connect(self.update_defect_transform)
        self.scale_label = QLabel("1.0x")
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(self.scale_slider)
        scale_layout.addWidget(self.scale_label)
        transform_layout.addLayout(scale_layout)
        
        # Rotation slider
        transform_layout.addWidget(QLabel("Rotation:"))
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(-180, 180)
        self.rotation_slider.setValue(0)
        self.rotation_slider.valueChanged.connect(self.update_defect_transform)
        self.rotation_label = QLabel("0°")
        rotation_layout = QHBoxLayout()
        rotation_layout.addWidget(self.rotation_slider)
        rotation_layout.addWidget(self.rotation_label)
        transform_layout.addLayout(rotation_layout)
        
        # Opacity slider
        transform_layout.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(70)
        self.opacity_slider.valueChanged.connect(self.update_defect_transform)
        self.opacity_label = QLabel("0.7")
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(self.opacity_slider)
        opacity_layout.addWidget(self.opacity_label)
        transform_layout.addLayout(opacity_layout)
        
        # [Removed] Live preview toggle
        
        transform_group.setLayout(transform_layout)
        layout.addWidget(transform_group)
        
        # [Removed] Blend mode controls
        
        # Selection Tool
        selection_group = QGroupBox("Selection Tool")
        selection_layout = QVBoxLayout()
        
        # Enable/Disable selection tool
        self.selection_enabled_cb = QCheckBox("Enable Selection Tool")
        self.selection_enabled_cb.toggled.connect(self.toggle_selection_tool)
        selection_layout.addWidget(self.selection_enabled_cb)
        
        # Selection mode
        selection_layout.addWidget(QLabel("Selection Mode:"))
        self.selection_mode = QComboBox()
        self.selection_mode.addItems(["Rectangle", "Freehand"])
        self.selection_mode.currentTextChanged.connect(self.on_selection_mode_changed)
        selection_layout.addWidget(self.selection_mode)
        
        # Copy selection button
        self.copy_selection_btn = QPushButton("Copy Selection")
        self.copy_selection_btn.clicked.connect(self.copy_selection)
        self.copy_selection_btn.setEnabled(False)
        selection_layout.addWidget(self.copy_selection_btn)
        
        # Clear selection button
        self.clear_selection_btn = QPushButton("Clear Selection")
        self.clear_selection_btn.clicked.connect(self.clear_selection)
        selection_layout.addWidget(self.clear_selection_btn)
        
        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)
        
        # Paint Brush Tool
        brush_group = QGroupBox("Paint Brush Tool")
        brush_layout = QVBoxLayout()
        
        # Brush mode selection
        brush_layout.addWidget(QLabel("Brush Mode:"))
        self.brush_mode = QComboBox()
        self.brush_mode.addItems(["Paint", "Erase"])
        self.brush_mode.currentTextChanged.connect(self.on_brush_mode_changed)
        brush_layout.addWidget(self.brush_mode)
        
        # Brush size
        brush_layout.addWidget(QLabel("Brush Size:"))
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setRange(1, 50)
        self.brush_size_slider.setValue(10)
        self.brush_size_slider.valueChanged.connect(self.update_brush_settings)
        self.brush_size_label = QLabel("10px")
        brush_size_layout = QHBoxLayout()
        brush_size_layout.addWidget(self.brush_size_slider)
        brush_size_layout.addWidget(self.brush_size_label)
        brush_layout.addLayout(brush_size_layout)
        
        # Brush opacity
        brush_layout.addWidget(QLabel("Brush Opacity:"))
        self.brush_opacity_slider = QSlider(Qt.Horizontal)
        self.brush_opacity_slider.setRange(10, 100)
        self.brush_opacity_slider.setValue(100)
        self.brush_opacity_slider.valueChanged.connect(self.update_brush_settings)
        self.brush_opacity_label = QLabel("100%")
        brush_opacity_layout = QHBoxLayout()
        brush_opacity_layout.addWidget(self.brush_opacity_slider)
        brush_opacity_layout.addWidget(self.brush_opacity_label)
        brush_layout.addLayout(brush_opacity_layout)
        
        # Brush color
        brush_layout.addWidget(QLabel("Brush Color:"))
        self.brush_color_btn = QPushButton("Choose Color")
        self.brush_color_btn.clicked.connect(self.choose_brush_color)
        self.brush_color_btn.setStyleSheet("background-color: black; color: white;")
        brush_layout.addWidget(self.brush_color_btn)
        
        # Enable/Disable brush tool
        self.brush_enabled_cb = QCheckBox("Enable Paint Brush")
        self.brush_enabled_cb.toggled.connect(self.toggle_brush_tool)
        brush_layout.addWidget(self.brush_enabled_cb)
        
        brush_group.setLayout(brush_layout)
        layout.addWidget(brush_group)
        
        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self.remove_selected_defect)
        self.remove_btn.clicked.connect(self._mark_unsaved)
        actions_layout.addWidget(self.remove_btn)
        
        self.clear_btn = QPushButton("Clear All Defects")
        self.clear_btn.clicked.connect(self.clear_all_defects)
        self.clear_btn.clicked.connect(self._mark_unsaved)
        actions_layout.addWidget(self.clear_btn)
        
        self.clear_paint_btn = QPushButton("Clear Paint Layer")
        self.clear_paint_btn.clicked.connect(self.clear_paint_layer)
        self.clear_paint_btn.clicked.connect(self._mark_unsaved)
        actions_layout.addWidget(self.clear_paint_btn)
        
        # [Removed] Preview Result button
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        return panel
        
    def create_right_panel(self):
        """Create right defect library panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Defect library
        defect_group = QGroupBox("Defect Library")
        defect_layout = QVBoxLayout()
        
        # Filter by type
        defect_layout.addWidget(QLabel("Filter by type:"))
        self.defect_filter = QComboBox()
        self.defect_filter.addItem("All")
        self.defect_filter.currentTextChanged.connect(self.filter_defects)
        defect_layout.addWidget(self.defect_filter)
        
        # Defect list
        defect_layout.addWidget(QLabel("Available defects:"))
        self.defect_list = QListWidget()
        self.defect_list.itemDoubleClicked.connect(self.add_defect_to_canvas)
        defect_layout.addWidget(self.defect_list)
        
        # Add defect button
        self.add_defect_btn = QPushButton("Add Selected Defect")
        self.add_defect_btn.clicked.connect(self.add_defect_to_canvas)
        self.add_defect_btn.clicked.connect(self._mark_unsaved)
        defect_layout.addWidget(self.add_defect_btn)
        
        defect_group.setLayout(defect_layout)
        layout.addWidget(defect_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        
        self.stats_label = QLabel("No dataset loaded")
        stats_layout.addWidget(self.stats_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        return panel
        
    def load_target_images(self):
        """Load target images from directory"""
        # Check for unsaved changes before loading new dataset
        if not self._check_unsaved_changes("loading new target images"):
            return
            
        self.target_images_dir = QFileDialog.getExistingDirectory(
            self, "Select Target Images Directory", os.getcwd()
        )
        
        if not self.target_images_dir:
            return
            
        # Scan for image files
        self.target_images = []
        for root, dirs, files in os.walk(self.target_images_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.target_images.append(os.path.join(root, file))
        
        if not self.target_images:
            QMessageBox.warning(self, "Warning", "No image files found in the selected directory.")
            return
            
        # Populate target list
        self.target_list.clear()
        for img_path in self.target_images[:50]:  # Limit to 50 for UI
            img_name = os.path.basename(img_path)
            item = QListWidgetItem(img_name)
            item.setData(Qt.UserRole, img_path)
            self.target_list.addItem(item)
            
        self.status_bar.showMessage(f"Loaded {len(self.target_images)} target images")
        
    def load_defect_images(self):
        """Load defect images from directory"""
        # Check for unsaved changes before loading new dataset
        if not self._check_unsaved_changes("loading new defect images"):
            return
            
        self.defect_images_dir = QFileDialog.getExistingDirectory(
            self, "Select Defect Images Directory", os.getcwd()
        )
        
        if not self.defect_images_dir:
            return
            
        # Scan for image files
        self.defect_images = []
        defect_types = set()
        for root, dirs, files in os.walk(self.defect_images_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    self.defect_images.append(img_path)
                    # Extract defect type from directory name
                    defect_type = os.path.basename(root) if root != self.defect_images_dir else 'defect'
                    defect_types.add(defect_type)
        
        if not self.defect_images:
            QMessageBox.warning(self, "Warning", "No defect image files found in the selected directory.")
            return
            
        # Update stats if masks are also loaded
        if self.defect_masks:
            self._update_stats()
        else:
            self.status_bar.showMessage(f"Loaded {len(self.defect_images)} defect images")
        
    def load_defect_masks(self):
        """Load defect masks from directory"""
        # Check for unsaved changes before loading new dataset
        if not self._check_unsaved_changes("loading new defect masks"):
            return
            
        self.defect_masks_dir = QFileDialog.getExistingDirectory(
            self, "Select Defect Masks Directory", os.getcwd()
        )
        
        if not self.defect_masks_dir:
            return
            
        # Scan for mask files
        self.defect_masks = []
        defect_types = set()
        for root, dirs, files in os.walk(self.defect_masks_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    mask_path = os.path.join(root, file)
                    self.defect_masks.append(mask_path)
                    # Extract defect type from directory name
                    defect_type = os.path.basename(root) if root != self.defect_masks_dir else 'defect'
                    defect_types.add(defect_type)
        
        if not self.defect_masks:
            QMessageBox.warning(self, "Warning", "No mask files found in the selected directory.")
            return
            
        # Populate defect list
        self.defect_list.clear()
        for mask_path in self.defect_masks:
            mask_name = os.path.basename(mask_path)
            # Extract defect type from directory
            relative_path = os.path.relpath(mask_path, self.defect_masks_dir)
            defect_type = os.path.dirname(relative_path) if os.path.dirname(relative_path) else 'defect'
            
            item_text = f"{defect_type} - {mask_name}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, mask_path)
            self.defect_list.addItem(item)
            
        # Update filter
        self.defect_filter.clear()
        self.defect_filter.addItem("All")
        for dtype in sorted(defect_types):
            self.defect_filter.addItem(dtype)
            
        # Update stats
        self._update_stats()
        
        self.status_bar.showMessage(f"Loaded {len(self.defect_masks)} defect masks")
    
    def _update_stats(self):
        """Update the statistics display"""
        defect_types = set()
        
        # Get defect types from masks if available
        if self.defect_masks:
            for mask_path in self.defect_masks:
                relative_path = os.path.relpath(mask_path, self.defect_masks_dir)
                defect_type = os.path.dirname(relative_path) if os.path.dirname(relative_path) else 'defect'
                defect_types.add(defect_type)
        
        # Get defect types from images if available
        if self.defect_images:
            for img_path in self.defect_images:
                relative_path = os.path.relpath(img_path, self.defect_images_dir)
                defect_type = os.path.dirname(relative_path) if os.path.dirname(relative_path) else 'defect'
                defect_types.add(defect_type)
        
        # Update stats display
        stats_text = f"Target Images: {len(self.target_images)}\n"
        if self.defect_images:
            stats_text += f"Defect Images: {len(self.defect_images)}\n"
        if self.defect_masks:
            stats_text += f"Defect Masks: {len(self.defect_masks)}\n"
        if defect_types:
            stats_text += f"Types: {', '.join(sorted(defect_types))}"
        
        self.stats_label.setText(stats_text)
            
    def on_target_selected(self, item):
        """Handle target image selection"""
        if not self.target_images:
            return
        
        # Check for unsaved changes before switching
        if not self._check_unsaved_changes("switching images"):
            return
            
        try:
            # Save current state before switching
            self.save_current_state_to_cache()
            
            # Get image path from item data
            image_path = item.data(Qt.UserRole)
            self.current_image_path = image_path
            
            # Load and display the image
            image_tensor = self._load_image_tensor(image_path)
            self.canvas.set_background_image(image_tensor)
            
            # Restore cached defects for this image, if any
            self.restore_state_from_cache()
            
            image_name = os.path.basename(image_path)
            self.status_bar.showMessage(f"Loaded target: {image_name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load target image:\n{str(e)}")
    
    def _load_image_tensor(self, image_path):
        """Load and convert image to tensor preserving original aspect ratio"""
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        return transform(image)
    
    def _load_mask_tensor(self, mask_path):
        """Load and convert mask to tensor preserving original aspect ratio"""
        mask = Image.open(mask_path).convert('L')
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        mask_tensor = transform(mask)
        return (mask_tensor > 0.5).float()
    
    def save_current_state_to_cache(self):
        """Persist current canvas defect items and paint layer into the cache for the active image."""
        if not self.current_image_path:
            return
        key = self.current_image_path
        items_state = []
        for item in self.canvas.defect_items:
            items_state.append({
                'type': item.defect_data.get('type', 'unknown'),
                'source': item.defect_data.get('source', 'unknown'),
                'mask_path': item.defect_data.get('mask_path', None),
                'defect_image_path': item.defect_data.get('defect_image_path', None),
                'x': float(item.x()),
                'y': float(item.y()),
                'scale': float(item.scale_factor),
                'rotation': float(item.rotation_angle),
                'opacity': float(item.opacity),
            })
        
        regions_state = []
        for item in self.canvas.region_items:
            regions_state.append({
                'type': item.region_data.get('type', 'selected_region'),
                'source': item.region_data.get('source', 'unknown'),
                'original_rect': item.region_data.get('original_rect', None),
                'x': float(item.x()),
                'y': float(item.y()),
                'scale': float(item.scale_factor),
                'rotation': float(item.rotation_angle),
                'opacity': float(item.opacity),
            })
        
        # Save paint layer if it exists and has content
        paint_layer_data = None
        if self.canvas.paint_layer and not self.canvas.paint_layer.isNull():
            # Check if paint layer has any non-transparent content
            paint_image = self.canvas.paint_layer.toImage()
            has_content = False
            for x in range(paint_image.width()):
                for y in range(paint_image.height()):
                    if paint_image.pixelColor(x, y).alpha() > 0:
                        has_content = True
                        break
                if has_content:
                    break
            
            if has_content:
                paint_layer_data = self.canvas.paint_layer.copy()
        
        self.augmentation_states[key] = {
            'items': items_state,
            'regions': regions_state,
            'paint_layer': paint_layer_data
        }
        
        # Also cache the paint layer separately for quick access
        if paint_layer_data:
            self.paint_layer_cache[key] = paint_layer_data
        elif key in self.paint_layer_cache:
            del self.paint_layer_cache[key]
    
    def restore_state_from_cache(self):
        """Restore cached defects and paint layer for the current image if present."""
        if not self.current_image_path:
            return
        key = self.current_image_path
        state = self.augmentation_states.get(key)
        if not state:
            return
        # Clear any existing
        self.canvas.clear_defects()
        self.canvas.clear_regions()
        self.canvas.clear_paint_layer()
        
        # Restore paint layer if it exists
        paint_layer_data = state.get('paint_layer')
        if paint_layer_data and not paint_layer_data.isNull():
            self.canvas.paint_layer = paint_layer_data.copy()
            self.canvas.paint_layer_item.setPixmap(self.canvas.paint_layer)
        
        for entry in state.get('items', []):
            # Try to find the mask by path or by type and source
            mask_path = entry.get('mask_path')
            defect_image_path = entry.get('defect_image_path')
            
            if not mask_path or not os.path.exists(mask_path):
                # Fallback: find by type and source name
                defect_type = entry['type']
                source_name = entry.get('source', '')
                mask_path = None
                for mask in self.defect_masks:
                    if defect_type in os.path.dirname(mask) and source_name in os.path.basename(mask):
                        mask_path = mask
                        break
                if not mask_path:
                    continue
            
            # If we don't have the defect image path, try to find it
            if not defect_image_path or not os.path.exists(defect_image_path):
                defect_image_path = self._find_corresponding_defect_image(mask_path)
                if not defect_image_path:
                    continue
            
            try:
                # Load the mask
                mask_tensor = self._load_mask_tensor(mask_path)
                
                # Load the defect image
                defect_image_tensor = self._load_image_tensor(defect_image_path)
                
                # Extract only the defect region using the mask
                defect_tensor = defect_image_tensor * mask_tensor
                
                # Crop to tight bounding box around the defect
                defect_tensor, mask_tensor = self._crop_to_defect_bounding_box(defect_tensor, mask_tensor)
                
                self.canvas.add_defect(
                    defect_tensor,
                    mask_tensor,
                    {
                        'type': entry['type'], 
                        'source': entry.get('source', 'unknown'),
                        'mask_path': mask_path,
                        'defect_image_path': defect_image_path
                    },
                    position=(entry['x'], entry['y']),
                    opacity_override=entry['opacity']
                )
                # Apply transform parameters
                self.canvas.selected_defect.update_transform(entry['scale'], entry['rotation'], entry['opacity'])
            except Exception:
                continue
        
        # Restore regions
        for entry in state.get('regions', []):
            try:
                # Create a simple region from the original rect
                original_rect = entry.get('original_rect')
                if not original_rect:
                    continue
                    
                x, y, w, h = original_rect
                
                # Extract region from current background image
                if self.canvas.background_item:
                    region_pixmap = self.canvas.background_item.pixmap().copy(x, y, w, h)
                    
                    # Create mask
                    mask_pixmap = QPixmap(w, h)
                    mask_pixmap.fill(Qt.white)
                    
                    # Create region item
                    region_item = SelectedRegionItem(
                        region_pixmap, 
                        mask_pixmap, 
                        {
                            'type': entry['type'],
                            'source': entry.get('source', 'unknown'),
                            'original_rect': original_rect
                        }
                    )
                    
                    # Set position and add to scene
                    region_item.setPos(entry['x'], entry['y'])
                    self.scene.addItem(region_item)
                    self.region_items.append(region_item)
                    
                    # Apply transform parameters
                    region_item.update_transform(entry['scale'], entry['rotation'], entry['opacity'])
            except Exception:
                continue
                
        self.has_unsaved_changes = True
                
    def add_defect_to_canvas(self, item=None):
        """Add selected defect to canvas"""
        if not item:
            item = self.defect_list.currentItem()
            
        if not item or not self.current_image_path:
            return
            
        # Get mask path from item data
        mask_path = item.data(Qt.UserRole)
        item_text = item.text()
        defect_type = item_text.split(" - ")[0]
        
        try:
            # Load the mask
            mask_tensor = self._load_mask_tensor(mask_path)
            
            # Find the corresponding defect image
            defect_image_path = self._find_corresponding_defect_image(mask_path)
            if not defect_image_path:
                QMessageBox.warning(self, "Warning", f"No corresponding defect image found for {os.path.basename(mask_path)}")
                return
                
            # Load the defect image
            defect_image_tensor = self._load_image_tensor(defect_image_path)
            
            # Extract only the defect region using the mask
            # The mask tells us where the defect is in the original image
            # Apply the mask to each channel and set background to transparent (0)
            defect_tensor = defect_image_tensor * mask_tensor
            
            # Crop to tight bounding box around the defect
            defect_tensor, mask_tensor = self._crop_to_defect_bounding_box(defect_tensor, mask_tensor)
            
            # Add to canvas
            self.canvas.add_defect(
                defect_tensor,
                mask_tensor,
                {
                    'type': defect_type,
                    'source': os.path.basename(mask_path),
                    'mask_path': mask_path,
                    'defect_image_path': defect_image_path
                }
            )
            
            # Reset transformation controls to defaults for the new defect
            self.scale_slider.setValue(100)
            self.rotation_slider.setValue(0)
            self.opacity_slider.setValue(70)
            self.scale_label.setText("1.0x")
            self.rotation_label.setText("0°")
            self.opacity_label.setText("0.7")

            self.status_bar.showMessage(f"Added defect: {defect_type}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load defect mask:\n{str(e)}")
                
    def _crop_to_defect_bounding_box(self, defect_tensor, mask_tensor):
        """Crop defect and mask to tight bounding box around the defect"""
        if mask_tensor.sum() == 0:
            return defect_tensor, mask_tensor
            
        # Find bounding box of non-zero mask values
        mask_binary = mask_tensor > 0.5
        coords = torch.where(mask_binary.squeeze(0))
        
        if len(coords[0]) == 0:
            return defect_tensor, mask_tensor
            
        y_min, y_max = coords[0].min().item(), coords[0].max().item()
        x_min, x_max = coords[1].min().item(), coords[1].max().item()
        
        # Add small margin for better visual appearance
        margin = 5
        y_min = max(0, int(y_min - margin))
        y_max = min(mask_tensor.shape[1], int(y_max + margin))
        x_min = max(0, int(x_min - margin))
        x_max = min(mask_tensor.shape[2], int(x_max + margin))
        
        # Crop both defect and mask
        cropped_defect = defect_tensor[:, y_min:y_max, x_min:x_max]
        cropped_mask = mask_tensor[:, y_min:y_max, x_min:x_max]
        
        return cropped_defect, cropped_mask

    def extract_defect(self, image, mask):
        """Extract defect region from image"""
        if mask.sum() == 0:
            return None, None
            
        # Find bounding box
        mask_binary = mask > 0.5
        coords = torch.where(mask_binary.squeeze(0))
        
        if len(coords[0]) == 0:
            return None, None
            
        y_min, y_max = coords[0].min().item(), coords[0].max().item()
        x_min, x_max = coords[1].min().item(), coords[1].max().item()
        
        # Add margin
        margin = 10
        y_min = max(0, y_min - margin)
        y_max = min(mask.shape[1], y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(mask.shape[2], x_max + margin)
        
        # Extract region
        defect_region = image[:, y_min:y_max, x_min:x_max]
        defect_mask = mask[:, y_min:y_max, x_min:x_max]
        
        return defect_region, defect_mask
        
    def update_defect_transform(self):
        """Update transformation of selected defect or region"""
        selected_item = self.canvas.selected_defect or self.canvas.selected_region
        if not selected_item:
            return
            
        scale = self.scale_slider.value() / 100.0
        rotation = self.rotation_slider.value()
        opacity = self.opacity_slider.value() / 100.0
        
        self.scale_label.setText(f"{scale:.1f}x")
        self.rotation_label.setText(f"{rotation}°")
        self.opacity_label.setText(f"{opacity:.1f}")
        
        # Update selected item (defect or region)
        selected_item.update_transform(scale, rotation, opacity)
        self._mark_unsaved()
        
    # [Removed] toggle_mask_display
        
    def remove_selected_defect(self):
        """Remove selected defect or region from canvas"""
        if self.canvas.selected_defect:
            self.canvas.remove_selected_defect()
            self.status_bar.showMessage("Removed selected defect")
        elif self.canvas.selected_region:
            self.canvas.remove_selected_region()
            self.status_bar.showMessage("Removed selected region")
        
    def clear_all_defects(self):
        """Clear all defects and regions from canvas"""
        # Check for unsaved changes before clearing
        if not self._check_unsaved_changes("clearing all defects and regions"):
            return
            
        self.canvas.clear_defects()
        self.canvas.clear_regions()
        self.status_bar.showMessage("Cleared all defects and regions")
    
    def clear_paint_layer(self):
        """Clear the paint layer"""
        # Check for unsaved changes before clearing paint
        if not self._check_unsaved_changes("clearing paint layer"):
            return
            
        self.canvas.clear_paint_layer()
        self.status_bar.showMessage("Cleared paint layer")
        
    def clear_all(self):
        """Clear everything"""
        # Check for unsaved changes before clearing everything
        if not self._check_unsaved_changes("clearing everything"):
            return
            
        self.canvas.clear_defects()
        self.canvas.clear_regions()
        self.canvas.clear_paint_layer()
        self.canvas.scene.clear()
        self.canvas.background_item = None
        self.canvas.selected_defect = None
        self.canvas.selected_region = None
        self.status_bar.showMessage("Cleared canvas")
    
    def _mark_unsaved(self):
        self.has_unsaved_changes = True
        # Mark current image as unsaved if it was previously saved
        if self.current_image_path and self.current_image_path in self.saved_images:
            self.saved_images.remove(self.current_image_path)
        # Update window title to show unsaved changes
        self._update_window_title()
    
    def _update_window_title(self):
        """Update window title to show unsaved changes"""
        base_title = "DefectPaste - Interactive Defect Placement Tool"
        if self.has_unsaved_changes:
            self.setWindowTitle(f"{base_title} *")
        else:
            self.setWindowTitle(base_title)
    
    def _check_unsaved_changes(self, action_name="this action"):
        """Check for unsaved changes and prompt user to save"""
        if not self.has_unsaved_changes:
            return True
        
        # Check if there are actual changes (defects, regions, or paint)
        has_defects = len(self.canvas.defect_items) > 0
        has_regions = len(self.canvas.region_items) > 0
        has_paint = self.has_unsaved_paint_changes()
        
        if not (has_defects or has_regions or has_paint):
            return True
        
        # Create appropriate message
        if has_defects and has_regions and has_paint:
            message = f"You have unsaved defects, regions, and paint changes. Do you want to save before {action_name}?"
        elif has_defects and has_regions:
            message = f"You have unsaved defects and regions. Do you want to save before {action_name}?"
        elif has_defects and has_paint:
            message = f"You have unsaved defects and paint changes. Do you want to save before {action_name}?"
        elif has_regions and has_paint:
            message = f"You have unsaved regions and paint changes. Do you want to save before {action_name}?"
        elif has_defects:
            message = f"You have unsaved defects. Do you want to save before {action_name}?"
        elif has_regions:
            message = f"You have unsaved regions. Do you want to save before {action_name}?"
        else:
            message = f"You have unsaved paint changes. Do you want to save before {action_name}?"
        
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            message,
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            self.save_augmented_image()
            return True
        elif reply == QMessageBox.No:
            return True
        else:  # Cancel
            return False
    
    def has_unsaved_paint_changes(self):
        """Check if there are unsaved paint changes"""
        if not self.canvas.paint_layer or self.canvas.paint_layer.isNull():
            return False
        
        # Check if paint layer has any non-transparent content
        paint_image = self.canvas.paint_layer.toImage()
        for x in range(paint_image.width()):
            for y in range(paint_image.height()):
                if paint_image.pixelColor(x, y).alpha() > 0:
                    return True
        return False
        
    # [Removed] on_blend_mode_changed (blend mode removed)

    # [Removed] on_live_preview_toggled
        
    def filter_defects(self, filter_type):
        """Filter defect list by type"""
        # Hide or show items based on filter
        first_visible_index = None
        for i in range(self.defect_list.count()):
            item = self.defect_list.item(i)
            is_visible = (filter_type == "All" or filter_type == item.text().split(" - ")[0])
            item.setHidden(not is_visible)
            if is_visible and first_visible_index is None:
                first_visible_index = i
        # Auto-select the first visible item so actions use the intended type
        if first_visible_index is not None:
            self.defect_list.setCurrentRow(first_visible_index)
                
    # [Removed] preview_result and show_preview
        
    def save_augmented_image(self):
        """Save the augmented image and mask"""
        result_image, result_mask = self.canvas.get_augmented_image()
        
        if result_image is None:
            QMessageBox.information(self, "Info", "No defects placed yet")
            return
            
        # Get save path
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Augmented Image", os.getcwd(), "PNG Files (*.png)"
        )
        
        if save_path:
            # Save image
            image_pil = TF.to_pil_image(result_image)
            image_pil.save(save_path)
            
            # Save mask
            mask_path = save_path.replace('.png', '_mask.png')
            mask_pil = TF.to_pil_image(result_mask)
            mask_pil.save(mask_path)
            
            # Save metadata
            metadata = {
                'target_image': os.path.basename(self.current_image_path),
                'target_image_path': self.current_image_path,
                'defects': [
                    {
                        'type': item.defect_data.get('type', 'unknown'),
                        'position': item.get_position(),
                        'scale': item.scale_factor,
                        'rotation': item.rotation_angle,
                        'opacity': item.opacity,
                        'mask_path': item.defect_data.get('mask_path'),
                        'defect_image_path': item.defect_data.get('defect_image_path')
                    }
                    for item in self.canvas.defect_items
                ],
                'regions': [
                    {
                        'type': item.region_data.get('type', 'selected_region'),
                        'position': item.get_position(),
                        'scale': item.scale_factor,
                        'rotation': item.rotation_angle,
                        'opacity': item.opacity,
                        'original_rect': item.region_data.get('original_rect'),
                        'source': item.region_data.get('source', 'unknown')
                    }
                    for item in self.canvas.region_items
                ]
            }
            
            meta_path = save_path.replace('.png', '_metadata.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Mark current image as saved
            if self.current_image_path:
                self.saved_images.add(self.current_image_path)
                self.has_unsaved_changes = False
                self._update_window_title()
            
            self.status_bar.showMessage(f"Saved to: {save_path}")
            QMessageBox.information(self, "Success", "Augmented image saved successfully!")
    
    def _find_next_index(self, directory: str, base_name: str) -> int:
        """Return next integer index to use for files named like base_name_#.png in directory."""
        try:
            existing = os.listdir(directory)
        except Exception:
            return 1
        prefix = f"{base_name}_"
        max_idx = 0
        for name in existing:
            if not name.lower().endswith('.png'):
                continue
            if not name.startswith(prefix):
                continue
            stem = name[:-4]
            parts = stem.split('_')
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[-1])
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                continue
        return max_idx + 1

    def _get_image_index_by_path(self, image_path: str) -> Optional[int]:
        try:
            return self.target_images.index(image_path)
        except ValueError:
            return None
    
    def _find_corresponding_defect_image(self, mask_path: str) -> Optional[str]:
        """Find the corresponding defect image for a given mask path by matching filenames"""
        if not self.defect_images_dir or not self.defect_images:
            return None
            
        # Get the mask filename without extension
        mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
        
        # Try exact filename match first
        for defect_img in self.defect_images:
            defect_filename = os.path.splitext(os.path.basename(defect_img))[0]
            if mask_filename == defect_filename:
                return defect_img
        
        # Try partial filename matching (in case of slight naming differences)
        for defect_img in self.defect_images:
            defect_filename = os.path.splitext(os.path.basename(defect_img))[0]
            # Check if one filename contains the other (case insensitive)
            if (mask_filename.lower() in defect_filename.lower() or 
                defect_filename.lower() in mask_filename.lower()):
                return defect_img
        
        # If no match found, return None
        return None

    def save_all_augmentations(self):
        """Save all cached augmentations to a chosen folder with a base name and incremental indices."""
        if not self.augmentation_states:
            QMessageBox.information(self, "Info", "No augmentations to save.")
            return
        # Choose output dir
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory for All Augmented", os.getcwd())
        if not output_dir:
            return
        # Choose base name
        from PyQt5.QtWidgets import QInputDialog
        base_name, ok = QInputDialog.getText(self, "Base filename", "Enter base filename:", text="augmented")
        if not ok or not base_name:
            return
        start_idx = self._find_next_index(output_dir, base_name)
        current_idx = start_idx
        
        # Remember current context to restore later
        orig_image_path = self.current_image_path
        
        num_saved = 0
        # Iterate deterministically
        for key in list(self.augmentation_states.keys()):
            state = self.augmentation_states.get(key)
            if not state or not state.get('items'):
                continue
            if not os.path.exists(key):
                continue
            # Load target image
            self.current_image_path = key
            image_tensor = self._load_image_tensor(key)
            self.canvas.set_background_image(image_tensor)
            # Restore this image's state
            self.restore_state_from_cache()
            # Render
            result_image, result_mask = self.canvas.get_augmented_image()
            if result_image is None:
                continue
            # Save files
            img_filename = f"{base_name}_{current_idx}.png"
            mask_filename = f"{base_name}_{current_idx}_mask.png"
            img_path = os.path.join(output_dir, img_filename)
            mask_path = os.path.join(output_dir, mask_filename)
            image_pil = TF.to_pil_image(result_image)
            image_pil.save(img_path)
            mask_pil = TF.to_pil_image(result_mask)
            mask_pil.save(mask_path)
            # Save metadata
            metadata = {
                'target_image': os.path.basename(key),
                'target_image_path': key,
                'defects': [
                    {
                        'type': item.defect_data.get('type', 'unknown'),
                        'position': item.get_position(),
                        'scale': item.scale_factor,
                        'rotation': item.rotation_angle,
                        'opacity': item.opacity,
                        'mask_path': item.defect_data.get('mask_path'),
                        'defect_image_path': item.defect_data.get('defect_image_path')
                    }
                    for item in self.canvas.defect_items
                ],
                'regions': [
                    {
                        'type': item.region_data.get('type', 'selected_region'),
                        'position': item.get_position(),
                        'scale': item.scale_factor,
                        'rotation': item.rotation_angle,
                        'opacity': item.opacity,
                        'original_rect': item.region_data.get('original_rect'),
                        'source': item.region_data.get('source', 'unknown')
                    }
                    for item in self.canvas.region_items
                ]
            }
            meta_path = os.path.join(output_dir, f"{base_name}_{current_idx}_metadata.json")
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            current_idx += 1
            num_saved += 1
        
        # Restore original context
        if orig_image_path is not None:
            self.current_image_path = orig_image_path
            image_tensor = self._load_image_tensor(orig_image_path)
            self.canvas.set_background_image(image_tensor)
            self.restore_state_from_cache()
        
        if num_saved == 0:
            QMessageBox.information(self, "Info", "No augmentations were saved.")
        else:
            # Mark all saved images as saved
            for key in list(self.augmentation_states.keys()):
                if key in self.augmentation_states and self.augmentation_states[key].get('items'):
                    self.saved_images.add(key)
            self.has_unsaved_changes = False
            self._update_window_title()
            QMessageBox.information(self, "Success", f"Saved {num_saved} augmentations to {output_dir}.")
            
    # [Removed] export_batch
        
    def on_defect_placed(self, info):
        """Handle defect placement signal"""
        self.status_bar.showMessage(
            f"Placed {info['type']} at ({info['position'][0]:.0f}, {info['position'][1]:.0f})"
        )
        self._mark_unsaved()
    
    def on_brush_mode_changed(self, mode):
        """Handle brush mode change"""
        self.canvas.brush_mode = mode
        self.status_bar.showMessage(f"Brush mode changed to: {mode}")
    
    def update_brush_settings(self):
        """Update brush settings from UI controls"""
        size = self.brush_size_slider.value()
        opacity = self.brush_opacity_slider.value()
        
        self.brush_size_label.setText(f"{size}px")
        self.brush_opacity_label.setText(f"{opacity}%")
        
        # Update canvas brush settings
        self.canvas.set_brush_settings(
            self.brush_enabled_cb.isChecked(),
            self.brush_mode.currentText(),
            size,
            opacity,
            self.canvas.brush_color
        )
    
    def choose_brush_color(self):
        """Open color dialog to choose brush color"""
        color = QColorDialog.getColor(self.canvas.brush_color, self, "Choose Brush Color")
        if color.isValid():
            self.canvas.brush_color = color
            # Update button color
            self.brush_color_btn.setStyleSheet(f"background-color: {color.name()}; color: {'white' if color.lightness() < 128 else 'black'};")
            self.update_brush_settings()
    
    def toggle_brush_tool(self, enabled):
        """Toggle brush tool on/off"""
        self.canvas.brush_enabled = enabled
        self.update_brush_settings()
        
        if enabled:
            self.status_bar.showMessage("Paint brush enabled - Click and drag to paint")
        else:
            self.status_bar.showMessage("Paint brush disabled")
    
    def toggle_selection_tool(self, enabled):
        """Toggle selection tool on/off"""
        self.canvas.selection_enabled = enabled
        
        if enabled:
            self.status_bar.showMessage("Selection tool enabled - Click and drag to select region (auto-creates draggable region)")
            # Disable brush tool when selection is enabled
            self.brush_enabled_cb.setChecked(False)
            self.canvas.brush_enabled = False
            self.update_brush_settings()
            # Update cursor for selection mode
            self.canvas.setCursor(Qt.CrossCursor)
        else:
            self.status_bar.showMessage("Selection tool disabled")
            self.canvas.clear_selection()
            self.canvas.setCursor(Qt.ArrowCursor)
    
    def on_selection_mode_changed(self, mode):
        """Handle selection mode change"""
        self.canvas.selection_mode = mode
        self.status_bar.showMessage(f"Selection mode changed to: {mode}")
    
    def copy_selection(self):
        """Copy the current selection"""
        self.canvas.copy_selection()
        self.status_bar.showMessage("Selection copied as draggable region")
        self._mark_unsaved()
    
    def clear_selection(self):
        """Clear the current selection"""
        self.canvas.clear_selection()
        self.status_bar.showMessage("Selection cleared")
    
    def on_region_placed(self, info):
        """Handle region placement signal"""
        if 'has_selection' in info:
            self.copy_selection_btn.setEnabled(info['has_selection'])
        else:
            self.status_bar.showMessage(
                f"Region created and placed at ({info['position'][0]:.0f}, {info['position'][1]:.0f}) - You can now drag and transform it!"
            )
            self._mark_unsaved()

    def closeEvent(self, event):
        """Prompt to save changes on close if needed."""
        try:
            self.save_current_state_to_cache()
        except Exception:
            pass
        
        # Use the comprehensive save checking system
        if not self._check_unsaved_changes("exiting the application"):
            event.ignore()
        else:
            event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    # Set application info
    app.setApplicationName("DefectPaste")
    app.setOrganizationName("Defect Augmentation")
    
    # Create and show main window
    window = DefectPlacementTool()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()