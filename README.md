# DefectPaste 🎯

**Interactive Copy-Paste Tool for Augmentation**

DefectPaste is a GUI application that allows you to manually place defects on objects with drag-and-drop functionality, perfect for data augmentation and defect analysis workflows.

## ✨ Features

- **Interactive Defect Placement**: Drag and drop defects onto target images
- **Selection Tool**: Select and copy portions of images to create custom regions
- **Real-time Transformations**: Scale, rotate, and adjust opacity of defects and regions
- **Multiple Defect Types**: Support for various defect categories
- **Paint Brush Tool**: Manual painting and erasing capabilities
- **Batch Processing**: Save multiple augmented images at once

## 🚀 Quick Start

### Installation

1. **Clone or download the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python defectpaste.py
   ```

## 🎮 How to Use

### 1. Load Your Data
- **Load Defect Images**: Click "Load Defect Images" and select your directory
- **Load Defect Masks**: Click "Load Defect Masks" and select your directory  
- **Load Target Images**: Click "Load Target Images" and select your directory

### 2. Place Defects
- **Select a target image** from the left panel
- **Choose a defect** from the defect library on the right
- **Double-click or click "Add Selected Defect"** to place it on the canvas
- **Drag the defect** to position it where you want

### 3. Transform Defects
- **Scale**: Use the scale slider (25% - 200%)
- **Rotate**: Use the rotation slider (-180° to +180°)
- **Opacity**: Adjust transparency (10% - 100%)

### 4. Manage Defects
- **Select**: Click on a defect to select it
- **Remove**: Click "Remove Selected" to delete the selected defect
- **Clear All**: Click "Clear All Defects" to remove all defects

### 5. Selection Tool
- **Enable Selection Tool**: Check the "Enable Selection Tool" checkbox in the left panel
- **Choose Mode**: Select "Rectangle" or "Freehand" from the dropdown
- **Rectangle Selection**: Click and drag to create a rectangular selection
- **Freehand Selection**: Click and drag to draw a custom selection shape
- **Auto-Create Region**: Release mouse to automatically create a draggable region and unselect the tool box
- **Transform Region**: Use the same controls as defects (scale, rotation, opacity)

### 6. Paint Brush Tool
- **Enable Brush**: Check "Enable Paint Brush" in the left panel
- **Choose Mode**: Select "Paint" or "Erase" from the dropdown
- **Adjust Settings**: Set brush size, opacity, and color
- **Paint**: Click and drag to paint or erase on the canvas
- **Clear Paint**: Use "Clear Paint Layer" to remove all paint

### 7. Save Results
- **Save Single**: Click "Save Augmented" to save the current image
- **Save All**: Click "Save All Augmented..." to batch save all images with their defects

## 💾 File Outputs

When you save an augmented image, DefectPaste creates:

- **`filename.png`**: The augmented image with defects
- **`filename_mask.png`**: The corresponding mask
- **`filename_metadata.json`**: Detailed information about the augmentation

### Metadata Format
```json
{
  "target_image": "",
  "target_image_path": "/path/to/image",
  "defects": [
    {
      "type": "",
      "position": [],
      "scale": 1.0,
      "rotation": 0,
      "opacity": 0.7,
      "mask_path": "/path/to/mask",
      "defect_image_path": "/path/to/defect"
    }
  ],
  "regions": [
    {
      "type": "selected_region",
      "position": [100, 150],
      "scale": 1.2,
      "rotation": 45,
      "opacity": 0.8,
      "original_rect": [50, 75, 100, 80],
      "source": "region_0"
    }
  ]
}
```

## 📋 Requirements

```
PyQt5>=5.15.0
torch>=1.8.0
torchvision>=0.9.0
Pillow>=8.0.0
numpy>=1.19.0
```

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve DefectPaste!

## 📄 License

This project is open source and available under the MIT License.
