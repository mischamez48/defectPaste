# DefectPaste 🎯

**Interactive Defect Placement Tool for MVTec AD Dataset**

DefectPaste is a GUI application that allows you to manually place defects on objects with drag-and-drop functionality, perfect for data augmentation and defect analysis workflows.

## ✨ Features

- **Interactive Defect Placement**: Drag and drop defects onto target images
- **Real-time Transformations**: Scale, rotate, and adjust opacity of defects
- **Transparent Overlay**: Defects blend seamlessly with background images
- **Multiple Defect Types**: Support for various defect categories
- **Batch Processing**: Save multiple augmented images at once
- **State Management**: Preserve defect placements when switching between images

## 🚀 Quick Start

### Installation

1. **Clone or download the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python interactive_defect_tool.py
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

### 5. Save Results
- **Save Single**: Click "Save Augmented" to save the current image
- **Save All**: Click "Save All Augmented..." to batch save all images with their defects

## 🛠️ Controls

### Left Panel - Image & Transform Controls
- **Target Image List**: Select which image to work on
- **Scale Slider**: Resize defects (25% - 200%)
- **Rotation Slider**: Rotate defects (-180° to +180°)
- **Opacity Slider**: Control transparency (10% - 100%)
- **Action Buttons**: Remove selected or clear all defects

### Right Panel - Defect Library
- **Filter Dropdown**: Filter defects by type
- **Defect List**: Browse available defects
- **Add Button**: Place selected defect on canvas
- **Statistics**: View loaded data counts

### Canvas - Main Work Area
- **Drag & Drop**: Move defects around
- **Click to Select**: Select defects for transformation
- **Real-time Preview**: See changes as you make them

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
      "scale": ,
      "rotation": ,
      "opacity": ,
      "mask_path": "/path/to/mask",
      "defect_image_path": "/path/to/defect"
    }
  ]
}
```

## 🔧 Technical Details

### Supported Formats
- **Images**: PNG, JPG, JPEG
- **Masks**: PNG (grayscale)
- **Output**: PNG with transparency support

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

---

**DefectPaste** - Copy-paste defects with precision! 🎯
