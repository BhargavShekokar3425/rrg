import os
import json
import cv2
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from matplotlib.patches import Polygon
from pathlib import Path
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET


class DICOMVisualizerWithMalignancy:
    def __init__(self):
        self.dicom_data = {}
        self.annotations = {}

    # --- updated helper: fixed lung windowing ---
    def _window_image_for_lung(self, ds, use_dicom_window=False):
        """
        Return uint8 image with lung windowing suitable for lung nodule segmentation.

        If use_dicom_window == False (default): force a standard lung window
            WL = -650 HU, WW = 1200 HU

        If use_dicom_window == True: try DICOM window first, otherwise fall back
        to lung window.
        """
        img = ds.pixel_array.astype(np.float32)

        # Apply rescale slope/intercept if present
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        img = img * slope + intercept

        # Standard lung window (inside your recommended ranges)
        lung_wc = -650.0   # Window level
        lung_ww = 1200.0   # Window width

        wc = lung_wc
        ww = lung_ww

        if use_dicom_window:
            # Try to read window from DICOM, fall back to lung window if invalid
            dicom_wc = getattr(ds, "WindowCenter", None)
            dicom_ww = getattr(ds, "WindowWidth", None)

            def _to_scalar(v):
                if v is None:
                    return None
                try:
                    if isinstance(v, (list, tuple)) or hasattr(v, "__len__"):
                        return float(v[0])
                    return float(v)
                except Exception:
                    return None

            wc_tmp = _to_scalar(dicom_wc)
            ww_tmp = _to_scalar(dicom_ww)

            if wc_tmp is not None and ww_tmp is not None and ww_tmp > 0:
                wc, ww = wc_tmp, ww_tmp

        # Windowing
        low = wc - ww / 2.0
        high = wc + ww / 2.0
        img = np.clip(img, low, high)

        # Normalize to 0–255 uint8
        img = (img - low) / (high - low) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def load_dicom_series(self, dicom_folder):
        """Load DICOM series from folder"""
        dicom_files = []
        for file_path in Path(dicom_folder).glob("*.dcm"):
            try:
                ds = pydicom.dcmread(file_path)
                dicom_files.append((ds.SOPInstanceUID, ds))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # Sort by instance number if available
        try:
            dicom_files.sort(key=lambda x: int(x[1].InstanceNumber))
        except:
            pass
            
        self.dicom_data = {sop_id: ds for sop_id, ds in dicom_files}
        print(f"Loaded {len(self.dicom_data)} DICOM files")
        return self.dicom_data
    
    def parse_lidc_xml(self, xml_path):
        """Parse LIDC XML file and return nodules data"""
        ns = {'lidc': 'http://www.nih.gov'}
        root = ET.parse(xml_path).getroot()

        def txt(elem, tag):
            t = elem.find(f'lidc:{tag}', ns)
            return t.text.strip() if t is not None and t.text is not None else None

        nodules = []
        for session in root.findall('lidc:readingSession', ns):
            for n in session.findall('lidc:unblindedReadNodule', ns):
                nid = txt(n, 'noduleID')
                c = n.find('lidc:characteristics', ns)
                characteristics = {}
                if c is not None:
                    characteristics = {
                        'malignancy': txt(c, 'malignancy'),
                        'spiculation': txt(c, 'spiculation'),
                        'sphericity': txt(c, 'sphericity'),
                        'margin': txt(c, 'margin'),
                        'calcification': txt(c, 'calcification')
                    }

                rois = []
                for roi in n.findall('lidc:roi', ns):
                    z = txt(roi, 'imageZposition')
                    sop = txt(roi, 'imageSOP_UID') or txt(roi, 'imageSOP_UID ')
                    inc = txt(roi, 'inclusion')
                    pts = []
                    for em in roi.findall('lidc:edgeMap', ns):
                        x = int(em.find('lidc:xCoord', ns).text)
                        y = int(em.find('lidc:yCoord', ns).text)
                        pts.append((x, y))
                    
                    rois.append({
                        'z': float(z) if z is not None else None,
                        'sop': sop,
                        'inclusion': (inc == 'TRUE'),
                        'points': pts
                    })

                nodules.append({
                    'noduleID': nid,
                    'rois': rois,
                    'characteristics': characteristics
                })
        return nodules
    
    def parse_annotations(self, nodules_data):
        """Parse LIDC annotation data into organized structure"""
        self.annotations = {}
        
        for nodule in nodules_data:
            nodule_id = nodule['noduleID']
            characteristics = nodule.get('characteristics', {})
            
            for roi in nodule['rois']:
                sop_id = roi['sop']
                z_coord = roi['z']
                points = roi['points']
                inclusion = roi.get('inclusion', True)
                
                if sop_id not in self.annotations:
                    self.annotations[sop_id] = {'nodules': [], 'characteristics': {}}
                
                self.annotations[sop_id]['nodules'].append({
                    'id': nodule_id,
                    'points': points,
                    'z': z_coord,
                    'inclusion': inclusion,
                    'characteristics': characteristics
                })
                
                # Store characteristics at slice level
                if not self.annotations[sop_id]['characteristics']:
                    self.annotations[sop_id]['characteristics'] = characteristics
    
    def get_malignancy_color(self, malignancy_score):
        """Get color based on malignancy score"""
        if not malignancy_score:
            return 'gray', 'white'
            
        try:
            score = int(malignancy_score)
            if score <= 2:
                return 'green', 'white'
            elif score == 3:
                return 'orange', 'black'
            else:
                return 'red', 'white'
        except:
            return 'gray', 'white'
    
    def save_slice_as_png(self, sop_id, output_folder, slice_name, save_groundtruth=True):
        """Save DICOM slice as PNG with optional groundtruth mask and annotations"""
        if sop_id not in self.dicom_data:
            print(f"DICOM with SOP ID {sop_id} not found")
            return None

        ds = self.dicom_data[sop_id]

        # --- use robust lung windowing here ---
        image = self._window_image_for_lung(ds)

        # Create output directories
        images_dir = os.path.join(output_folder, 'images')
        masks_dir = os.path.join(output_folder, 'masks')
        annotations_dir = os.path.join(output_folder, 'annotations')

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)

        # Save original image
        image_path = os.path.join(images_dir, f"{slice_name}.png")
        cv2.imwrite(image_path, image)

        # Create and save groundtruth mask
        annotations = self.annotations.get(sop_id, {'nodules': [], 'characteristics': {}})
        nodule_mask = np.zeros(image.shape, dtype=np.uint8)

        annotation_data = {
            'image': f"{slice_name}.png",
            'mask': f"{slice_name}_mask.png",
            'nodules': []
        }

        mask_path = None
        if save_groundtruth and annotations['nodules']:
            for nodule in annotations['nodules']:
                points = nodule['points']
                if len(points) >= 3 and nodule['inclusion']:
                    pts = np.array(points, dtype=np.int32)
                    cv2.fillPoly(nodule_mask, [pts], 255)

                    characteristics = nodule.get('characteristics', {})
                    annotation_data['nodules'].append({
                        'nodule_id': nodule['id'],
                        'points': points,
                        'malignancy': characteristics.get('malignancy'),
                        'spiculation': characteristics.get('spiculation'),
                        'sphericity': characteristics.get('sphericity'),
                        'margin': characteristics.get('margin'),
                        'calcification': characteristics.get('calcification')
                    })

            mask_path = os.path.join(masks_dir, f"{slice_name}_mask.png")
            cv2.imwrite(mask_path, nodule_mask)

        # Save annotation JSON
        annotation_path = os.path.join(annotations_dir, f"{slice_name}.json")
        with open(annotation_path, 'w') as f:
            json.dump(annotation_data, f, indent=2)

        return {
            'image_path': image_path,
            'mask_path': mask_path,
            'annotation_path': annotation_path
        }

    def visualize_slice(self, sop_id, figsize=(15, 8)):
        """Visualize a single DICOM slice with annotations and malignancy info"""
        if sop_id not in self.dicom_data:
            print(f"DICOM with SOP ID {sop_id} not found")
            return None

        ds = self.dicom_data[sop_id]

        # --- use same lung windowing as for saving ---
        image = self._window_image_for_lung(ds)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        annotations = self.annotations.get(sop_id, {'nodules': [], 'characteristics': {}})

        # Display original image with annotations
        ax1.imshow(image, cmap='gray')
        ax1.set_title(f'DICOM Slice with Malignancy Labels\nSOP: {sop_id[:20]}...')
        ax1.axis('off')
        
        # Create masks
        nodule_mask = np.zeros(image.shape, dtype=np.uint8)
        
        # Group nodules by ID
        nodule_groups = {}
        for nodule in annotations['nodules']:
            nid = nodule['id']
            if nid not in nodule_groups:
                nodule_groups[nid] = []
            nodule_groups[nid].append(nodule)
        
        # Collect all nodule positions
        all_nodule_bounds = []
        for nodule_list in nodule_groups.values():
            for nodule in nodule_list:
                if len(nodule['points']) >= 3:
                    x_coords = [p[0] for p in nodule['points']]
                    y_coords = [p[1] for p in nodule['points']]
                    all_nodule_bounds.append({
                        'min_x': min(x_coords), 'max_x': max(x_coords),
                        'min_y': min(y_coords), 'max_y': max(y_coords),
                        'centroid_x': np.mean(x_coords), 'centroid_y': np.mean(y_coords)
                    })
        
        used_labels = []
        
        def find_best_label_position(nodule_bounds, label_text, attempt=0):
            centroid_x = nodule_bounds['centroid_x']
            centroid_y = nodule_bounds['centroid_y']
            min_x, max_x = nodule_bounds['min_x'], nodule_bounds['max_x']
            min_y, max_y = nodule_bounds['min_y'], nodule_bounds['max_y']
            
            label_width = len(label_text.split('\n')[0]) * 8 + 20
            label_height = len(label_text.split('\n')) * 12 + 10
            
            base_offset = 40 + (attempt * 20)
            
            candidates = [
                (max_x + base_offset, centroid_y, 'right'),
                (min_x - base_offset, centroid_y, 'left'),
                (centroid_x, min_y - base_offset, 'top'),
                (centroid_x, max_y + base_offset, 'bottom'),
                (max_x + base_offset, min_y, 'top-right'),
                (max_x + base_offset, max_y, 'bottom-right'),
                (min_x - base_offset, min_y, 'top-left'),
                (min_x - base_offset, max_y, 'bottom-left'),
            ]
            
            for x, y, pos_type in candidates:
                if (x < label_width/2 + 10 or y < label_height/2 + 10 or 
                    x > image.shape[1] - label_width/2 - 10 or y > image.shape[0] - label_height/2 - 10):
                    continue
                    
                collision = False
                for used_x, used_y, used_w, used_h in used_labels:
                    if (abs(x - used_x) < (label_width + used_w) / 2 + 15 and 
                        abs(y - used_y) < (label_height + used_h) / 2 + 15):
                        collision = True
                        break
                
                nodule_collision = False
                for bounds in all_nodule_bounds:
                    nodule_margin = 15
                    expanded_min_x = bounds['min_x'] - nodule_margin
                    expanded_max_x = bounds['max_x'] + nodule_margin  
                    expanded_min_y = bounds['min_y'] - nodule_margin
                    expanded_max_y = bounds['max_y'] + nodule_margin
                    
                    label_min_x = x - label_width/2
                    label_max_x = x + label_width/2
                    label_min_y = y - label_height/2
                    label_max_y = y + label_height/2
                    
                    if (label_max_x > expanded_min_x and label_min_x < expanded_max_x and
                        label_max_y > expanded_min_y and label_min_y < expanded_max_y):
                        nodule_collision = True
                        break
                
                if not collision and not nodule_collision:
                    return x, y, label_width, label_height
            
            if attempt < 4:
                return find_best_label_position(nodule_bounds, label_text, attempt + 1)
            
            max_nodule_x = max(all_nodule_bounds, key=lambda b: b['max_x'])['max_x']
            fallback_x = max_nodule_x + 100 + (attempt * 25)
            fallback_y = 60 + len(used_labels) * 35
            return fallback_x, fallback_y, label_width, label_height
        
        # Plot nodule annotations
        for nodule_id, nodule_list in nodule_groups.items():
            for j, nodule in enumerate(nodule_list):
                points = nodule['points']
                if len(points) >= 3:
                    characteristics = nodule.get('characteristics', {})
                    malignancy = characteristics.get('malignancy')
                    
                    bbox_color, text_color = self.get_malignancy_color(malignancy)
                    edge_color = bbox_color if malignancy else ('red' if nodule['inclusion'] else 'blue')
                    
                    poly = Polygon(points, closed=True, alpha=0.15,
                                 facecolor='none',  
                                 edgecolor=edge_color,
                                 linewidth=2)
                    ax1.add_patch(poly)
                    
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    nodule_bounds = {
                        'min_x': min(x_coords), 'max_x': max(x_coords),
                        'min_y': min(y_coords), 'max_y': max(y_coords),
                        'centroid_x': np.mean(x_coords), 'centroid_y': np.mean(y_coords)
                    }
                    
                    if len(nodule_list) > 1:
                        label_text = f"N{nodule_id}-R{j+1}"
                        if malignancy:
                            label_text += f"\nM:{malignancy}"
                    else:
                        label_text = f"N{nodule_id}"
                        if malignancy:
                            label_text += f"\nM:{malignancy}"
                    
                    label_x, label_y, label_w, label_h = find_best_label_position(nodule_bounds, label_text)
                    used_labels.append((label_x, label_y, label_w, label_h))
                    
                    ax1.plot([nodule_bounds['centroid_x'], label_x], [nodule_bounds['centroid_y'], label_y], 
                            color=bbox_color, linewidth=1, alpha=0.6, linestyle='--')
                    
                    ax1.text(label_x, label_y, label_text, 
                            color=text_color, fontsize=8, ha='center', va='center', weight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor=bbox_color, 
                                    alpha=0.9, edgecolor='black', linewidth=1))
                    
                    if nodule['inclusion'] and len(points) >= 3:
                        pts = np.array(points, dtype=np.int32)
                        cv2.fillPoly(nodule_mask, [pts], 255)
        
        # Create legend
        if annotations['nodules']:
            legend_elements = [
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                          markersize=10, alpha=0.9, label='Low Malignancy (1-2)', markeredgecolor='black'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', 
                          markersize=10, alpha=0.9, label='Moderate Malignancy (3)', markeredgecolor='black'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                          markersize=10, alpha=0.9, label='High Malignancy (4-5)', markeredgecolor='black'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                          markersize=10, alpha=0.9, label='No Malignancy Data', markeredgecolor='black')
            ]
            ax1.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)
        
        ax2.imshow(nodule_mask, cmap='hot')
        ax2.set_title('Inclusion Masks Only')
        ax2.axis('off')
        
        ax3.imshow(image, cmap='gray')
        ax3.imshow(nodule_mask, cmap='hot', alpha=0.4)
        ax3.set_title('Overlay (Inclusion Only)')
        ax3.axis('off')
        
        plt.tight_layout()
        return fig, nodule_mask
    
    def get_slice_with_annotations(self, max_slices=None):
        """Get SOP IDs of slices that have annotations"""
        annotated_slices = []
        for sop_id, annotations in self.annotations.items():
            if annotations['nodules']:
                annotated_slices.append(sop_id)
                if max_slices and len(annotated_slices) >= max_slices:
                    break
        return annotated_slices


def process_single_folder(folder_path, output_base="dataset", split_name=None, visualize=False):
    """Process a single folder containing DICOM files and XML"""
    visualizer = DICOMVisualizerWithMalignancy()
    
    # Find XML file
    xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
    if not xml_files:
        print(f"No XML file found in {folder_path}")
        return 0
    
    xml_path = os.path.join(folder_path, xml_files[0])
    
    # Load data
    print(f"\nProcessing: {folder_path}")
    annotations = visualizer.parse_lidc_xml(xml_path)
    visualizer.load_dicom_series(folder_path)
    visualizer.parse_annotations(annotations)
    
    # Get annotated slices
    annotated_slices = visualizer.get_slice_with_annotations()
    print(f"Found {len(annotated_slices)} annotated slices")
    
    if len(annotated_slices) == 0:
        print("No annotated slices found, skipping folder")
        return 0
    
    # Determine output folder
    folder_name = os.path.basename(folder_path)
    if split_name:
        output_folder = os.path.join(output_base, split_name)
    else:
        output_folder = output_base
    
    # Save slices
    saved_count = 0
    for i, sop_id in enumerate(annotated_slices):
        if sop_id in visualizer.dicom_data:
            slice_name = f"{folder_name}_slice_{i:04d}"
            result = visualizer.save_slice_as_png(sop_id, output_folder, slice_name)
            if result:
                saved_count += 1
                print(f"  Saved: {slice_name}")
    
    # Visualize if requested
    if visualize and len(annotated_slices) > 0:
        print(f"\nVisualizing first 3 slices from {folder_path}...")
        for i, sop_id in enumerate(annotated_slices[:3]):
            if sop_id in visualizer.dicom_data:
                fig, mask = visualizer.visualize_slice(sop_id)
                plt.show()
    
    return saved_count


def process_all_folders(base_path="data", output_base="dataset", train_ratio=0.8, val_ratio=0.1):
    """Process all folders and split into train/val/test"""
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    if len(folders) == 0:
        print(f"No folders found in {base_path}")
        return
    
    print(f"Found {len(folders)} folders to process")
    
    # Split into train/val/test
    train_folders, temp_folders = train_test_split(folders, train_size=train_ratio, random_state=42)
    val_folders, test_folders = train_test_split(temp_folders, train_size=val_ratio/(1-train_ratio), random_state=42)
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_folders)} folders ({len(train_folders)/len(folders)*100:.1f}%)")
    print(f"  Val: {len(val_folders)} folders ({len(val_folders)/len(folders)*100:.1f}%)")
    print(f"  Test: {len(test_folders)} folders ({len(test_folders)/len(folders)*100:.1f}%)")
    
    # Process each split
    total_saved = 0
    
    print("\n" + "="*60)
    print("Processing TRAIN set...")
    print("="*60)
    for folder in train_folders:
        folder_path = os.path.join(base_path, folder)
        count = process_single_folder(folder_path, output_base, split_name="train")
        total_saved += count
    
    print("\n" + "="*60)
    print("Processing VAL set...")
    print("="*60)
    for folder in val_folders:
        folder_path = os.path.join(base_path, folder)
        count = process_single_folder(folder_path, output_base, split_name="val")
        total_saved += count
    
    print("\n" + "="*60)
    print("Processing TEST set...")
    print("="*60)
    for folder in test_folders:
        folder_path = os.path.join(base_path, folder)
        count = process_single_folder(folder_path, output_base, split_name="test")
        total_saved += count
    
    print("\n" + "="*60)
    print(f"Summary:")
    print(f"  Total slices saved: {total_saved}")
    print(f"  Output directory: {output_base}/")
    print(f"  Structure:")
    print(f"    {output_base}/train/images/")
    print(f"    {output_base}/train/masks/")
    print(f"    {output_base}/train/annotations/")
    print(f"    {output_base}/val/...")
    print(f"    {output_base}/test/...")
    print("="*60)


def visualize_from_dataset(dataset_path="dataset", split="train", num_samples=3):
    """Visualize samples from the processed dataset"""
    images_dir = os.path.join(dataset_path, split, 'images')
    masks_dir = os.path.join(dataset_path, split, 'masks')
    annotations_dir = os.path.join(dataset_path, split, 'annotations')
    
    if not os.path.exists(images_dir):
        print(f"Dataset path {images_dir} not found!")
        return
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    
    if len(image_files) == 0:
        print(f"No images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images in {split} set")
    print(f"Visualizing {min(num_samples, len(image_files))} samples...\n")
    
    for i, img_file in enumerate(image_files[:num_samples]):
        base_name = img_file.replace('.png', '')
        
        # Load image
        img_path = os.path.join(images_dir, img_file)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Load mask
        mask_file = f"{base_name}_mask.png"
        mask_path = os.path.join(masks_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None
        
        # Load annotations
        annotation_file = f"{base_name}.json"
        annotation_path = os.path.join(annotations_dir, annotation_file)
        annotations = None
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title(f'Image: {img_file}')
        axes[0].axis('off')
        
        # Mask
        if mask is not None:
            axes[1].imshow(mask, cmap='hot')
            axes[1].set_title('Ground Truth Mask')
        else:
            axes[1].text(0.5, 0.5, 'No mask available', ha='center', va='center')
            axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image, cmap='gray')
        if mask is not None:
            axes[2].imshow(mask, cmap='hot', alpha=0.4)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        # Add annotation info as text
        if annotations and annotations.get('nodules'):
            info_text = f"Nodules: {len(annotations['nodules'])}\n"
            for nodule in annotations['nodules'][:3]:  # Show first 3
                mal = nodule.get('malignancy', 'N/A')
                info_text += f"  N{nodule['nodule_id']}: M={mal}\n"
            fig.text(0.02, 0.02, info_text, fontsize=9, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()


def display_menu():
    """Display menu options"""
    print("\n" + "="*60)
    print("DICOM to Dataset Converter (VLM-Ready)")
    print("="*60)
    print("1. Process all folders and create train/val/test dataset")
    print("   (Splits data 8:1:1 and saves as PNG with annotations)")
    print("\n2. Process single folder and visualize")
    print("   (Convert one folder to PNG and show results)")
    print("\n3. Visualize from existing dataset")
    print("   (View already processed data from 'dataset' folder)")
    print("\n4. Exit")
    print("="*60)


def main():
    """Main menu-driven program"""
    
    while True:
        display_menu()
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Process all folders
            print("\n--- Process All Folders ---")
            base_path = input("Enter base data folder path (default: data): ").strip()
            if not base_path:
                base_path = "data"
            
            output_path = input("Enter output dataset folder (default: dataset): ").strip()
            if not output_path:
                output_path = "dataset"
            
            train_ratio = input("Enter train ratio (default: 0.8): ").strip()
            train_ratio = float(train_ratio) if train_ratio else 0.8
            
            val_ratio = input("Enter validation ratio (default: 0.1): ").strip()
            val_ratio = float(val_ratio) if val_ratio else 0.1
            
            print(f"\n⚠️  This will process all folders in: {base_path}")
            print(f"   Output will be saved to: {output_path}/")
            confirm = input("Proceed? (yes/no): ").strip().lower()
            
            if confirm == 'yes':
                process_all_folders(base_path, output_path, train_ratio, val_ratio)
            else:
                print("Operation cancelled.")
        
        elif choice == '2':
            # Process single folder
            print("\n--- Process Single Folder ---")
            folder_path = input("Enter folder path: ").strip()
            
            if not os.path.exists(folder_path):
                print(f"Folder {folder_path} not found!")
                continue
            
            output_path = input("Enter output folder (default: dataset_single): ").strip()
            if not output_path:
                output_path = "dataset_single"
            
            visualize = input("Visualize results? (yes/no, default: yes): ").strip().lower()
            visualize = visualize != 'no'
            
            count = process_single_folder(folder_path, output_path, visualize=visualize)
            print(f"\nProcessed {count} slices from {folder_path}")
        
        elif choice == '3':
            # Visualize from dataset
            print("\n--- Visualize from Dataset ---")
            dataset_path = input("Enter dataset path (default: dataset): ").strip()
            if not dataset_path:
                dataset_path = "dataset"
            
            split = input("Enter split (train/val/test, default: train): ").strip().lower()
            if split not in ['train', 'val', 'test']:
                split = 'train'
            
            num_samples = input("Number of samples to visualize (default: 3): ").strip()
            num_samples = int(num_samples) if num_samples else 3
            
            visualize_from_dataset(dataset_path, split, num_samples)
        
        elif choice == '4':
            print("\nExiting program. Goodbye!")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        
        # Ask if user wants to continue
        continue_choice = input("\nPress Enter to continue or type 'exit' to quit: ").strip().lower()
        if continue_choice == 'exit':
            print("\nExiting program. Goodbye!")
            break


if __name__ == "__main__":
    main()