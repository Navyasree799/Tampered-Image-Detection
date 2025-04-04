import sys
import os
import cv2
import numpy as np
import uuid
import time
import csv
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                           QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                           QMessageBox, QProgressBar, QScrollArea, QGroupBox,
                           QTabWidget, QSlider, QSizePolicy, QRadioButton,
                           QButtonGroup)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings
import absl.logging
import exiftool
from tensorflow.keras.models import load_model

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
absl.logging.set_verbosity(absl.logging.ERROR)

class DetectionWorker(QThread):
    update_progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    batch_processed = pyqtSignal(int, int)
    error_occurred = pyqtSignal(str)

    def __init__(self, model, input_path, threshold=0.5, batch_size=10, output_dir="output"):
        super().__init__()
        self.model = model
        self.input_path = input_path
        self.threshold = threshold
        self.cancel_flag = False
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.real_dir = os.path.join(self.output_dir, "real")
        self.fake_dir = os.path.join(self.output_dir, "fake")
        os.makedirs(self.real_dir, exist_ok=True)
        os.makedirs(self.fake_dir, exist_ok=True)

    def run(self):
        try:
            results = {"original": [], "tampered": [], "errors": []}
            
            if os.path.isdir(self.input_path):
                files = [f for f in os.listdir(self.input_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                total = len(files)
                
                for batch_start in range(0, total, self.batch_size):
                    if self.cancel_flag:
                        break
                        
                    batch_end = min(batch_start + self.batch_size, total)
                    batch_files = files[batch_start:batch_end]
                    
                    for i, filename in enumerate(batch_files):
                        if self.cancel_flag:
                            break
                            
                        progress = int(((batch_start + i)/total)*100)
                        self.update_progress.emit(progress, filename)
                        result = self.process_image(os.path.join(self.input_path, filename))
                        self.categorize_result(result, results, filename)
                        if "error" not in result:
                            self.save_result_image(result)
                    
                    self.batch_processed.emit(batch_end, total)
            else:
                self.update_progress.emit(50, "Processing image")
                result = self.process_image(self.input_path)
                self.categorize_result(result, results, os.path.basename(self.input_path))
                if "error" not in result:
                    self.save_result_image(result)
                    
            self.finished.emit(results)
            
        except Exception as e:
            self.finished.emit({"error": str(e)})
            self.error_occurred.emit(str(e))

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Failed to read image", "path": image_path}
            
        h, w = img.shape[:2]
        if h != 256 or w != 256:
            img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img.copy()
            
        img_normalized = np.expand_dims(img_resized, axis=0) / 255.0
        
        ela = self.apply_ela(image_path, quality=75)
        if ela is None:
            return {"error": "ELA processing failed", "path": image_path}
            
        ela_resized = cv2.resize(ela, (256, 256), interpolation=cv2.INTER_AREA)
        ela_normalized = np.expand_dims(ela_resized, axis=0) / 255.0
        
        metadata = self.check_metadata(image_path)
        metadata_score = self.analyze_metadata(metadata)
        
        noise_score = self.analyze_noise(img_resized)
        
        pred = self.model.predict([img_normalized, ela_normalized])
        confidence = float(pred[0][0])
        
        adjusted_confidence = self.adjust_confidence(
            confidence, 
            metadata_score, 
            noise_score
        )
        
        return {
            "path": image_path,
            "image": img,
            "ela_image": ela,
            "confidence": adjusted_confidence,
            "tampered": adjusted_confidence > self.threshold,
            "metadata": metadata,
            "noise": noise_score
        }

    def apply_ela(self, image_path, quality=75):
        temp_path = f"temp_ela_{os.getpid()}.jpg"
        try:
            original = cv2.imread(image_path)
            if original is None:
                return None
                
            cv2.imwrite(temp_path, original, [cv2.IMWRITE_JPEG_QUALITY, quality])
            compressed = cv2.imread(temp_path)
            
            if compressed is None:
                return None
                
            ela = cv2.absdiff(original, compressed)
            ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
            ela_gray = cv2.equalizeHist(ela_gray)
            ela = cv2.cvtColor(ela_gray, cv2.COLOR_GRAY2BGR)
            
            return ela
        except Exception as e:
            print(f"ELA processing error: {str(e)}")
            return None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def check_metadata(self, image_path):
        try:
            with exiftool.ExifTool() as et:
                metadata = et.get_metadata(image_path)
            return metadata
        except:
            return {}

    def analyze_metadata(self, metadata):
        score = 0
        red_flags = [
            'Photoshop', 'Edited', 'Software',
            'CreationDate', 'ModifyDate'
        ]
        
        for flag in red_flags:
            if flag in str(metadata):
                score += 0.2
                
        return min(score, 1.0)

    def analyze_noise(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise = cv2.Laplacian(gray, cv2.CV_64F).var()
        normalized_noise = min(noise / 1000, 1.0)
        return normalized_noise

    def adjust_confidence(self, model_confidence, metadata_score, noise_score):
        if metadata_score > 0.5:
            model_confidence *= 0.8
            
        if noise_score < 0.2:
            model_confidence *= 1.2
            
        return min(max(model_confidence, 0), 1.0)

    def categorize_result(self, result, results, filename):
        if "error" in result:
            results["errors"].append((filename, result["error"]))
        elif result["tampered"]:
            results["tampered"].append(result)
        else:
            results["original"].append(result)

    def save_result_image(self, result):
        filename = os.path.basename(result["path"])
        base, ext = os.path.splitext(filename)
        unique_id = uuid.uuid4().hex[:6]
        
        if result["tampered"]:
            output_path = os.path.join(self.fake_dir, f"{base}_{unique_id}{ext}")
        else:
            output_path = os.path.join(self.real_dir, f"{base}_{unique_id}{ext}")
        
        cv2.imwrite(output_path, result["image"])
        
        ela_path = os.path.join(os.path.dirname(output_path), f"{base}_{unique_id}_ela{ext}")
        cv2.imwrite(ela_path, result["ela_image"])
        
        meta_path = os.path.join(os.path.dirname(output_path), f"{base}_{unique_id}_meta.txt")
        with open(meta_path, 'w') as f:
            f.write(f"Status: {'Fake' if result['tampered'] else 'Real'}\n")
            f.write(f"Confidence: {result['confidence']*100:.2f}%\n")
            f.write(f"Original Path: {result['path']}\n")
            f.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n=== Metadata Analysis ===\n")
            for k, v in result.get('metadata', {}).items():
                f.write(f"{k}: {v}\n")
            f.write(f"\nNoise Score: {result.get('noise', 0):.2f}")

    def cancel(self):
        self.cancel_flag = True

class TamperDetectionApp(QMainWindow):
    model_loaded = pyqtSignal(bool)
    model_load_error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.worker = None
        self.current_results = None
        self.settings = QSettings("TamperDetection", "ImageDetector")
        
        self.setWindowTitle("Image Authenticity Analyzer - Loading...")
        self.setGeometry(100, 100, 400, 150)
        
        self.loading_widget = QWidget()
        self.setCentralWidget(self.loading_widget)
        loading_layout = QVBoxLayout()
        
        self.loading_label = QLabel("Loading AI model...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("font-size: 16px;")
        
        self.loading_bar = QProgressBar()
        self.loading_bar.setRange(0, 0)
        
        loading_layout.addWidget(self.loading_label)
        loading_layout.addWidget(self.loading_bar)
        self.loading_widget.setLayout(loading_layout)
        
        self.model_loaded.connect(self.on_model_loaded)
        self.model_load_error.connect(self.show_model_error)
        QTimer.singleShot(0, self.load_model)

    def load_model(self):
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'tampered_detection.h5')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {os.path.abspath(model_path)}")
                
            self._model = load_model(model_path)
            
            if not all(inp.shape[1:] == (256,256,3) for inp in self._model.inputs):
                raise ValueError("Invalid model input dimensions")
                
            self.model_loaded.emit(True)
            
        except Exception as e:
            self.model_loaded.emit(False)
            self.model_load_error.emit(str(e))

    def on_model_loaded(self, success):
        if success:
            self.loading_widget.deleteLater()
            self.setWindowTitle("Image Authenticity Analyzer")
            self.setGeometry(100, 100, 1400, 900)
            self.setup_main_ui()
            self.statusBar().showMessage("Model loaded successfully. Ready to analyze images.")
        else:
            pass

    def setup_main_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        self.header = QLabel("""
            <div style='text-align:center'>
                <h1 style='color:#6a1b9a; margin-bottom:5px;'>IMAGE AUTHENTICITY ANALYZER</h1>
                <p style='color:#757575;'>Detect manipulated images using Error Level Analysis</p>
            </div>
        """)
        self.header.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.header)
        
        self.setup_control_panel()
        self.setup_results_area()
        self.statusBar().setFont(QFont("Segoe UI", 9))
        self.setup_style()

    def setup_control_panel(self):
        self.control_panel = QGroupBox("Analysis Controls")
        control_layout = QVBoxLayout()
        
        input_type_group = QGroupBox("Input Type")
        input_type_layout = QHBoxLayout()
        
        self.single_image_radio = QRadioButton("Single Image")
        self.folder_radio = QRadioButton("Folder of Images")
        self.folder_radio.setChecked(True)
        
        self.input_type_group = QButtonGroup()
        self.input_type_group.addButton(self.single_image_radio)
        self.input_type_group.addButton(self.folder_radio)
        
        input_type_layout.addWidget(self.single_image_radio)
        input_type_layout.addWidget(self.folder_radio)
        input_type_group.setLayout(input_type_layout)
        
        threshold_group = QGroupBox("Detection Sensitivity")
        threshold_layout = QVBoxLayout()
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(30, 90)
        self.threshold_slider.setValue(50)
        self.threshold_slider.setTickInterval(5)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        
        threshold_display = QHBoxLayout()
        self.threshold_min_label = QLabel("30% (Lenient)")
        self.threshold_value_label = QLabel("50%")
        self.threshold_value_label.setAlignment(Qt.AlignCenter)
        self.threshold_value_label.setStyleSheet("font-weight: bold; color: #6a1b9a;")
        self.threshold_max_label = QLabel("90% (Strict)")
        
        threshold_display.addWidget(self.threshold_min_label)
        threshold_display.addWidget(self.threshold_value_label)
        threshold_display.addWidget(self.threshold_max_label)
        
        threshold_layout.addLayout(threshold_display)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_group.setLayout(threshold_layout)
        
        button_group = QGroupBox("Actions")
        button_layout = QHBoxLayout()
        
        self.input_btn = QPushButton("Select Input")
        self.process_btn = QPushButton("Analyze")
        self.export_btn = QPushButton("Export Results")
        self.export_btn.setEnabled(False)
        
        button_layout.addWidget(self.input_btn)
        button_layout.addWidget(self.process_btn)
        button_layout.addWidget(self.export_btn)
        button_group.setLayout(button_layout)
        
        control_layout.addWidget(input_type_group)
        control_layout.addWidget(threshold_group)
        control_layout.addWidget(button_group)
        self.control_panel.setLayout(control_layout)
        self.main_layout.addWidget(self.control_panel)
        
        self.threshold_slider.valueChanged.connect(self.update_threshold_display)
        self.input_btn.clicked.connect(self.select_input)
        self.process_btn.clicked.connect(self.process_input)
        self.export_btn.clicked.connect(self.export_results)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel()
        self.progress_label.setVisible(False)
        self.batch_label = QLabel()
        self.batch_label.setVisible(False)
        
        progress_layout = QVBoxLayout()
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.batch_label)
        
        self.main_layout.addLayout(progress_layout)

    def setup_results_area(self):
        self.tabs = QTabWidget()
        
        self.original_tab = QWidget()
        self.original_layout = QVBoxLayout(self.original_tab)
        original_scroll = QScrollArea()
        original_scroll.setWidgetResizable(True)
        original_scroll.setWidget(self.original_tab)
        
        self.tampered_tab = QWidget()
        self.tampered_layout = QVBoxLayout(self.tampered_tab)
        tampered_scroll = QScrollArea()
        tampered_scroll.setWidgetResizable(True)
        tampered_scroll.setWidget(self.tampered_tab)
        
        self.tabs.addTab(original_scroll, "Authentic Images (0)")
        self.tabs.addTab(tampered_scroll, "Tampered Images (0)")
        
        self.main_layout.addWidget(self.tabs)

    def setup_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #6a1b9a;
                font-weight: bold;
            }
            QPushButton {
                background-color: #7e57c2;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #9575cd;
            }
            QPushButton:pressed {
                background-color: #5e35b1;
            }
            QPushButton:disabled {
                background-color: #b39ddb;
                color: #e0e0e0;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #e0e0e0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                width: 20px;
                height: 20px;
                margin: -6px 0;
                background: #6a1b9a;
                border-radius: 10px;
            }
            QTabWidget::pane {
                border: 1px solid #e0e0e0;
                border-radius: 0 0 4px 4px;
                background: white;
            }
            QTabBar::tab {
                background: #f5f5f5;
                border: 1px solid #e0e0e0;
                border-bottom: none;
                padding: 8px 16px;
            }
            QTabBar::tab:selected {
                background: white;
                color: #6a1b9a;
                font-weight: bold;
            }
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                text-align: center;
                background: white;
            }
            QProgressBar::chunk {
                background-color: #7e57c2;
            }
            QRadioButton {
                spacing: 8px;
            }
        """)

        header_font = QFont("Segoe UI Semibold", 18)
        self.header.setFont(header_font)
        
        button_font = QFont("Segoe UI", 10, QFont.Medium)
        for btn in self.findChildren(QPushButton):
            btn.setFont(button_font)

    def show_model_error(self, error_msg):
        QMessageBox.critical(self, "Model Error", 
            f"Failed to load model:\n{error_msg}\n\n"
            f"Please ensure 'tampered_detection.h5' exists in:\n"
            f"{os.path.dirname(__file__)}")
        self.close()

    def update_threshold_display(self, value):
        self.threshold_value_label.setText(f"{value}%")
        
        if value < 50:
            color = "#4caf50"
        elif value < 70:
            color = "#ff9800"
        else:
            color = "#f44336"
            
        self.threshold_value_label.setStyleSheet(f"""
            font-weight: bold; 
            font-size: 14px;
            color: {color};
        """)

    def select_input(self):
        options = QFileDialog.Options()
        
        if self.single_image_radio.isChecked():
            input_path, _ = QFileDialog.getOpenFileName(
                self, "Select Image", "", 
                "Images (*.png *.jpg *.jpeg)", options=options)
        else:
            input_path = QFileDialog.getExistingDirectory(
                self, "Select Image Folder")
                
        if input_path:
            self.process_input(input_path)

    def process_input(self, input_path):
        if not input_path:
            return
            
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Processing", "Another operation is already in progress")
            return
            
        self.clear_results()
        self.export_btn.setEnabled(False)
        
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.batch_label.setVisible(True)
        self.progress_bar.setValue(0)
        
        if hasattr(self, 'input_label'):
            self.input_label.deleteLater()
            
        input_type = "file" if self.single_image_radio.isChecked() else "folder"
        display_text = os.path.basename(input_path) if input_type == "file" else input_path
        self.input_label = QLabel(f"Input ({input_type}): {display_text}")
        self.main_layout.insertWidget(2, self.input_label)
        
        threshold = self.threshold_slider.value() / 100
        self.worker = DetectionWorker(
            self.model, 
            input_path, 
            threshold=threshold,
            batch_size=10,
            output_dir="output"
        )
        
        self.worker.update_progress.connect(self.update_progress)
        self.worker.batch_processed.connect(self.update_batch_status)
        self.worker.finished.connect(self.handle_results)
        self.worker.error_occurred.connect(self.show_error)
        
        self.worker.start()
        self.statusBar().showMessage(f"Processing: {os.path.basename(input_path) if input_type == 'file' else input_path}")

    def update_progress(self, percent, filename):
        self.progress_bar.setValue(percent)
        self.progress_label.setText(f"Processing: {filename}")

    def update_batch_status(self, current, total):
        self.batch_label.setText(f"Processed: {current}/{total} images")

    def handle_results(self, results):
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.batch_label.setVisible(False)
        self.current_results = results
        
        if "error" in results:
            QMessageBox.critical(self, "Error", results["error"])
            return
            
        if results["errors"]:
            error_msg = "\n".join([f"{f}: {e}" for f, e in results["errors"]])
            QMessageBox.warning(self, "Processing Errors", 
                               f"Some files couldn't be processed:\n\n{error_msg}")
                               
        for item in results["original"]:
            self.original_layout.addWidget(self.create_image_card(item))
            
        for item in results["tampered"]:
            self.tampered_layout.addWidget(self.create_image_card(item))
            
        self.tabs.setTabText(0, f"Authentic Images ({len(results['original'])})")
        self.tabs.setTabText(1, f"Tampered Images ({len(results['tampered'])})")
            
        orig_count = len(results["original"])
        tampered_count = len(results["tampered"])
        self.statusBar().showMessage(
            f"Done! Authentic: {orig_count}, Tampered: {tampered_count}, Errors: {len(results['errors'])}", 
            5000)
            
        if orig_count + tampered_count > 0:
            self.export_btn.setEnabled(True)

    def create_image_card(self, result):
        card = QGroupBox()
        card.setStyleSheet("""
            QGroupBox {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                margin: 10px;
                padding: 15px;
                background: white;
            }
        """)
        layout = QVBoxLayout()
        
        filename = os.path.basename(result["path"])
        name_label = QLabel(filename)
        name_label.setStyleSheet("""
            font-weight: bold;
            font-size: 14px;
            color: #424242;
            padding-bottom: 8px;
            border-bottom: 1px solid #eeeeee;
        """)
        layout.addWidget(name_label)
        
        img_comparison = QHBoxLayout()
        
        orig_frame = QGroupBox("Original")
        orig_layout = QVBoxLayout()
        orig_img = self.create_image_label(result["image"])
        orig_layout.addWidget(orig_img)
        orig_frame.setLayout(orig_layout)
        
        ela_frame = QGroupBox("ELA Analysis")
        ela_layout = QVBoxLayout()
        ela_img = self.create_image_label(result["ela_image"])
        ela_layout.addWidget(ela_img)
        ela_frame.setLayout(ela_layout)
        
        img_comparison.addWidget(orig_frame)
        img_comparison.addWidget(ela_frame)
        layout.addLayout(img_comparison)
        
        confidence = result["confidence"] * 100
        confidence_bar = QProgressBar()
        confidence_bar.setRange(0, 100)
        confidence_bar.setValue(int(confidence))
        confidence_bar.setFormat(f"Confidence: {confidence:.1f}%")
        confidence_bar.setStyleSheet("""
            QProgressBar {
                text-align: center;
                height: 24px;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #7e57c2;
                border-radius: 4px;
            }
        """)
        layout.addWidget(confidence_bar)
        
        card.setLayout(layout)
        return card

    def create_image_label(self, img):
        height, width, _ = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("background: #f5f5f5; border-radius: 4px;")
        return label

    def clear_results(self):
        while self.original_layout.count():
            child = self.original_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        while self.tampered_layout.count():
            child = self.tampered_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        if hasattr(self, 'input_label'):
            self.input_label.deleteLater()
            
        self.tabs.setTabText(0, "Authentic Images (0)")
        self.tabs.setTabText(1, "Tampered Images (0)")

    def export_results(self):
        if not self.current_results:
            QMessageBox.warning(self, "Export Error", "No results to export")
            return
            
        options = QFileDialog.Options()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "CSV Files (*.csv)", options=options)
            
        if path:
            try:
                with open(path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Filename", "Status", "Confidence", "Path", "Output Path"])
                    
                    for item in self.current_results["original"]:
                        writer.writerow([
                            os.path.basename(item["path"]),
                            "Authentic",
                            f"{item['confidence']*100:.1f}%",
                            item["path"],
                            os.path.abspath(os.path.join('output', 'real'))
                        ])
                        
                    for item in self.current_results["tampered"]:
                        writer.writerow([
                            os.path.basename(item["path"]),
                            "Tampered",
                            f"{item['confidence']*100:.1f}%",
                            item["path"],
                            os.path.abspath(os.path.join('output', 'fake'))
                        ])
                        
                QMessageBox.information(self, "Export Successful", 
                    f"Results exported to {path}\n\nImages saved to:\n{os.path.abspath('output')}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", 
                    f"Failed to export results: {str(e)}")

    def show_error(self, error_msg):
        self.statusBar().showMessage(f"Error: {error_msg}", 5000)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Processing Running",
                "A detection process is still running. Do you want to cancel it and quit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker.cancel()
                self.worker.wait(2000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    @property
    def model(self):
        return self._model

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont("Segoe UI", 10))
    
    window = TamperDetectionApp()
    window.show()
    
    sys.exit(app.exec_())
