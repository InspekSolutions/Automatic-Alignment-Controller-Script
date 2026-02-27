# Updated PyQt5 Vimba X viewer with asynchronous frame acquisition and alignment features

import cv2
import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QHBoxLayout, QGridLayout)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from vmbpy import VmbSystem, PixelFormat, VmbTimeout, Camera, Stream, Frame
import threading

import sys
from os.path import expanduser
import time
import pythonnet
pythonnet.load('coreclr')
import clr
try:
    clr.AddReference("srgmc")
    print("DLL Loaded Successfully.")
except Exception as e:
    print(f"Failed to load DLL: {e}")
# Importing the srgmc namespace
import SurugaSeiki.Motion as SSM
import math
import numpy as np


# Import functions from detection.py
from detection import find_edge_lines, angle_between_lines, find_chip_channels, find_fa_channels
from alignment_functions import SurugaController


class FrameHandler(QObject):
    frame_ready = pyqtSignal(object)  # Signal to emit when frame is ready
    
    def __init__(self, camera: Camera):
        super().__init__()
        self.camera = camera
        
    def frame_handler(self, cam: Camera, stream: Stream, frame: Frame):
        try:
            # Convert frame to numpy array
            img = frame.as_numpy_ndarray()
            h, w, c = img.shape
            qimg = QImage(img.data, w, h, QImage.Format_BGR888)
            
            # Emit signal with the new frame
            self.frame_ready.emit(qimg)
            
            # Queue the frame back for reuse
            cam.queue_frame(frame)
        except Exception as e:
            print("Frame handler error:", e)
            # Still queue the frame back even if processing failed
            cam.queue_frame(frame)


class VimbaViewer(QMainWindow):
    trajectory_color_signal = pyqtSignal(object)
    trajectory_update_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Alvium 1800 U-2050c Live View")
        self.resize(1280, 720)

        # Create image label with border
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)
        # self.image_label.setStyleSheet("border: 3px solid green;")
        
        # Create buttons
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.find_angle_button = QPushButton("Find Angle")
        self.move_angle_button = QPushButton("Move Angle")
        self.find_channels_button = QPushButton("Find Channels")
        self.align_channels_button = QPushButton("Align Channels")
        self.move_to_chip_button = QPushButton("Move to chip")
        self.fullscreen_button = QPushButton("Full Screen")
        
        # Connect button signals
        self.play_button.clicked.connect(self.start_streaming)
        self.pause_button.clicked.connect(self.pause_streaming)
        self.find_angle_button.clicked.connect(self.find_angle_mode)
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.find_channels_button.clicked.connect(self.perform_channel_recognition)
        self.align_channels_button.clicked.connect(self.perform_channel_alignment)
        self.move_to_chip_button.clicked.connect(self.move_fa_to_chip)

        
        # Create button layouts
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.fullscreen_button)
        
        alignment_layout = QHBoxLayout()
        alignment_layout.addWidget(self.find_angle_button)
        # alignment_layout.addWidget(self.move_angle_button)
        alignment_layout.addWidget(self.find_channels_button)
        alignment_layout.addWidget(self.align_channels_button)
        alignment_layout.addWidget(self.move_to_chip_button)
        
        # Combine layouts
        button_layout = QVBoxLayout()
        button_layout.addLayout(control_layout)
        button_layout.addLayout(alignment_layout)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.image_label)
        layout.addLayout(button_layout)
        self.setCentralWidget(central_widget)

        # Initialize Vimba system and camera
        self.vmb = VmbSystem.get_instance()
        self.vmb.__enter__()
        self.cam = self.vmb.get_all_cameras()[0]
        self.cam.__enter__()

        if PixelFormat.Bgr8 in self.cam.get_pixel_formats():
            self.cam.set_pixel_format(PixelFormat.Bgr8)
        else:
            raise RuntimeError("Camera does not support BGR8.")

        # Initialize frame handler
        self.frame_handler_obj = FrameHandler(self.cam)
        self.frame_handler_obj.frame_ready.connect(self.display_image)
        
        self.current_image = None
        self.streaming = False
        self.alignment_mode = False
        self.edge_lines = None
        self.edge_angle = None
        self.is_fullscreen = False
        self.fa_channels = None
        self.chip_channels = None
        self.trajectory = None
        self.trajectory_color = QColor(255, 165, 0)  # Orange by default
        self.suruga_controller = None
        try:
            self.suruga_controller = SurugaController()
            print("SurugaController initialized.")
        except Exception as e:
            print(f"Failed to initialize SurugaController: {e}")
        self.trajectory_color_signal.connect(self.set_trajectory_color)
        self.trajectory_update_signal.connect(self.update_trajectory_display)

    def start_streaming(self):
        if not self.streaming:
            try:
                self.cam.start_streaming(self.frame_handler_obj.frame_handler)
                self.streaming = True
                print("Streaming started")
            except Exception as e:
                print("Error starting streaming:", e)

    def pause_streaming(self):
        if self.streaming:
            try:
                self.cam.stop_streaming()
                self.streaming = False
                print("Streaming stopped")
            except Exception as e:
                print("Error stopping streaming:", e)

    def find_angle_mode(self):
        """Toggle camera alignment mode and perform edge detection"""
        self.pause_streaming()  # Freeze the frame before processing
        if not self.alignment_mode:
            # Enter alignment mode
            self.alignment_mode = True
            self.image_label.setStyleSheet("border: 3px solid orange;")
            self.find_angle_button.setText("Exit Alignment")
            
            # Perform edge detection on current frame
            if self.current_image is not None:
                self.perform_edge_detection()
        else:
            # Exit alignment mode
            self.alignment_mode = False
            self.image_label.setStyleSheet("border: 3px solid green;")
            self.find_angle_button.setText("Find Angle")
            self.edge_lines = None
            self.edge_angle = None

    def perform_edge_detection(self):
        """Perform edge detection on the current frame"""
        try:
            # Convert QImage to numpy array for processing
            qimg = self.current_image
            width = qimg.width()
            height = qimg.height()
            
            # Convert QImage to numpy array
            ptr = qimg.bits()
            ptr.setsize(height * width * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
            
            # Convert BGR to RGB for OpenCV processing
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            
            # Perform edge detection using functions from detection.py
            left_line, right_line = find_edge_lines(img_bgr, debug=False)
            self.edge_lines = (left_line, right_line)
            self.edge_angle = angle_between_lines(left_line, right_line)
            
            print(f"Edge detection completed. Angle between edges: {self.edge_angle:.4f}°")
            
            # Update display with edge lines
            self.display_image_with_edges()
            
        except Exception as e:
            print(f"Edge detection error: {e}")

    def display_image_with_edges(self):
        """Display the current image with detected edge lines (minimal overlay)"""
        if self.current_image is None or self.edge_lines is None:
            return
        qimg = self.current_image.copy()
        painter = QPainter(qimg)
        # Minimal: draw left and right edge lines
        left_line, right_line = self.edge_lines
        painter.setPen(QPen(QColor(0, 0, 255), 3*4))  # Blue for left line
        painter.drawLine(left_line[0], left_line[1], left_line[2], left_line[3])
        painter.setPen(QPen(QColor(255, 0, 0), 3*4))  # Red for right line
        painter.drawLine(right_line[0], right_line[1], right_line[2], right_line[3])
        # Draw angle text (minimal)
        if self.edge_angle is not None:
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setFont(self.font())
            painter.drawText(10, 30, f"Angle: {self.edge_angle:.2f}°")
        painter.end()
        self.display_image(qimg)
        self.image_label.repaint()

    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showNormal()
            self.is_fullscreen = False
            self.fullscreen_button.setText("Full Screen")
        else:
            self.showFullScreen()
            self.is_fullscreen = True
            self.fullscreen_button.setText("Exit Full Screen")

    def display_image(self, qimg: QImage):
        self.current_image = qimg
        scaled_img = qimg.scaled(self.image_label.contentsRect().size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(QPixmap.fromImage(scaled_img))

    def resizeEvent(self, event):
        if self.current_image:
            if self.alignment_mode and self.edge_lines is not None:
                self.display_image_with_edges()
            else:
                self.display_image(self.current_image)
        super().resizeEvent(event)

    def closeEvent(self, event):
        self.pause_streaming()
        self.cam.__exit__(None, None, None)
        self.vmb.__exit__(None, None, None)
        event.accept()

    def perform_channel_recognition(self):
        """Detect channels on FA and chip, overlay them on the image"""
        self.pause_streaming()  # Freeze the frame before processing
        if self.current_image is None or self.edge_lines is None:
            print("No image or edge lines for channel recognition.")
            return
        try:
            qimg = self.current_image
            width = qimg.width()
            height = qimg.height()
            ptr = qimg.bits()
            ptr.setsize(height * width * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            left_line, right_line = self.edge_lines
            self.fa_channels = find_fa_channels(img_bgr, left_line, debug=False)
            self.chip_channels = find_chip_channels(img_bgr, right_line, debug=False)
            print(f"Detected {len(self.fa_channels)} FA channels, {len(self.chip_channels)} chip channels.")
            self.display_image_with_channels()
        except Exception as e:
            print(f"Channel recognition error: {e}")

    def display_image_with_channels(self):
        """Display the current image with detected channels overlaid (minimal overlay)"""
        if self.current_image is None or self.fa_channels is None or self.chip_channels is None:
            return
        qimg = self.current_image.copy()
        painter = QPainter(qimg)
        # Minimal: draw circles at channel positions
        left_line = self.edge_lines[0]
        left_x = max(left_line[0], left_line[2])
        painter.setPen(QPen(QColor(0, 0, 255), 3*4))  # Blue for FA
        for idx, y in enumerate(self.fa_channels):
            painter.drawEllipse(left_x-7, int(y)-7, 14, 14)
            if idx == 0:
                painter.drawText(left_x-25, int(y)+5, "1")
        right_line = self.edge_lines[1]
        right_x = min(right_line[0], right_line[2])
        painter.setPen(QPen(QColor(255, 0, 0), 3*4))  # Red for chip
        for idx, y in enumerate(self.chip_channels):
            painter.drawEllipse(right_x-7, int(y)-7, 14, 14)
            if idx == 0:
                painter.drawText(right_x+10, int(y)+5, "1")
        painter.end()
        self.display_image(qimg)

    def perform_channel_alignment(self):
        """Calculate and display the trajectory from FA channel 1 to chip channel 1, then move FA along the trajectory with live feedback."""
        if (self.fa_channels is None or len(self.fa_channels) == 0 or
            self.chip_channels is None or len(self.chip_channels) == 0 or
            self.edge_lines is None):
            print("No channels or edge lines for alignment.")
            return
        qimg = self.current_image.copy()
        painter = QPainter(qimg)
        self.trajectory_color = QColor(255, 165, 0)  # Orange
        painter.setPen(QPen(self.trajectory_color, 3*4, Qt.SolidLine))
        # Start: FA channel 1 (left edge)
        left_line = self.edge_lines[0]
        left_x = max(left_line[0], left_line[2])
        fa_y = int(self.fa_channels[0])
        start = (left_x, fa_y)
        # End: chip channel 1 (right edge)
        right_line = self.edge_lines[1]
        right_x = min(right_line[0], right_line[2])
        chip_y = int(self.chip_channels[0])
        end = (right_x, chip_y)
        # Draw trajectory (3 segments)
        mid_x = (start[0] + end[0]) // 2
        painter.drawLine(start[0], start[1], mid_x, start[1])
        painter.drawLine(mid_x, start[1], mid_x, end[1])
        painter.drawLine(mid_x, end[1], end[0], end[1])
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.setFont(QFont("Arial", 40));
        painter.drawText(end[0], end[1] + 20, "Trajectory")
        painter.end()
        self.display_image(qimg)
        # Start streaming and move FA along the trajectory
        self.start_streaming()
        # Start threaded movement
        import threading
        threading.Thread(target=self.move_fa_to_chip, args=(True,), daemon=True).start()

    def move_fa_to_chip(self, live=True):
        """Move the FA along the calculated trajectory (3 segments) using SurugaController, converting pixels to microns using chip channel pitch. If live=True, keep the trajectory orange during movement, then set to green and update display. Runs in a thread if called from perform_channel_alignment."""
        if self.suruga_controller is None:
            print("SurugaController not initialized.")
            return
        if (self.fa_channels is None or len(self.fa_channels) == 0 or
            self.chip_channels is None or len(self.chip_channels) < 2 or
            self.edge_lines is None):
            print("No channels or edge lines for movement.")
            return
        # Get start and end points
        left_line = self.edge_lines[0]
        left_x = max(left_line[0], left_line[2])
        fa_y = int(self.fa_channels[0])
        start = (left_x, fa_y)
        right_line = self.edge_lines[1]
        right_x = min(right_line[0], right_line[2])
        chip_y = int(self.chip_channels[0])
        end = (right_x, chip_y)
        # Calculate mid x
        mid_x = (start[0] + end[0]) // 2
        # 1. Move horizontally from start to (mid_x, start[1])
        dx1 = mid_x - start[0]
        dy1 = 0
        # 2. Move vertically from (mid_x, start[1]) to (mid_x, end[1])
        dx2 = 0
        dy2 = end[1] - start[1]
        # 3. Move horizontally from (mid_x, end[1]) to end
        dx3 = end[0] - mid_x
        dy3 = 0
        # Calculate pixel-to-micron scale using chip channel pitch
        pixel_pitch = np.mean(np.diff(self.chip_channels))
        microns_per_pixel = 250.0 / pixel_pitch
        print(f"Pixel pitch: {pixel_pitch:.2f} px, scale: {microns_per_pixel:.4f} µm/px")
        # Convert all dx/dy to microns
        dx1_um = dx1 * microns_per_pixel
        dy1_um = dy1 * microns_per_pixel
        dx2_um = dx2 * microns_per_pixel
        dy2_um = dy2 * microns_per_pixel
        dx3_um = dx3 * microns_per_pixel*0.95
        dy3_um = dy3 * microns_per_pixel
        print(f"Moving FA (in µm): dx1={dx1_um:.2f}, dy1={dy1_um:.2f}; dx2={dx2_um:.2f}, dy2={dy2_um:.2f}; dx3={dx3_um:.2f}, dy3={dy3_um:.2f}")
        # axis2d = SSM.Axis2D(9,7)
        # axis2d.SetSpeed(1000)
        try:
            # Axis 9: Z, Axis 8: Y
            if live:
                self.trajectory_color_signal.emit(QColor(255, 165, 0))  # Orange
                self.trajectory_update_signal.emit()
            self.suruga_controller.move_relative(9, dx1_um)
            while self.suruga_controller.AxisComponents[9].IsMoving() or self.suruga_controller.AxisComponents[7].IsMoving():
                time.sleep(0.01)
                if live:
                    self.trajectory_update_signal.emit()
            self.suruga_controller.move_relative(7, dy1_um)
            while self.suruga_controller.AxisComponents[9].IsMoving() or self.suruga_controller.AxisComponents[7].IsMoving():
                time.sleep(0.01)
                if live:
                    self.trajectory_update_signal.emit()
            print("First movement to chip complete.")
            self.suruga_controller.move_relative(9, dx2_um)
            while self.suruga_controller.AxisComponents[9].IsMoving() or self.suruga_controller.AxisComponents[7].IsMoving():
                time.sleep(0.01)
                if live:
                    self.trajectory_update_signal.emit()
            self.suruga_controller.move_relative(7, dy2_um)
            while self.suruga_controller.AxisComponents[9].IsMoving() or self.suruga_controller.AxisComponents[7].IsMoving():
                time.sleep(0.01)
                if live:
                    self.trajectory_update_signal.emit()
            print("Second movement to chip complete.")
            self.suruga_controller.move_relative(9, dx3_um)
            while self.suruga_controller.AxisComponents[9].IsMoving() or self.suruga_controller.AxisComponents[7].IsMoving():
                time.sleep(0.01)
                if live:
                    self.trajectory_update_signal.emit()
            self.suruga_controller.move_relative(7, dy3_um)
            while self.suruga_controller.AxisComponents[9].IsMoving() or self.suruga_controller.AxisComponents[7].IsMoving():
                time.sleep(0.01)
                if live:
                    self.trajectory_update_signal.emit()
            print("Movement to chip complete.")
            if live:
                self.trajectory_color_signal.emit(QColor(0, 255, 0))  # Green
                self.trajectory_update_signal.emit()
        except Exception as e:
            print(f"Movement error: {e}")

    def display_trajectory_live(self, start, mid_x, end):
        """Display the trajectory line in the current color during/after movement."""
        if self.current_image is None:
            return
        qimg = self.current_image.copy()
        painter = QPainter(qimg)
        painter.setPen(QPen(self.trajectory_color, 3*4, Qt.SolidLine))
        painter.drawLine(start[0], start[1], mid_x, start[1])
        painter.drawLine(mid_x, start[1], mid_x, end[1])
        painter.drawLine(mid_x, end[1], end[0], end[1])
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.setFont(QFont("Arial", 40));
        painter.drawText(end[0], end[1] + 20, "Trajectory")
        painter.end()
        self.display_image(qimg)

    def set_trajectory_color(self, color):
        self.trajectory_color = color

    def update_trajectory_display(self):
        if (self.fa_channels is None or len(self.fa_channels) == 0 or
            self.chip_channels is None or len(self.chip_channels) == 0 or
            self.edge_lines is None):
            return
        left_line = self.edge_lines[0]
        left_x = max(left_line[0], left_line[2])
        fa_y = int(self.fa_channels[0])
        start = (left_x, fa_y)
        right_line = self.edge_lines[1]
        right_x = min(right_line[0], right_line[2])
        chip_y = int(self.chip_channels[0])
        end = (right_x, chip_y)
        mid_x = (start[0] + end[0]) // 2
        self.display_trajectory_live(start, mid_x, end)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = VimbaViewer()
    viewer.show()
    sys.exit(app.exec_())