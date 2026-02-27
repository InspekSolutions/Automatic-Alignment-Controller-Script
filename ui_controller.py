import random
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QFormLayout, QLabel, QDoubleSpinBox,
    QComboBox, QPushButton, QTextEdit, QApplication,
    QDialog, QInputDialog, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt, QObject
from PyQt5 import uic
import alignment_functions
import time
import json
import os
import traceback
import numpy as np
import cv2
from vmbpy import VmbSystem, PixelFormat, VmbTimeout, Camera, Stream, Frame
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from detection import find_edge_lines, angle_between_lines, find_chip_channels, find_fa_channels

# Use the real SurugaController from alignment_functions.
# Controller is created when user clicks Connect in the UI (see MainWindow).

def standardize_config(config, controller=None):
    """
    Standardizes configuration format to a dictionary with x, y, and rz keys.
    
    Args:
        config: Either a tuple (x, y), (x, y, rz) or a dictionary with x, y keys
        controller: Controller instance to get rz position if needed
        
    Returns:
        Dictionary with standardized format {'x': x, 'y': y, 'rz': rz}
    """
    if isinstance(config, tuple):
        if len(config) == 2:  # 2D config (x,y)
            if controller:
                current_rz = controller.AxisComponents[12].GetActualPosition()
            else:
                current_rz = 0
            return {'x': config[0], 'y': config[1], 'rz': current_rz}
        else:  # 3D config (x,y,rz)
            return {'x': config[0], 'y': config[1], 'rz': config[2]}
    elif isinstance(config, dict):
        # Ensure the dict has all required keys
        if 'rz' not in config and controller:
            config['rz'] = controller.AxisComponents[12].GetActualPosition()
        return config
    else:
        # Handle unexpected cases
        return {'x': 0, 'y': 0, 'rz': 0}

class StepDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi('step_dialog.ui', self)
        self.step_type_combo.currentTextChanged.connect(self.on_step_type_changed)
        self.on_step_type_changed(self.step_type_combo.currentText())

    def on_step_type_changed(self, step_type):
        # Show/hide relevant parameters based on step type
        is_move = step_type == "Move to Position"
        is_scan = step_type in ["Scan2D", "Spiral Scan", "2Channel Alignment", "Hill Climb"]
        is_max = step_type == "Move to Max Signal"
        is_rot_align = step_type in ["Align Tz2", "Align Tx2"]
        is_move_axis = step_type == "Move axis"

        self.x_label.setVisible(is_move or is_scan)
        self.x_spin.setVisible(is_move or is_scan)
        self.y_label.setVisible(is_move or is_scan)
        self.y_spin.setVisible(is_move or is_scan)
        self.rz_label.setVisible(is_move or step_type == "2Channel Alignment")
        self.rz_spin.setVisible(is_move or step_type == "2Channel Alignment")
        self.range_label.setVisible(is_scan)
        self.range_spin.setVisible(is_scan)
        self.step_label.setVisible(is_scan)
        self.step_spin.setVisible(is_scan)
        self.speed_label.setVisible(is_move or is_scan or is_move_axis)
        self.speed_spin.setVisible(is_move or is_scan or is_move_axis)
        self.rotation_range_label.setVisible(is_rot_align)
        self.rotation_range_spin.setVisible(is_rot_align)
        self.rotation_step_label.setVisible(is_rot_align)
        self.rotation_step_spin.setVisible(is_rot_align)
        self.distance_label.setVisible(is_move_axis)
        self.distance_spin.setVisible(is_move_axis)
        self.axis_label.setVisible(is_move_axis)
        self.axis_combo.setVisible(is_move_axis)

    def get_step_data(self):
        step_type = self.step_type_combo.currentText()
        num_ax = {
            'X': 7,
            'Y': 8,
            'Z': 9
        }
        data = {
            'type': step_type,
            'x': self.x_spin.value(),
            'y': self.y_spin.value(),
            'rz': self.rz_spin.value(),
            'range': self.range_spin.value(),
            'step': self.step_spin.value(),
            'speed': self.speed_spin.value(),
            'rotation_range': self.rotation_range_spin.value(),
            'rotation_step': self.rotation_step_spin.value()
        }
        if step_type == "Move axis":
            data['distance'] = self.distance_spin.value()
            data['axis_number'] = num_ax[self.axis_combo.currentText()]
        return data

class CustomRoutineWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, controller, steps):
        super().__init__()
        self.controller = controller
        self.steps = steps
        self.is_running = True
        self.last_scan_result = None  # Add this line to track scan results
        
        # Track the last check time for periodic interruption checks
        self.last_interrupt_check = time.time()
        self.interrupt_check_interval = 0.2  # seconds

    def stop(self):
        self.is_running = False

    def check_interrupt(self):
        """
        Check if we should interrupt the operation.
        Call this frequently during long operations.
        """
        # Only check periodically to avoid performance impact
        current_time = time.time()
        if current_time - self.last_interrupt_check >= self.interrupt_check_interval:
            self.last_interrupt_check = current_time
            # Process events to keep UI responsive
            QApplication.processEvents()
            # Return True if we should continue
            return self.is_running
        return True

    def run(self):
        try:
            for i, step in enumerate(self.steps, 1):
                if not self.is_running:
                    break

                self.progress.emit(f"Executing step {i}/{len(self.steps)}: {step['type']}")
                
                if step['type'] == "Move to Position":
                    self.controller.axis2d.MoveAbsolute(step['x'], step['y'])
                    while self.controller.axis2d.IsMoving() and self.is_running:
                        time.sleep(0.01)
                    
                    if 'rz' in step:
                        self.controller.AxisComponents[12].MoveAbsolute(step['rz'])
                        while self.controller.AxisComponents[12].IsMoving() and self.is_running:
                            time.sleep(0.01)

                elif step['type'] == "Scan2D":
                    result = alignment_functions.scan2D(
                        self.controller,
                        axis_x=7,
                        axis_y=8,
                        search_range_x=step['range'],
                        search_range_y=step['range'],
                        field_pitch_x=step['step'],
                        field_pitch_y=step['step'],
                        speed_x=step['speed'],
                        speed_y=step['speed'],
                        alpha = step.get('alpha', 0), # no alpha in the routines for now
                    )
                    if result:
                        best_config, best_signal, data_list = result
                        self.last_scan_result = (best_config, best_signal, data_list)
                        self.progress.emit(f"Scan2D complete. Best signal: {best_signal:.2f}")

                elif step['type'] == "Spiral Scan":
                    result = alignment_functions.spiral(
                        self.controller,
                        axis_x=7,
                        axis_y=8,
                        search_range_x=step['range'],
                        search_range_y=step['range'],
                        field_pitch_x=step['step'],
                        field_pitch_y=step['step'],
                        speed_x=step['speed'],
                        speed_y=step['speed'],
                        alpha=step.get('alpha', 0)
                    )
                    if result:
                        best_config, best_signal, data_list = result
                        self.last_scan_result = (best_config, best_signal, data_list)
                        self.progress.emit(f"Spiral scan complete. Best signal: {best_signal:.2f}")

                elif step['type'] == "Hill Climb":
                    result = alignment_functions.hill_climb_2channel(
                        self.controller,
                        axis_x=7,
                        axis_y=8,
                        step_size=step['step'],
                        search_range=step['range'],
                        speed=step['speed'],
                        alpha=0, # no alpha in the routines for now
                        max_iterations=3,
                        progress_callback=lambda config, sig1, sig2: self.progress.emit(
                            f"Hill climbing... Position: x={config['x']:.2f}, y={config['y']:.2f}, "
                            f"Signals: {sig1:.2f}, {sig2:.2f}"
                        ),
                        steps_per_direction=3
                    )
                    if result:
                        best_config, best_signals, data_list = result
                        self.last_scan_result = (best_config, best_signals, data_list)
                        self.progress.emit(f"Hill climb complete. Best signal: {best_signals:.2f}")

                elif step['type'] == "Align Tz2":
                    result = alignment_functions.align_Tz2(
                        self.controller,
                        search_range=step['rotation_range'],
                        step=step['rotation_step'],
                        signal_channel=step.get('signal_channel', 1),
                        progress_callback=lambda angle, sigs: self.progress.emit(
                            f"Align Tz2... Angle: {angle:.3f}, Signals: {sigs[0]:.3f}, {sigs[1]:.3f}, Total: {sigs[0]+sigs[1]:.3f}")
                    )
                    if result:
                        best_angle, best_signals, data_list = result
                        # self.last_scan_result = (best_angle, best_signals, data_list)
                        self.progress.emit(f"Tz2 alignment complete. Best angle: {best_angle:.3f}, Best signals: {best_signals[0]:.3f}, {best_signals[1]:.3f}")

                # elif step

                elif step['type'] == "Align Tx2":
                    result = alignment_functions.align_Tx2(
                        self.controller,
                        search_range=step['rotation_range'],
                        step=step['rotation_step'],
                        signal_channel=step.get('signal_channel', 1),
                        progress_callback=lambda angle, sigs: self.progress.emit(
                            f"Align Tx2... Angle: {angle:.3f}, Signals: {sigs[0]:.3f}, {sigs[1]:.3f}, Total: {sigs[0]+sigs[1]:.3f}")
                    )
                    if result:
                        best_angle, best_signals, data_list = result
                        # self.last_scan_result = (best_angle, best_signals, data_list)
                        self.progress.emit(f"Tx2 alignment complete. Best angle: {best_angle:.3f}, Best signals: {best_signals[0]:.3f}, {best_signals[1]:.3f}")
                
                elif step['type'] == "Move axis":
                    result = alignment_functions.move_axis(
                        self.controller,
                        axis_number = step['axis_number'],
                        distance=step['distance'],
                        speed=step['speed'])
                    if result:
                        # self.last_scan_result = (result, None, None)
                        self.progress.emit(f"Axis {step['axis_number']} movement complete. Final position: {result:.3f}")

                elif step['type'] == "Move to Max Signal":
                    if self.last_scan_result is not None:
                        best_config = self.last_scan_result[0]
                        # If best_config is a dict, use keys; if tuple, use indices
                        if isinstance(best_config, dict):
                            x = best_config.get('x', 0)
                            y = best_config.get('y', 0)
                            rz = best_config.get('rz', None)
                        elif isinstance(best_config, (list, tuple)):
                            # Assume (x, y) or (x, y, rz)
                            x = best_config[0]
                            y = best_config[1]
                            rz = best_config[2] if len(best_config) > 2 else None
                        else:
                            x = y = 0
                            rz = None
                        self.controller.axis2d.MoveAbsolute(x, y)
                        while self.controller.axis2d.IsMoving() and self.is_running:
                            time.sleep(0.01)
                    else:
                        self.error.emit("No previous scan results available for Move to Max Signal")
                        return

            if self.is_running:
                self.progress.emit("Custom routine completed successfully")
                self.finished.emit()
            else:
                self.progress.emit("Custom routine interrupted by user")

        except Exception as e:
            self.error.emit(f"Error during custom routine: {str(e)}")

class AlignmentWorker(QThread):
    result_ready = pyqtSignal(object)
    log_signal = pyqtSignal(str)
    measurement_update = pyqtSignal(dict, float, float)

    def __init__(self, controller, mode, translation_speed, rotation_speed, search_range, translation_step, rotation_range, rotation_step, alpha):
        super().__init__()
        self.controller = controller
        # self.controller.AxisComponents[7].SetSineMotion(False)
        # self.controller.AxisComponents[8].SetSineMotion(False)

        self.mode = mode
        self.translation_speed = translation_speed
        self.search_range = search_range
        self.translation_step = translation_step
        self.rotation_range = rotation_range
        self.rotation_step = rotation_step
        self.rotation_speed = rotation_speed
        self.is_running = True  # Flag to control the process
        self.alpha = alpha
        
        # Track the last check time for periodic interruption checks
        self.last_interrupt_check = time.time()
        self.interrupt_check_interval = 0.2  # seconds

    def stop(self):
        """Safely stop the alignment process"""
        self.is_running = False
        self.log_signal.emit("Stopping alignment process...")
        
    def check_interrupt(self):
        """
        Check if we should interrupt the operation.
        Call this frequently during long operations.
        """
        # Only check periodically to avoid performance impact
        current_time = time.time()
        if current_time - self.last_interrupt_check >= self.interrupt_check_interval:
            self.last_interrupt_check = current_time
            # Process events to keep UI responsive
            QApplication.processEvents()
            # Return False if we should stop
            return self.is_running
        return True
        
    def run(self):
        # Convert all configurations to dictionary format
        def progress_callback(config, signal1, signal2=None):
            # Check if we should stop immediately
            if not self.is_running:
                return False  # Return False to indicate interruption
            
            config_dict = config     
            self.measurement_update.emit(config_dict, signal1, signal2)
            return True  # Return True to continue

        try:
           
            if self.mode == "Scan2D":
                
                # Check for interruption frequently
                def scan_progress_callback(config, signal1, signal2=None):
                    # Always check for interruption in callbacks
                    if not self.is_running:
                        return False
                    # Call the standard progress callback
                    return progress_callback(config, signal1, signal2)
                
                result = alignment_functions.scan2D(
                    self.controller,
                    axis_x=7,
                    axis_y=8,
                    search_range_x=self.search_range,
                    search_range_y=self.search_range,
                    field_pitch_x=self.translation_step,
                    field_pitch_y=self.translation_step,
                    speed_x=self.translation_speed,
                    speed_y=self.translation_speed,
                    alpha = self.alpha,
                    progress_callback=scan_progress_callback
                )
                
                # Check if we should stop
                if not self.check_interrupt():
                    self.log_signal.emit("Process stopped by user")
                    self.result_ready.emit(None)
                    return
                
                # Convert to standard format
                if result:
                    config, signal, data_list = result
                    # self.log_signal.emit("Processing scan results")
                    result = (config, signal, data_list)

            elif self.mode == "Scan Tz,Y":
                # self.log_signal.emit("Running 2D Scan...")
                
                # Check for interruption frequently
                def scan_progress_callback(config, signal1, signal2=None):
                    # Always check for interruption in callbacks
                    if not self.is_running:
                        return False
                    # Call the standard progress callback
                    return progress_callback(config, signal1, signal2)
                
                result = alignment_functions.scanTz_Y(
                    self.controller,
                    axis_tz=12,
                    axis_y=8,
                    rotation_range_tz=self.rotation_range,
                    search_range_y=self.search_range,
                    rotation_step_tz=self.rotation_step,
                    translation_step_y=self.translation_step,
                    speed_tz=self.rotation_speed,
                    speed_y=self.translation_speed,
                    alpha = self.alpha,
                    progress_callback=scan_progress_callback
                )
                # self.log_signal.emit(f"Scan2D result obtained: {len(result) if result else 'None'}")
                
                # Check if we should stop
                if not self.check_interrupt():
                    self.log_signal.emit("Process stopped by user")
                    self.result_ready.emit(None)
                    return
                
                # Convert to standard format
                if result:
                    config, signal, data_list = result
                    # self.log_signal.emit("Processing scan results")
                    result = (config, signal, data_list)        

            elif self.mode == "Scan Tx,Y":
                def scan_progress_callback(config, signal1, signal2=None):
                    # Always check for interruption in callbacks
                    if not self.is_running:
                        return False
                    # Call the standard progress callback
                    return progress_callback(config, signal1, signal2)
                
                result = alignment_functions.scanTx_Y(
                    self.controller,
                    axis_tx=10,
                    axis_y=8,
                    rotation_range_tx=self.rotation_range,
                    search_range_y=self.search_range,
                    rotation_step_tx=self.rotation_step,
                    translation_step_y=self.translation_step,
                    speed_tx=self.rotation_speed,
                    speed_y=self.translation_speed,
                    alpha = self.alpha,
                    progress_callback=scan_progress_callback
                )
                # self.log_signal.emit(f"Scan2D result obtained: {len(result) if result else 'None'}")
                
                # Check if we should stop
                if not self.check_interrupt():
                    self.log_signal.emit("Process stopped by user")
                    self.result_ready.emit(None)
                    return
                
                # Convert to standard format
                if result:
                    config, signal, data_list = result
                    # self.log_signal.emit("Processing scan results")
                    result = (config, signal, data_list)

            elif self.mode == "Spiral Scan":
                self.log_signal.emit("Running Spiral Scan...")
                
                # Check for interruption frequently
                def scan_progress_callback(config, signal1, signal2=None):
                    # Always check for interruption in callbacks
                    if not self.is_running:
                        return False
                    # Call the standard progress callback
                    return progress_callback(config, signal1, signal2)
                
                result = alignment_functions.spiral(
                    self.controller,
                    axis_x=7,
                    axis_y=8,
                    search_range_x=self.search_range,
                    search_range_y=self.search_range,
                    field_pitch_x=self.translation_step,
                    field_pitch_y=self.translation_step,
                    speed_x=self.translation_speed,
                    speed_y=self.translation_speed,
                    alpha=self.alpha,
                    progress_callback=scan_progress_callback
                )
                self.log_signal.emit(f"Spiral result obtained: {len(result) if result else 'None'}")
                
                # Check if we should stop
                if not self.check_interrupt():
                    self.log_signal.emit("Process stopped by user")
                    self.result_ready.emit(None)
                    return
                
                # Convert to standard format
                if result:
                    config, signal, data_list = result
                    self.log_signal.emit("Processing scan results")
                    result = (config, signal, data_list)

            elif self.mode == "Hill Climb":
                self.log_signal.emit("Running Hill Climb...")
                
                # Check for interruption frequently
                def hill_climb_progress_callback(config, signal1, signal2=None):
                    # Always check for interruption in callbacks
                    if not self.is_running:
                        return False
                    # Call the standard progress callback
                    return progress_callback(config, signal1, signal2)
                
                result = alignment_functions.hill_climb_2channel(
                    self.controller,
                    axis_x=7,
                    axis_y=8,
                    step_size=self.translation_step,
                    search_range=self.search_range,
                    speed=self.translation_speed,
                    alpha=self.alpha,
                    max_iterations=3,
                    progress_callback=hill_climb_progress_callback,
                    steps_per_direction=3
                )
                
                # Check if we should stop
                if not self.check_interrupt():
                    self.log_signal.emit("Process stopped by user")
                    self.result_ready.emit(None)
                    return
                
                # Convert config to standard format
                if result:
                    config, signal, data_list = result
                    config = standardize_config(config, self.controller)
                    result = (config, signal, data_list)
                    
            elif self.mode == "Align Tz2":
                def angle_progress_callback(angle, sigs):
                    if not self.is_running:
                        return False
                    # Emit progress signal, but do not update heatmap
                    self.log_signal.emit(f"Align Tz2... Angle: {angle:.3f}, Signals: {sigs[0]:.3f}, {sigs[1]:.3f}, Total: {sigs[0]+sigs[1]:.3f}")
                    return True
                result = alignment_functions.align_Tz2(
                    self.controller,
                    search_range=self.rotation_range,
                    step=self.rotation_step,
                    progress_callback=angle_progress_callback
                )
                if not self.check_interrupt():
                    self.log_signal.emit("Process stopped by user")
                    self.result_ready.emit(None)
                    return
                if result:
                    config, signals, data_list = result
                    self.log_signal.emit(f"Align Tz2 complete. Best angle: {config:.3f}, Best signals: {signals[0]:.3f}, {signals[1]:.3f}")
                    self.result_ready.emit((config, signals, data_list))
                return

            elif self.mode == "Align Tx2":
                def angle_progress_callback(angle, sigs):
                    if not self.is_running:
                        return False
                    self.log_signal.emit(f"Align Tx2... Angle: {angle:.3f}, Signals: {sigs[0]:.3f}, {sigs[1]:.3f}, Total: {sigs[0]+sigs[1]:.3f}")
                    return True
                result = alignment_functions.align_Tx2(
                    self.controller,
                    search_range=self.rotation_range,
                    step=self.rotation_step,
                    progress_callback=angle_progress_callback
                )
                if not self.check_interrupt():
                    self.log_signal.emit("Process stopped by user")
                    self.result_ready.emit(None)
                    return
                if result:
                    config, signals, data_list = result
                    self.log_signal.emit(f"Align Tx2 complete. Best angle: {config:.3f}, Best signals: {signals[0]:.3f}, {signals[1]:.3f}")
                    self.result_ready.emit((config, signals, data_list))
                return
            else:
                self.log_signal.emit("Mode not supported")
                result = None

            if not self.is_running:
                self.log_signal.emit("Alignment process interrupted by user")
                result = None

            # Convert result to use dictionaries and standard format
            if isinstance(result, tuple):
                if len(result) == 3:  # (best_config, best_signals, data_list)
                    best_config, best_signals, data_list = result
                    if isinstance(best_config, tuple):
                        best_config = {'x': best_config[0], 'y': best_config[1], 
                                     'rz': best_config[2] if len(best_config) > 2 else 0}
                    result = (best_config, best_signals, data_list)

            self.result_ready.emit(result)
            
        except Exception as e:
            self.log_signal.emit(f"Error during alignment: {str(e)}")
            self.result_ready.emit(None)

# Matplotlib imports for visualization.
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Canvas for line plots.
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        
        # Create subplots for x, y, and rz positions
        self.ax_x = self.fig.add_subplot(211)  # X position plot
        self.ax_y = self.fig.add_subplot(212)  # Y position plot
        # self.ax_rz = self.fig.add_subplot(313)  # RZ position plot
        
        # Initialize the canvas
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        
        # Set titles and labels
        self.ax_x.set_title("X Position Evolution")
        self.ax_x.set_xlabel("Adjustment Index")
        self.ax_x.set_ylabel("X Position")
        
        self.ax_y.set_title("Y Position Evolution")
        self.ax_y.set_xlabel("Adjustment Index")
        self.ax_y.set_ylabel("Y Position")
        
        # self.ax_rz.set_title("RZ Position Evolution")
        # self.ax_rz.set_xlabel("Adjustment Index")
        # self.ax_rz.set_ylabel("RZ Position")
        
        # Adjust layout to prevent overlap
        # self.fig.tight_layout()
        
    def clear_plots(self):
        """Clear all plots"""
        self.ax_x.clear()
        self.ax_y.clear()
        # self.ax_rz.clear()
        self.draw()

    def update_plots(self, history):
        if history is None or history.size == 0: 
            return
            
        indices = list(range(len(history)))
        
        # Clear previous plots
        self.ax_x.clear()
        self.ax_y.clear()
        # self.ax_rz.clear()
        
        # Plot x position
        self.ax_x.plot(indices, history['x'], marker='o', color='blue')
        self.ax_x.set_title("X Position Evolution")
        self.ax_x.set_xlabel("Adjustment Index")
        self.ax_x.set_ylabel("X Position")
        
        # Plot y position
        self.ax_y.plot(indices, history['y'], marker='o', color='green')
        self.ax_y.set_title("Y Position Evolution")
        self.ax_y.set_xlabel("Adjustment Index")
        self.ax_y.set_ylabel("Y Position")
        
        # # Plot rz position
        # self.ax_rz.plot(indices, history['rz'], marker='o', color='red')
        # self.ax_rz.set_title("RZ Position Evolution")
        # self.ax_rz.set_xlabel("Adjustment Index")
        # self.ax_rz.set_ylabel("RZ Position")
        
        # Adjust layout
        self.fig.tight_layout()
        self.draw()

# Canvas for the dynamic signal heatmap.
class HeatmapCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        # Use constrained_layout instead of tight_layout for better spacing
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
         # Use tight_layout for better spacing
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Initialize plot elements
        self._sc = None  # Scatter plot
        self._cbar = None  # Colorbar
        
        # Set default labels and title
        self.ax.set_title("Signal Heatmap")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.set_aspect('auto')
        
        # Initialize data
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.step_x = None
        self.step_y = None
        self.initialized = False
        self.mode = "default"  # Track the current mode
        
        # Store clicked coordinates
        self.clicked_x = None
        self.clicked_y = None
        
        # Connect mouse click event
        self.mpl_connect('button_press_event', self.on_click)
        
        # Add crosshair marker for clicked point
        self.click_marker = None
        
    def set_mode(self, mode):
        """Set the visualization mode and update labels accordingly"""
        self.mode = mode
        if mode == "Scan Tz,Y":
            self.ax.set_xlabel("Tz Position")
            self.ax.set_ylabel("Y Position")
        elif mode == "Scan Tx,Y":
            self.ax.set_xlabel("Tx Position")
            self.ax.set_ylabel("Y Position")
        else:
            self.ax.set_xlabel("X Position")
            self.ax.set_ylabel("Y Position")
        self.draw()

    def on_click(self, event):
        """Handle mouse click events on the heatmap"""
        if event.inaxes != self.ax:
            return
            
        # Store the clicked coordinates
        self.clicked_x = event.xdata
        self.clicked_y = event.ydata
        
        # Update the marker
        self.update_click_marker()
        
        # Print coordinates for debugging
        print(f"Clicked at x={self.clicked_x:.2f}, y={self.clicked_y:.2f}")
        
    def update_click_marker(self):
        """Update the marker showing the clicked position"""
        # Remove previous marker if it exists
        if self.click_marker:
            self.click_marker.remove()
            self.click_marker = None
            
        # Add new marker if we have coordinates
        if self.clicked_x is not None and self.clicked_y is not None:
            self.click_marker = self.ax.plot(
                self.clicked_x, 
                self.clicked_y, 
                'rx',  # red x marker
                markersize=12,
                markeredgewidth=2
            )[0]
            self.draw()
        
    def reset(self):
        """Reset the heatmap to its initial state"""
        # self.initialized = False
        # self.ax.clear()
        
        # # Note: We intentionally DON'T clear the colorbar here to maintain consistency
        # # This helps prevent layout shifts when restarting scans
        
        # self.ax.set_title("Signal Heatmap")
        # self.ax.set_xlabel("X Position")
        # self.ax.set_ylabel("Y Position")
        # self.ax.set_aspect('auto')
        
        # Reset clicked coordinates
        self.clicked_x = None
        self.clicked_y = None
        self.click_marker = None
        
        self.draw()
        
    def set_range_params(self, x_min, x_max, y_min, y_max, step_x, step_y):
        """Set fixed parameters for the heatmap visualization"""
        # Reset previous state
        self.reset()
        
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.step_x = step_x
        self.step_y = step_y
        
        # Set axis limits
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        
        self.initialized = True
        
    def update_heatmap(self, heatmap_data):
        """Update the heatmap visualization with new data using scatter plot"""
        try:
            if heatmap_data is None or len(heatmap_data) == 0:
                return
                
            data = np.array(heatmap_data)
            
            # Initialize grid if not already done
            if not self.initialized:
                    return
            
            # Clear the axis but keep the colorbar
            self.ax.clear()

            x = data[:, 0]
            y = data[:, 1]
            signal = data[:, 2]
            
            # num_x = len(np.unique(x))
            # num_y = len(np.unique(y))

            # bbox = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            # width, height = bbox.width * self.fig.dpi, bbox.height * self.fig.dpi

            # marker_width = width / num_x
            # marker_height = height / num_y
            # marker_size_pixels = min(marker_width, marker_height)
            # marker_size_points = marker_size_pixels * 72.0 / self.fig.dpi
            # s = marker_size_points ** 2
            
            # Create scatter plot with color mapped to signal intensity
            sc = self.ax.scatter(x, y, c=signal, cmap='jet', 
                               s=400, # marker size
                               marker='s', # square markers
                               alpha=0.9) # transparency
            
            # Create colorbar only if it doesn't exist yet, otherwise update it
            if self._cbar is None:
                self._cbar = self.fig.colorbar(sc, ax=self.ax)
            else:
                # Update existing colorbar with new data range
                self._cbar.update_normal(sc)
            
            # Set labels and title
            self.ax.set_title("Signal Heatmap")
            self.set_mode(self.mode)
            self.ax.set_aspect('auto')
            self.ax.set_box_aspect(1)
            
            # Set axis limits to ensure consistent view
            self.ax.set_xlim(self.x_min, self.x_max)
            self.ax.set_ylim(self.y_min, self.y_max)
            
            # Draw the updated figure
            self.fig.tight_layout()
            self.draw()
            
        except Exception as e:
            raise RuntimeError(f"Error in update_heatmap: {str(e)}")
            # raise RuntimeError(f"Traceback: {traceback.format_exc()}")
            
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Load the UI file
        uic.loadUi('main_window.ui', self)
        
        # Create canvas instances with larger size (motor tab is archived; canvas only if layout exists)
        if hasattr(self, 'visualization_layout'):
            self.canvas = MplCanvas(width=8, height=6)
            self.visualization_layout.addWidget(self.canvas)
        else:
            self.canvas = None  # Motor movement tab archived
        self.heatmap_canvas = HeatmapCanvas(width=8, height=6)
            
        if hasattr(self, 'heatmap_layout'):
            self.heatmap_layout.addWidget(self.heatmap_canvas)
        else:
            self.log_message("Warning: heatmap_layout not found in UI")

        # Controller connection: start with no controller; user connects via UI
        self.controller = None
        self._setup_connection_ui()

        # Initialize data storage
        self.measurement_data = None  # Will be initialized in start_alignment
        self.data_index = 0
        self.last_scan_result = None
        self.update_counter = 0
        self.update_interval = 1  # Update heatmap every row initially every point

        # Initialize control elements that are displayed
        self.on_mode_changed(self.mode_input.currentText())
        
        # Initialize saved position
        self.saved_position = None

        # Custom routine data
        self.current_routine = []
        self.presets_file = "custom_routines.json"
        self.load_presets()

        # Connect signals
        self.run_button.clicked.connect(self.start_alignment)
        self.stop_button.clicked.connect(self.stop_alignment)
        self.move_to_max_button.clicked.connect(self.move_to_max_signal)
        self.manual_move_button.clicked.connect(self.manual_move)
        self.mode_input.currentTextChanged.connect(self.on_mode_changed)
        self.save_position_button.clicked.connect(self.save_position)
        self.go_to_saved_button.clicked.connect(self.go_to_saved_position)
        self.streaming_button.clicked.connect(self.start_streaming)
        
        # Connect the grab coordinates button
        self.grab_coordinates_button.clicked.connect(self.grab_heatmap_coordinates)
        
        # Custom routine signals
        self.add_step_button.clicked.connect(self.add_step)
        self.remove_step_button.clicked.connect(self.remove_step)
        self.move_up_button.clicked.connect(self.move_step_up)
        self.move_down_button.clicked.connect(self.move_step_down)
        self.save_preset_button.clicked.connect(self.save_preset)
        self.run_routine_button.clicked.connect(self.run_custom_routine)
        self.preset_combo.currentTextChanged.connect(self.load_preset)

        # Timer for updating status
        self.ui_timer = QTimer()
        self.ui_timer.setInterval(500)
        self.ui_timer.start()

        self.alignment_worker = None
        self.routine_worker = None

        # Initialize logging
        self.log_display.setReadOnly(True)

        # --- Camera Integration ---
        # Access camera_view and its children
        self.camera_view = self.findChild(QWidget, 'camera_view')
        self.camera_canvas = self.findChild(QWidget, 'camera_canvas')
        self.find_angle_button = self.findChild(QPushButton, 'find_angle_button')
        self.move_angle_button = self.findChild(QPushButton, 'move_angle_button')
        self.find_channels_button = self.findChild(QPushButton, 'find_channels_button')
        self.align_channels_button = self.findChild(QPushButton, 'align_channels_button')
        # Add QLabel for image display if not present
        self.camera_image_label = QLabel()
        self.camera_image_label.setAlignment(Qt.AlignCenter)
        self.camera_image_label.setScaledContents(False)
        camera_canvas_layout = QVBoxLayout(self.camera_canvas)
        camera_canvas_layout.setContentsMargins(0, 0, 0, 0)
        camera_canvas_layout.addWidget(self.camera_image_label)
        self.camera_canvas.setLayout(camera_canvas_layout)
        # Camera state
        self.vmb = None
        self.cam = None
        self.frame_handler_obj = None
        self.current_image = None
        self.streaming = False
        self.alignment_mode = False
        self.edge_lines = None
        self.edge_angle = None
        self.fa_channels = None
        self.chip_channels = None
        self.trajectory = None
        self.trajectory_color = QColor(255, 165, 0)
        # Camera button connections
        self.find_angle_button.clicked.connect(self.find_angle_mode)
        self.move_angle_button.clicked.connect(self.move_angle)
        self.find_channels_button.clicked.connect(self.perform_channel_recognition)
        self.align_channels_button.clicked.connect(self.perform_channel_alignment)
        self.complete_al_button.clicked.connect(self.complete_alignment_sequence)
        # Camera initialization
        self.init_camera()
        self.start_streaming()

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def log_message(self, message):
        """Standardized logging method that updates both the log display and prints to console"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_display.append(formatted_message)
        print(formatted_message)
        # Ensure the log display is scrolled to the bottom
        self.log_display.verticalScrollBar().setValue(
            self.log_display.verticalScrollBar().maximum()
        )

    def _setup_connection_ui(self):
        """Wire connection group: suggest IP from config, connect button creates controller."""
        suggested = alignment_functions.get_suggested_ams_net_id()
        if suggested:
            self.ams_net_id_input.setText(suggested)
        self.suggest_ip_button.clicked.connect(self._suggest_ip)
        self.connect_button.clicked.connect(self._try_connect)
        self.set_connection_state(False)

    def _suggest_ip(self):
        """Fill AMS Net ID from last saved config (suggested address)."""
        suggested = alignment_functions.get_suggested_ams_net_id()
        if suggested:
            self.ams_net_id_input.setText(suggested)
            self.log_message("Suggested AMS Net ID loaded from config.")
        else:
            self.log_message("No saved address in config. Enter AMS Net ID manually.")

    def _try_connect(self):
        """Create SurugaController with current AMS Net ID; enable UI on success."""
        ip = (self.ams_net_id_input.text() or "").strip()
        if not ip:
            QMessageBox.warning(self, "Connection", "Please enter an AMS Net ID (e.g. 5.146.68.196.1.1).")
            return
        self.connect_button.setEnabled(False)
        self.log_message(f"Connecting to {ip}...")
        try:
            self.controller = alignment_functions.SurugaController(ip)
            alignment_functions.save_ams_net_id(ip)
            self.connection_status_label.setText("Connected")
            self.set_connection_state(True)
            self.log_message("Controller connected successfully.")
        except Exception as e:
            self.controller = None
            self.connection_status_label.setText("Not connected")
            self.set_connection_state(False)
            self.log_message(f"Connection failed: {e}")
            QMessageBox.critical(self, "Connection failed", str(e))
        finally:
            self.connect_button.setEnabled(True)

    def set_connection_state(self, connected):
        """Enable or disable controls that require the controller."""
        for w in (
            self.run_button, self.stop_button, self.move_to_max_button,
            self.manual_move_button, self.save_position_button, self.go_to_saved_button,
            self.grab_coordinates_button, self.run_routine_button,
            self.add_step_button, self.remove_step_button, self.move_up_button, self.move_down_button,
        ):
            w.setEnabled(connected)
        if hasattr(self, "control_group"):
            self.control_group.setEnabled(connected)
        if hasattr(self, "manual_group"):
            self.manual_group.setEnabled(connected)
        if hasattr(self, "custom_routine_group"):
            self.custom_routine_group.setEnabled(connected)

    def load_presets(self):
        try:
            if os.path.exists(self.presets_file):
                with open(self.presets_file, 'r') as f:
                    self.presets = json.load(f)
                    self.preset_combo.clear()
                    self.preset_combo.addItems(self.presets.keys())
            else:
                self.presets = {}
        except Exception as e:
            self.log_message(f"Error loading presets: {e}")
            self.presets = {}

    def save_presets(self):
        try:
            with open(self.presets_file, 'w') as f:
                json.dump(self.presets, f, indent=4)
        except Exception as e:
            self.log_message(f"Error saving presets: {e}")

    def save_preset(self):
        if not self.current_routine:
            QMessageBox.warning(self, "Warning", "No steps in the current routine to save")
            return

        name, ok = QInputDialog.getText(self, "Save Preset", "Enter preset name:")
        if ok and name:
            self.presets[name] = self.current_routine
            self.preset_combo.addItem(name)
            self.save_presets()
            self.log_message(f"Saved preset: {name}")

    def load_preset(self, name):
        if name and name in self.presets:
            self.current_routine = self.presets[name].copy()
            self.update_routine_list()
            self.log_message(f"Loaded preset: {name}")

    def add_step(self):
        dialog = StepDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            step_data = dialog.get_step_data()
            self.current_routine.append(step_data)
            self.update_routine_list()
            self.log_message(f"Added step: {step_data['type']}")

    def remove_step(self):
        current_row = self.routine_list.currentRow()
        if current_row >= 0:
            self.current_routine.pop(current_row)
            self.update_routine_list()
            self.log_message("Removed selected step")

    def move_step_up(self):
        current_row = self.routine_list.currentRow()
        if current_row > 0:
            self.current_routine[current_row], self.current_routine[current_row - 1] = \
                self.current_routine[current_row - 1], self.current_routine[current_row]
            self.update_routine_list()
            self.routine_list.setCurrentRow(current_row - 1)

    def move_step_down(self):
        current_row = self.routine_list.currentRow()
        if current_row < len(self.current_routine) - 1:
            self.current_routine[current_row], self.current_routine[current_row + 1] = \
                self.current_routine[current_row + 1], self.current_routine[current_row]
            self.update_routine_list()
            self.routine_list.setCurrentRow(current_row + 1)

    def update_routine_list(self):
        self.routine_list.clear()
        for step in self.current_routine:
            self.routine_list.addItem(f"{step['type']}")

    def run_custom_routine(self):
        if not self.current_routine:
            QMessageBox.warning(self, "Warning", "No steps in the current routine to run")
            return
        if not self.controller:
            self.log_message("Please connect to the controller first (AMS Net ID → Connect).")
            return

        self.run_routine_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log_message("Starting custom routine...")

        self.routine_worker = CustomRoutineWorker(self.controller, self.current_routine)
        self.routine_worker.progress.connect(self.log_message)
        self.routine_worker.finished.connect(self.routine_complete)
        self.routine_worker.error.connect(self.routine_error)
        self.routine_worker.start()

    def routine_complete(self):
        self.run_routine_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_message("Custom routine completed")

    def routine_error(self, message):
        self.run_routine_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_message(f"Error in custom routine: {message}")

    def on_mode_changed(self, mode):
        # Enable/disable rotation controls based on mode
        is_rot = mode in ["Align Tz2", "Scan Tz,Y", "Scan Tx,Y", "Align Tx2"]
        is_transl = mode in ["Scan2D", "Scan Tz,Y", "Scan Tx,Y", "Hill Climb", "Spiral Scan"]
        self.translation_range_input.setVisible(is_transl)
        self.translation_step_input.setVisible(is_transl)
        self.speed_input.setVisible(is_transl)
        self.translation_range_label.setVisible(is_transl)
        self.translation_step_label.setVisible(is_transl)
        self.speed_label.setVisible(is_transl)

        self.rotation_range_input.setVisible(is_rot)
        self.rotation_step_input.setVisible(is_rot)
        self.rotation_speed_input.setVisible(is_rot)
        self.rotation_range_label.setVisible(is_rot)
        self.rotation_step_label.setVisible(is_rot)
        self.rotation_speed_label.setVisible(is_rot)


    def start_alignment(self):
        if not self.controller:
            self.log_message("Please connect to the controller first (AMS Net ID → Connect).")
            return
        translation_speed = self.speed_input.value()
        rotation_speed = self.rotation_speed_input.value()
        search_range = self.translation_range_input.value()
        rotation_range = self.rotation_range_input.value()
        mode = self.mode_input.currentText()
        self.mode = self.mode_input.currentText()
        translation_step = self.translation_step_input.value()
        rotation_step = self.rotation_step_input.value()
        self.alpha = self.alpha_spin.value()
        alpha = self.alpha
        
        self.log_message(f"Starting {mode} alignment...")
        self.log_message(f"Range: {search_range}, Step: {translation_step}, Speed: {translation_speed}")
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Calculate grid size based on translation range and step size
        num_steps = int(search_range / translation_step) + 1
        self.update_interval = num_steps
        if mode == "Scan Tx,Y" or mode == "Scan Tz,Y":
            num_steps_tx = int(rotation_range / rotation_step) + 1
            grid_size = num_steps * num_steps_tx
        else:
            grid_size = num_steps * num_steps  # Square grid
        
        # Initialize structured array for all measurement data with NaN values
        dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('tz', np.float32),
            ('tx', np.float32),
            ('signal1', np.float32),
            ('signal2', np.float32)
        ])
        self.measurement_data = np.full(grid_size, np.nan, dtype=dtype)
        self.data_index = 0
        self.last_scan_result = None
        self.move_to_max_button.setEnabled(False)
        
        # Reset visualizations (motor tab may be archived)
        if hasattr(self, 'position_evolution_group'):
            self.position_evolution_group.show()
        if self.canvas:
            self.canvas.clear_plots()

        # Set heatmap mode and axis ranges
        if mode == "Scan Tz,Y":
            self.heatmap_canvas.set_mode("Scan Tz,Y")
            current_tz = self.controller.AxisComponents[12].GetActualPosition()
            current_y = self.controller.AxisComponents[8].GetActualPosition()
            tz_min = current_tz - rotation_range/2
            tz_max = current_tz + rotation_range/2
            y_min = current_y - search_range/2
            y_max = current_y + search_range/2
            self.heatmap_canvas.set_range_params(
                x_min=tz_min,
                x_max=tz_max,
                y_min=y_min,
                y_max=y_max,
                step_x=rotation_step,
                step_y=translation_step
            )
        elif mode == "Scan Tx,Y":
            self.heatmap_canvas.set_mode("Scan Tx,Y")
            current_tx = self.controller.AxisComponents[10].GetActualPosition()
            current_y = self.controller.AxisComponents[8].GetActualPosition()
            tx_min = current_tx - rotation_range/2
            tx_max = current_tx + rotation_range/2
            y_min = current_y - search_range/2
            y_max = current_y + search_range/2
            self.heatmap_canvas.set_range_params(
                x_min=tx_min,
                x_max=tx_max,
                y_min=y_min,
                y_max=y_max,
                step_x=rotation_step,
                step_y=translation_step
            )
        else:
            self.heatmap_canvas.set_mode("Scan2D")
            current_x = self.controller.AxisComponents[7].GetActualPosition()
            current_y = self.controller.AxisComponents[8].GetActualPosition()
            x_min = current_x - search_range/2
            x_max = current_x + search_range/2
            y_min = current_y - search_range/2
            y_max = current_y + search_range/2
            self.heatmap_canvas.set_range_params(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                step_x=translation_step,
                step_y=translation_step
            )
        
        self.alignment_worker = AlignmentWorker(
            self.controller,
            mode,
            translation_speed,
            rotation_speed,
            search_range,
            translation_step,
            rotation_range,
            rotation_step,
            alpha
        )
        self.alignment_worker.log_signal.connect(self.log_message)
        self.alignment_worker.measurement_update.connect(self.on_measurement_update)
        self.alignment_worker.result_ready.connect(self.alignment_complete)
        self.alignment_worker.start()

    def stop_alignment(self):
        """Safely stop the alignment process by turning off servos and exiting loops"""
        if self.alignment_worker and self.alignment_worker.isRunning():
            self.stop_button.setEnabled(False)
            # self.log_message("Stopping servos...")
            self.controller.stop_requested = True
            # Turn off servos first for safety
            try:
                # Turn off the servo motors to immediately stop movement
                
                for axis in self.controller.AxisComponents.values():
                    axis.Stop()
                # self.log_message("Servos stopped")
            except Exception as e:
                self.log_message(f"Error stopping servos: {e}")
            
            # Set the flag to stop algorithm loops
            self.alignment_worker.stop()
            
            # Setup timer to check when process is done normally
            self.stop_check_timer = QTimer(self)
            self.stop_check_timer.setInterval(100)  # Check every 100ms
            
            def on_timer_tick():
                if not self.alignment_worker.isRunning():
                    self.stop_check_timer.stop()
                    self.run_button.setEnabled(True)
                    # self.log_message("Alignment process stopped normally.")
            
            self.stop_check_timer.timeout.connect(on_timer_tick)
            self.stop_check_timer.start()
            
        elif self.routine_worker and self.routine_worker.isRunning():
            self.stop_button.setEnabled(False)
            # self.log_message("Stopping custom routine and stopping servos...")
            
            # Turn off servos first for safety
            try:
                # Turn off the servo motors to immediately stop movement
                for axis in self.controller.AxisComponents.values():
                    axis.Stop()
                self.log_message("Servos stopped")
            except Exception as e:
                self.log_message(f"Error stopping servos: {e}")
                
            # Set the flag to stop algorithm loops
            self.routine_worker.stop()
            
            # Setup timer for routine worker completion
            self.stop_check_timer = QTimer(self)
            self.stop_check_timer.setInterval(100)  # Check every 100ms
            
            def on_routine_timer_tick():
                if not self.routine_worker.isRunning():
                    self.stop_check_timer.stop()
                    self.run_routine_button.setEnabled(True)
                    self.log_message("Custom routine stopped normally.")
            
            self.stop_check_timer.timeout.connect(on_routine_timer_tick)
            self.stop_check_timer.start()
            
        else:
            self.log_message("No process running.")

    def alignment_complete(self, result):
        """Handle completion of alignment process"""
        try:
            if result is None:
                self.log_message(f"Alignment failed, was cancelled, or was interrupted.")
                return
            
            best_config, best_signals, data_list = result
            
            # Format the configuration string
            config_str = ""
            for i in best_config:
                # append e.g. "x=12.34, "
                config_str += f"{i}={best_config[i]:.2f}, "
            # strip off the trailing comma+space
            config_str = config_str.rstrip(", ")
            
            # Format signals string based on whether it's a tuple (multi-channel) or single value
            if isinstance(best_signals, (tuple, list)):
                if len(best_signals) >= 2:
                    signal_str = f"PM1={best_signals[0]:.2f}, PM2={best_signals[1]:.2f}, Total={sum(best_signals):.2f}"
                else:
                    signal_str = f"Signal={best_signals[0]:.2f}"
            else:
                signal_str = f"Signal={best_signals:.2f}"
                
            self.log_message(f"Alignment Complete.\nBest Config: {config_str}\nBest signal: {signal_str}")
            
            # Update the measurement data array
            self.data_index = 0
            for entry in data_list:
                if self.data_index < len(self.measurement_data):
                    # Handle numpy array format from scan2D and spiral functions
                    if isinstance(entry, np.ndarray) or (isinstance(entry, (list, tuple)) and len(entry) == 4):
                        # Format from scan2D/spiral: [x, y, signal1, signal2]
                        self.measurement_data[self.data_index] = (
                            entry[0],  # x
                            entry[1],  # y
                            self.controller.AxisComponents[12].GetActualPosition(),  # rz (not included in scan data)
                            self.controller.AxisComponents[10].GetActualPosition(),  # tx (not included in scan data)
                            entry[2],  # signal1
                            entry[3]   # signal2
                        )
                        self.data_index += 1
                    # Handle the original expected format
                    elif isinstance(entry, (tuple, list)) and len(entry) >= 1:
                        config = entry[0]
                        if isinstance(config, tuple):
                            self.measurement_data[self.data_index] = (
                                config[0],
                                config[1],
                                config[2] if len(config) > 2 else 0,
                                entry[1],
                                entry[2] if len(entry) > 2 else 0
                            )
                            self.data_index += 1
                        elif isinstance(config, dict): # thats what is expected
                            self.measurement_data[self.data_index] = (
                                config['x'],
                                config['y'],
                                config.get('rz', 0),
                                entry[1],
                                entry[2] if len(entry) > 1 else 0
                            )
                            self.data_index += 1
                        else:
                            self.log_message(f"Warning: Unexpected config format in result data_list")
                    else:
                        self.log_message(f"Warning: Unexpected data entry format in result data_list")
                else:
                    break  # Stop if we've reached the end of measurement_data array
            
            self.last_scan_result = result
            self.move_to_max_button.setEnabled(True)
            
            # Update UI state (motor tab may be archived)
            current_data = self.measurement_data[:self.data_index]
            if self.canvas:
                self.canvas.update_plots(current_data[['x', 'y']]) #, 'rz']])
            self.ui_timer.stop()
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.controller.stop_requested = False            
        except Exception as e:
            self.log_message(f"Error in alignment completion: {str(e)}")
            self.log_message(f"Traceback: {traceback.format_exc()}")
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def on_measurement_update(self, config, signal1, signal2=None):
        """Handle measurement updates from alignment process"""
        try:
            config_dict = config

            # Store all data in the structured array
            if self.data_index < len(self.measurement_data):
                self.measurement_data[self.data_index] = (
                    config_dict.get('x', np.nan),
                    config_dict.get('y', np.nan),
                    config_dict.get('tz', np.nan),
                    config_dict.get('tx', np.nan),
                    signal1,
                    signal2 if signal2 is not None else 0
                )
            
            # Update the heatmap visualization
            if self.data_index % int(self.update_interval) == 0:
                # Extract the correct axes for the heatmap
                try:
                    current_data = self.measurement_data[:self.data_index]
                    if len(current_data) > 0:
                        if self.heatmap_canvas.mode == "Scan Tz,Y":
                            heatmap_points = np.column_stack((
                                current_data['tz'],
                                current_data['y'],
                                current_data['signal1'] + current_data['signal2'] - self.alpha * abs(current_data['signal1'] - current_data['signal2'])
                            ))
                        elif self.heatmap_canvas.mode == "Scan Tx,Y":
                            heatmap_points = np.column_stack((
                                current_data['tx'],
                                current_data['y'],
                                current_data['signal1'] + current_data['signal2'] - self.alpha * abs(current_data['signal1'] - current_data['signal2'])
                            ))
                        else:
                            heatmap_points = np.column_stack((
                                current_data['x'],
                                current_data['y'],
                                current_data['signal1'] + current_data['signal2'] -  self.alpha* abs(current_data['signal1'] - current_data['signal2'])
                            ))
                        self.heatmap_canvas.update_heatmap(heatmap_points)
                except Exception as e:
                    raise RuntimeError(f"Error creating heatmap points: {str(e)}")
            
            self.data_index += 1
            
            # Update status display
            config_text = f"Current Position: X={config_dict.get('x', np.nan):.2f}, Y={config_dict.get('y', np.nan):.2f}, Tz={config_dict.get('tz', np.nan):.2f}, Tx={config_dict.get('tx', np.nan):.2f}"
            signal_text = ""
            if signal2 is not None:
                signal_text += f"\nSignals: PM1={signal1:.2f}, PM2={signal2:.2f}, Total={signal1 + signal2:.2f}"
            else:
                signal_text += f"\nSignal: {signal1:.2f}"
            self.position_label.setText(config_text)
            self.signal_label.setText(signal_text)
            
        except Exception as e:
            self.log_message(f"Error updating measurement display: {str(e)}")
            self.log_message(f"Traceback: {traceback.format_exc()}")
            

    def manual_move(self):
        try:
            target_x = self.manual_x_input.value()
            target_y = self.manual_y_input.value()
            target_tz = self.manual_tz_input.value()
            target_tx = self.manual_tx_input.value()
            current_x = self.controller.AxisComponents[7].GetActualPosition()
            current_y = self.controller.AxisComponents[8].GetActualPosition()
            
            delta_x = target_x - current_x
            delta_y = target_y - current_y
            self.controller.axis2d.MoveAbsolute(target_x, target_y)
            self.controller.AxisComponents[12].MoveAbsolute(target_tz)
            self.controller.AxisComponents[10].MoveAbsolute(target_tx)
            while self.controller.axis2d.IsMoving() or self.controller.AxisComponents[12].IsMoving() or self.controller.AxisComponents[10].IsMoving():
                time.sleep(0.07)
            QApplication.processEvents()
            new_signal = self.controller.Alignment.GetVoltage(1) + self.controller.Alignment.GetVoltage(2) - self.alpha * abs(self.controller.Alignment.GetVoltage(1) - self.controller.Alignment.GetVoltage(2))

            self.log_message(f"Manual move to x: {target_x}, y: {target_y} (Δx: {delta_x}, Δy: {delta_y}) with signal {new_signal:.2f}")
        except Exception as e:
            self.log_message(f"Manual move error: {e}")

    def move_to_max_signal(self):
        if self.last_scan_result is None:
            self.log_message("No scan results available")
            return

        best_config, best_signals, _ = self.last_scan_result
        try:
            # Move to the position with maximum signal
            mode = self.mode # _input.currentText()
            if mode == "Scan Tz,Y":
                # Move Tz (axis 12) and Y (axis 8)
                tz = best_config.get('tz', None)
                y = best_config.get('y', None)
                if tz is not None and y is not None:
                    self.controller.AxisComponents[12].MoveAbsolute(tz)
                    self.controller.AxisComponents[8].MoveAbsolute(y)
                    while self.controller.AxisComponents[12].IsMoving() or self.controller.AxisComponents[8].IsMoving():
                        time.sleep(0.01)
                    current_power = self.controller.Alignment.GetVoltage(1) + self.controller.Alignment.GetVoltage(2) - self.alpha * abs(self.controller.Alignment.GetVoltage(1) - self.controller.Alignment.GetVoltage(2))
                    self.log_message(f"Moved to max power position: tz={tz:.4f}, y={y:.2f}\nCurrent power: {current_power:.2f} dBm")
                else:
                    self.log_message("Best config missing tz or y for Scan Tz,Y")
            elif mode == "Scan Tx,Y":
                tx = best_config.get('tx', None)
                y = best_config.get('y', None)
                if tx is not None and y is not None:
                    self.controller.AxisComponents[10].MoveAbsolute(tx)
                    self.controller.AxisComponents[8].MoveAbsolute(y)
                    while self.controller.AxisComponents[10].IsMoving() or self.controller.AxisComponents[8].IsMoving():
                        time.sleep(0.01)
                    current_power = self.controller.Alignment.GetVoltage(1) + self.controller.Alignment.GetVoltage(2) - self.alpha * abs(self.controller.Alignment.GetVoltage(1) - self.controller.Alignment.GetVoltage(2))
                    self.log_message(f"Moved to max power position: tx={tx:.4f}, y={y:.2f}\nCurrent power: {current_power:.2f} dBm")
                else:
                    self.log_message("Best config missing tx or y for Scan Tx,Y")
            else:
                x = best_config.get('x', None)
                y = best_config.get('y', None)
                if x is not None and y is not None:
                    self.controller.axis2d.MoveAbsolute(x, y)
                    while self.controller.axis2d.IsMoving():
                        time.sleep(0.01)
                    current_power = self.controller.Alignment.GetPower(1) + self.controller.Alignment.GetPower(2)
                    self.log_message(f"Moved to max power position: x={x:.2f}, y={y:.2f}" +
                            (f", tz={best_config.get('tz', 0):.4f}" if 'tz' in best_config else "") +
                            f"\nCurrent power: {current_power:.2f} dBm")
                else:
                    self.log_message("Best config missing x or y for Scan2D")
        except Exception as e:
            self.log_message(f"Error moving to max power position: {e}")
            
    def reset_system(self):
        """Reset all variables and visualizations without closing the program"""
        try:
            # Reset data structures
            self.measurement_data = None
            self.data_index = 0
            self.last_scan_result = None
            self.current_routine.clear()
            
            # Reset visualizations (motor tab may be archived)
            if self.canvas:
                self.canvas.clear_plots()
            self.heatmap_canvas.update_heatmap([])
            
            # Reset UI elements
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.move_to_max_button.setEnabled(False)
            
            self.log_message("Data reset complete")
        except Exception as e:
            self.log_message(f"Error during reset: {e}")

    def save_position(self):
        """Save the current position of the motors"""
        try:
            x_pos = self.controller.AxisComponents[7].GetActualPosition()
            y_pos = self.controller.AxisComponents[8].GetActualPosition()
            tz_pos = self.controller.AxisComponents[12].GetActualPosition()
            
            self.saved_position = {
                'x': x_pos,
                'y': y_pos,
                'tz': tz_pos
            }
            
            self.log_message(f"Position saved: x={x_pos:.2f}, y={y_pos:.2f}, tz={tz_pos:.4f}")
            self.go_to_saved_button.setEnabled(True)
        except Exception as e:
            self.log_message(f"Error saving position: {str(e)}")

    def go_to_saved_position(self):
        """Move motors to the saved position"""
        if self.saved_position is None:
            self.log_message("No position has been saved yet")
            return
            
        try:
            self.log_message(f"Moving to saved position: x={self.saved_position['x']:.2f}, y={self.saved_position['y']:.2f}")
            
            # Move XY first
            self.controller.axis2d.MoveAbsolute(self.saved_position['x'], self.saved_position['y'])
            while self.controller.axis2d.IsMoving():
                time.sleep(0.01)
            
            # Then move TZ
            self.controller.AxisComponents[12].MoveAbsolute(self.saved_position['tz'])
            while self.controller.AxisComponents[12].IsMoving():
                time.sleep(0.01)
                
            self.log_message("Movement to saved position completed")
        except Exception as e:
            self.log_message(f"Error moving to saved position: {str(e)}")

    def grab_heatmap_coordinates(self):
        """Grab the coordinates from the heatmap"""
        try:
            mode = self.mode
            if self.heatmap_canvas.clicked_x is not None and self.heatmap_canvas.clicked_y is not None:
                if mode == "Scan Tz,Y":
                    self.log_message(f"Grabbed coordinates: tz={self.heatmap_canvas.clicked_x:.4f}, y={self.heatmap_canvas.clicked_y:.2f}")
                    self.manual_tz_input.setValue(self.heatmap_canvas.clicked_x)
                    self.manual_y_input.setValue(self.heatmap_canvas.clicked_y)
                    self.manual_x_input.setValue(self.controller.AxisComponents[7].GetActualPosition())
                    self.manual_tx_input.setValue(self.controller.AxisComponents[10].GetActualPosition())
                elif mode == "Scan Tx,Y":
                    self.log_message(f"Grabbed coordinates: tx={self.heatmap_canvas.clicked_x:.4f}, y={self.heatmap_canvas.clicked_y:.2f}")
                    self.manual_tx_input.setValue(self.heatmap_canvas.clicked_x)
                    self.manual_y_input.setValue(self.heatmap_canvas.clicked_y)
                    self.manual_tz_input.setValue(self.controller.AxisComponents[12].GetActualPosition())
                    self.manual_x_input.setValue(self.controller.AxisComponents[7].GetActualPosition())
                else:
                    self.log_message(f"Grabbed coordinates: x={self.heatmap_canvas.clicked_x:.2f}, y={self.heatmap_canvas.clicked_y:.2f}")
                    self.manual_x_input.setValue(self.heatmap_canvas.clicked_x)
                    self.manual_y_input.setValue(self.heatmap_canvas.clicked_y)
                    self.manual_tz_input.setValue(self.controller.AxisComponents[12].GetActualPosition())
                    self.manual_tx_input.setValue(self.controller.AxisComponents[10].GetActualPosition())
            else:
                self.log_message("No coordinates clicked on the heatmap yet")
        except Exception as e:
            self.log_message(f"Error grabbing coordinates: {str(e)}")

    def init_camera(self):
        try:
            self.vmb = VmbSystem.get_instance()
            self.vmb.__enter__()
            self.cam = self.vmb.get_all_cameras()[0]
            self.cam.__enter__()
            if PixelFormat.Bgr8 in self.cam.get_pixel_formats():
                self.cam.set_pixel_format(PixelFormat.Bgr8)
            else:
                raise RuntimeError("Camera does not support BGR8.")
            self.frame_handler_obj = self.FrameHandler(self.cam, self)
        except Exception as e:
            self.log_message(f"Camera initialization error: {e}")

    class FrameHandler(QObject):
        frame_ready = pyqtSignal(object)
        def __init__(self, camera, main_window):
            super().__init__()
            self.camera = camera
            self.main_window = main_window
        def frame_handler(self, cam, stream, frame):
            try:
                img = frame.as_numpy_ndarray()
                h, w, c = img.shape
                qimg = QImage(img.data, w, h, QImage.Format_BGR888)
                self.frame_ready.emit(qimg)
                cam.queue_frame(frame)
            except Exception as e:
                print("Frame handler error:", e)
                cam.queue_frame(frame)

    def start_streaming(self):
        if not self.streaming:
            try:
                self.cam.start_streaming(self.frame_handler_obj.frame_handler)
                self.frame_handler_obj.frame_ready.connect(self.display_image)
                self.streaming = True
                self.log_message("Camera streaming started")
            except Exception as e:
                self.log_message(f"Error starting streaming: {e}")

    def pause_streaming(self):
        if self.streaming:
            try:
                self.cam.stop_streaming()
                self.streaming = False
                self.log_message("Camera streaming stopped")
            except Exception as e:
                self.log_message(f"Error stopping streaming: {e}")

    def display_image(self, qimg):
        self.current_image = qimg
        scaled_img = qimg.scaled(self.camera_image_label.contentsRect().size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_image_label.setPixmap(QPixmap.fromImage(scaled_img))

    def resizeEvent(self, event):
        if hasattr(self, 'current_image') and self.current_image is not None:
            self.display_image(self.current_image)
        super().resizeEvent(event)

    def find_angle_mode(self):
        self.pause_streaming()
        if not self.alignment_mode:
            self.alignment_mode = True
            self.camera_image_label.setStyleSheet("border: 3px solid orange;")
            self.find_angle_button.setText("Exit Alignment")
            if self.current_image is not None:
                self.perform_edge_detection()
        else:
            self.alignment_mode = False
            self.camera_image_label.setStyleSheet("border: 3px solid green;")
            self.find_angle_button.setText("Find Angle")
            self.edge_lines = None
            self.edge_angle = None

    def perform_edge_detection(self):
        try:
            qimg = self.current_image
            width = qimg.width()
            height = qimg.height()
            ptr = qimg.bits()
            ptr.setsize(height * width * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            left_line, right_line = find_edge_lines(img_bgr, debug=False)
            self.edge_lines = (left_line, right_line)
            self.edge_angle = angle_between_lines(left_line, right_line)
            self.log_message(f"Angle: {self.edge_angle}°")
            self.display_image_with_edges()
        except Exception as e:
            self.log_message(f"Edge detection error: {e}")

    def display_image_with_edges(self):
        if self.current_image is None or self.edge_lines is None:
            return
        qimg = self.current_image.copy()
        painter = QPainter(qimg)
        left_line, right_line = self.edge_lines
        painter.setPen(QPen(QColor(0, 0, 255), 12))
        painter.drawLine(left_line[0], left_line[1], left_line[2], left_line[3])
        painter.setPen(QPen(QColor(255, 0, 0), 12))
        painter.drawLine(right_line[0], right_line[1], right_line[2], right_line[3])
        if self.edge_angle is not None:
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            font = self.font()
            font.setPointSize(font.pointSize() * 4)
            painter.setFont(font)
            painter.drawText(10, 30, f"Angle: {self.edge_angle:.2f}°")
        painter.end()
        self.display_image(qimg)
        # self.camera_image_label.repaint()

    def move_angle(self):
        # Placeholder for move angle logic
        self.log_message("Move Angle button pressed (not implemented)")

    def perform_channel_recognition(self):
        self.pause_streaming()
        if self.current_image is None or self.edge_lines is None:
            self.log_message("No image or edge lines for channel recognition.")
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
            self.display_image_with_channels()
        except Exception as e:
            self.log_message(f"Channel recognition error: {e}")

    def display_image_with_channels(self):
        if self.current_image is None or self.fa_channels is None or self.chip_channels is None:
            return
        qimg = self.current_image.copy()
        painter = QPainter(qimg)
        left_line = self.edge_lines[0]
        left_x = max(left_line[0], left_line[2])
        painter.setPen(QPen(QColor(0, 0, 255), 12))
        for idx, y in enumerate(self.fa_channels):
            painter.drawEllipse(left_x-7, int(y)-7, 14, 14)
            # if idx == 0:
                # painter.drawText(left_x+10, int(y)+5, "1")
            painter.drawText(left_x-25, int(y)+5, f"{idx+1}")
        right_line = self.edge_lines[1]
        right_x = min(right_line[0], right_line[2])
        painter.setPen(QPen(QColor(255, 0, 0), 12))
        for idx, y in enumerate(self.chip_channels):
            painter.drawEllipse(right_x-7, int(y)-7, 14, 14)
            # if idx == 0:
                # painter.drawText(right_x+10, int(y)+5, "1")
            painter.drawText(right_x+10, int(y)+5, f"{idx+1}")
        painter.end()
        self.display_image(qimg)

    def perform_channel_alignment(self):
        """Calculate and display the trajectory from FA channel 1 to chip channel 1, then move FA along the trajectory with live feedback."""
        if (self.fa_channels is None or len(self.fa_channels) == 0 or
            self.chip_channels is None or len(self.chip_channels) == 0 or
            self.edge_lines is None):
            self.log_message("No channels or edge lines for alignment.")
            return
        qimg = self.current_image.copy()
        painter = QPainter(qimg)
        self.trajectory_color = QColor(255, 165, 0)  # Orange
        painter.setPen(QPen(self.trajectory_color, 12, Qt.SolidLine))
        # Start: FA channel 1 (left edge)
        left_line = self.edge_lines[0]
        left_x = max(left_line[0], left_line[2])
        fa_y = int(self.fa_channels[-1])
        start = (left_x, fa_y)
        # End: chip channel 1 (right edge)
        right_line = self.edge_lines[1]
        right_x = min(right_line[0], right_line[2])
        chip_y = int(self.chip_channels[-1])
        end = (right_x, chip_y)
        # Draw trajectory (3 segments)
        mid_x = (start[0] + end[0]) // 2
        painter.drawLine(start[0], start[1], mid_x, start[1])
        painter.drawLine(mid_x, start[1], mid_x, end[1])
        painter.drawLine(mid_x, end[1], end[0], end[1])
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.setFont(QFont("Arial", 40))
        painter.drawText(end[0], end[1] + 20, "Trajectory")
        painter.end()
        self.display_image(qimg)
        # Start streaming and move FA along the trajectory
        self.start_streaming()
        import threading
        threading.Thread(target=self.move_fa_to_chip, args=(True,), daemon=True).start()

    def move_fa_to_chip(self, live=True):
        """Move the FA along the calculated trajectory (3 segments) using the already connected controller, converting pixels to microns using chip channel pitch. If live=True, keep the trajectory orange during movement, then set to green and update display. Runs in a thread if called from perform_channel_alignment."""
        import time
        start_move = time.time()
        if self.controller is None:
            self.log_message("Controller not initialized.")
            return
        if (self.fa_channels is None or len(self.fa_channels) == 0 or
            self.chip_channels is None or len(self.chip_channels) < 2 or
            self.edge_lines is None):
            self.log_message("No channels or edge lines for movement.")
            return
        # Get start and end points
        left_line = self.edge_lines[0]
        left_x = max(left_line[0], left_line[2])
        fa_y = int(self.fa_channels[-1])
        start = (left_x, fa_y)
        right_line = self.edge_lines[1]
        right_x = min(right_line[0], right_line[2])
        chip_y = int(self.chip_channels[-1])
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
        self.log_message(f"Pixel pitch: {pixel_pitch:.2f} px, scale: {microns_per_pixel:.4f} µm/px")
        # Convert all dx/dy to microns
        dx1_um = dx1 * microns_per_pixel
        dy1_um = dy1 * microns_per_pixel
        dx2_um = dx2 * microns_per_pixel
        dy2_um = dy2 * microns_per_pixel
        dx3_um = dx3 * microns_per_pixel * 0.95
        dy3_um = dy3 * microns_per_pixel
        
        dx_um = dx1_um + dx2_um + dx3_um
        dy_um = dy1_um + dy2_um + dy3_um
        
        
        self.log_message(f"Moving FA (in µm): dx={dx_um:.2f}, dy={dy_um:.2f}")
        try:
            # Use Axis2D for XY moves
            speed = self.speed_input.value()
            self.controller.axis2d_zx.SetSpeed(100) # TODO: Set the speed fixed after testing
            # 1. Move horizontally from start to (mid_x, start[1])
            self.controller.axis2d_zx.MoveRelative(dx_um, dy_um)
            while self.controller.axis2d_zx.IsMoving():
                time.sleep(0.01)
            self.log_message("Movement to chip complete.")
            duration = time.time() - start_move
            self.log_message(f"[Move FA to Chip] Duration: {duration:.2f} seconds.")
        except Exception as e:
            self.log_message(f"Movement error: {e}")
        
        # self.log_message(f"Moving FA (in µm): dx1={dx1_um:.2f}, dy1={dy1_um:.2f}; dx2={dx2_um:.2f}, dy2={dy2_um:.2f}; dx3={dx3_um:.2f}, dy3={dy3_um:.2f}")
        # try:
        #     # Use Axis2D for XY moves
        #     speed = self.speed_input.value()
        #     self.controller.axis2d_zx.SetSpeed(speed) # TODO: Set the speed fixed after testing
        #     # 1. Move horizontally from start to (mid_x, start[1])
        #     self.controller.axis2d_zx.MoveRelative(dx1_um, dy1_um)
        #     while self.controller.axis2d_zx.IsMoving():
        #         time.sleep(0.01)
        #     # 2. Move vertically from (mid_x, start[1]) to (mid_x, end[1])
        #     self.controller.axis2d_zx.MoveRelative(0, dy2_um)
        #     while self.controller.axis2d_zx.IsMoving():
        #         time.sleep(0.01)
        #     # 3. Move horizontally from (mid_x, end[1]) to end
        #     self.controller.axis2d_zx.MoveRelative(dx3_um, 0)
        #     while self.controller.axis2d_zx.IsMoving():
        #         time.sleep(0.01)
        #     self.log_message("Movement to chip complete.")
        # except Exception as e:
        #     self.log_message(f"Movement error: {e}")

    def complete_alignment_sequence(self):
        """Run line detection, channel recognition, and move FA to chip in sequence."""
        import time
        try:
            start_total = time.time()
            self.log_message("[Complete Alignment] Starting edge (line) detection...")
            start = time.time()
            self.find_angle_mode()
            duration = time.time() - start
            self.log_message(f"[Complete Alignment] Edge detection took {duration:.2f} seconds.")

            self.log_message("[Complete Alignment] Running channel recognition...")
            start = time.time()
            self.perform_channel_recognition()
            duration = time.time() - start
            self.log_message(f"[Complete Alignment] Channel recognition took {duration:.2f} seconds.")

            self.log_message("[Complete Alignment] Moving FA to chip...")
            start = time.time()
            self.perform_channel_alignment()  # This calls move_fa_to_chip
            duration = time.time() - start
            self.log_message(f"[Complete Alignment] Channel alignment/move took {duration:.2f} seconds.")

            total_duration = time.time() - start_total
            self.log_message(f"[Complete Alignment] Sequence complete. Total time: {total_duration:.2f} seconds.")
        except Exception as e:
            self.log_message(f"[Complete Alignment] Error: {e}")
