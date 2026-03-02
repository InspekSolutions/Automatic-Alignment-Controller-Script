"""
alignment_functions.py

This module implements alignment algorithms for optical fiber alignment in different modes.

    For first light search we use:
  - "xymax": Uses only x and y axes.
  - "snake": Covers the fixed area with a snake-like movement.
  - "spiral": Covers the fixed area with a spiral movement.

Each function uses a coordinate-descent approach that moves the motor through candidate positions
and selects the configuration that maximizes the signal measured by the power meter.
A progress_callback parameter (optional) allows reporting every new best configuration.
"""

import sys
import os
import json
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

# ----------------------------------------
# Connection config: suggested AMS Net ID (saved last-used address)
# ----------------------------------------
CONNECTION_CONFIG_FILE = "connection_config.json"


def get_suggested_ams_net_id():
    """
    Return a suggested AMS Net ID for the Suruga controller:
    - If connection_config.json exists, returns the last saved 'ams_net_id'.
    - Otherwise returns a placeholder (user must enter or paste an address).
    """
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONNECTION_CONFIG_FILE)
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("ams_net_id", "") or ""
    except Exception:
        pass
    return ""


def get_suggested_machine():
    """Return the last saved machine label (e.g. 'Machine 1') from config, or empty string."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONNECTION_CONFIG_FILE)
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("machine", "") or ""
    except Exception:
        pass
    return ""


def save_ams_net_id(ams_net_id, machine=None):
    """Save the given AMS Net ID (and optional machine label) to connection_config.json."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONNECTION_CONFIG_FILE)
        payload = {"ams_net_id": ams_net_id}
        if machine is not None:
            payload["machine"] = machine
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


# ----------------------------------------
# SurugaController: wrapper around srgmc.dll
# ----------------------------------------
# Machine 1: physical axes 7-12. Machine 2: physical axes 1-6 (mapped to logical 7-12 in code).
MACHINE_AXIS_RANGES = {
    1: (7, 12),   # Machine 1: channels 7-12
    2: (1, 6),    # Machine 2: channels 1-6
}


class SurugaController:
    def __init__(self, ip_address, machine=1):
        """
        Initialize and connect to the SurugaSeiki alignment system via srgmc.dll
        using the specified AMS Net ID (ip_address).
        machine: 1 = axes 7-12, 2 = axes 1-6 (exposed as logical 7-12 to callers).
        """
        if not (ip_address and str(ip_address).strip()):
            raise ValueError("AMS Net ID (ip_address) is required.")
        self._machine = int(machine)
        first, last = MACHINE_AXIS_RANGES.get(self._machine, (7, 12))

        self.alignSystem = SSM.System.Instance
        self.alignSystem.SetAddress(str(ip_address).strip())

        time.sleep(1)
        if not self.alignSystem.Connected:
            raise RuntimeError("Failed to connect to Suruga alignment system.")

        # Create only the AxisComponents valid for this machine (avoids "out of range" error)
        self.AxisComponents = {}
        if self._machine == 1:
            for axis_number in range(first, last + 1):
                self.AxisComponents[axis_number] = SSM.AxisComponents(axis_number)
            self.axis2d = SSM.Axis2D(7, 8)
            self.axis2d_zx = SSM.Axis2D(9, 7)
        else:
            # Machine 2: physical 1-6, expose as logical 7-12 so existing code works
            _logical_to_physical = {7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6}
            for logical, physical in _logical_to_physical.items():
                self.AxisComponents[logical] = SSM.AxisComponents(physical)
            self.axis2d = SSM.Axis2D(1, 2)       # physical X,Y
            self.axis2d_zx = SSM.Axis2D(3, 1)   # physical Z,X

        self.Alignment = SSM.Alignment()
        self.stop_requested = False
        for axis in self.AxisComponents.values():
            if not axis.IsServoOn():
                axis.TurnOnServo()

    def move_relative(self, axis_number, distance):
        """
        Simple wrapper for MoveRelative on a single axis.
        """
        axis_comp = self.AxisComponents[axis_number]
        axis_comp.MoveRelative(distance)
        # Optionally wait for motion to finish, etc.
        # while axis_comp.IsMoving():
        #     time.sleep(0.1)

    def set_flat_parameter(self, param):
        """
        Call Alignment.SetFlat(...) with a SurugaSeiki 'FlatParameter' object.
        param must be an instance of SSM.Alignment.FlatParameter.
        """
        self.Alignment.SetFlat(param)

    def start_flat_alignment(self):
        """
        Start a flat alignment with the parameters that were previously set.
        """
        err = self.Alignment.StartFlat()
        if err != None:
            raise RuntimeError(f"StartFlat returned error: {err}")

    def wait_for_alignment_done(self, poll_interval=0.5):
        """
        Wait for alignment to complete or fail.
        """
        while True:
            status = self.Alignment.GetStatus()
            if status == SSM.Alignment.Status.Success:
                return  # done
            elif status == SSM.Alignment.Status.Aligning:
                # Optionally read aligning status
                align_status = self.Alignment.GetAligningStatus()
                # print(f"Aligning status: {align_status}")
                time.sleep(poll_interval)
            else:
                # Some error or other condition
                raise RuntimeError(f"Alignment ended with status: {status}")


def check_error(err, msg=""):
    # Assuming 0 is the "no error" code.
    if err != 1:
        raise RuntimeError(f"{msg} error: {err}")
        
def spiral(controller, axis_x=7, axis_y=8, 
          search_range_x=100, search_range_y=100,
          field_pitch_x=1, field_pitch_y=1,
          speed_x=1000, speed_y=1000, alpha=0,
          progress_callback=None):
    """
    Spiral scan of a 2D grid in x,y from -search_range/2 to search_range/2
    (relative to the current x,y), measuring the signal at each point.
    The scan starts from the center and expands outward in a spiral pattern.
    
    Returns:
      (best_config, best_signal, data_list)
      where data_list is a numpy array with shape (n, 4) containing [x, y, signal1, signal2] for each point.
    """
    best_signal = -float('inf')
    best_config = None

    # Use the current position as the center reference point
    center_pos = controller.axis2d.GetActualPosition()

    # Compute the number of steps in each direction (half range in each direction)
    num_steps_x = int(search_range_x / field_pitch_x) + 1
    num_steps_y = int(search_range_y / field_pitch_y) + 1
    half_steps_x = num_steps_x // 2
    half_steps_y = num_steps_y // 2

    # Calculate the total number of points in the spiral
    max_radius = max(half_steps_x, half_steps_y)
    total_points = 1  # Center point
    for radius in range(1, max_radius + 1):
        total_points += 8 * radius  # Points at this radius
    
    # Preallocate a static matrix to store the scan data
    # Each row holds [x, y, signal1, signal2]
    data_list = np.zeros((total_points, 4), dtype=float)
    count = 0

    # Move to center position first
    controller.axis2d.SetSpeed(speed_x)
    controller.axis2d.MoveAbsolute(center_pos.X, center_pos.Y)
    while controller.axis2d.IsMoving():
        time.sleep(0.07)

    # Spiral pattern: start from center and expand outward
    for radius in range(max_radius + 1):
        if radius == 0:
            # Just measure at center point
            target_x = center_pos.X
            target_y = center_pos.Y
            err = controller.axis2d.MoveAbsolute(target_x, target_y)
            move_start_time = time.time()
            while controller.axis2d.IsMoving():
                time.sleep(0.01)
                # Check if we should interrupt move after 100ms intervals
                if progress_callback and time.time() - move_start_time > 0.1:
                    move_start_time = time.time()  # Reset timer
                    should_continue = progress_callback((target_x, target_y), 0, 0)  # Use zeros as placeholders
                    if should_continue is False:
                        print("Movement interrupted by user")
                        return best_config, best_signal, data_list[:count]

            # Get the actual position after move (more accurate than target)
            current_pos = controller.axis2d.GetActualPosition()
            signal1 = controller.Alignment.GetVoltage(1)
            signal2 = controller.Alignment.GetVoltage(2)
            signal_sum = signal1 + signal2
            metric = signal_sum - alpha*abs(signal1-signal2)

            # Store data in numpy array
            data_list[count] = [current_pos.X, current_pos.Y, signal1, signal2]
            count += 1

            if metric > best_signal:
                best_signal = metric
                best_config = {'x': current_pos.X, 'y': current_pos.Y}

            if progress_callback:
                progress_callback({'x': current_pos.X, 'y': current_pos.Y}, signal1, signal2)
        else:
            # For each radius, measure points in a spiral pattern
            # Calculate the number of points in this radius
            points = 8 * radius if radius > 0 else 1
            for i in range(points):
                # Calculate angle for this point
                angle = (2 * 3.14159 * i) / points
                
                # Calculate x and y offsets
                x_offset = radius * field_pitch_x * math.cos(angle)
                y_offset = radius * field_pitch_y * math.sin(angle)
                
                # Calculate target position
                target_x = center_pos.X + x_offset
                target_y = center_pos.Y + y_offset

                # Move to the target position
                err = controller.axis2d.MoveAbsolute(target_x, target_y)
                move_start_time = time.time()
                while controller.axis2d.IsMoving():
                    if controller.stop_requested:
                        return best_config, best_signal, data_list[:count, :]
                    time.sleep(0.01)

                # Get the actual position after move (more accurate than target)
                current_pos = controller.axis2d.GetActualPosition()
                
                # Measure both signals
                signal1 = controller.Alignment.GetVoltage(1)
                signal2 = controller.Alignment.GetVoltage(2)
                signal_sum = signal1 + signal2
                metric = signal_sum - alpha*abs(signal1-signal2)

                # Store data in numpy array
                if count < total_points:
                    data_list[count] = [current_pos.X, current_pos.Y, signal1, signal2]
                    count += 1

                if metric > best_signal:
                    best_signal = metric
                    best_config = {'x': current_pos.X, 'y': current_pos.Y}

                if progress_callback:
                    progress_callback({'x': current_pos.X, 'y': current_pos.Y}, signal1, signal2)

    # Return only the filled portion of the data list
    return best_config, best_signal, data_list

def scan2D(controller, axis_x=7, axis_y=8, 
          search_range_x=100, search_range_y=100,
          field_pitch_x=1, field_pitch_y=1,
          speed_x=1000, speed_y=1000, alpha=0,
          progress_callback=None):
    """
    2D scan of a grid in x,y from -search_range/2 to search_range/2
    (relative to the current x,y), measuring the signal at each point.
    The scan starts from the bottom left corner and moves in alternating rows.
    
    Returns:
      (best_config, best_signal, data_list)
      where data_list is a numpy array with shape (n, 4) containing [x, y, signal1, signal2] for each point.
    """
    best_signal = -float('inf')
    best_config = None

    # Use the current position as the center reference point
    center_pos = controller.axis2d.GetActualPosition()

    # Compute the number of steps in each direction (half range in each direction)
    num_steps_x = int(search_range_x / field_pitch_x) + 1
    num_steps_y = int(search_range_y / field_pitch_y) + 1
    total_points = num_steps_x * num_steps_y

    # Preallocate a static matrix to store the scan data.
    # Each row holds [target_x, target_y, signal1, signal2]
    data_list = np.empty((total_points, 4), dtype=float)

    count = 0
    half_steps_x = num_steps_x // 2
    half_steps_y = num_steps_y // 2

    # Calculate the bottom left corner position
    start_x = center_pos.X - half_steps_x * field_pitch_x
    start_y = center_pos.Y - half_steps_y * field_pitch_y

    # Move to bottom left corner first
    controller.axis2d.SetSpeed(speed_x)
    controller.axis2d.MoveAbsolute(start_x, start_y)
    
    while controller.axis2d.IsMoving():
        if not controller.AxisComponents[axis_x].IsServoOn() or not controller.AxisComponents[axis_y].IsServoOn():
            return best_config, best_signal, data_list[:count, :]
        time.sleep(0.07)
    time.sleep(2.0)  # wait 2 s while the ranges in powermeters change and the voltage stabilizes
    # Start scanning from bottom left, alternating row direction
    for row in range(num_steps_y):
        # For snake pattern: even rows left->right, odd rows right->left
        if row % 2 == 0:
            x_indices = range(num_steps_x)
        else:
            x_indices = range(num_steps_x - 1, -1, -1)

        for col in x_indices:
            # Calculate target position
            target_x = start_x + col * field_pitch_x
            target_y = start_y + row * field_pitch_y

            # Move to the target position
            controller.axis2d.MoveAbsolute(target_x, target_y)
            while controller.axis2d.IsMoving():
                if controller.stop_requested:
                    return best_config, best_signal, data_list[:count, :]
                time.sleep(0.01)

            # Measure both signals
            signal1 = controller.Alignment.GetVoltage(1)
            signal2 = controller.Alignment.GetVoltage(2)
            current_pos = controller.axis2d.GetActualPosition()
            signal_sum = signal1 + signal2
            metric = signal_sum - alpha*abs(signal1-signal2)
            config = {'x': current_pos.X, 'y': current_pos.Y}
            data_list[count, :] = [current_pos.X, current_pos.Y, signal1, signal2]
            count += 1

            # Update the best signal if sum is higher
            if metric > best_signal:
                best_signal = metric
                best_config = config

            # Call the progress callback if provided
            if progress_callback:
                progress_callback(config, signal1, signal2)

    return best_config, best_signal, data_list

def scanTz_Y(controller, axis_tz=12, axis_y=8, 
        rotation_range_tz=1, search_range_y=10,
        rotation_step_tz=0.02, translation_step_y=0.2,
        speed_tz=0.01, speed_y=1000, alpha=0,
        progress_callback=None):
    """
    2D scan of a grid in tz,y in area [-rotation_range/2, rotation_range/2] x [-search_range/2, search_range/2]
    (relative to the current tz,y), measuring the signal at each point.
    The scan starts from the bottom left corner and moves in alternating rows.
    Args:
        controller: SurugaController instance
        axis_tz: Tz-axis number
        axis_y: Y-axis number
        rotation_range_tz: Initial search range for Tz-axis
        search_range_y: Initial search range for Y-axis
        rotation_step_tz: Initial step size for Tz-axis
        translation_step_y: Initial step size for Y-axis
        speed_tz: Movement speed for Tz-axis
        speed_y: Movement speed for Y-axis
        progress_callback: Optional callback function for progress updates
        iterations: Number of iterations to perform

    Parameters:
        alpha : weight of difference of the signals
    Returns:
        None at testing
        or
        (best_config, best_signal, data_list)
        where data_list is a numpy array with shape (n, 4) containing [x, y, signal1, signal2] for each point.
    """
    best_signal = -float('inf')
    best_config = None

    # alpha = 0.5 # Empirical
    # Use the current position as the center reference point
    current_tz = controller.AxisComponents[axis_tz].GetActualPosition()
    current_y = controller.AxisComponents[axis_y].GetActualPosition()

    # Compute the number of steps in each direction (half range in each direction)
    num_steps_tz = int(rotation_range_tz / rotation_step_tz) + 1
    num_steps_y = int(search_range_y / translation_step_y) + 1
    total_points = num_steps_tz * num_steps_y

    # Preallocate a static matrix to store the scan data.
    # Each row holds [target_tz, target_y, signal1, signal2]
    data_list = np.empty((total_points, 4), dtype=float)

    count = 0
    half_steps_tz = num_steps_tz // 2
    half_steps_y = num_steps_y // 2

    # Calculate the bottom left corner position
    start_tz = current_tz - half_steps_tz * rotation_step_tz
    start_y = current_y - half_steps_y * translation_step_y
    
    controller.AxisComponents[axis_y].SetMaxSpeed(speed_y)
    controller.AxisComponents[axis_tz].SetMaxSpeed(speed_tz)

    # Move to bottom left corner first
    controller.AxisComponents[axis_y].MoveAbsolute(start_y)
    controller.AxisComponents[axis_tz].MoveAbsolute(start_tz)

    while controller.AxisComponents[axis_y].IsMoving() or controller.AxisComponents[axis_tz].IsMoving():
        if not controller.AxisComponents[axis_tz].IsServoOn() or not controller.AxisComponents[axis_y].IsServoOn():
            return None #best_config, best_signal, data_list[:count, :] # Return what we have if servos stop from the software
        time.sleep(0.07)
    time.sleep(0.5)  # wait 0.5 s while the ranges in powermeters change and the voltage stabilizes
    # Start scanning from bottom left, alternating row direction
    for row in range(num_steps_y):
        # For snake pattern: even rows left->right, odd rows right->left
        if row % 2 == 0:
            tz_indices = range(num_steps_tz)
        else:
            tz_indices = range(num_steps_tz - 1, -1, -1)

        for col in tz_indices:
            # Calculate target position
            target_tz = start_tz + col * rotation_step_tz
            target_y = start_y + row * translation_step_y

            # Move to the target position
            controller.AxisComponents[axis_tz].MoveAbsolute(target_tz)
            controller.AxisComponents[axis_y].MoveAbsolute(target_y)

            while controller.AxisComponents[axis_tz].IsMoving() or controller.AxisComponents[axis_y].IsMoving():
                if controller.stop_requested:
                    return best_config, best_signal, data_list[:count, :]
                time.sleep(0.01)

            # Measure both signals
            signal1 = controller.Alignment.GetVoltage(1)
            signal2 = controller.Alignment.GetVoltage(2)
            current_tz = controller.AxisComponents[axis_tz].GetActualPosition()
            current_y = controller.AxisComponents[axis_y].GetActualPosition()
            # current_pos = controller.axis2d.GetActualPosition()
            signal_sum = signal1 + signal2
            metric = signal_sum - alpha*abs(signal1-signal2)
            config = {'tz': current_tz, 'y': current_y}
            data_list[count, :] = [current_tz, current_y, signal1, signal2]
            count += 1

            # Update the best signal if sum is higher
            if metric > best_signal:
                best_signal = metric
                best_config = config

            # Call the progress callback if provided
            if progress_callback:
                progress_callback(config, signal1, signal2)

    return best_config, best_signal, data_list

def scanTx_Y(controller, axis_tx=10, axis_y=8, 
        rotation_range_tx=1, search_range_y=10,
        rotation_step_tx=0.02, translation_step_y=0.2,
        speed_tx=0.01, speed_y=1000, alpha=0,
        progress_callback=None):
    """
    2D scan of a grid in tx,y in area [-rotation_range/2, rotation_range/2] x [-search_range/2, search_range/2]
    (relative to the current tx,y), measuring the signal at each point.
    The scan starts from the bottom left corner and moves in alternating rows.
    Args:
        controller: SurugaController instance
        axis_tx: Tx-axis number
        axis_y: Y-axis number
        rotation_range_tx: Initial search range for Tx-axis
        search_range_y: Initial search range for Y-axis
        rotation_step_tx: Initial step size for Tx-axis
        translation_step_y: Initial step size for Y-axis
        speed_tx: Movement speed for Tx-axis
        speed_y: Movement speed for Y-axis
        progress_callback: Optional callback function for progress updates
        iterations: Number of iterations to perform
    Parameters:
        alpha : weight of difference of the signals
    Returns:
        (best_config, best_signal, data_list)
        where data_list is a numpy array with shape (n, 4) containing [tx, y, signal1, signal2] for each point.
    """
    best_signal = -float('inf')
    best_config = None

    # Use the current position as the center reference point
    current_tx = controller.AxisComponents[axis_tx].GetActualPosition()
    current_y = controller.AxisComponents[axis_y].GetActualPosition()

    # Compute the number of steps in each direction (half range in each direction)
    num_steps_tx = int(rotation_range_tx / rotation_step_tx) + 1
    num_steps_y = int(search_range_y / translation_step_y) + 1
    total_points = num_steps_tx * num_steps_y

    # Preallocate a static matrix to store the scan data.
    # Each row holds [target_tx, target_y, signal1, signal2]
    data_list = np.empty((total_points, 4), dtype=float)

    count = 0
    half_steps_tx = num_steps_tx // 2
    half_steps_y = num_steps_y // 2

    # Calculate the bottom left corner position
    start_tx = current_tx - half_steps_tx * rotation_step_tx
    start_y = current_y - half_steps_y * translation_step_y
    
    controller.AxisComponents[axis_y].SetMaxSpeed(speed_y)
    controller.AxisComponents[axis_tx].SetMaxSpeed(speed_tx)

    # Move to bottom left corner first
    controller.AxisComponents[axis_y].MoveAbsolute(start_y)
    controller.AxisComponents[axis_tx].MoveAbsolute(start_tx)

    while controller.AxisComponents[axis_y].IsMoving() or controller.AxisComponents[axis_tx].IsMoving():
        if not controller.AxisComponents[axis_tx].IsServoOn() or not controller.AxisComponents[axis_y].IsServoOn():
            return None
        time.sleep(0.07)
    time.sleep(0.5)
    # Start scanning from bottom left, alternating row direction
    for row in range(num_steps_y):
        # For snake pattern: even rows left->right, odd rows right->left
        if row % 2 == 0:
            tx_indices = range(num_steps_tx)
        else:
            tx_indices = range(num_steps_tx - 1, -1, -1)

        for col in tx_indices:
            # Calculate target position
            target_tx = start_tx + col * rotation_step_tx
            target_y = start_y + row * translation_step_y

            # Move to the target position
            controller.AxisComponents[axis_tx].MoveAbsolute(target_tx)
            controller.AxisComponents[axis_y].MoveAbsolute(target_y)

            while controller.AxisComponents[axis_tx].IsMoving() or controller.AxisComponents[axis_y].IsMoving():
                if controller.stop_requested:
                    return best_config, best_signal, data_list[:count, :]
                time.sleep(0.01)

            # Measure both signals
            signal1 = controller.Alignment.GetVoltage(1)
            signal2 = controller.Alignment.GetVoltage(2)
            current_tx = controller.AxisComponents[axis_tx].GetActualPosition()
            current_y = controller.AxisComponents[axis_y].GetActualPosition()
            signal_sum = signal1 + signal2
            metric = signal_sum - alpha*abs(signal1-signal2)
            config = {'tx': current_tx, 'y': current_y}
            data_list[count, :] = [current_tx, current_y, signal1, signal2]
            count += 1

            # Update the best signal if sum is higher
            if metric > best_signal:
                best_signal = metric
                best_config = config

            # Call the progress callback if provided
            if progress_callback:
                progress_callback(config, signal1, signal2)

    return best_config, best_signal, data_list

def hill_climb_2channel(controller, axis_x=7, axis_y=8,
                       step_size=0.1,
                       search_range=20,
                       speed=1000,
                       alpha=0,
                       max_iterations=3, progress_callback=None,
                       steps_per_direction=3):
    """
    Fine alignment using hill climbing algorithm for 2-channel alignment.
    Uses smaller steps than the coarse search for precise positioning.
    Only moves in X and Y directions, preserving RZ position.
    
    Args:
        controller: SurugaController instance
        axis_x: X-axis component index
        axis_y: Y-axis component index
        step_size: Step size for X-axis movement
        search_range: Search range for X-axis movement
        speed: Movement speed
        alpha: Weight of difference of the signals
        max_iterations: Maximum number of iterations
        progress_callback: Callback function for progress updates
        steps_per_direction: Number of steps to take in each direction (default: 3)
        
    Returns:
        (best_config, best_signal, data_list)
        where data_list is a numpy array with shape (n, 4) containing [x, y, signal1, signal2] for each point.
    """
    best_signal = -float('inf')
    best_config = None

    # Get initial position
    center_pos = controller.axis2d.GetActualPosition()
    current_x = center_pos.X
    current_y = center_pos.Y

    # Get initial signals
    signal1 = controller.Alignment.GetVoltage(1)
    signal2 = controller.Alignment.GetVoltage(2)
    signal_sum = signal1 + signal2
    metric = signal_sum - alpha*abs(signal1-signal2)
    
    # Initialize best values with current position
    best_signal = metric
    best_config = {'x': current_x, 'y': current_y}
    
    # Estimate the maximum number of data points we might collect
    data_list = []
    # Store initial position and signals
    data_list.append([current_x, current_y, signal1, signal2])
    
    # Define movement directions for hill climbing (only X and Y)
    directions = [
        {'x': 1, 'y': 0},
        {'x': -1, 'y': 0},
        {'x': 0, 'y': 1},
        {'x': 0, 'y': -1}
    ]
    
    iteration = 0
    while iteration < max_iterations:
        improved = False
        
        # Try each direction
        for direction in directions:
            # Take multiple steps in this direction
            for step_multiplier in range(1, steps_per_direction + 1):
                # Calculate new position with scaled step size
                new_x = current_x + direction['x'] * step_size * step_multiplier
                new_y = current_y + direction['y'] * step_size * step_multiplier
                
                # Check if we're at boundaries
                if abs(new_x - center_pos.X) > search_range/2 or \
                   abs(new_y - center_pos.Y) > search_range/2:
                    break  # Stop exploring in this direction
                
                # Move to new position
                controller.axis2d.MoveAbsolute(new_x, new_y)
                while controller.axis2d.IsMoving():
                    if not controller.AxisComponents[axis_x].IsServoOn() or not controller.AxisComponents[axis_y].IsServoOn():
                        # Return the actual number of data points we've collected
                        return best_config, best_signal, np.array(data_list)
                    time.sleep(0.1)
                
                # Get signals at new position
                signal1 = controller.Alignment.GetVoltage(1)
                signal2 = controller.Alignment.GetVoltage(2)
                current_pos = controller.axis2d.GetActualPosition()
                signal_sum = signal1 + signal2
                metric = signal_sum - alpha*abs(signal1-signal2)
                config = {'x': current_pos.X, 'y': current_pos.Y}
                data_list.append([current_pos.X, current_pos.Y, signal1, signal2])
            
                if metric > best_signal:
                    best_signal = metric
                    best_config = config
                    current_x, current_y = current_pos.X, current_pos.Y
                    improved = True
                    break  # Found an improvement, move to the next direction
                print(f"[DEBUG] Iteration {iteration}, Direction {direction}, Step {step_multiplier}, Metric: {metric:.3f}, Best Signal: {best_signal:.3f}")
                # Update progress
                if progress_callback:
                    if not progress_callback(config, signal1, signal2):
                        # Return the actual number of data points we've collected
                        return best_config, best_signal, np.array(data_list)
            
            if improved:
                controller.axis2d.MoveAbsolute(best_config['x'], best_config['y'])
                while controller.axis2d.IsMoving():
                    time.sleep(0.01)
                break  # If we found an improvement, break out of the direction loop
        # iteration += 1
        
        # If no improvement in any direction, we've reached a local maximum
        if not improved:
            iteration += 1
            
    
    # Move to the best position found
    controller.axis2d.MoveAbsolute(best_config['x'], best_config['y'])
    while controller.axis2d.IsMoving():
        time.sleep(0.01)
    
    # Return the actual number of data points we've collected
    return best_config, best_signal, np.array(data_list)

def align_angle_axis(controller, axis_component, search_range, step, signal_channel=1, progress_callback=None):
    """
    Scan the specified axis (angle) within a given range and step, measure the signal, and set axis to maximize the signal.
    Args:
        controller: SurugaController instance
        axis_component: Axis number (e.g., 10 for Tx2, 12 for Tz2)
        search_range: Range to scan (centered at current position)
        step: Step size
        signal_channel: Power meter channel (default 1)
        progress_callback: Optional callback for progress reporting
    Returns:
        best_angle: Angle value where signal is maximized
        best_signal: Maximum measured signal
        data_list: Numpy array with shape (n, 2) containing [angle, signal] for each point
    """
    axis = controller.AxisComponents[axis_component]
    center_angle = axis.GetActualPosition()
    print(f"[DEBUG] Starting angle alignment scan on axis {axis_component} (center: {center_angle:.5f}, range: {search_range}, step: {step})")
    num_steps = int(np.round((search_range * 2) / step)) + 1
    # Store: angle, signal1, signal2
    data_list = np.zeros((num_steps, 3), dtype=float)
    best_signal1 = -np.inf
    best_signal2 = -np.inf
    best_angle1 = None
    best_angle2 = None
    import datetime
    now = datetime.datetime.now()
    axis.MoveAbsolute(center_angle - search_range)
    if axis.IsMoving():
        print(f"[DEBUG] Moving")
    while axis.IsMoving():
        time.sleep(0.01)
        
    print(f"[DEBUG] Wait before scan: {now.time()}")
    time.sleep(0.5)
    print(f"[DEBUG] Start running {now.time()}") 

    for idx, i in enumerate(range(num_steps)):
        target_angle = center_angle - search_range + i * step
        axis.MoveAbsolute(target_angle)
        while axis.IsMoving():
            time.sleep(0.01)
        # Read signals
        signal1 = controller.Alignment.GetVoltage(1)
        signal2 = controller.Alignment.GetVoltage(2)
        data_list[idx, 0] = target_angle
        data_list[idx, 1] = signal1
        data_list[idx, 2] = signal2
        if signal1 > best_signal1:
            best_signal1 = signal1
            best_angle1 = target_angle
        if signal2 > best_signal2:
            best_signal2 = signal2
            best_angle2 = target_angle
        if progress_callback:
            progress_callback(target_angle, (signal1, signal2))
    time.sleep(0.3)
    print(f"[DEBUG] Scan complete. Max CH1 at {best_angle1:.5f} ({best_signal1:.5f}), Max CH2 at {best_angle2:.5f} ({best_signal2:.5f})")
    # Move to the middle of the two max angles
    if best_angle1 is not None and best_angle2 is not None:
        optimal_angle = (best_angle1 + best_angle2) / 2
        print(f"[DEBUG] Moving to optimal angle (midpoint): {optimal_angle:.5f}")
    elif best_angle1 is not None:
        optimal_angle = best_angle1
        print(f"[DEBUG] Moving to optimal angle (CH1 max): {optimal_angle:.5f}")
    elif best_angle2 is not None:
        optimal_angle = best_angle2
        print(f"[DEBUG] Moving to optimal angle (CH2 max): {optimal_angle:.5f}")
    else:
        optimal_angle = center_angle
        print(f"[DEBUG] No max found, returning to center angle: {optimal_angle:.5f}")
    # Move to the optimal angle
    axis.MoveAbsolute(optimal_angle)
    while axis.IsMoving():
        time.sleep(0.01)
    print(f"[DEBUG] Alignment complete. Final angle: {optimal_angle:.5f}")
    return optimal_angle, (best_signal1, best_signal2), data_list


def align_Tz2(controller, search_range, step, signal_channel=1, progress_callback=None):
    """
    Align Tz2 (axis 12) by scanning in the given range and step, maximizing the signal.
    """
    return align_angle_axis(controller, 12, search_range, step, signal_channel, progress_callback)


def align_Tx2(controller, search_range, step, signal_channel=1, progress_callback=None):
    """
    Align Tx2 (axis 10) by scanning in the given range and step, maximizing the signal.
    """
    return align_angle_axis(controller, 10, search_range, step, signal_channel, progress_callback)

def move_axis(controller, axis_number, distance, speed=0.05):
    axis = controller.AxisComponents[axis_number]
    axis.SetMaxSpeed(speed)
    axis.MoveRelative(distance)
    while axis.IsMoving():
        time.sleep(0.01)
    return axis.GetActualPosition()
