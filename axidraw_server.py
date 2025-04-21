import serial
import time
import signal
import sys
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
import json
import logging
from pathlib import Path

# Configure logging to reduce noise
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

app = FastAPI(title="AxiDraw Control Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the settings file path
SETTINGS_FILE = "axidraw_settings.json"

# Default settings
DEFAULT_SETTINGS = {
    "steps_per_mm": 80,
    "speed": 1000,
    "speed_up": 4000,
    "speed_down": 4000,
    "pen_up_angle": 100,
    "pen_down_angle": 60,
    "pen_up_delay": 150,
    "pen_down_delay": 150,
    "paper_size": "A4",
    "bounds": {
        "width": 297,
        "height": 210
    }
}

# Add this near the top of the file, after the DEFAULT_SETTINGS definition
_settings_cache = None

def load_settings():
    """Load settings from cache or file, using defaults if neither exists"""
    global _settings_cache
    
    if _settings_cache is not None:
        return _settings_cache
        
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                _settings_cache = json.load(f)
                print(f"Loaded settings from {SETTINGS_FILE}")
        else:
            print(f"Settings file not found, using defaults")
            _settings_cache = DEFAULT_SETTINGS.copy()
        return _settings_cache
    except Exception as e:
        print(f"Error loading settings: {e}")
        _settings_cache = DEFAULT_SETTINGS.copy()
        return _settings_cache

def save_settings(settings):
    """Save settings to file and update cache"""
    global _settings_cache
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        _settings_cache = settings.copy()
        print(f"Settings saved to {SETTINGS_FILE}")
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

# Paper size definitions (in mm)
PAPER_SIZES = {
    "LETTER": {"width": 279.4, "height": 215.9},  # 8.5 x 11 inches
    "HALF_LETTER": {"width": 215.9, "height": 139.7},  # 5.5 x 8.5 inches
    "A4": {"width": 297, "height": 210},
    "A5": {"width": 210, "height": 148},
    "A6": {"width": 148, "height": 105},
    "A7": {"width": 105, "height": 74},
    "A8": {"width": 74, "height": 52}
}

# AxiDraw Control Class
class AxiDraw:
    def __init__(self, port=None, baudrate=38400):
        self.port = port or os.getenv('AXIDRAW_PORT', '/dev/tty.usbmodem101')
        self.baudrate = baudrate
        self.serial = None
        self.is_connected = False
        self.current_position = [0, 0]  # Current position in absolute coordinates
        self.pen_up = True
        self.motors_on_state = False  # Track if motors are currently on
        self.debug_mode = True  # Enable debug mode by default
        
        # Machine bounds (physical limits in mm)
        self.machine_bounds = {
            'width': 430,  # AxiDraw V3 width
            'height': 297  # AxiDraw V3 height
        }
        
        # Default paper size (A4 in mm)
        self.paper_bounds = {
            'width': 297,  # A4 width
            'height': 210  # A4 height
        }
        
        # Load settings from file
        settings = load_settings()
        
        # Initialize settings from loaded values
        self.speed = settings.get("speed", 1000)
        self.speed_up = settings.get("speed_up", 3000)
        self.speed_down = settings.get("speed_down", 1000)
        self.acceleration = settings.get("acceleration", 1000)
        self.steps_per_mm = settings.get("steps_per_mm", 100)
        self.pen_up_angle = settings.get("pen_up_angle", 100)
        self.pen_down_angle = settings.get("pen_down_angle", 60)
        self.pen_up_delay = settings.get("pen_up_delay", 150)
        self.pen_down_delay = settings.get("pen_down_delay", 150)
        self.pen_limits = [self.pen_down_angle, self.pen_up_angle]  # Store pen limits

    def connect(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            self.is_connected = True
            print(f"Successfully connected to AxiDraw on port {self.port}")
            self.initialize_machine()
            return True
        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            print("Running in debug mode - commands will be printed to console")
            self.is_connected = False
            return False

    def disconnect(self):
        if self.serial and self.serial.is_open:
            self.pen_up = True
            self.send_command("SP,1")  # Ensure pen is up before disconnecting
            time.sleep(0.5)
            self.motors_off()  # Turn off motors before disconnecting
            self.serial.close()
        self.is_connected = False

    def initialize_machine(self):
        """Initialize the AxiDraw with proper settings"""
        commands = [
            "EM,1,1",  # Enable motors
            "SP,1",    # Set pen state to up
        ]
        
        # Configure pen settings
        self.pen_config(self.pen_down_angle, self.pen_up_angle)
        
        for cmd in commands:
            self.send_command(cmd)
            time.sleep(0.1)  # Small delay between commands

    def send_command(self, command: str):
        """Send a command to the AxiDraw and wait for a response"""
        if not self.is_connected:
            print(f"[DEBUG] Would send command: {command}")
            return True
            
        try:
            # Add more detailed logging for pen commands
            if command.startswith("SP"):
                print(f"Sending pen command: {command}")
                
            # Send the command with a carriage return
            self.serial.write(f"{command}\r\n".encode())
            self.serial.flush()
            
            # Wait for a response
            response = self.serial.readline().decode().strip()
            print(f"Response: '{response}' for command '{command}'")
            
            # Check if the response indicates success
            if response.startswith("OK"):
                return True
            else:
                print(f"Warning: Unexpected response '{response}' for command '{command}'")
                return False
        except Exception as e:
            print(f"Error sending command '{command}': {e}")
            return False

    def motors_on(self):
        """Turn on the motors"""
        self.send_command("EM,1,1")
        self.motors_on_state = True

    def motors_off(self):
        """Turn off the motors"""
        self.send_command("EM,0,0")
        self.motors_on_state = False

    def home_machine(self):
        """Home the machine and reset position to absolute (0,0)"""
        if not self.is_connected:
            print("[DEBUG] Would home machine")
            self.current_position = [0, 0]
            return

        # Ensure pen is up before homing
        if not self.pen_up:
            self.pen_up_command()
            time.sleep(0.5)
            self.pen_up = True

        # Move to absolute position 0,0 with pen up
        print("Moving to home position (0,0)...")
        self.move_to(0, 0, pen_up=True, speed=2000)  # Use slower speed for homing
        
        # Reset position tracking
        self.current_position = [0, 0]
        
        # Turn off motors after homing
        self.motors_off()

    def pen_config(self, down=None, up=None):
        """Configure pen up/down angles"""
        if down is not None:
            self.pen_down_angle = down
            self.pen_limits[0] = down
            self.send_pen_config(5, down)
            
        if up is not None:
            self.pen_up_angle = up
            self.pen_limits[1] = up
            self.send_pen_config(4, up)
            
        # Set servo timing
        self.send_command("SC,10,65535")
        
        print(f"Pen configuration updated: down={self.pen_down_angle}, up={self.pen_up_angle}")
        return True
        
    def send_pen_config(self, id, angle):
        """Send pen configuration command
        Based on the official AxiDraw implementation:
        https://github.com/thi-ng/umbrella/blob/develop/packages/axidraw/src/axidraw.ts
        """
        # Convert angle to servo value (7500 + 175 * angle)
        servo_value = int(7500 + 175 * angle)
        self.send_command(f"SC,{id},{servo_value}")
        print(f"Sent pen config: SC,{id},{servo_value} for angle {angle}")
        
    def pen_up_command(self, delay=None):
        """Raise the pen"""
        if delay is None:
            delay = self.pen_up_delay
            
        print(f"Sending pen up command with delay {delay}ms")
        
        # First ensure pen up angle is configured
        self.send_pen_config(4, self.pen_up_angle)
        
        # Then send the pen up command
        self.send_command(f"SP,1,{delay}")
        self.pen_up = True
        return delay

    def pen_down_command(self, delay=None):
        """Lower the pen"""
        if delay is None:
            delay = self.pen_down_delay
            
        print(f"Sending pen down command with delay {delay}ms")
        
        # First ensure pen down angle is configured
        self.send_pen_config(5, self.pen_down_angle)
        
        # Then send the pen down command
        self.send_command(f"SP,0,{delay}")
        self.pen_up = False
        return delay

    def move_to(self, x, y, pen_up=False, duration=None, speed=None, acceleration=None):
        """Move to absolute coordinates (x, y)"""
        # Store original coordinates
        original_x, original_y = x, y
        
        # First, clamp to machine bounds (physical limits)
        x = max(0, min(x, self.machine_bounds['width']))
        y = max(0, min(y, self.machine_bounds['height']))
        
        # Then, check if movement is within paper bounds
        if (x > self.paper_bounds['width'] or y > self.paper_bounds['height']):
            # If pen is down and trying to move outside paper, lift pen
            if not pen_up and not self.pen_up:
                print(f"Warning: Lifting pen as movement goes outside paper bounds")
                self.pen_up_command()
                pen_up = True
        
        # Log any position adjustments
        if x != original_x or y != original_y:
            if x != original_x and y != original_y:
                print(f"Warning: Movement clamped from ({original_x}, {original_y}) to ({x}, {y}) due to machine bounds")
            elif x != original_x:
                print(f"Warning: X movement clamped from {original_x} to {x} due to machine bounds")
            else:
                print(f"Warning: Y movement clamped from {original_y} to {y} due to machine bounds")
        
        # Turn on motors before movement
        self.motors_on()
        
        # Update speed if provided
        if speed is not None:
            self.speed = speed
        else:
            # Use appropriate speed based on pen state
            speed = self.speed_up if pen_up else self.speed_down

        # Truncate coordinates to 3 decimal places
        x = round(x, 3)
        y = round(y, 3)

        # Calculate relative movement from current position
        rel_x = x - self.current_position[0]
        rel_y = y - self.current_position[1]
        
        # If no actual movement is needed, return early
        if abs(rel_x) < 0.001 and abs(rel_y) < 0.001:
            return
        
        # Truncate relative coordinates to 3 decimal places
        rel_x = round(rel_x, 3)
        rel_y = round(rel_y, 3)

        # Convert relative coordinates to steps
        x_steps = int(rel_x * self.steps_per_mm)
        y_steps = int(rel_y * self.steps_per_mm)

        # Handle pen up/down
        if pen_up != self.pen_up:
            if pen_up:
                self.pen_up_command()
            else:
                # Only allow pen down if within paper bounds
                if x <= self.paper_bounds['width'] and y <= self.paper_bounds['height']:
                    self.pen_down_command()
                else:
                    print(f"Warning: Prevented pen down at ({x}, {y}) as it's outside paper bounds")
                    pen_up = True
            time.sleep(0.5)  # Wait for pen movement

        # Calculate movement duration based on distance and speed
        if duration is None:
            distance = ((rel_x)**2 + (rel_y)**2)**0.5
            # Ensure minimum duration to prevent too fast movements
            duration = max(75, int((distance / (speed / 60)) * 1000))  # Convert to milliseconds

        # Move to position - use relative coordinates
        print(f"Moving to absolute X:{x} Y:{y} (relative X:{rel_x} Y:{rel_y}) with duration:{duration}ms")
        self.send_command(f"XM,{duration},{x_steps},{y_steps}")
        
        # Wait for movement to complete
        time.sleep(duration / 1000)  # Convert back to seconds
        
        # Update current position
        self.current_position = [x, y]
        
    def move_relative(self, dx, dy, pen_up=False, duration=None, speed=None, acceleration=None):
        """Move by relative amount (dx, dy) from current position"""
        # Truncate relative movements to 3 decimal places
        dx = round(dx, 3)
        dy = round(dy, 3)
        
        # Calculate absolute target position
        target_x = self.current_position[0] + dx
        target_y = self.current_position[1] + dy
        
        # Use move_to with the calculated absolute position
        return self.move_to(target_x, target_y, pen_up, duration, speed, acceleration)

    def update_settings(self, steps_per_mm=None, speed=None, speed_up=None, speed_down=None, acceleration=None, pen_up_angle=None, pen_down_angle=None, pen_up_delay=None, pen_down_delay=None):
        """Update machine settings"""
        # Create a dictionary to track changes
        settings_changed = {}
        
        if steps_per_mm is not None:
            self.steps_per_mm = steps_per_mm
            settings_changed["steps_per_mm"] = steps_per_mm
            print(f"Updated steps per mm: {steps_per_mm}")
            
        if speed is not None:
            self.speed = speed
            settings_changed["speed"] = speed
            print(f"Updated speed: {speed} mm/min")
            
        if speed_up is not None:
            self.speed_up = speed_up
            settings_changed["speed_up"] = speed_up
            print(f"Updated pen up speed: {speed_up} mm/min")
            
        if speed_down is not None:
            self.speed_down = speed_down
            settings_changed["speed_down"] = speed_down
            print(f"Updated pen down speed: {speed_down} mm/min")
            
        if acceleration is not None:
            self.acceleration = acceleration
            settings_changed["acceleration"] = acceleration
            print(f"Updated acceleration: {acceleration} mm/s²")
            
        if pen_up_angle is not None:
            self.pen_up_angle = pen_up_angle
            self.pen_limits[1] = pen_up_angle
            self.send_pen_config(4, pen_up_angle)
            settings_changed["pen_up_angle"] = pen_up_angle
            print(f"Updated pen up angle: {pen_up_angle} degrees")
            
        if pen_down_angle is not None:
            self.pen_down_angle = pen_down_angle
            self.pen_limits[0] = pen_down_angle
            self.send_pen_config(5, pen_down_angle)
            settings_changed["pen_down_angle"] = pen_down_angle
            print(f"Updated pen down angle: {pen_down_angle} degrees")
            
        if pen_up_delay is not None:
            self.pen_up_delay = pen_up_delay
            settings_changed["pen_up_delay"] = pen_up_delay
            print(f"Updated pen up delay: {pen_up_delay} ms")
            
        if pen_down_delay is not None:
            self.pen_down_delay = pen_down_delay
            settings_changed["pen_down_delay"] = pen_down_delay
            print(f"Updated pen down delay: {pen_down_delay} ms")
        
        # If any settings were changed, save to file
        if settings_changed:
            # Load current settings
            current_settings = load_settings()
            # Update with new values
            current_settings.update(settings_changed)
            # Save back to file
            save_settings(current_settings)
            
        return True

# Create AxiDraw controller instance
controller = AxiDraw()

# Flag to track if server is shutting down
is_shutting_down = False

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global is_shutting_down
    if not is_shutting_down:
        is_shutting_down = True
        print("Shutting down gracefully...")
        controller.disconnect()
        sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.on_event("startup")
async def startup_event():
    # Load settings into cache on startup
    load_settings()
    if not controller.connect():
        print("Warning: Failed to connect to AxiDraw on startup - running in debug mode")

@app.on_event("shutdown")
async def shutdown_event():
    controller.disconnect()

@app.get("/")
async def root():
    return {"message": "Welcome to AxiDraw Control Server. Visit /static/index.html for the web interface."}

class MovementCommand(BaseModel):
    x: Optional[float] = None
    y: Optional[float] = None
    dx: Optional[float] = None  # Relative X movement
    dy: Optional[float] = None  # Relative Y movement
    points: Optional[List[Dict[str, float]]] = None
    pen_up: bool = False
    duration: Optional[float] = None
    speed: Optional[float] = 1000
    is_relative: bool = False  # Whether to use relative coordinates

class SettingsCommand(BaseModel):
    steps_per_mm: int = 80
    speed: float = 1000
    speed_up: Optional[float] = None
    speed_down: Optional[float] = None
    pen_up_angle: Optional[float] = None
    pen_down_angle: Optional[float] = None
    pen_up_delay: Optional[int] = None
    pen_down_delay: Optional[int] = None
    paper_size: Optional[str] = None

@app.post("/move")
async def move(command: MovementCommand):
    try:
        # Check if movement is within bounds
        current_settings = load_settings()
        bounds = current_settings["bounds"]
        
        if command.points:
            # Check all points in the stroke
            for point in command.points:
                if not (0 <= point['x'] <= bounds['width'] and 0 <= point['y'] <= bounds['height']):
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Movement out of bounds. Point ({point['x']}, {point['y']}) exceeds paper size {bounds['width']}x{bounds['height']}mm"
                    )
        else:
            # Check single point movement
            x = command.x if command.x is not None else controller.current_position[0]
            y = command.y if command.y is not None else controller.current_position[1]
            
            if command.is_relative:
                x = controller.current_position[0] + (command.dx or 0)
                y = controller.current_position[1] + (command.dy or 0)
            
            if not (0 <= x <= bounds['width'] and 0 <= y <= bounds['height']):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Movement out of bounds. Point ({x}, {y}) exceeds paper size {bounds['width']}x{bounds['height']}mm"
                )
        
        # Continue with existing movement logic
        if command.points:
            controller.motors_on()
            for i, point in enumerate(command.points):
                if i == 0:
                    controller.move_to(point['x'], point['y'], pen_up=True, speed=command.speed)
                    controller.move_to(point['x'], point['y'], pen_up=False, speed=command.speed)
                else:
                    controller.move_to(point['x'], point['y'], pen_up=False, speed=command.speed)
            
            last_point = command.points[-1]
            controller.move_to(last_point['x'], last_point['y'], pen_up=True, speed=command.speed)
            controller.motors_off()
            return {"status": "success", "message": "Stroke completed"}
        else:
            if command.is_relative:
                if command.dx is not None or command.dy is not None:
                    controller.move_relative(
                        command.dx or 0, 
                        command.dy or 0, 
                        command.pen_up, 
                        command.duration,
                        command.speed
                    )
                    controller.motors_off()
                    return {"status": "success", "position": {"x": controller.current_position[0], "y": controller.current_position[1]}}
                else:
                    raise HTTPException(status_code=400, detail="Relative movement requires dx and/or dy values")
            else:
                if command.x is not None or command.y is not None:
                    controller.move_to(
                        command.x or controller.current_position[0], 
                        command.y or controller.current_position[1], 
                        command.pen_up, 
                        command.duration,
                        command.speed
                    )
                    controller.motors_off()
                    return {"status": "success", "position": {"x": command.x, "y": command.y}}
                else:
                    raise HTTPException(status_code=400, detail="Absolute movement requires x and/or y values")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/home")
async def home():
    try:
        controller.home_machine()
        return {"status": "success", "message": "Machine homed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/settings")
async def update_settings(settings: SettingsCommand):
    try:
        # Load current settings
        current_settings = load_settings()
        
        # Update settings
        if settings.paper_size:
            if settings.paper_size not in PAPER_SIZES:
                raise HTTPException(status_code=400, detail=f"Invalid paper size. Available sizes: {', '.join(PAPER_SIZES.keys())}")
            current_settings["paper_size"] = settings.paper_size
            current_settings["bounds"] = PAPER_SIZES[settings.paper_size]
        
        controller.update_settings(
            steps_per_mm=settings.steps_per_mm,
            speed=settings.speed,
            speed_up=settings.speed_up,
            speed_down=settings.speed_down,
            pen_up_angle=settings.pen_up_angle,
            pen_down_angle=settings.pen_down_angle,
            pen_up_delay=settings.pen_up_delay,
            pen_down_delay=settings.pen_down_delay
        )
        
        # Save updated settings
        save_settings(current_settings)
        
        return {"status": "success", "settings": current_settings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset_settings")
async def reset_settings():
    """Reset all settings to default values"""
    try:
        # Save default settings to file
        save_settings(DEFAULT_SETTINGS)
        
        # Update controller with default settings
        controller.update_settings(
            steps_per_mm=DEFAULT_SETTINGS["steps_per_mm"],
            speed=DEFAULT_SETTINGS["speed"],
            speed_up=DEFAULT_SETTINGS["speed_up"],
            speed_down=DEFAULT_SETTINGS["speed_down"],
            pen_up_angle=DEFAULT_SETTINGS["pen_up_angle"],
            pen_down_angle=DEFAULT_SETTINGS["pen_down_angle"],
            pen_up_delay=DEFAULT_SETTINGS["pen_up_delay"],
            pen_down_delay=DEFAULT_SETTINGS["pen_down_delay"]
        )
        
        return {"status": "success", "message": "Settings reset to defaults", "settings": DEFAULT_SETTINGS}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/shutdown")
async def shutdown(background_tasks: BackgroundTasks):
    """Gracefully shutdown the server"""
    try:
        # Disconnect from AxiDraw
        controller.disconnect()
        
        # Schedule the server shutdown
        background_tasks.add_task(shutdown_server)
        
        return {"status": "success", "message": "Server shutting down"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def shutdown_server():
    """Background task to shutdown the server"""
    await asyncio.sleep(1)  # Give time for the response to be sent
    
    # Get the running server instance
    for task in asyncio.all_tasks():
        if task.get_name() == "uvicorn.main":
            task.cancel()
    
    # Force exit after a short delay
    await asyncio.sleep(0.5)
    sys.exit(0)

@app.get("/status")
async def get_status():
    # Load current settings to get bounds information
    current_settings = load_settings()
    
    return {
        "connected": controller.is_connected,
        "current_position": controller.current_position,
        "pen_up": controller.pen_up,
        "motors_on": controller.motors_on_state,
        "debug_mode": not controller.is_connected,
        "speed": controller.speed,
        "speed_up": controller.speed_up,
        "speed_down": controller.speed_down,
        "steps_per_mm": controller.steps_per_mm,
        "pen_up_angle": controller.pen_up_angle,
        "pen_down_angle": controller.pen_down_angle,
        "pen_up_delay": controller.pen_up_delay,
        "pen_down_delay": controller.pen_down_delay,
        "paper_size": current_settings["paper_size"],
        "bounds": current_settings["bounds"]
    }

@app.post("/toggle_pen")
async def toggle_pen():
    """Toggle the pen state (up/down) without moving"""
    try:
        print(f"Current pen state: {'up' if controller.pen_up else 'down'}")
        if controller.pen_up:
            print("Toggling pen down")
            controller.pen_down_command()
            print("Pen down command sent")
        else:
            print("Toggling pen up")
            controller.pen_up_command()
            print("Pen up command sent")
        return {"status": "success", "pen_up": controller.pen_up}
    except Exception as e:
        print(f"Error toggling pen: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/toggle_motors")
async def toggle_motors():
    """Toggle motors on and off"""
    try:
        # Check if motors are currently on by sending a status command
        # For simplicity, we'll use a global variable to track motor state
        # In a real implementation, you might want to query the device
        
        # Toggle motors state
        if hasattr(controller, 'motors_on_state'):
            controller.motors_on_state = not controller.motors_on_state
        else:
            controller.motors_on_state = True
            
        if controller.motors_on_state:
            print("Turning motors on")
            controller.motors_on()
        else:
            print("Turning motors off")
            controller.motors_off()
            
        return {"status": "success", "motors_on": controller.motors_on_state}
    except Exception as e:
        print(f"Error toggling motors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/paper_sizes")
async def get_paper_sizes():
    """Return available paper sizes"""
    return PAPER_SIZES

if __name__ == "__main__":
    # Use uvicorn with reload for development
    uvicorn.run(
        "axidraw_server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        reload_dirs=["."]
    ) 