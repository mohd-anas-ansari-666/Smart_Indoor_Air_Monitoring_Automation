
"""
Smart Air Quality Monitoring System - Backend Server
Includes:
- FastAPI REST API
- MQTT client for sensor data and device control
- AI modules for mold risk assessment and anomaly detection
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import numpy as np
import paho.mqtt.client as mqtt
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import threading
import queue

# Neural network implementation with numpy
class SimpleANN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)  # hidden layer activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self._sigmoid(self.z2)  # output layer activation
        return self.a2
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # In a real system, you would train this model with data
    # For this example, we'll use pre-defined weights

# Load pre-trained weights for mold model
def load_mold_model():
    model = SimpleANN(2, 4, 1)  # 2 inputs (temp, humidity), 4 hidden neurons, 1 output
    # Pre-defined weights based on domain knowledge (simplified)
    model.W1 = np.array([
        [0.2, 0.15, -0.1, 0.05],
        [0.3, 0.25, 0.2, 0.15]
    ])
    model.W2 = np.array([
        [0.25],
        [0.3],
        [0.2],
        [0.25]
    ])
    return model

# Create and initialize the models
mold_model = load_mold_model()

# Initialize FastAPI app
app = FastAPI(title="Smart Air Quality API", description="API for air quality monitoring system")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MQTT configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_USERNAME = "mqttuser"  # Optional
MQTT_PASSWORD = "mqttpass"  # Optional

# MQTT topics
TOPIC_TEMPERATURE = "sensor/data/temperature"
TOPIC_HUMIDITY = "sensor/data/humidity"
TOPIC_AIR_QUALITY = "sensor/data/air_quality"

TOPIC_RELAY_FAN = "device/control/relay1"
TOPIC_RELAY_PURIFIER = "device/control/relay2"
TOPIC_ALARM = "device/control/alarm"

# Global state storage
sensor_data = {
    "temperature": 0.0,
    "humidity": 0.0,
    "air_quality": 0,
    "aqi_category": "Good",
    "mold_risk": 0.0,
    "last_updated": datetime.now().isoformat()
}

device_state = {
    "fan": False,
    "purifier": False,
    "alarm": False
}

device_usage = {
    "fan": 0,  # seconds
    "purifier": 0,  # seconds
    "alarm": 0  # seconds
}

# Store device state change timestamps
device_state_changes = {
    "fan": datetime.now(),
    "purifier": datetime.now(),
    "alarm": datetime.now()
}

# History storage
history_data = []
MAX_HISTORY = 1000  # Maximum number of history points to keep

# Message queue for thread communication
message_queue = queue.Queue()

# Class for device control requests
class DeviceControlRequest(BaseModel):
    device: str  # "fan", "purifier", or "alarm"
    state: bool  # true = on, false = off
    duration: Optional[int] = None  # Optional duration in seconds

# MQTT client callbacks
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    # Subscribe to sensor topics
    client.subscribe(TOPIC_TEMPERATURE)
    client.subscribe(TOPIC_HUMIDITY)
    client.subscribe(TOPIC_AIR_QUALITY)

def on_message(client, userdata, msg):
    # Process received message from sensors
    topic = msg.topic
    payload = msg.payload.decode()
    print(f"Received message on topic {topic}: {payload}")
    
    # Put the message in the queue for processing by the main thread
    message_queue.put((topic, payload))

# Function to map AQI value to category
def get_aqi_category(aqi_value):
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200: 
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# Process sensor data and make decisions
def process_sensor_data():
    global sensor_data, device_state, history_data
    
    # Extract current sensor readings
    temperature = sensor_data["temperature"]
    humidity = sensor_data["humidity"]
    gas_reading = sensor_data["air_quality"]
    
    # Apply processing logic
    
    # 1. Direct AQI from sensor reading
    if gas_reading > 0:
        # Use gas_reading directly as AQI value
        aqi_value = gas_reading
        
        # Map AQI value to category
        aqi_category = get_aqi_category(aqi_value)
        
        sensor_data["aqi_category"] = aqi_category
        
        # 2. Mold risk prediction
        if temperature > 0 and humidity > 0:
            # Normalize inputs
            temp_norm = temperature / 50.0  # Assuming max temp is 50°C
            humidity_norm = humidity / 100.0
            
            input_data = np.array([[temp_norm, humidity_norm]])
            mold_risk = float(mold_model.forward(input_data)[0][0])
            sensor_data["mold_risk"] = round(mold_risk * 100, 2)  # Convert to percentage
        
        # Save to history
        timestamp = datetime.now().isoformat()
        sensor_data["last_updated"] = timestamp
        
        history_entry = {
            "timestamp": timestamp,
            "temperature": temperature,
            "humidity": humidity,
            "air_quality": gas_reading,
            "aqi_category": aqi_category,
            "mold_risk": sensor_data["mold_risk"]
        }
        
        history_data.append(history_entry)
        
        # Trim history if it gets too long
        if len(history_data) > MAX_HISTORY:
            history_data = history_data[-MAX_HISTORY:]
        
        # 3. Make control decisions
        make_control_decisions(mqtt_client)

def make_control_decisions(client):
    global sensor_data, device_state
    
    # Get current data
    temperature = sensor_data["temperature"]
    humidity = sensor_data["humidity"]
    gas_reading = sensor_data["air_quality"]
    aqi_category = sensor_data["aqi_category"]
    mold_risk = sensor_data["mold_risk"]
    
    # Decision rules for fan
    turn_on_fan = False
    if temperature > 27:  # Turn on fan if temperature is high
        turn_on_fan = True
    if aqi_category in ["Unhealthy", "Very Unhealthy", "Hazardous"]:  # Turn on fan if air quality is bad
        turn_on_fan = True
    
    # Decision rules for purifier
    turn_on_purifier = False
    if aqi_category not in ["Good", "Moderate"]:  # Turn on purifier if air quality is not good
        turn_on_purifier = True
    
    # Decision rules for alarm
    turn_on_alarm = False
    if aqi_category == "Hazardous":  # Trigger alarm for hazardous conditions
        turn_on_alarm = True
    if mold_risk > 75:  # High mold risk
        turn_on_alarm = True
    if gas_reading > 300:  # Very high AQI reading (hazardous)
        turn_on_alarm = True
    
    # Send control commands if state changes
    if turn_on_fan != device_state["fan"]:
        command = "ON" if turn_on_fan else "OFF"
        client.publish(TOPIC_RELAY_FAN, command)
        device_state["fan"] = turn_on_fan
        device_state_changes["fan"] = datetime.now()
        print(f"Fan turned {command}")
    
    if turn_on_purifier != device_state["purifier"]:
        command = "ON" if turn_on_purifier else "OFF"
        client.publish(TOPIC_RELAY_PURIFIER, command)
        device_state["purifier"] = turn_on_purifier
        device_state_changes["purifier"] = datetime.now()
        print(f"Purifier turned {command}")
    
    if turn_on_alarm != device_state["alarm"]:
        command = "ON" if turn_on_alarm else "OFF"
        client.publish(TOPIC_ALARM, command)
        device_state["alarm"] = turn_on_alarm
        device_state_changes["alarm"] = datetime.now()
        print(f"Alarm turned {command}")

# Update device usage statistics
def update_device_usage():
    now = datetime.now()
    
    for device in ["fan", "purifier", "alarm"]:
        if device_state[device]:
            # If device is on, calculate time since last state change
            elapsed = (now - device_state_changes[device]).total_seconds()
            device_usage[device] += elapsed
            device_state_changes[device] = now

# Thread to process messages from the queue
def message_processing_thread():
    global sensor_data
    
    while True:
        try:
            # Get message from queue with timeout
            topic, payload = message_queue.get(timeout=1)
            
            # Update sensor data based on topic
            if topic == TOPIC_TEMPERATURE:
                sensor_data["temperature"] = float(payload)
            elif topic == TOPIC_HUMIDITY:
                sensor_data["humidity"] = float(payload)
            elif topic == TOPIC_AIR_QUALITY:
                sensor_data["air_quality"] = int(payload)
            
            # Process data after receiving all sensor readings
            process_sensor_data()
            
            # Mark task as done
            message_queue.task_done()
            
        except queue.Empty:
            # Queue is empty, continue
            pass
        
        # Update device usage statistics
        update_device_usage()
        
        # Sleep briefly to prevent high CPU usage
        time.sleep(0.1)

# Initialize MQTT client
mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

# Connect to MQTT broker
try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
except Exception as e:
    print(f"Failed to connect to MQTT broker: {e}")

# Start message processing thread
thread = threading.Thread(target=message_processing_thread, daemon=True)
thread.start()

# API endpoints
@app.get("/")
def read_root():
    return {"message": "Smart Air Quality Monitoring API"}

@app.get("/api/status")
def get_status():
    return {
        "sensor_data": sensor_data,
        "device_state": device_state
    }

@app.post("/api/control")
def control_device(request: DeviceControlRequest, background_tasks: BackgroundTasks):
    # Validate device
    if request.device not in ["fan", "purifier", "alarm"]:
        raise HTTPException(status_code=400, detail="Invalid device specified")
    
    # Map device to MQTT topic
    topic_map = {
        "fan": TOPIC_RELAY_FAN,
        "purifier": TOPIC_RELAY_PURIFIER,
        "alarm": TOPIC_ALARM
    }
    
    # Send command
    command = "ON" if request.state else "OFF"
    mqtt_client.publish(topic_map[request.device], command)
    
    # Update state
    device_state[request.device] = request.state
    device_state_changes[request.device] = datetime.now()
    
    # If duration is specified, schedule turning off
    if request.duration and request.state:
        def turn_off_later():
            time.sleep(request.duration)
            mqtt_client.publish(topic_map[request.device], "OFF")
            device_state[request.device] = False
            device_state_changes[request.device] = datetime.now()
        
        background_tasks.add_task(turn_off_later)
    
    return {"success": True, "device": request.device, "state": request.state}

@app.get("/api/history")
def get_history(limit: int = 100):
    # Return the most recent entries up to the limit
    return {"history": history_data[-limit:]}

@app.get("/api/usage")
def get_device_usage():
    return {"usage": device_usage}

# Run the application (when executed directly)
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)