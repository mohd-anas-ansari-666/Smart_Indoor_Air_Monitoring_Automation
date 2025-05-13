/*
 * Smart Air Quality Monitoring System
 * ESP32 Firmware
 * 
 * This code implements a smart indoor air quality monitoring system using:
 * - DHT11 for temperature and humidity
 * - MQ135 for gas/air quality
 * - 3 relays for fan, purifier, and alarm control
 * - MQTT communication with a local broker
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <DHT.h>

// WiFi credentials
const char* ssid = "JARVIS";
const char* password = "jarvis$100";

// MQTT Broker settings
const char* mqtt_server = "192.168.43.4"; // Change to your MQTT broker IP
const int mqtt_port = 1883;
// const char* mqtt_username = "mqttuser";    // Optional
// const char* mqtt_password = "mqttpass";    // Optional

// MQTT Topics
const char* topic_temp = "sensor/data/temperature";
const char* topic_humidity = "sensor/data/humidity";
const char* topic_air_quality = "sensor/data/air_quality";

const char* topic_relay_fan = "device/control/relay1";
const char* topic_relay_purifier = "device/control/relay2";
const char* topic_alarm = "device/control/alarm";

// Pin definitions
#define DHTPIN 4        // Digital pin connected to the DHT sensor
#define DHTTYPE DHT11   // DHT 11
#define MQ135_PIN 34    // Analog pin for MQ135 gas sensor

#define RELAY_FAN 16     // Fan relay
#define RELAY_PURIFIER 17  // Purifier relay
#define RELAY_ALARM 18     // Alarm relay

// Variables for sensor readings
float temperature = 0.0;
float humidity = 0.0;
int airQuality = 0;

// Timers
unsigned long lastSensorReadTime = 0;
const long sensorInterval = 10000; // Read sensors every 10 seconds

unsigned long lastReconnectAttempt = 0;
const long reconnectInterval = 5000; // Try to reconnect every 5 seconds

// WiFi and MQTT client instances
WiFiClient espClient;
PubSubClient client(espClient);
DHT dht(DHTPIN, DHTTYPE);

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  
  // Initialize sensors
  dht.begin();
  
  // Initialize relay pins as outputs
  pinMode(RELAY_FAN, OUTPUT);
  pinMode(RELAY_PURIFIER, OUTPUT);
  pinMode(RELAY_ALARM, OUTPUT);
  
  // Set initial relay states (LOW = off)
  digitalWrite(RELAY_FAN, LOW);
  digitalWrite(RELAY_PURIFIER, LOW);
  digitalWrite(RELAY_ALARM, LOW);
  
  // Setup WiFi
  setupWiFi();
  
  // Setup MQTT
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

void loop() {
  // Check WiFi connection
  if (WiFi.status() != WL_CONNECTED) {
    setupWiFi();
  }
  
  // Check MQTT connection
  if (!client.connected()) {
    reconnectMQTT();
  }
  
  // MQTT client loop
  client.loop();
  
  // Read sensors and publish data at specified intervals
  unsigned long currentMillis = millis();
  if (currentMillis - lastSensorReadTime >= sensorInterval) {
    lastSensorReadTime = currentMillis;
    
    // Read sensors
    readSensors();
    
    // Publish sensor data
    publishSensorData();
  }
}

void setupWiFi() {
  Serial.println("Connecting to WiFi...");
  
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nFailed to connect to WiFi");
  }
}

void reconnectMQTT() {
  unsigned long currentMillis = millis();
  
  // Attempt to reconnect only at specified intervals
  if (currentMillis - lastReconnectAttempt >= reconnectInterval) {
    lastReconnectAttempt = currentMillis;
    
    Serial.print("Attempting MQTT connection...");
    
    // Create a random client ID
    String clientId = "ESP32Client-";
    clientId += String(random(0xffff), HEX);
    
    // Attempt to connect
    if (client.connect(clientId.c_str())) {
      Serial.println("connected");
      
      // Subscribe to control topics
      client.subscribe(topic_relay_fan);
      client.subscribe(topic_relay_purifier);
      client.subscribe(topic_alarm);
      
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
    }
  }
}

void callback(char* topic, byte* payload, unsigned int length) {
  // Convert payload to string
  String message;
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("] ");
  Serial.println(message);
  
  // Control relays based on topics
  if (String(topic) == topic_relay_fan) {
    digitalWrite(RELAY_FAN, message == "ON" ? HIGH : LOW);
    Serial.println(message == "ON" ? "Fan turned ON" : "Fan turned OFF");
  } 
  else if (String(topic) == topic_relay_purifier) {
    digitalWrite(RELAY_PURIFIER, message == "ON" ? HIGH : LOW);
    Serial.println(message == "ON" ? "Purifier turned ON" : "Purifier turned OFF");
  } 
  else if (String(topic) == topic_alarm) {
    digitalWrite(RELAY_ALARM, message == "ON" ? HIGH : LOW);
    Serial.println(message == "ON" ? "Alarm turned ON" : "Alarm turned OFF");
  }
}

void readSensors() {
  // Read DHT11 sensor
  humidity = dht.readHumidity();
  temperature = dht.readTemperature();
  
  // Check if DHT11 reading failed
  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("Failed to read from DHT sensor!");
    return;
  }
  
  // Read MQ135 gas sensor
  airQuality = analogRead(MQ135_PIN);
  
  // Print sensor readings to Serial Monitor
  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.print(" °C, Humidity: ");
  Serial.print(humidity);
  Serial.print(" %, Air Quality: ");
  Serial.println(airQuality);
}

void publishSensorData() {
  // Convert sensor readings to strings
  char tempStr[10];
  char humStr[10];
  char aqStr[10];
  
  // Format to 2 decimal places for temperature and humidity
  dtostrf(temperature, 4, 2, tempStr);
  dtostrf(humidity, 4, 2, humStr);
  
  // Convert air quality value to string
  sprintf(aqStr, "%d", airQuality);
  
  // Publish sensor data to MQTT topics
  client.publish(topic_temp, tempStr);
  client.publish(topic_humidity, humStr);
  client.publish(topic_air_quality, aqStr);
  
  Serial.println("Sensor data published to MQTT");
}