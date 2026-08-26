/*
 * FallGuard AI - ESP32 + MPU6050 Wearable Sensor Firmware
 * Samples 6-Axis IMU (Acc X,Y,Z + Gyro X,Y,Z) at 50 Hz and streams via HTTP/WebSocket
 * 
 * Hardware Wiring:
 * - ESP32 3.3V  -> MPU6050 VCC
 * - ESP32 GND   -> MPU6050 GND
 * - ESP32 GPIO21 -> MPU6050 SDA
 * - ESP32 GPIO22 -> MPU6050 SCL
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <ArduinoJson.h>

// Wi-Fi Credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// FallGuard AI Server Endpoint
const char* serverUrl = "http://192.168.1.100:8000/api/predict";
const char* deviceId = "WEARABLE_ESP32_01";

// MPU6050 I2C Address
const int MPU_ADDR = 0x68;

// Sampling parameters: 50 Hz = 20ms period
const unsigned long SAMPLE_INTERVAL_MS = 20;
unsigned long lastSampleTime = 0;

// Rolling Window Buffer: 128 samples (2.56 seconds)
const int WINDOW_SIZE = 128;
float acc_x[WINDOW_SIZE];
float acc_y[WINDOW_SIZE];
float acc_z[WINDOW_SIZE];
float gyro_x[WINDOW_SIZE];
float gyro_y[WINDOW_SIZE];
float gyro_z[WINDOW_SIZE];
int window_index = 0;

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22); // SDA = GPIO21, SCL = GPIO22

  // Initialize MPU6050 (Wake up from sleep mode)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); // PWR_MGMT_1 register
  Wire.write(0x00); // Set to 0 to wake up
  Wire.endTransmission(true);

  // Configure Accelerometer Scale: ±8g (AFS_SEL = 2 -> 0x10)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C); // ACCEL_CONFIG
  Wire.write(0x10);
  Wire.endTransmission(true);

  // Configure Gyroscope Scale: ±1000 deg/s (FS_SEL = 2 -> 0x10)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B); // GYRO_CONFIG
  Wire.write(0x10);
  Wire.endTransmission(true);

  Serial.println("\n--- FallGuard AI ESP32 Wearable Node ---");
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected! IP Address: " + WiFi.localIP().toString());
}

void loop() {
  unsigned long currentTime = millis();

  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = currentTime;

    // Read 14 bytes from MPU6050 (Acc X,Y,Z, Temp, Gyro X,Y,Z)
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 14, true);

    int16_t raw_ax = (Wire.read() << 8) | Wire.read();
    int16_t raw_ay = (Wire.read() << 8) | Wire.read();
    int16_t raw_az = (Wire.read() << 8) | Wire.read();
    int16_t raw_temp = (Wire.read() << 8) | Wire.read();
    int16_t raw_gx = (Wire.read() << 8) | Wire.read();
    int16_t raw_gy = (Wire.read() << 8) | Wire.read();
    int16_t raw_gz = (Wire.read() << 8) | Wire.read();

    // Scale raw values to physical units:
    // ±8g range -> 4096 LSB/g
    // ±1000 deg/s range -> 32.8 LSB/(deg/s) -> convert to rad/s (* PI / 180.0)
    float ax_g = (float)raw_ax / 4096.0f;
    float ay_g = (float)raw_ay / 4096.0f;
    float az_g = (float)raw_az / 4096.0f;

    float gx_rad = ((float)raw_gx / 32.8f) * (3.14159265f / 180.0f);
    float gy_rad = ((float)raw_gy / 32.8f) * (3.14159265f / 180.0f);
    float gz_rad = ((float)raw_gz / 32.8f) * (3.14159265f / 180.0f);

    acc_x[window_index] = ax_g;
    acc_y[window_index] = ay_g;
    acc_z[window_index] = az_g;
    gyro_x[window_index] = gx_rad;
    gyro_y[window_index] = gy_rad;
    gyro_z[window_index] = gz_rad;

    window_index++;

    // When buffer is full (128 samples / 2.56s), send HTTP POST payload
    if (window_index >= WINDOW_SIZE) {
      sendWindowPayload();
      
      // 50% Sliding overlap: slide window forward by 64 samples
      for (int i = 0; i < 64; i++) {
        acc_x[i] = acc_x[i + 64];
        acc_y[i] = acc_y[i + 64];
        acc_z[i] = acc_z[i + 64];
        gyro_x[i] = gyro_x[i + 64];
        gyro_y[i] = gyro_y[i + 64];
        gyro_z[i] = gyro_z[i + 64];
      }
      window_index = 64;
    }
  }
}

void sendWindowPayload() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi disconnected. Skipping packet.");
    return;
  }

  HTTPClient http;
  http.begin(serverUrl);
  http.addHeader("Content-Type", "application/json");

  StaticJsonDocument<8192> doc;
  doc["device_id"] = deviceId;
  doc["sampling_rate"] = 50.0;

  JsonArray j_ax = doc.createNestedArray("acc_x");
  JsonArray j_ay = doc.createNestedArray("acc_y");
  JsonArray j_az = doc.createNestedArray("acc_z");
  JsonArray j_gx = doc.createNestedArray("gyro_x");
  JsonArray j_gy = doc.createNestedArray("gyro_y");
  JsonArray j_gz = doc.createNestedArray("gyro_z");

  for (int i = 0; i < WINDOW_SIZE; i++) {
    j_ax.add(acc_x[i]);
    j_ay.add(acc_y[i]);
    j_az.add(acc_z[i]);
    j_gx.add(gyro_x[i]);
    j_gy.add(gyro_y[i]);
    j_gz.add(gyro_z[i]);
  }

  String requestBody;
  serializeJson(doc, requestBody);

  int httpResponseCode = http.POST(requestBody);
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.printf("[HTTP %d] Response: %s\n", httpResponseCode, response.c_str());
  } else {
    Serial.printf("Error on HTTP POST: %s\n", http.errorToString(httpResponseCode).c_str());
  }
  http.end();
}
