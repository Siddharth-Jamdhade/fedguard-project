/*
  FedGuard Node Firmware — streams MPU6050 accelerometer readings over Serial as CSV.
  Use a Python serial-logger (see log_serial.py) to capture normal/abnormal runs to CSV.

  Wiring: VCC->3.3V, GND->GND, SCL->GPIO22, SDA->GPIO21

  Libraries needed: Adafruit MPU6050, Adafruit Unified Sensor, Wire (built-in)
*/

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  if (!mpu.begin()) {
    Serial.println("MPU6050 not found — check wiring!");
    while (1) delay(10);
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  Serial.println("t_ms,ax,ay,az"); // CSV header
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  Serial.print(millis());
  Serial.print(",");
  Serial.print(a.acceleration.x, 4);
  Serial.print(",");
  Serial.print(a.acceleration.y, 4);
  Serial.print(",");
  Serial.println(a.acceleration.z, 4);

  delay(20); // ~50 Hz sampling
}
