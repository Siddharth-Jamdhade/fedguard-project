#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;
const int POT_PIN = 34;       // potentiometer wiper -> ESP32 ADC pin
const float MAX_SIMULATED_WATTS = 2500.0;  // turn the knob: 0W to 2500W load

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22); // SDA, SCL
  if (!mpu.begin()) {
    Serial.println("MPU6050 not found");
    while (1) delay(10);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  analogReadResolution(12); // 0-4095
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  int adc_raw = analogRead(POT_PIN);
  float power_w = (adc_raw / 4095.0) * MAX_SIMULATED_WATTS;

  // CSV line: rx,ry,rz,adc_raw,power_w
  Serial.print(a.acceleration.x); Serial.print(",");
  Serial.print(a.acceleration.y); Serial.print(",");
  Serial.print(a.acceleration.z); Serial.print(",");
  Serial.print(adc_raw); Serial.print(",");
  Serial.println(power_w, 1);

  delay(500);
}
