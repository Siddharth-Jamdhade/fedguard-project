#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// =====================================================
// A4988 PINS  (unchanged)
// =====================================================
const int STEP_PIN = 16;
const int DIR_PIN  = 17;

const int STEP_INTERVAL_US = 1200;
unsigned long lastStepTime = 0;

// =====================================================
// MPU6050  (unchanged)
// =====================================================
Adafruit_MPU6050 mpu;

// =====================================================
// POTENTIOMETER
// SIG -> GPIO34
// VCC -> 3.3V
// GND -> GND
// =====================================================
const int POT_PIN = 34;

// =====================================================
// POWER SIMULATION SETTINGS
// =====================================================
const float MIN_POWER_W = 500.0;
const float MAX_POWER_W = 1200.0;

// =====================================================
// SETUP
// =====================================================
void setup() {

  Serial.begin(115200);

  // MPU6050 I2C (SDA=GPIO21, SCL=GPIO22)
  Wire.begin(21, 22);

  // A4988
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN,  OUTPUT);
  digitalWrite(DIR_PIN, HIGH);

  // Potentiometer — input-only on GPIO34, no pinMode needed
  analogReadResolution(12);   // 0–4095

  // Start MPU6050
  if (!mpu.begin()) {
    Serial.println("# MPU6050 not found - check wiring!");
    while (1) delay(10);
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  // -----------------------------------------------
  // CSV header — matches energy_local_model.py
  // -----------------------------------------------
  Serial.println("# FedGuard PS-13 Energy Node — ready");
  Serial.println("t_ms,power_w,ax,ay,az,vibration_rms");
}


// =====================================================
// LOOP
// =====================================================
void loop() {

  // ===================================================
  // STEP MOTOR  (unchanged)
  // ===================================================
  if (micros() - lastStepTime >= STEP_INTERVAL_US) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(5);
    digitalWrite(STEP_PIN, LOW);
    lastStepTime = micros();
  }


  // ===================================================
  // SENSOR READING  ~50 Hz
  // ===================================================
  static unsigned long lastPrint = 0;

  if (millis() - lastPrint >= 20) {

    // -------------------------------------------------
    // MPU6050  (unchanged)
    // -------------------------------------------------
    sensors_event_t acceleration, gyro, temperature;
    mpu.getEvent(&acceleration, &gyro, &temperature);

    float ax = acceleration.acceleration.x;
    float ay = acceleration.acceleration.y;
    float az = acceleration.acceleration.z;

    // vibration_rms — needed by energy_local_model.py
    float vibration_rms = sqrt(ax*ax + ay*ay + az*az);

    // -------------------------------------------------
    // POTENTIOMETER → power_w
    // map() is integer-only, use float math instead
    // 0 → 500 W  |  4095 → 1200 W
    // -------------------------------------------------
    int   potRaw    = analogRead(POT_PIN);
    float power_w   = MIN_POWER_W
                    + (float(potRaw) / 4095.0)
                    * (MAX_POWER_W - MIN_POWER_W);

    // -------------------------------------------------
    // CSV output — 6 columns expected by the pipeline:
    // t_ms, power_w, ax, ay, az, vibration_rms
    // -------------------------------------------------
    Serial.print(millis());        Serial.print(",");
    Serial.print(power_w,   2);    Serial.print(",");
    Serial.print(ax,        4);    Serial.print(",");
    Serial.print(ay,        4);    Serial.print(",");
    Serial.print(az,        4);    Serial.print(",");
    Serial.println(vibration_rms,  4);

    lastPrint = millis();
  }
}
