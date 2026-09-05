/*
  Smart Bio-Crate — Sentinel Crate Firmware (ESP32-S3)
  =====================================================
  Reference implementation of the Deterioration Risk Score (DRS) fusion
  algorithm described in risk_score_simulation.py, written for deployment
  on an ESP32-S3 sentinel node (1 sentinel per ~10-12 crates).

  Hardware assumed:
    - ESP32-S3-WROOM-1 (dual-core, sufficient RAM for this workload)
    - SHT31 temp/RH sensor (I2C)
    - HX711 + load cell (crate weight -> mass-loss %)
    - OV2640 camera (colour sampling only, not full-frame ML — see note)
    - MQ-135 or SGX MiCS VOC sensor (analog)
    - IR break-beam or low-res PIR-style pest/insect activity counter
    - SIM800L / similar GSM module (UART) for store-and-forward sync
    - Micro-servo (ventilation louvre)
    - 6V solar panel + 18650 Li-ion pack + TP4056 charge controller,
      with a 12V-vehicle-power fallback input (per governance doc's
      monsoon power-gap mitigation)

  Design principles carried over from the concept doc:
    - Offline-first: all readings logged to onboard flash (LittleFS);
      only a compact summary is pushed over GSM when signal is available.
    - Explainable model: linear weighted fusion, not a black-box network,
      so field staff and NIPHM auditors can trace exactly why an alert
      fired (needed for the quality-pricing dispute-resolution process).
    - Crop profile (weights + safe ranges) is loaded from a config struct
      that can be updated in the field via a signed OTA/USB config push
      after each NIPHM/NERDC calibration cycle, WITHOUT re-flashing the
      whole firmware.
    - Hysteresis on the ventilation servo to avoid motor wear from
      bumpy-road vibration triggering rapid open/close cycling.
*/

#include <Wire.h>
#include <Adafruit_SHT31.h>
#include "HX711.h"
#include <LittleFS.h>
#include <ArduinoJson.h>

// ---------------------------------------------------------------------
// Pin map
// ---------------------------------------------------------------------
#define PIN_HX711_DT      4
#define PIN_HX711_SCK     5
#define PIN_VOC_ANALOG     6
#define PIN_PEST_SENSOR    7
#define PIN_SERVO_LOUVRE   8
#define PIN_GSM_RX        16
#define PIN_GSM_TX        17

Adafruit_SHT31 sht31 = Adafruit_SHT31();
HX711 scale;

// ---------------------------------------------------------------------
// Crop profile — loaded from /config/crop_profile.json at boot.
// This example shows the papaya (Tier 2) defaults from the NIPHM pilot.
// ---------------------------------------------------------------------
struct CropProfile {
  char name[16];
  float tempSafeMin, tempSafeMax, tempCritical;
  float rhSafeMin, rhSafeMax;
  float massLossWarnPct, massLossCriticalPct;
  float vocWarnDelta, vocCriticalDelta;
  int   pestWarnPerHr, pestCriticalPerHr;
  float wTemp, wRh, wMass, wVoc, wPest;   // must sum to 1.0
};

CropProfile crop = {
  "papaya",
  7.0, 13.0, 20.0,
  85.0, 90.0,
  4.0, 8.0,
  80.0, 200.0,
  2, 6,
  0.30, 0.15, 0.20, 0.25, 0.10
};

const float ALERT_THRESHOLD = 55.0;
const float WATCH_THRESHOLD = 35.0;

float baselineMassG = 20000.0;   // set at loading (20 kg nominal)
float vocBaselinePpm = 50.0;     // captured during 60s calibration at load time
unsigned long tripStartMillis = 0;

// Sampling cadence: 5 min normal, drops to 1 min once WATCH threshold hit
unsigned long sampleIntervalMs = 5UL * 60UL * 1000UL;
const unsigned long WATCH_INTERVAL_MS = 60UL * 1000UL;
unsigned long lastSampleMillis = 0;

// Servo hysteresis state
bool louvreOpen = false;
const float LOUVRE_OPEN_TEMP = 14.0;   // open above this
const float LOUVRE_CLOSE_TEMP = 11.5;  // close below this (hysteresis gap)

// ---------------------------------------------------------------------
// Stress-score helpers (mirrors risk_score_simulation.py exactly, so
// firmware behaviour can be validated offline before flashing)
// ---------------------------------------------------------------------
float clip01(float v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }

float stressScore(float value, float safeMin, float safeMax, float critical) {
  if (value >= safeMin && value <= safeMax) return 0.0;
  if (value > safeMax) {
    float span = max(critical - safeMax, 1e-6f);
    return clip01((value - safeMax) / span);
  }
  float span = max(safeMin * 0.5f, 1e-6f);
  return clip01((safeMin - value) / span);
}

float massLossScore(float pct, float warn, float critical) {
  if (pct <= warn) return 0.0;
  return clip01((pct - warn) / (critical - warn));
}

float vocScore(float deltaPpm, float warn, float critical) {
  if (deltaPpm <= warn) return 0.0;
  return clip01((deltaPpm - warn) / (critical - warn));
}

float pestScore(float eventsPerHr, float warn, float critical) {
  if (eventsPerHr <= warn) return 0.0;
  return clip01((eventsPerHr - warn) / (critical - warn));
}

// ---------------------------------------------------------------------
// Core fusion function — returns DRS 0-100 and logs component breakdown
// ---------------------------------------------------------------------
float computeDRS(float tempC, float rhPct, float massG, float vocPpm,
                  float pestEventsPerHr, JsonDocument &breakdownOut) {
  float massLossPct = ((baselineMassG - massG) / baselineMassG) * 100.0;
  float vocDelta = vocPpm - vocBaselinePpm;

  float sT = stressScore(tempC, crop.tempSafeMin, crop.tempSafeMax, crop.tempCritical);
  float sRh = stressScore(rhPct, crop.rhSafeMin, crop.rhSafeMax, 100.0);
  float sM = massLossScore(massLossPct, crop.massLossWarnPct, crop.massLossCriticalPct);
  float sV = vocScore(vocDelta, crop.vocWarnDelta, crop.vocCriticalDelta);
  float sP = pestScore(pestEventsPerHr, crop.pestWarnPerHr, crop.pestCriticalPerHr);

  float drs = 100.0 * (crop.wTemp * sT + crop.wRh * sRh + crop.wMass * sM +
                        crop.wVoc * sV + crop.wPest * sP);

  breakdownOut["temp"] = sT;
  breakdownOut["rh"] = sRh;
  breakdownOut["mass"] = sM;
  breakdownOut["voc"] = sV;
  breakdownOut["pest"] = sP;
  breakdownOut["mass_loss_pct"] = massLossPct;
  breakdownOut["voc_delta_ppm"] = vocDelta;

  return drs;
}

// ---------------------------------------------------------------------
// Ventilation control with hysteresis (avoids servo wear on rough roads)
// ---------------------------------------------------------------------
void updateLouvre(float tempC, float rhPct) {
  bool shouldOpen = louvreOpen
      ? (tempC > LOUVRE_CLOSE_TEMP || rhPct > crop.rhSafeMax)
      : (tempC > LOUVRE_OPEN_TEMP || rhPct > crop.rhSafeMax + 3.0);

  if (shouldOpen != louvreOpen) {
    louvreOpen = shouldOpen;
    // servo.write(louvreOpen ? OPEN_ANGLE : CLOSED_ANGLE);  // actual servo call
  }
}

// ---------------------------------------------------------------------
// Offline-first log write; GSM sync attempted opportunistically
// ---------------------------------------------------------------------
void logReading(float drs, JsonDocument &breakdown, float tempC, float rhPct) {
  JsonDocument doc;
  doc["t_ms"] = millis() - tripStartMillis;
  doc["drs"] = drs;
  doc["temp"] = tempC;
  doc["rh"] = rhPct;
  doc["breakdown"] = breakdown;

  File f = LittleFS.open("/trip_log.jsonl", "a");
  if (f) {
    serializeJson(doc, f);
    f.println();
    f.close();
  }

  // Attempt GSM push only if signal present; otherwise queued for next window.
  // gsmPushSummaryIfAvailable(doc);  // implementation depends on SIM800L lib
}

void triggerAlert(float drs) {
  // Sends a compact SMS/HTTP payload: crate ID, DRS, GPS (if available),
  // nearest-DEC hint. Dispatcher then arranges priority unloading —
  // this is the direct firmware response to the documented failure mode
  // of produce waiting 2-3 days in vehicles before unloading.
  // gsmSendAlert(crateId, drs, "PRIORITY_UNLOAD_RECOMMENDED");
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  sht31.begin(0x44);
  scale.begin(PIN_HX711_DT, PIN_HX711_SCK);
  LittleFS.begin(true);

  // loadCropProfileFromConfig(crop);  // reads /config/crop_profile.json
  // vocBaselinePpm = calibrateVocBaseline();  // 60s average at loading dock
  // baselineMassG = scale.get_units(10);

  tripStartMillis = millis();
  lastSampleMillis = millis();
}

void loop() {
  unsigned long now = millis();
  if (now - lastSampleMillis < sampleIntervalMs) return;
  lastSampleMillis = now;

  float tempC = sht31.readTemperature();
  float rhPct = sht31.readHumidity();
  float massG = scale.get_units(5);
  float vocPpm = analogRead(PIN_VOC_ANALOG) * (100.0 / 4095.0); // placeholder scaling, needs sensor-specific calibration curve
  float pestEventsPerHr = digitalRead(PIN_PEST_SENSOR); // placeholder: real impl accumulates counts/hour

  JsonDocument breakdown;
  float drs = computeDRS(tempC, rhPct, massG, vocPpm, pestEventsPerHr, breakdown);

  updateLouvre(tempC, rhPct);
  logReading(drs, breakdown, tempC, rhPct);

  if (drs >= WATCH_THRESHOLD) {
    sampleIntervalMs = WATCH_INTERVAL_MS;  // increase monitoring frequency
  }
  if (drs >= ALERT_THRESHOLD) {
    triggerAlert(drs);
  }
}
