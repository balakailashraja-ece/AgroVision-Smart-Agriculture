/*
  smart_agri_esp32.ino  -  AI-driven adaptive irrigation, fully on the ESP32 (no PC / server needed at run time)
  ---------------------------------------------------------------------------------------------------------------
  Sketch folder must contain:  smart_agri_esp32.ino  +  agri_logic.h  +  ml_model.h     (open the .ino, the others appear as tabs)
  Board     : ESP32 DevKit (Arduino-ESP32 core 2.x or 3.x)
  Libraries : "DHT sensor library" + "Adafruit Unified Sensor" (Adafruit), "LiquidCrystal_I2C" (only if USE_LCD = 1)

  WHAT RUNS ON THE ESP32
   1. Sensors      soil moisture, DHT11 temperature/humidity, LDR light, rain sensor           (your original pins)
   2. History      every 15 minutes (on the clock: :00 :15 :30 :45) -> 24 h of moisture + air temperature
   3. ML           17 features -> Ridge model -> forecast of soil moisture after 1 h / 6 h / 24 h  (ml_model.h)
   4. Rule engine  port of irrigation_core.py: validation, rain lock, soak, cool-down, daily budget, episode limit,
                   drying-rate estimate, ineffective-pulse lock-out, adaptive pulse length (learns "pp per second of pumping")
   5. Pump         timed pulses only, hard-capped, relay OFF at boot / on any fault
   6. ThingSpeak   fields 1-8 (see UPLOAD below)

  HONEST STATUS OF THE ML (read this)
   * The bundled models were trained on a different sensor, soil and city (Kolkata, ~11 % moisture) and, according to your own
     evaluation, none of them reliably beats "moisture stays the same". Therefore ML_ALLOW_CONTROL = 0: forecasts are only
     LOGGED (Serial / LCD / ThingSpeak field 8) - they can never switch the pump.
   * With your sensor (35-60 %) the inputs are far outside the training range, so the firmware reports "out of range"
     instead of printing a meaningless number. After you collect 8-12 weeks of your own data: retrain.py -> export_esp32_model.py
     -> new ml_model.h -> re-upload. Only a model promoted by retrain.py (may_control_pump = true) can ever influence the pump.
   * What really adapts the irrigation today is the on-device learner (pulse length from measured outcomes) and the rules.

  STATUS: written to match the tested Python code (agri_logic.h was compared against it on a PC: features, forecasts,
          drying rate and 29 decision scenarios agree). The .ino itself has NOT been compiled for an ESP32 or run on hardware by the
          author. Bench-test with the PUMP DISCONNECTED (watch the relay LED / Serial Monitor) before connecting water.

  WIRING (your original sketch)   Soil AOUT -> GPIO34 | LDR -> GPIO35 | DHT11 -> GPIO4 | Rain DO -> GPIO27 (LOW = rain)
                                  Relay IN -> GPIO18   | LCD I2C SDA 21 / SCL 22
                                  OPTIONAL emergency button GPIO25 -> GND (internal pull-up). Press = latch pump OFF, hold 5 s = clear.
  Serial Monitor 115200 baud, commands:  s = status   x = emergency OFF   r = clear emergency + ineffective-pulse lock-out
  PUMP POWER: separate supply, common GND, flyback diode across a DC pump, fuse; keep mains and water away from the ESP32.
*/
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <DHT.h>
#include <Preferences.h>
#include <time.h>
#include <esp_task_wdt.h>
#include "agri_logic.h"

// ============================ USER SETTINGS ============================
const char* WIFI_SSID   = "Kailash";
const char* WIFI_PASS   = "1234567890";
const char* TS_API_KEY  = "VYWXD5CSBJD7FZ0T";
const char* TS_URL      = "http://api.thingspeak.com/update";
const long  TZ_OFFSET_S = 19800;                               // IST (UTC+5:30). Only used for the hour-of-day feature.
#define USE_LCD 1                                              // 1 = 16x2 I2C LCD at 0x27

// ---- crop moisture thresholds, in the % of YOUR calibrated sensor (see SOIL CALIBRATION below)
//      Only SOIL_MIN_PCT = 35 comes from your original sketch. TARGET and MAX are placeholders.
//      Take real values from a cited reference (FAO-56 / ICAR / state agri. university) for YOUR crop + soil, then set the flag to 1.
#define THRESHOLDS_CONFIRMED 0          // 0 => pump stays OFF and the LCD says CONFIG REQUIRED
const float SOIL_MIN_PCT    = 35.0f;    // start irrigating below this
const float SOIL_TARGET_PCT = 45.0f;    // pulses continue until the soil reaches this
const float SOIL_MAX_PCT    = 60.0f;    // above this = too wet

// ---- machine learning
#define ML_ALLOW_CONTROL 0              // 0 = forecasts are logged only.  1 = a model whose registry flag may_control_pump is true
                                        //     (validated by retrain.py on YOUR data) may start a pulse early. Leave 0 for the bundled models.

// ---- pins (as in your original sketch)
#define SOIL_PIN   34
#define DHT_PIN    4
#define DHT_TYPE   DHT11
#define LDR_PIN    35
#define RAIN_PIN   27
#define RELAY_PIN  18
#define BTN_PIN    25                   // optional emergency button to GND
#define RELAY_ON   LOW
#define RELAY_OFF  HIGH

// ---- SOIL CALIBRATION (your values). Dry = sensor in air, Wet = sensor in saturated soil. Watch the "adc=" value on Serial.
int soilDry = 3000;
int soilWet = 1300;

// ---- hard limits enforced HERE, independent of the rule engine
const uint32_t HARD_MAX_PULSE_S  = 60;       // no pulse longer than this, ever
const uint32_t LOCAL_MAX_DAILY_S = 600;      // pump seconds per rolling 24 h
const uint32_t SENSE_MS          = 2000;
const uint32_t EVAL_MS           = 60000;    // run the decision engine once a minute
const uint32_t UPLOAD_MS         = 20000;    // ThingSpeak (free accounts: >= 15 s)
const uint32_t HTTP_TIMEOUT_MS   = 5000;

#if USE_LCD
#include <LiquidCrystal_I2C.h>
LiquidCrystal_I2C lcd(0x27, 16, 2);
#endif

using namespace agri;

// ============================ STATE ============================
DHT         dht(DHT_PIN, DHT_TYPE);
Preferences prefs;
Config      CF;                    // safety parameters (agri_logic.h)
History     HIST;                  // 15-min history for the ML features
CycleLog    LOG;                   // pump cycles: budget, episode, learner (persisted in flash)
Forecast    FC[ML_N_MODELS];       // 1 h, 6 h, 24 h
Decision    DEC;                   // last decision
bool        haveDecision = false;

struct SensorSnap {
  int   adc = 0; float pct = NAN;  // pct = NAN when the soil sensor is invalid
  float t = NAN, h = NAN; int ldr = 0; bool rain = false; bool dhtOk = false;
} S;

enum PumpState { PUMP_IDLE, PUMP_RUNNING };
PumpState pump = PUMP_IDLE;
uint32_t pumpStartMs = 0, pumpTargetMs = 0, pumpStartEpoch = 0;
float    pumpPrePct = NAN;
bool     emergencyLatched = false;
bool     awaitingPost = false; uint32_t postDueMs = 0; bool rainSincePulse = false, pulseAborted = false;
uint32_t lastRainMs = 0;           // millis() when rain was last seen (0 = never since boot)
uint32_t lastSlot = 0;
uint32_t tSense = 0, tEval = 0, tUpload = 0, tLcd = 0, tPrint = 0, tWifi = 0;
bool     ntpStarted = false;
int      lcdPage = 0;

// ============================ RELAY (fail-safe) ============================
inline void relayWrite(bool on) { digitalWrite(RELAY_PIN, on ? RELAY_ON : RELAY_OFF); }

// ============================ TIME ============================
bool     timeValid() { return time(nullptr) > 1700000000; }
uint32_t nowEpoch()  { return (uint32_t)time(nullptr); }

// ============================ SENSORS ============================
int readMedian(int pin, int n) {
  int v[31]; if (n > 31) n = 31;
  for (int i = 0; i < n; i++) { v[i] = analogRead(pin); delay(3); }
  for (int i = 1; i < n; i++) { int k = v[i], j = i - 1; while (j >= 0 && v[j] > k) { v[j + 1] = v[j]; j--; } v[j + 1] = k; }
  return v[n / 2];
}
bool soilAdcValid(int adc) { return adc > CF.adc_rail_low && adc < CF.adc_rail_high; }
float soilPercent(int adc) {                       // two-point linear calibration (same as your map()+constrain)
  if (soilDry == soilWet) return NAN;
  float p = (float)(soilDry - adc) / (float)(soilDry - soilWet) * 100.0f;
  return p < 0 ? 0 : (p > 100 ? 100 : p);
}
bool readRainWet() {                               // debounced: 3 of 3 samples LOW (module output: LOW = rain)
  int wet = 0; for (int i = 0; i < 3; i++) { if (digitalRead(RAIN_PIN) == LOW) wet++; delay(5); }
  return wet == 3;
}
void readSensors() {
  S.adc = readMedian(SOIL_PIN, 15);
  S.pct = soilAdcValid(S.adc) ? soilPercent(S.adc) : NAN;
  S.ldr = readMedian(LDR_PIN, 9);
  float t = dht.readTemperature(), h = dht.readHumidity();
  S.dhtOk = !isnan(t) && !isnan(h);
  S.t = S.dhtOk ? t : NAN; S.h = S.dhtOk ? h : NAN;
  S.rain = readRainWet();
  if (S.rain) { lastRainMs = millis() ? millis() : 1; if (pump == PUMP_RUNNING || awaitingPost) rainSincePulse = true; }
}

// ============================ PERSISTENCE ============================
void saveLog() { prefs.putBytes("log", &LOG, sizeof(LOG)); }
void loadLog() {
  LOG.clear();
  if (prefs.getBytesLength("log") == sizeof(CycleLog)) {
    CycleLog tmp; prefs.getBytes("log", &tmp, sizeof(tmp));
    if (tmp.n >= 0 && tmp.n <= MAX_CYCLES) LOG = tmp;
  }
}

// ============================ EMERGENCY / RESET ============================
void latchEmergency(const char* why) {
  if (!emergencyLatched) { emergencyLatched = true; prefs.putBool("estop", true); }
  relayWrite(false);
  Serial.print("EMERGENCY OFF: "); Serial.println(why);
}
void resetLockouts() {                              // clears emergency latch + ineffective-pulse lock-out (old outcomes stop counting for learning)
  emergencyLatched = false; prefs.putBool("estop", false);
  for (int i = 0; i < LOG.n; i++) LOG.c[i].valid = 0;   // keeps them for the 24 h budget and the episode counter (safety)
  saveLog(); Serial.println("Lock-outs cleared");
}

// ============================ PUMP ============================
void stopPump(const char* why, bool aborted);
double runtimeLast24h() {
  double used = 0; if (!timeValid()) return 1e9; double now = nowEpoch();
  for (int i = 0; i < LOG.n; i++) if ((double)LOG.c[i].start_epoch > now - 86400.0) used += LOG.c[i].runtime_s;
  return used;
}
const char* refuseReason(uint32_t secs) {            // "" = allowed. Independent second opinion on every pulse request.
  if (emergencyLatched) return "emergency latched";
  if (pump != PUMP_IDLE) return "pump busy";
  if (secs == 0) return "zero length";
  if (!timeValid()) return "clock not set";
  if (S.rain || (lastRainMs && (millis() - lastRainMs) < (uint32_t)CF.rain_lockout_s * 1000UL)) return "rain lockout";
  int adc = analogRead(SOIL_PIN); if (!soilAdcValid(adc)) return "soil sensor invalid";
  if (LOG.n > 0) { double since = (double)nowEpoch() - ((double)LOG.c[LOG.n - 1].start_epoch + LOG.c[LOG.n - 1].runtime_s); if (since < CF.cooldown_s) return "cooldown"; }
  if (runtimeLast24h() + secs > LOCAL_MAX_DAILY_S) return "daily budget";
  return "";
}
void startPump(uint32_t secs) {
  if (secs > HARD_MAX_PULSE_S) secs = HARD_MAX_PULSE_S;
  const char* why = refuseReason(secs);
  if (why[0]) { Serial.print("PUMP REFUSED: "); Serial.println(why); return; }
  pumpStartEpoch = nowEpoch(); pumpPrePct = S.pct; pumpStartMs = millis(); pumpTargetMs = secs * 1000UL;
  rainSincePulse = false; pulseAborted = false; awaitingPost = false;
  pump = PUMP_RUNNING; relayWrite(true);
  Serial.print("PUMP ON for "); Serial.print(secs); Serial.println(" s");
}
void stopPump(const char* why, bool aborted) {
  relayWrite(false);
  if (pump != PUMP_RUNNING) return;
  pump = PUMP_IDLE;
  uint32_t ms = millis() - pumpStartMs; uint32_t rt = (ms + 500) / 1000; if (rt == 0) rt = 1;
  Cycle c; memset(&c, 0, sizeof(c));
  c.start_epoch = pumpStartEpoch; c.runtime_s = (uint16_t)rt; c.pre_pct = isnan(pumpPrePct) ? 0 : pumpPrePct; c.valid = 0; c.post_known = 0;
  LOG.add(c); saveLog();
  pulseAborted = aborted; awaitingPost = true; postDueMs = millis() + (uint32_t)CF.soak_s * 1000UL;
  Serial.print("PUMP OFF ("); Serial.print(why); Serial.print(") after "); Serial.print(rt); Serial.println(" s");
}
void measurePostOutcome() {                          // after the soak period: how much did the pulse really add?  -> learner
  awaitingPost = false; readSensors();
  if (LOG.n == 0) return;
  Cycle& c = LOG.c[LOG.n - 1];
  if (!isnan(S.pct)) {
    c.post_pct = S.pct; c.post_known = 1;
    c.valid = (!rainSincePulse && !pulseAborted && !isnan(pumpPrePct) && c.runtime_s > 0) ? 1 : 0;   // clean outcomes only
    Serial.print("Outcome: "); Serial.print(c.pre_pct, 1); Serial.print(" -> "); Serial.print(c.post_pct, 1);
    Serial.print(" % in "); Serial.print(c.runtime_s); Serial.println(c.valid ? " s (used for learning)" : " s (NOT used: rain/abort/sensor)");
  }
  saveLog();
}
void buttonWatch() {
  static uint32_t t0 = 0; static bool down = false, cleared = false;
  bool p = digitalRead(BTN_PIN) == LOW;
  if (p && !down) { down = true; cleared = false; t0 = millis(); latchEmergency("button"); if (pump == PUMP_RUNNING) stopPump("emergency button", true); }
  else if (p && down && !cleared && millis() - t0 >= 5000) { cleared = true; resetLockouts(); }
  else if (!p) down = false;
}
void pumpWatch() {                                   // every loop iteration: the ONLY place a pulse ends
  buttonWatch();
  if (emergencyLatched) { relayWrite(false); if (pump == PUMP_RUNNING) stopPump("emergency", true); return; }
  if (pump != PUMP_RUNNING) { relayWrite(false); return; }   // belt and braces
  uint32_t el = millis() - pumpStartMs;
  static uint32_t lastChk = 0;
  if (millis() - lastChk > 500) {
    lastChk = millis();
    if (readRainWet()) { lastRainMs = millis() ? millis() : 1; rainSincePulse = true; stopPump("rain detected", true); return; }
    if (!soilAdcValid(analogRead(SOIL_PIN))) { stopPump("soil sensor fault", true); return; }
  }
  if (el >= pumpTargetMs)             { stopPump("completed", false); return; }
  if (el >= HARD_MAX_PULSE_S * 1000UL){ stopPump("hard max pulse", true); return; }
}

// ============================ HISTORY + ML ============================
void runForecasts(uint32_t slotEpoch) {
  float x[ML_N_FEATURES];
  float hour = (float)(((slotEpoch + (uint32_t)TZ_OFFSET_S) % 86400UL)) / 3600.0f;
  bool ok = makeFeatures(HIST, hour, x);
  for (int k = 0; k < ML_N_MODELS; k++) FC[k] = forecast(ML_MODELS[k], x, ok, CF);
  for (int k = 0; k < ML_N_MODELS; k++) {
    Serial.print("ML "); Serial.print(ML_MODELS[k].horizon_h); Serial.print("h: ");
    if (FC[k].status == FC_OK) {
      Serial.print(FC[k].pred, 2); Serial.print(" % (90% range "); Serial.print(FC[k].lo, 2); Serial.print(".."); Serial.print(FC[k].hi, 2); Serial.print(") ");
      Serial.println((ML_ALLOW_CONTROL && FC[k].may_control) ? "[ACTIVE]" : "[shadow: logged only]");
    } else if (FC[k].status == FC_OUT_OF_RANGE) {
      Serial.print("out of range (max |z| = "); Serial.print(FC[k].max_abs_z, 1); Serial.println("): model trained on a different sensor scale -> retrain on your data");
    } else {
      Serial.print("warm-up ("); Serial.print(HIST.n); Serial.print("/"); Serial.print(HIST_N); Serial.println(" samples, needs 24 h)");
    }
  }
}
void sampleHistory() {                               // once per 15-minute clock slot
  if (!timeValid()) return;
  uint32_t slot = nowEpoch() / 900UL; if (slot == lastSlot) return;
  if (lastSlot != 0 && slot > lastSlot + 1) {          // missed slots become NaN so the time grid never shifts silently
    uint32_t miss = slot - lastSlot - 1; if (miss > 96) miss = 96;
    for (uint32_t k = 0; k < miss; k++) HIST.push(NAN, NAN, (slot - miss + k) * 900UL, CF);
  }
  HIST.push(S.pct, S.t, slot * 900UL, CF); lastSlot = slot;
  runForecasts(slot * 900UL);
}

// ============================ DECISION ============================
void evaluate() {
  Inputs in; memset(&in, 0, sizeof(in));
  in.r.moisture_pct = S.pct; in.r.adc = S.adc; in.r.time_valid = timeValid(); in.r.rain_now = S.rain;
  in.r.air_temp = S.t; in.r.humidity = S.h; in.r.pump_on = (pump == PUMP_RUNNING); in.r.manual_off = emergencyLatched;
  in.last_m = HIST.n ? HIST.m[HIST.n - 1] : NAN;
  in.sec_since_rain = lastRainMs ? (int32_t)((millis() - lastRainMs) / 1000UL) : -1;
  in.now_epoch = nowEpoch(); in.log = &LOG; in.hist = &HIST;
  in.thr.configured = THRESHOLDS_CONFIRMED && thresholdsValid(SOIL_MIN_PCT, SOIL_TARGET_PCT, SOIL_MAX_PCT);
  in.thr.lo = SOIL_MIN_PCT; in.thr.tgt = SOIL_TARGET_PCT; in.thr.hi = SOIL_MAX_PCT;
  in.ineffective_latched = false;
  in.f6 = FC[1]; in.ml_may_control = ML_ALLOW_CONTROL;
  decide(in, CF, DEC); haveDecision = true;
  Serial.print("DECISION "); Serial.print(codeName(DEC.code)); Serial.print(" (code "); Serial.print(DEC.code); Serial.print(") | "); Serial.println(DEC.reason);
  if (DEC.code == IRRIGATION_REQUIRED && DEC.pump_s > 0 && pump == PUMP_IDLE) startPump((uint32_t)DEC.pump_s);
}

// ============================ DISPLAY / SERIAL ============================
void lcdPrint(const char* a, const char* b) {
#if USE_LCD
  char l[17]; snprintf(l, sizeof(l), "%-16.16s", a); lcd.setCursor(0, 0); lcd.print(l);
  snprintf(l, sizeof(l), "%-16.16s", b); lcd.setCursor(0, 1); lcd.print(l);
#endif
}
void updateLcd() {
  char a[48], b[48];
  if (pump == PUMP_RUNNING) { snprintf(a, sizeof(a), "PUMP ON  %lus", (unsigned long)((pumpTargetMs - (millis() - pumpStartMs)) / 1000UL)); snprintf(b, sizeof(b), "Soil:%d%%", isnan(S.pct) ? -1 : (int)S.pct); lcdPrint(a, b); return; }
  if (emergencyLatched) { lcdPrint("EMERGENCY OFF", "hold btn 5s/'r'"); return; }
  lcdPage = (lcdPage + 1) % 3;
  if (lcdPage == 0) {
    if (isnan(S.pct)) snprintf(a, sizeof(a), "Soil:ERR"); else snprintf(a, sizeof(a), "Soil:%d%%", (int)S.pct);
    if (S.dhtOk) snprintf(a + strlen(a), sizeof(a) - strlen(a), " T:%dC", (int)S.t); else snprintf(a + strlen(a), sizeof(a) - strlen(a), " T:ERR");
    snprintf(b, sizeof(b), "%s L:%d", S.rain ? "Rain" : "Dry ", S.ldr); lcdPrint(a, b);
  } else if (lcdPage == 1) {
    lcdPrint("Decision:", haveDecision ? codeName(DEC.code) : "starting...");
  } else {
    if (FC[1].status == FC_OK) { snprintf(a, sizeof(a), "6h: %.1f%%", FC[1].pred); lcdPrint(a, "ML shadow only"); }
    else if (FC[1].status == FC_OUT_OF_RANGE) lcdPrint("ML: out of range", "retrain on data");
    else { snprintf(a, sizeof(a), "ML warm-up %d/%d", HIST.n, HIST_N); lcdPrint(a, "needs 24 h"); }
  }
}
void printStatus() {
  Serial.println("------------------------");
  Serial.print("adc="); Serial.print(S.adc); Serial.print("  soil="); if (isnan(S.pct)) Serial.print("INVALID"); else { Serial.print(S.pct, 1); Serial.print("%"); }
  Serial.print("  T="); if (S.dhtOk) Serial.print(S.t, 1); else Serial.print("ERR");
  Serial.print("C  H="); if (S.dhtOk) Serial.print(S.h, 0); else Serial.print("ERR");
  Serial.print("%  light="); Serial.print(S.ldr); Serial.print("  rain="); Serial.print(S.rain ? "YES" : "no");
  Serial.print("  pump="); Serial.print(pump == PUMP_RUNNING ? "ON" : "OFF");
  Serial.print("  wifi="); Serial.print(WiFi.status() == WL_CONNECTED ? "ok" : "down"); Serial.print("  clock="); Serial.println(timeValid() ? "ok" : "not set");
  float g; int nv;
  if (gainEstimate(LOG, CF, &g, &nv)) { Serial.print("learned gain: "); Serial.print(g, 3); Serial.print(" pp per second (from "); Serial.print(nv); Serial.println(" clean cycles)"); }
  else { Serial.print("learner: "); Serial.print(nv); Serial.print("/"); Serial.print(CF.min_cycles_for_gain); Serial.println(" clean cycles -> initial pulse length"); }
  Serial.print("pump seconds in last 24 h: "); Serial.print(runtimeLast24h(), 0); Serial.print(" / "); Serial.println(LOCAL_MAX_DAILY_S);
}
void serialCommands() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 's') printStatus();
    else if (c == 'x') { latchEmergency("serial"); if (pump == PUMP_RUNNING) stopPump("emergency serial", true); }
    else if (c == 'r') resetLockouts();
  }
}

// ============================ WIFI / THINGSPEAK ============================
void wifiMaintain() {
  if (WiFi.status() == WL_CONNECTED) {
    if (!ntpStarted) { configTime(0, 0, "pool.ntp.org", "time.google.com"); ntpStarted = true; }
    return;
  }
  if (millis() - tWifi > 30000UL) { tWifi = millis(); WiFi.disconnect(); WiFi.begin(WIFI_SSID, WIFI_PASS); }   // non-blocking retry
}
void uploadThingSpeak() {
  if (pump != PUMP_IDLE || WiFi.status() != WL_CONNECTED) return;     // never block the loop while the pump runs
  String body = "api_key=" + String(TS_API_KEY);
  if (!isnan(S.pct))  body += "&field1=" + String(S.pct, 1);
  if (S.dhtOk)        body += "&field2=" + String(S.t, 1) + "&field3=" + String(S.h, 1);
  body += "&field4=" + String(S.ldr) + "&field5=" + String(S.rain ? 1 : 0) + "&field6=" + String(pump == PUMP_RUNNING ? 1 : 0);
  if (haveDecision)   body += "&field7=" + String((int)DEC.code);     // 0 not required, 1 irrigate, 2 wait, 3 rain, 4 sensor invalid, 5 config required, 6 lock-out, 7 manual off
  if (FC[1].status == FC_OK) body += "&field8=" + String(FC[1].pred, 2);   // 6 h forecast (shadow)
  WiFiClient client; HTTPClient http; http.setTimeout(HTTP_TIMEOUT_MS);
  if (!http.begin(client, TS_URL)) return;
  http.addHeader("Content-Type", "application/x-www-form-urlencoded");
  int code = http.POST(body);
  if (code > 0) { String r = http.getString(); Serial.print("ThingSpeak: HTTP "); Serial.print(code); Serial.print(" entry "); Serial.println(r); }
  else { Serial.print("ThingSpeak error "); Serial.println(code); }
  http.end();
}

// ============================ ARDUINO ============================
void setup() {
  relayWrite(false);                                   // set the OFF level BEFORE the pin becomes an output (no glitch)
  pinMode(RELAY_PIN, OUTPUT); relayWrite(false);
  pinMode(RAIN_PIN, INPUT); pinMode(BTN_PIN, INPUT_PULLUP); pinMode(SOIL_PIN, INPUT); pinMode(LDR_PIN, INPUT);
  Serial.begin(115200);
  analogReadResolution(12);
  dht.begin();
#if USE_LCD
  Wire.begin(21, 22); lcd.init(); lcd.backlight(); lcdPrint("Smart Agriculture", "booting...");
#endif
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  esp_task_wdt_config_t wc = {.timeout_ms = 30000, .idle_core_mask = 0, .trigger_panic = true}; esp_task_wdt_reconfigure(&wc);
#else
  esp_task_wdt_init(30, true);
#endif
  esp_task_wdt_add(NULL);                              // reboot (=> relay OFF) if the loop ever hangs
  prefs.begin("agri", false);
  emergencyLatched = prefs.getBool("estop", false); loadLog();
  HIST.reset(); for (int k = 0; k < ML_N_MODELS; k++) { memset(&FC[k], 0, sizeof(Forecast)); FC[k].status = FC_NOT_READY; }
  Serial.println("\nSmart agriculture ESP32 - pump OFF");
  if (!THRESHOLDS_CONFIRMED) Serial.println("NOTE: THRESHOLDS_CONFIRMED = 0 -> the pump will not run until you confirm the crop thresholds in the sketch.");
  Serial.print("Loaded "); Serial.print(LOG.n); Serial.println(" pump cycles from flash.");
  WiFi.mode(WIFI_STA); WiFi.setAutoReconnect(true); WiFi.begin(WIFI_SSID, WIFI_PASS);
  uint32_t t0 = millis(); while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000UL) { delay(250); esp_task_wdt_reset(); Serial.print("."); }
  Serial.println(WiFi.status() == WL_CONNECTED ? "\nWiFi connected" : "\nWiFi not connected - running offline (pump stays OFF until the clock is set)");
  wifiMaintain();
  tEval = millis() + 15000UL;                          // first decision after the sensors have settled
  readSensors();
}

void loop() {
  esp_task_wdt_reset();
  pumpWatch();                                         // ends pulses on time, every iteration
  serialCommands();
  wifiMaintain();
  uint32_t ms = millis();

  if (pump == PUMP_IDLE && ms - tSense >= SENSE_MS) { tSense = ms; readSensors(); sampleHistory(); }
  if (awaitingPost && pump == PUMP_IDLE && (int32_t)(ms - postDueMs) >= 0) measurePostOutcome();
  if (pump == PUMP_IDLE && !awaitingPost && (int32_t)(ms - tEval) >= 0) { tEval = ms + EVAL_MS; evaluate(); }
  if (pump == PUMP_IDLE && ms - tUpload >= UPLOAD_MS) { tUpload = ms; uploadThingSpeak(); }
  if (ms - tLcd >= 3000UL) { tLcd = ms; updateLcd(); }
  if (ms - tPrint >= 10000UL) { tPrint = ms; printStatus(); }
  delay(20);
}
