#include <ArduinoJson.h>
#include <ESP32Servo.h>
#include <math.h>
#include <string.h>

// ======================================================
// SERIAL SETTINGS
// ======================================================
constexpr unsigned long BAUD_RATE = 115200;
constexpr size_t MAX_LINE_LENGTH = 512;
constexpr unsigned long DATA_TIMEOUT_MS = 500;
constexpr unsigned long SERVO_STEP_INTERVAL_MS = 15;

// ======================================================
// SERVO SETTINGS
// ======================================================
constexpr int SERVO_STEP = 10;
constexpr int INITIAL_POSITION = 90;
constexpr int SERVO_MIN_PULSE = 500;
constexpr int SERVO_MAX_PULSE = 2400;

// ======================================================
// SERVO PINS
// ======================================================
constexpr int LEFT_SHOULDER_PIN = 18;
constexpr int LEFT_ELBOW_HORIZONTAL_PIN = 19;
constexpr int LEFT_ELBOW_VERTICAL_PIN = 23;

constexpr int RIGHT_SHOULDER_PIN = 21;
constexpr int RIGHT_ELBOW_HORIZONTAL_PIN = 22;
constexpr int RIGHT_ELBOW_VERTICAL_PIN = 25;

constexpr int LEFT_HIP_PIN = 26;
constexpr int RIGHT_HIP_PIN = 27;

// ======================================================
// MOTOR PINS
// ======================================================
constexpr int LEFT_MOTOR_EN = 16;
constexpr int LEFT_MOTOR_IN1 = 13;
constexpr int LEFT_MOTOR_IN2 = 14;

constexpr int RIGHT_MOTOR_EN = 17;
constexpr int RIGHT_MOTOR_IN1 = 32;
constexpr int RIGHT_MOTOR_IN2 = 33;

// ======================================================
// SERVO OBJECTS
// ======================================================
Servo leftShoulder;
Servo leftElbowHorizontal;
Servo leftElbowVertical;
Servo rightShoulder;
Servo rightElbowHorizontal;
Servo rightElbowVertical;
Servo leftHip;
Servo rightHip;

// ======================================================
// LAST SERVO TARGETS
// ======================================================
int lastLeftShoulder = INITIAL_POSITION;
int lastLeftElbowHorizontal = INITIAL_POSITION;
int lastLeftElbowVertical = INITIAL_POSITION;
int lastRightShoulder = INITIAL_POSITION;
int lastRightElbowHorizontal = INITIAL_POSITION;
int lastRightElbowVertical = INITIAL_POSITION;
int lastLeftHip = INITIAL_POSITION;
int lastRightHip = INITIAL_POSITION;

int currentLeftShoulder = INITIAL_POSITION;
int currentLeftElbowHorizontal = INITIAL_POSITION;
int currentLeftElbowVertical = INITIAL_POSITION;
int currentRightShoulder = INITIAL_POSITION;
int currentRightElbowHorizontal = INITIAL_POSITION;
int currentRightElbowVertical = INITIAL_POSITION;
int currentLeftHip = INITIAL_POSITION;
int currentRightHip = INITIAL_POSITION;
unsigned long lastServoStepTime = 0;

// ======================================================
// SERIAL BUFFER AND CONNECTION STATUS
// ======================================================
char inputBuffer[MAX_LINE_LENGTH];
size_t bufferIndex = 0;
bool discardLine = false;
unsigned long lastDataTime = 0;
bool dataReceived = false;

bool forwardSignal = false;
bool backwardSignal = false;
bool leftTurnSignal = false;
bool rightTurnSignal = false;

// ======================================================
// ANGLE PROCESSING
// Clip upward to the next 10-degree step first.
// For reversed servos, reverse the clipped result afterwards.
// ======================================================
int processServoAngle(float angle, bool invert = false) {
    if (!isfinite(angle)) {
        return INITIAL_POSITION;
    }

    angle = constrain(angle, 0.0f, 180.0f);

    int clippedAngle = (int)ceil(angle / (float)SERVO_STEP) * SERVO_STEP;
    clippedAngle = constrain(clippedAngle, 0, 180);

    if (invert) {
        return 180 - clippedAngle;
    }

    return clippedAngle;
}

// ======================================================
// SERVO UPDATE
// A JSON null means the landmark was unavailable. In that
// case, keep the servo at its previous target.
// ======================================================
void updateServo(
    JsonDocument& doc,
    const char* key,
    int& lastAngle,
    bool invert = false
) {
    if (doc[key].isNull()) {
        return;
    }

    if (!doc[key].is<float>() && !doc[key].is<int>()) {
        return;
    }

    float receivedAngle = doc[key].as<float>();
    if (!isfinite(receivedAngle)) {
        return;
    }

    int targetAngle = processServoAngle(receivedAngle, invert);
    lastAngle = targetAngle;
}

void updateServosSmoothly() {
    unsigned long now = millis();
    if (now - lastServoStepTime < SERVO_STEP_INTERVAL_MS) {
        return;
    }
    lastServoStepTime = now;

    Servo* servos[] = {
        &leftShoulder, &leftElbowHorizontal, &leftElbowVertical,
        &rightShoulder, &rightElbowHorizontal, &rightElbowVertical,
        &leftHip, &rightHip
    };
    int* currentAngles[] = {
        &currentLeftShoulder, &currentLeftElbowHorizontal, &currentLeftElbowVertical,
        &currentRightShoulder, &currentRightElbowHorizontal, &currentRightElbowVertical,
        &currentLeftHip, &currentRightHip
    };
    int* targetAngles[] = {
        &lastLeftShoulder, &lastLeftElbowHorizontal, &lastLeftElbowVertical,
        &lastRightShoulder, &lastRightElbowHorizontal, &lastRightElbowVertical,
        &lastLeftHip, &lastRightHip
    };

    for (size_t i = 0; i < sizeof(servos) / sizeof(servos[0]); ++i) {
        if (*currentAngles[i] < *targetAngles[i]) {
            *currentAngles[i] = min(*currentAngles[i] + SERVO_STEP, *targetAngles[i]);
        } else if (*currentAngles[i] > *targetAngles[i]) {
            *currentAngles[i] = max(*currentAngles[i] - SERVO_STEP, *targetAngles[i]);
        }
        servos[i]->write(*currentAngles[i]);
    }
}

// ======================================================
// READ HIGH/LOW MOTOR SIGNAL
// ======================================================
bool readSignal(JsonVariantConst value, bool& result) {
    if (value.is<bool>()) {
        result = value.as<bool>();
        return true;
    }

    if (value.is<const char*>()) {
        const char* signal = value.as<const char*>();
        if (strcmp(signal, "HIGH") == 0) {
            result = true;
            return true;
        }
        if (strcmp(signal, "LOW") == 0) {
            result = false;
            return true;
        }
    }

    return false;
}

// ======================================================
// UPDATE MOTOR SIGNALS
// These names match the Python JSON exactly.
// ======================================================
bool updateControlSignals(JsonDocument& doc) {
    bool newForward;
    bool newBackward;
    bool newLeftTurn;
    bool newRightTurn;

    if (!readSignal(doc["forward"], newForward)) return false;
    if (!readSignal(doc["backward"], newBackward)) return false;
    if (!readSignal(doc["left_turn"], newLeftTurn)) return false;
    if (!readSignal(doc["right_turn"], newRightTurn)) return false;

    forwardSignal = newForward;
    backwardSignal = newBackward;
    leftTurnSignal = newLeftTurn;
    rightTurnSignal = newRightTurn;
    return true;
}

// ======================================================
// MOTOR OUTPUT
// direction: 1 = forward, -1 = backward, 0 = stop
// ======================================================
void setMotor(int enablePin, int in1, int in2, int direction) {
    if (direction == 1) {
        digitalWrite(in1, HIGH);
        digitalWrite(in2, LOW);
        digitalWrite(enablePin, HIGH);
    } else if (direction == -1) {
        digitalWrite(in1, LOW);
        digitalWrite(in2, HIGH);
        digitalWrite(enablePin, HIGH);
    } else {
        digitalWrite(in1, LOW);
        digitalWrite(in2, LOW);
        digitalWrite(enablePin, LOW);
    }
}

void updateMotors() {
    int leftDirection = 0;
    int rightDirection = 0;

    // Priority if more than one signal is HIGH:
    // forward, backward, left turn, then right turn.
    if (forwardSignal) {
        leftDirection = 1;
        rightDirection = 1;
    } else if (backwardSignal) {
        leftDirection = -1;
        rightDirection = -1;
    } else if (leftTurnSignal) {
        leftDirection = 1;
    } else if (rightTurnSignal) {
        rightDirection = 1;
    }

    setMotor(LEFT_MOTOR_EN, LEFT_MOTOR_IN1, LEFT_MOTOR_IN2, leftDirection);
    setMotor(RIGHT_MOTOR_EN, RIGHT_MOTOR_IN1, RIGHT_MOTOR_IN2, rightDirection);
}

void stopMotors() {
    forwardSignal = false;
    backwardSignal = false;
    leftTurnSignal = false;
    rightTurnSignal = false;

    setMotor(LEFT_MOTOR_EN, LEFT_MOTOR_IN1, LEFT_MOTOR_IN2, 0);
    setMotor(RIGHT_MOTOR_EN, RIGHT_MOTOR_IN1, RIGHT_MOTOR_IN2, 0);
}

void moveAllServosToInitialPosition() {
    leftShoulder.write(INITIAL_POSITION);
    leftElbowHorizontal.write(INITIAL_POSITION);
    leftElbowVertical.write(INITIAL_POSITION);
    rightShoulder.write(INITIAL_POSITION);
    rightElbowHorizontal.write(INITIAL_POSITION);
    rightElbowVertical.write(INITIAL_POSITION);
    leftHip.write(INITIAL_POSITION);
    rightHip.write(INITIAL_POSITION);

    lastLeftShoulder = INITIAL_POSITION;
    lastLeftElbowHorizontal = INITIAL_POSITION;
    lastLeftElbowVertical = INITIAL_POSITION;
    lastRightShoulder = INITIAL_POSITION;
    lastRightElbowHorizontal = INITIAL_POSITION;
    lastRightElbowVertical = INITIAL_POSITION;
    lastLeftHip = INITIAL_POSITION;
    lastRightHip = INITIAL_POSITION;
    currentLeftShoulder = INITIAL_POSITION;
    currentLeftElbowHorizontal = INITIAL_POSITION;
    currentLeftElbowVertical = INITIAL_POSITION;
    currentRightShoulder = INITIAL_POSITION;
    currentRightElbowHorizontal = INITIAL_POSITION;
    currentRightElbowVertical = INITIAL_POSITION;
    currentLeftHip = INITIAL_POSITION;
    currentRightHip = INITIAL_POSITION;
}

// ======================================================
// VALIDATE SERVO FIELDS
// A field may be a finite number or JSON null.
// ======================================================
bool validServoField(JsonVariantConst value) {
    if (value.isNull()) {
        return true;
    }
    if (!value.is<float>() && !value.is<int>()) {
        return false;
    }
    return isfinite(value.as<float>());
}

bool hasValidServoData(JsonDocument& doc) {
    const char* keys[] = {
        "left_shoulder_vertical_deg",
        "left_wrist_horizontal_deg",
        "left_wrist_vertical_deg",
        "right_shoulder_vertical_deg",
        "right_wrist_horizontal_deg",
        "right_wrist_vertical_deg",
        "left_hip_deg",
        "right_hip_deg"
    };

    for (const char* key : keys) {
        if (!doc.containsKey(key) || !validServoField(doc[key])) {
            return false;
        }
    }
    return true;
}

// ======================================================
// PROCESS ONE JSON LINE
// ======================================================
void processJsonLine(char* jsonLine) {
    StaticJsonDocument<512> doc;
    DeserializationError error = deserializeJson(doc, jsonLine);

    if (error) {
        Serial.print("JSON error: ");
        Serial.println(error.c_str());
        return;
    }

    if (!hasValidServoData(doc)) {
        Serial.println("Missing or invalid servo fields");
        return;
    }

    if (!updateControlSignals(doc)) {
        Serial.println("Missing or invalid motor signals");
        return;
    }

    // LEFT ARM
    updateServo(doc, "left_shoulder_vertical_deg", lastLeftShoulder);
    updateServo(doc, "left_wrist_horizontal_deg", lastLeftElbowHorizontal);

    // LEFT ELBOW VERTICAL: REVERSED
    updateServo(doc, "left_wrist_vertical_deg", lastLeftElbowVertical, true);

    // RIGHT ARM
    // RIGHT SHOULDER: REVERSED
    updateServo(doc, "right_shoulder_vertical_deg", lastRightShoulder, true);
    updateServo(doc, "right_wrist_horizontal_deg", lastRightElbowHorizontal, true);
    updateServo(doc, "right_wrist_vertical_deg", lastRightElbowVertical);

    // HIPS
    updateServo(doc, "left_hip_deg", lastLeftHip);
    updateServo(doc, "left_hip_deg", lastRightHip, true);

    updateMotors();
    lastDataTime = millis();
    dataReceived = true;
}

// ======================================================
// READ SERIAL JSON LINES
// ======================================================
void readSerialLines() {
    while (Serial.available() > 0) {
        char receivedChar = (char)Serial.read();

        if (receivedChar == '\r') {
            continue;
        }

        if (receivedChar == '\n') {
            if (!discardLine && bufferIndex > 0) {
                inputBuffer[bufferIndex] = '\0';
                processJsonLine(inputBuffer);
            }
            bufferIndex = 0;
            discardLine = false;
            continue;
        }

        if (!discardLine) {
            if (bufferIndex < MAX_LINE_LENGTH - 1) {
                inputBuffer[bufferIndex++] = receivedChar;
            } else {
                discardLine = true;
                bufferIndex = 0;
                Serial.println("JSON line too long");
            }
        }
    }
}

// ======================================================
// CONNECTION TIMEOUT
// ======================================================
void checkConnectionTimeout() {
    if (dataReceived && millis() - lastDataTime > DATA_TIMEOUT_MS) {
        Serial.println("Connection lost. Stopping motors and resetting servos.");
        moveAllServosToInitialPosition();
        stopMotors();
        dataReceived = false;
    }
}

// ======================================================
// SETUP
// ======================================================
void setup() {
    Serial.begin(BAUD_RATE);

    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    ESP32PWM::allocateTimer(2);
    ESP32PWM::allocateTimer(3);

    leftShoulder.setPeriodHertz(50);
    leftElbowHorizontal.setPeriodHertz(50);
    leftElbowVertical.setPeriodHertz(50);
    rightShoulder.setPeriodHertz(50);
    rightElbowHorizontal.setPeriodHertz(50);
    rightElbowVertical.setPeriodHertz(50);
    leftHip.setPeriodHertz(50);
    rightHip.setPeriodHertz(50);

    leftShoulder.attach(LEFT_SHOULDER_PIN, SERVO_MIN_PULSE, SERVO_MAX_PULSE);
    leftElbowHorizontal.attach(LEFT_ELBOW_HORIZONTAL_PIN, SERVO_MIN_PULSE, SERVO_MAX_PULSE);
    leftElbowVertical.attach(LEFT_ELBOW_VERTICAL_PIN, SERVO_MIN_PULSE, SERVO_MAX_PULSE);
    rightShoulder.attach(RIGHT_SHOULDER_PIN, SERVO_MIN_PULSE, SERVO_MAX_PULSE);
    rightElbowHorizontal.attach(RIGHT_ELBOW_HORIZONTAL_PIN, SERVO_MIN_PULSE, SERVO_MAX_PULSE);
    rightElbowVertical.attach(RIGHT_ELBOW_VERTICAL_PIN, SERVO_MIN_PULSE, SERVO_MAX_PULSE);
    leftHip.attach(LEFT_HIP_PIN, SERVO_MIN_PULSE, SERVO_MAX_PULSE);
    rightHip.attach(RIGHT_HIP_PIN, SERVO_MIN_PULSE, SERVO_MAX_PULSE);

    pinMode(LEFT_MOTOR_EN, OUTPUT);
    pinMode(LEFT_MOTOR_IN1, OUTPUT);
    pinMode(LEFT_MOTOR_IN2, OUTPUT);
    pinMode(RIGHT_MOTOR_EN, OUTPUT);
    pinMode(RIGHT_MOTOR_IN1, OUTPUT);
    pinMode(RIGHT_MOTOR_IN2, OUTPUT);

    moveAllServosToInitialPosition();
    stopMotors();

    Serial.println("Humanoid Robot Controller Ready");
}

// ======================================================
// MAIN LOOP
// ======================================================
void loop() {
    readSerialLines();
    checkConnectionTimeout();
    updateServosSmoothly();
}
