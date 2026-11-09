# Arm Angles, Leg Movement, and Head Movement

This program uses webcam video and MediaPipe Pose to track both arms. For each
arm, it displays only the vertical angle from shoulder to elbow: down is 0,
horizontal is 90, and up is 180 degrees.

It also displays horizontal and vertical angles for the forearm, measured
from elbow to wrist. Before movement tracking starts, stand still with both
shoulders, elbows, wrists, and hips visible. The program waits until all eight
servo measurements are available together, then averages 15 consecutive clear
frames for calibration. If any required measurement is missing, calibration
progress resets and the display asks you to stand still and make the required
landmarks visible. The calibrated pose maps to 90 degrees; subsequent shoulder
and forearm angle changes are measured relative to that pose and clamped to
0-180 degrees. If the elbow and wrist overlap, both forearm angles are 90
degrees. These angles are 2D estimates from the camera view.

It also estimates torso bend for each side using the 3D shoulder and hip
landmarks. The torso baseline is captured during the same 15-frame servo
calibration. The display reads near 0 degrees at the calibrated pose and
increases as the torso bends, including forward/backward movement that may be
missed by a 2D image angle. Keep the shoulder and hip visible during
calibration and movement.

It also averages the image-pixel distances from the nose to both ears, then
compares that apparent face scale with the first clearly detected frame. The
display reports the change in pixels and as a percentage, with a larger scale
indicating that the head likely moved closer. This is an estimate, not a
physical distance measurement; head rotation and camera perspective affect
the result, and it cannot reliably report centimeters or meters without
calibration. Face the camera for the most consistent estimate.

The display also tracks both legs. Stand still with both legs visible during
the first 15 clear frames to calibrate the neutral position. Lifting the
right knee signals forward, and lifting the left knee signals backward. If no
knee lift is detected, forward or backward movement is detected in estimated
3D depth. Moving a leg outward signals a turn to that side.
Movements are detected relative to the calibrated pose, with a threshold of
18% of the leg length. The camera overlay shows `RIGHT TURN`, `LEFT TURN`,
`FORWARD`, and `BACKWARD` as `HIGH` or `LOW`.

The program sends a newline-delimited JSON object to the ESP over USB serial
at `115200` baud, limited to 10 updates per second. Servo values are sent as
`null` until calibration completes; after that, unavailable measurements are
sent as `null` so the ESP32 holds the previous servo target. Update `SERIAL_PORT` in
`main.py` to the COM port Windows assigns to the ESP. Each message contains
`left_shoulder_vertical_deg`, `left_wrist_horizontal_deg`,
`left_wrist_vertical_deg`,
`right_shoulder_vertical_deg`, `right_wrist_horizontal_deg`,
`right_wrist_vertical_deg`, `left_hip_deg`, and `right_hip_deg`. Values are
`null` when their landmarks are not visible. Hip-servo angles are based on
torso bend, with upright mapped to 90 degrees and bend increasing toward 180.
Each message also includes `right_turn`, `left_turn`, `forward`, and
`backward` as `HIGH` or `LOW`. The motors run at full speed when a valid
movement signal is active; PWM speed control is not used.
Torso-bend and head-movement readings remain on screen but are not sent
separately.

For an ESP32 using Arduino IDE, install the **ArduinoJson** and **ESP32Servo**
libraries and upload [`esp32_receiver.ino`](./esp32_receiver.ino). It uses
eight servos: shoulder and forearm servos for both arms, plus one hip servo per
side. Connect the hip servos to GPIO 26 and 27. Hip servos use 90 degrees as
upright and follow the torso-bend estimate.

The sketch also controls two DC motors through an L298N. Connect ENA to GPIO
16, IN1/IN2 to GPIO 13/14; connect ENB to GPIO 17, IN3/IN4 to GPIO 32/33.
The enable pins are switched fully on/off, so active motor commands run at
full speed.
Connect the ESP32 ground and L298N ground together, power the motors from the
driver's motor supply, and do not power motors from the ESP32. Forward and
backward drive both motors in the same direction. A right turn drives only
the right motor forward; a left turn drives only the left motor forward.
Forward takes priority over backward, and both take priority over turns if
multiple signals are active. Verify GPIO availability for your exact ESP32
board and test with the wheels lifted before placing the robot on the ground.

Servo commands map to the 0-180 degree range in 10-degree steps. Servos move
toward their latest targets by at most 10 degrees every 15 ms without blocking
serial or motor updates. The firmware returns servos to 90 degrees and stops
the motors if serial data stops for 500 ms. Actual servo travel depends on the
servo's supported range and pulse-width calibration; check that the mechanism
does not bind at either endpoint before using the full range.

Install the Python dependencies with `uv sync`, close any serial monitor using
the ESP's COM port, then run `uv run main.py`. Check in Windows Device Manager
that the ESP32 is assigned `COM3`, or update `SERIAL_PORT` in `main.py`. The
serial port can only be opened by one program at a time.

Press **Q** to close the camera window.
