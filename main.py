import json
import math
import time

import cv2
import mediapipe as mp
import serial

VISIBILITY_THRESHOLD = 0.6
BASELINE_CALIBRATION_FRAMES = 15
LEG_MOVEMENT_THRESHOLD = 0.18
SERIAL_PORT = "COM7"
SERIAL_BAUD_RATE = 115200
SERIAL_SEND_INTERVAL_SECONDS = 1 / 10
SERVO_ANGLE_STEP_DEGREES = 10
MOTOR_MIN_PWM = 80
MOTOR_MAX_PWM = 255
SERVO_MEASUREMENT_KEYS = (
    "left_shoulder_vertical_deg",
    "left_wrist_horizontal_deg",
    "left_wrist_vertical_deg",
    "right_shoulder_vertical_deg",
    "right_wrist_horizontal_deg",
    "right_wrist_vertical_deg",
    "left_hip_deg",
    "right_hip_deg",
)
ARMS = {
    "Left": {
        "shoulder": 11,
        "elbow": 13,
        "wrist": 15,
        "hip": 23,
    },
    "Right": {
        "shoulder": 12,
        "elbow": 14,
        "wrist": 16,
        "hip": 24,
    },
}
LEGS = {
    "Left": {
        "hip": 23,
        "knee": 25,
        "ankle": 27,
    },
    "Right": {
        "hip": 24,
        "knee": 26,
        "ankle": 28,
    },
}

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


def calculate_vertical_angle(start, end):
    """Return an image-plane angle: down 0, horizontal 90, up 180 degrees."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    if math.hypot(dx, dy) < 1e-6:
        return 90.0

    return math.degrees(math.atan2(abs(dx), dy))


def calculate_wrist_angles(elbow, wrist, inward_direction):
    """Return forearm horizontal and vertical angles relative to the body."""
    dx = wrist[0] - elbow[0]
    dy = wrist[1] - elbow[1]

    if math.hypot(dx, dy) < 1e-6:
        return 90.0, 90.0

    horizontal_angle = math.degrees(
        math.atan2(abs(dy), dx * inward_direction)
    )
    vertical_angle = math.degrees(math.atan2(abs(dx), dy))
    return horizontal_angle, vertical_angle


def calculate_vector_angle(reference, current):
    """Return the angle between two 3D vectors, or None for a zero vector."""
    reference_length = math.sqrt(sum(component**2 for component in reference))
    current_length = math.sqrt(sum(component**2 for component in current))
    if reference_length < 1e-6 or current_length < 1e-6:
        return None

    dot_product = sum(
        reference[index] * current[index] for index in range(3)
    )
    cosine = dot_product / (reference_length * current_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def calculate_face_scale(nose, left_ear, right_ear, frame_width, frame_height):
    """Return average nose-to-ear distance in image pixels."""
    ear_distances = []
    for ear in (left_ear, right_ear):
        dx = (nose.x - ear.x) * frame_width
        dy = (nose.y - ear.y) * frame_height
        ear_distances.append(math.hypot(dx, dy))

    return sum(ear_distances) / len(ear_distances)


def calculate_leg_relative_position(hip, knee, ankle):
    """Return the ankle position relative to the hip, scaled by leg length."""
    upper_leg_length = math.dist(
        (hip.x, hip.y, hip.z),
        (knee.x, knee.y, knee.z),
    )
    lower_leg_length = math.dist(
        (knee.x, knee.y, knee.z),
        (ankle.x, ankle.y, ankle.z),
    )
    leg_length = upper_leg_length + lower_leg_length
    if leg_length < 1e-6:
        return None

    return tuple(
        (getattr(ankle, axis) - getattr(hip, axis)) / leg_length
        for axis in ("x", "y", "z")
    )


def quantize_servo_angle(angle):
    """Round a servo command to the nearest 10 degrees."""
    return int(
        math.floor(angle / SERVO_ANGLE_STEP_DEGREES + 0.5)
        * SERVO_ANGLE_STEP_DEGREES
    )


serial_port = serial.Serial(SERIAL_PORT, SERIAL_BAUD_RATE, timeout=1)
cap = cv2.VideoCapture(0)
head_scale_baseline = None
torso_baselines = {side: None for side in ARMS}
torso_baseline_samples = {side: [] for side in ARMS}
leg_position_baselines = {side: None for side in LEGS}
leg_position_samples = {side: [] for side in LEGS}
leg_lift_baselines = {side: None for side in LEGS}
leg_lift_samples = {side: [] for side in LEGS}
last_serial_send_time = 0.0

try:
    if not cap.isOpened():
        raise RuntimeError("Could not open camera 0")

    with mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as pose:
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            measurements = {
                f"{side.lower()}_{angle_name}": None
                for side in ARMS
                for angle_name in (
                    "shoulder_vertical_deg",
                    "wrist_horizontal_deg",
                    "wrist_vertical_deg",
                    "torso_bend_deg",
                )
            }
            measurements.update(
                {
                    "head_scale_change_px": None,
                    "head_scale_change_percent": None,
                    "head_direction": None,
                    "left_hip_deg": None,
                    "right_hip_deg": None,
                }
            )
            motor_labels = ["Motor: leg pose not detected"]
            command_signals = {
                "right_turn": "LOW",
                "left_turn": "LOW",
                "forward": "LOW",
                "backward": "LOW",
            }
            movement_candidates = []
            knee_lift_candidates = []
            movement_strength = 0.0

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                height, width = frame.shape[:2]

                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

                world_landmarks = (
                    results.pose_world_landmarks.landmark
                    if results.pose_world_landmarks
                    else None
                )
                if world_landmarks is not None:
                    motor_labels = []
                    hip_midpoint_x = (
                        world_landmarks[LEGS["Left"]["hip"]].x
                        + world_landmarks[LEGS["Right"]["hip"]].x
                    ) / 2
                    for side, indices in LEGS.items():
                        leg_landmarks = tuple(
                            landmarks[indices[name]]
                            for name in ("hip", "knee", "ankle")
                        )
                        if not all(
                            landmark.visibility > VISIBILITY_THRESHOLD
                            for landmark in leg_landmarks
                        ):
                            motor_labels.append(
                                f"{side} leg not detected clearly"
                            )
                            continue

                        hip_image, knee_image, ankle_image = leg_landmarks
                        image_leg_length = (
                            math.dist(
                                (hip_image.x * width, hip_image.y * height),
                                (knee_image.x * width, knee_image.y * height),
                            )
                            + math.dist(
                                (knee_image.x * width, knee_image.y * height),
                                (ankle_image.x * width, ankle_image.y * height),
                            )
                        )
                        if image_leg_length < 1e-6:
                            motor_labels.append(
                                f"{side} leg position unavailable"
                            )
                            continue
                        current_lift_position = (
                            (knee_image.y - hip_image.y) * height
                            / image_leg_length
                        )

                        leg_world_landmarks = tuple(
                            world_landmarks[indices[name]]
                            for name in ("hip", "knee", "ankle")
                        )
                        current_position = calculate_leg_relative_position(
                            *leg_world_landmarks
                        )
                        if current_position is None:
                            motor_labels.append(f"{side} leg position unavailable")
                            continue

                        baseline = leg_position_baselines[side]
                        lift_baseline = leg_lift_baselines[side]
                        if baseline is None:
                            samples = leg_position_samples[side]
                            samples.append(current_position)
                            if len(samples) >= BASELINE_CALIBRATION_FRAMES:
                                baseline = tuple(
                                    sum(sample[axis] for sample in samples)
                                    / len(samples)
                                    for axis in range(3)
                                )
                                leg_position_baselines[side] = baseline

                        if lift_baseline is None:
                            samples = leg_lift_samples[side]
                            samples.append(current_lift_position)
                            if len(samples) >= BASELINE_CALIBRATION_FRAMES:
                                lift_baseline = sum(samples) / len(samples)
                                leg_lift_baselines[side] = lift_baseline

                        if baseline is None or lift_baseline is None:
                            motor_labels.append(
                                f"{side} leg: calibrating, stand still"
                            )
                            continue

                        delta = tuple(
                            current_position[axis] - baseline[axis]
                            for axis in range(3)
                        )
                        outward_direction = (
                            1
                            if world_landmarks[indices["hip"]].x
                            > hip_midpoint_x
                            else -1
                        )
                        if (
                            delta[0] * outward_direction
                            >= LEG_MOVEMENT_THRESHOLD
                        ):
                            movement_strength = max(
                                movement_strength,
                                delta[0] * outward_direction,
                            )
                            if side == "Right":
                                command_signals["right_turn"] = "HIGH"
                                motor_labels.append("Right turn")
                            else:
                                command_signals["left_turn"] = "HIGH"
                                motor_labels.append("Left turn")

                        lifted = current_lift_position <= (
                            lift_baseline - LEG_MOVEMENT_THRESHOLD
                        )
                        if lifted:
                            direction = (
                                "forward" if side == "Right" else "backward"
                            )
                            lift_strength = lift_baseline - current_lift_position
                            movement_strength = max(
                                movement_strength,
                                lift_strength,
                            )
                            knee_lift_candidates.append(
                                (direction, lift_strength)
                            )
                            motor_labels.append(
                                f"{side} knee up: Motor {direction.upper()}"
                            )
                        elif delta[2] <= -LEG_MOVEMENT_THRESHOLD:
                            movement_strength = max(
                                movement_strength, -delta[2]
                            )
                            movement_candidates.append(
                                ("forward", -delta[2])
                            )
                            motor_labels.append(f"{side} leg: Motor FORWARD")
                        elif delta[2] >= LEG_MOVEMENT_THRESHOLD:
                            movement_strength = max(
                                movement_strength, delta[2]
                            )
                            movement_candidates.append(
                                ("backward", delta[2])
                            )
                            motor_labels.append(f"{side} leg: Motor BACKWARD")

                    direction_candidates = (
                        knee_lift_candidates or movement_candidates
                    )
                    if direction_candidates:
                        direction, _ = max(
                            direction_candidates,
                            key=lambda candidate: candidate[1],
                        )
                        command_signals[direction] = "HIGH"

                if world_landmarks is not None and not motor_labels:
                    motor_labels = ["Motor: STOP"]
                elif world_landmarks is None:
                    motor_labels = ["Motor: 3D leg tracking unavailable"]

                left_shoulder_x = landmarks[ARMS["Left"]["shoulder"]].x * width
                right_shoulder_x = landmarks[ARMS["Right"]["shoulder"]].x * width
                body_midpoint_x = (left_shoulder_x + right_shoulder_x) / 2

                for row, (side, indices) in enumerate(ARMS.items()):
                    shoulder = landmarks[indices["shoulder"]]
                    elbow = landmarks[indices["elbow"]]
                    shoulder_visible = all(
                        landmark.visibility > VISIBILITY_THRESHOLD
                        for landmark in (shoulder, elbow)
                    )

                    shoulder_point = (shoulder.x * width, shoulder.y * height)
                    elbow_point = (elbow.x * width, elbow.y * height)
                    if shoulder_visible:
                        shoulder_vertical_angle = calculate_vertical_angle(
                            shoulder_point, elbow_point
                        )
                        measurements[
                            f"{side.lower()}_shoulder_vertical_deg"
                        ] = shoulder_vertical_angle
                        shoulder_label = (
                            f"{side} shoulder vertical: "
                            f"{shoulder_vertical_angle:.1f} deg"
                        )
                        shoulder_color = (0, 255, 0)
                    else:
                        shoulder_label = f"{side} shoulder not detected clearly"
                        shoulder_color = (0, 0, 255)

                    wrist = landmarks[indices["wrist"]]
                    wrist_visible = all(
                        landmark.visibility > VISIBILITY_THRESHOLD
                        for landmark in (elbow, wrist)
                    )
                    if wrist_visible:
                        wrist_point = (wrist.x * width, wrist.y * height)
                        inward_direction = (
                            1 if body_midpoint_x >= shoulder_point[0] else -1
                        )
                        wrist_horizontal_angle, wrist_vertical_angle = (
                            calculate_wrist_angles(
                                elbow_point,
                                wrist_point,
                                inward_direction,
                            )
                        )
                        measurements[f"{side.lower()}_wrist_horizontal_deg"] = (
                            wrist_horizontal_angle
                        )
                        measurements[f"{side.lower()}_wrist_vertical_deg"] = (
                            wrist_vertical_angle
                        )
                        wrist_label = (
                            f"{side} wrist horizontal: "
                            f"{wrist_horizontal_angle:.1f} deg"
                            f" | vertical: {wrist_vertical_angle:.1f} deg"
                        )
                        wrist_color = (0, 255, 0)
                    else:
                        wrist_label = f"{side} wrist not detected clearly"
                        wrist_color = (0, 0, 255)

                    hip = landmarks[indices["hip"]]
                    torso_visible = all(
                        landmark.visibility > VISIBILITY_THRESHOLD
                        for landmark in (shoulder, hip)
                    ) and world_landmarks is not None
                    if torso_visible:
                        shoulder_world = world_landmarks[indices["shoulder"]]
                        hip_world = world_landmarks[indices["hip"]]
                        torso_vector = (
                            shoulder_world.x - hip_world.x,
                            shoulder_world.y - hip_world.y,
                            shoulder_world.z - hip_world.z,
                        )
                        if torso_baselines[side] is None:
                            samples = torso_baseline_samples[side]
                            samples.append(torso_vector)
                            if len(samples) >= BASELINE_CALIBRATION_FRAMES:
                                torso_baselines[side] = tuple(
                                    sum(sample[i] for sample in samples)
                                    / len(samples)
                                    for i in range(3)
                                )

                        if torso_baselines[side] is None:
                            hip_label = f"{side} torso: calibrating upright"
                        else:
                            torso_bend = calculate_vector_angle(
                                torso_baselines[side],
                                torso_vector,
                            )
                            measurements[f"{side.lower()}_torso_bend_deg"] = (
                                torso_bend
                            )
                            if torso_bend is not None:
                                measurements[f"{side.lower()}_hip_deg"] = (
                                    min(180.0, 90.0 + torso_bend)
                                )
                            hip_label = (
                                f"{side} torso bend: {torso_bend:.1f} deg"
                                if torso_bend is not None
                                else f"{side} torso bend: unavailable"
                            )
                    else:
                        hip_label = f"{side} torso not detected in 3D"

                    cv2.putText(
                        frame,
                        shoulder_label,
                        (20, 30 + row * 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        shoulder_color,
                        2,
                    )
                    cv2.putText(
                        frame,
                        wrist_label,
                        (20, 55 + row * 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        wrist_color,
                        2,
                    )
                    cv2.putText(
                        frame,
                        hip_label,
                        (20, 80 + row * 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 255, 255)
                        if torso_visible and torso_baselines[side] is not None
                        else (0, 0, 255),
                        2,
                    )

                nose = landmarks[0]
                left_ear = landmarks[7]
                right_ear = landmarks[8]
                if all(
                    landmark.visibility > VISIBILITY_THRESHOLD
                    for landmark in (nose, left_ear, right_ear)
                ):
                    height, width = frame.shape[:2]
                    head_scale = calculate_face_scale(
                        nose,
                        left_ear,
                        right_ear,
                        width,
                        height,
                    )
                    if head_scale_baseline is None:
                        head_scale_baseline = head_scale

                    pixel_change = head_scale - head_scale_baseline
                    percent_change = (
                        pixel_change / head_scale_baseline * 100
                        if head_scale_baseline > 1e-6
                        else 0.0
                    )
                    if abs(pixel_change) < 1:
                        head_label = "Head near baseline: 0.0 px (0.0%)"
                        direction = "baseline"
                    else:
                        direction = "closer" if pixel_change > 0 else "farther"
                        head_label = (
                            f"Head {direction}: {abs(pixel_change):.1f} px "
                            f"({percent_change:.1f}%)"
                        )
                    measurements["head_scale_change_px"] = pixel_change
                    measurements["head_scale_change_percent"] = percent_change
                    measurements["head_direction"] = direction
                else:
                    head_label = "Head movement: face not detected clearly"
            else:
                head_label = "Head movement: face not detected clearly"

            current_time = time.monotonic()
            if (
                current_time - last_serial_send_time
                >= SERIAL_SEND_INTERVAL_SECONDS
            ):
                servo_measurements = {
                    key: (
                        quantize_servo_angle(measurements[key])
                        if measurements[key] is not None
                        else None
                    )
                    for key in SERVO_MEASUREMENT_KEYS
                }
                serial_port.write(
                    (
                        json.dumps(
                            {
                                **servo_measurements,
                                **command_signals,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                )
                last_serial_send_time = current_time

            cv2.putText(
                frame,
                head_label,
                (20, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )

            signal_labels = (
                ("RIGHT TURN", command_signals["right_turn"]),
                ("LEFT TURN", command_signals["left_turn"]),
                ("FORWARD", command_signals["forward"]),
                ("BACKWARD", command_signals["backward"]),
            )
            for index, (signal_name, signal_value) in enumerate(signal_labels):
                cv2.putText(
                    frame,
                    f"{signal_name}: {signal_value}",
                    (20, 240 + index * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0) if signal_value == "HIGH" else (0, 0, 255),
                    2,
                )

            for index, motor_label in enumerate(motor_labels):
                cv2.putText(
                    frame,
                    motor_label,
                    (20, 375 + index * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0)
                    if "FORWARD" in motor_label
                    else (0, 0, 255)
                    if "BACKWARD" in motor_label
                    else (255, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                "Q: Quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Both Arm Angles and Head Movement", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
finally:
    cap.release()
    cv2.destroyAllWindows()
    serial_port.close()
