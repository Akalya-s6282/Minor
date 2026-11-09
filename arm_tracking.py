import json
import math
import statistics
from pathlib import Path

VISIBILITY_THRESHOLD = 0.6
CALIBRATION_FRAME_COUNT = 20
CALIBRATION_VERSION = 1
CALIBRATION_FILE = Path(__file__).with_name("arm_calibration.json")
SMOOTHING_ALPHA = 0.35

ARMS = {
    "left": {"shoulder": 11, "elbow": 13, "wrist": 15, "outward": -1},
    "right": {"shoulder": 12, "elbow": 14, "wrist": 16, "outward": 1},
}
REQUIRED_LANDMARKS = (11, 12, 13, 14, 15, 16, 23, 24)

CALIBRATION_STAGES = (
    ("neutral", "Stand straight; let both nearly straight arms hang naturally."),
    ("shoulder_mid", "Raise both straight arms sideways about halfway to shoulder height."),
    ("shoulder_horizontal", "Hold both arms straight out sideways at shoulder height."),
    ("elbow_intermediate", "Keep upper arms horizontal; bend both elbows about halfway."),
    ("elbow_bent", "Keep upper arms horizontal; bend both elbows as far as comfortable."),
    ("elbow_horizontal_neutral", "Keep upper arms horizontal; bend elbows about 90 degrees, forearms down."),
    ("elbow_cross_body", "Keep the same pose; move both forearms inward across your body."),
    ("elbow_outward", "Keep the same pose; move both forearms outward, away from your body."),
)

SERVO_CONFIG = {
    f"{side}_{axis}": {
        "minimum": 0.0,
        "maximum": 180.0,
        "neutral": 90.0,
        "inverted": False,
    }
    for side in ARMS
    for axis in ("shoulder_vertical", "elbow_vertical", "elbow_horizontal")
}

SERVO_FIELDS = {
    "left_shoulder_vertical": "left_shoulder_vertical_deg",
    "left_elbow_horizontal": "left_wrist_horizontal_deg",
    "left_elbow_vertical": "left_wrist_vertical_deg",
    "right_shoulder_vertical": "right_shoulder_vertical_deg",
    "right_elbow_horizontal": "right_wrist_horizontal_deg",
    "right_elbow_vertical": "right_wrist_vertical_deg",
}
def _angle_between(first, second):
    first_length = math.hypot(*first)
    second_length = math.hypot(*second)
    if first_length < 1e-8 or second_length < 1e-8:
        return None
    cosine = (
        first[0] * second[0] + first[1] * second[1]
    ) / (first_length * second_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _signed_angle(first, second):
    first_length = math.hypot(*first)
    second_length = math.hypot(*second)
    if first_length < 1e-8 or second_length < 1e-8:
        return None
    cross = first[0] * second[1] - first[1] * second[0]
    dot = first[0] * second[0] + first[1] * second[1]
    return math.degrees(math.atan2(cross, dot))


def _unwrap_degrees(value, reference):
    return reference + (value - reference + 180.0) % 360.0 - 180.0


def extract_pose(landmarks, frame_width, frame_height):
    """Build torso-normalized 2D body coordinates and arm measurements."""
    if landmarks is None or len(landmarks) <= max(REQUIRED_LANDMARKS):
        return None
    if any(
        not math.isfinite(landmarks[index].visibility)
        or landmarks[index].visibility < VISIBILITY_THRESHOLD
        for index in REQUIRED_LANDMARKS
    ):
        return None

    image_points = {
        index: (
            float(landmarks[index].x) * frame_width,
            float(landmarks[index].y) * frame_height,
        )
        for index in REQUIRED_LANDMARKS
    }
    left_shoulder = image_points[11]
    right_shoulder = image_points[12]
    shoulder_center = (
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2,
    )
    hip_center = (
        (image_points[23][0] + image_points[24][0]) / 2,
        (image_points[23][1] + image_points[24][1]) / 2,
    )
    body_up = (
        shoulder_center[0] - hip_center[0],
        shoulder_center[1] - hip_center[1],
    )
    torso_length = math.hypot(*body_up)
    if not math.isfinite(torso_length) or torso_length < 1.0:
        return None
    body_up = (body_up[0] / torso_length, body_up[1] / torso_length)

    shoulder_axis = (
        right_shoulder[0] - left_shoulder[0],
        right_shoulder[1] - left_shoulder[1],
    )
    lateral = (-body_up[1], body_up[0])
    if lateral[0] * shoulder_axis[0] + lateral[1] * shoulder_axis[1] < 0:
        lateral = (-lateral[0], -lateral[1])
    lateral_length = math.hypot(*lateral)
    lateral = (lateral[0] / lateral_length, lateral[1] / lateral_length)

    points = {}
    for index, point in image_points.items():
        offset = (point[0] - shoulder_center[0], point[1] - shoulder_center[1])
        points[index] = (
            (offset[0] * lateral[0] + offset[1] * lateral[1]) / torso_length,
            (offset[0] * body_up[0] + offset[1] * body_up[1]) / torso_length,
        )

    features = {}
    for side, indices in ARMS.items():
        shoulder = points[indices["shoulder"]]
        elbow = points[indices["elbow"]]
        wrist = points[indices["wrist"]]
        upper = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
        upper_length = math.hypot(*upper)
        forearm = (wrist[0] - elbow[0], wrist[1] - elbow[1])
        forearm_length = math.hypot(*forearm)
        if (
            not math.isfinite(upper_length)
            or not math.isfinite(forearm_length)
            or upper_length < 0.06
            or forearm_length < 0.06
            or upper_length > 1.5
            or forearm_length > 1.5
        ):
            return None

        shoulder_elevation = math.degrees(
            math.atan2(
                indices["outward"] * upper[0],
                -upper[1],
            )
        )
        shoulder_elevation = max(0.0, min(180.0, shoulder_elevation))
        internal_angle = _angle_between(
            (shoulder[0] - elbow[0], shoulder[1] - elbow[1]),
            forearm,
        )
        horizontal_angle = _signed_angle(
            (shoulder[0] - elbow[0], shoulder[1] - elbow[1]),
            forearm,
        )
        if internal_angle is None or horizontal_angle is None:
            return None
        features[side] = {
            "shoulder_vertical": shoulder_elevation,
            "elbow_vertical": 180.0 - internal_angle,
            "elbow_horizontal": horizontal_angle,
            "upper_length": upper_length,
            "forearm_length": forearm_length,
        }

    return {
        "points": {str(index): list(point) for index, point in points.items()},
        "features": features,
        "torso_length_px": torso_length,
    }


def _mean(values):
    return statistics.fmean(values)


def _circular_mean_degrees(values):
    sine = _mean(math.sin(math.radians(value)) for value in values)
    cosine = _mean(math.cos(math.radians(value)) for value in values)
    return math.degrees(math.atan2(sine, cosine))


def _mean_stage(samples):
    stage = {"points": {}, "features": {}, "torso_length_px": 0.0}
    point_keys = samples[0]["points"]
    for key in point_keys:
        stage["points"][key] = [
            _mean(sample["points"][key][axis] for sample in samples)
            for axis in range(2)
        ]
    for side in ARMS:
        stage["features"][side] = {}
        for feature in samples[0]["features"][side]:
            values = [
                sample["features"][side][feature] for sample in samples
            ]
            stage["features"][side][feature] = (
                _circular_mean_degrees(values)
                if feature == "elbow_horizontal"
                else _mean(values)
            )
    stage["torso_length_px"] = _mean(
        sample["torso_length_px"] for sample in samples
    )
    return stage


def _capture_is_stable(samples):
    for side in ARMS:
        for feature, limit in (
            ("shoulder_vertical", 12.0),
            ("elbow_vertical", 12.0),
            ("elbow_horizontal", 15.0),
        ):
            values = [sample["features"][side][feature] for sample in samples]
            if feature == "elbow_horizontal":
                reference = values[0]
                values = [_unwrap_degrees(value, reference) for value in values]
            if statistics.pstdev(values) > limit:
                return False
    return True


def _validate_stage(stage_name, stage):
    for side in ARMS:
        features = stage["features"][side]
        shoulder = features["shoulder_vertical"]
        flexion = features["elbow_vertical"]
        if stage_name == "shoulder_mid" and not 20 <= shoulder <= 75:
            return f"{side.title()} arm is not at an intermediate raised position."
        if stage_name == "shoulder_horizontal" and not 65 <= shoulder <= 115:
            return f"{side.title()} arm should be extended near shoulder height."
        if stage_name == "elbow_intermediate" and not 20 <= flexion <= 135:
            return f"{side.title()} elbow bend is not intermediate."
        if stage_name == "elbow_bent" and flexion < 40:
            return f"{side.title()} elbow needs a comfortable visible bend."

    if stage_name in ("elbow_cross_body", "elbow_outward"):
        neutral = stage["features"]
        if any(
            neutral[side]["elbow_vertical"] < 25
            or neutral[side]["elbow_vertical"] > 155
            for side in ARMS
        ):
            return "Keep both elbows bent while calibrating horizontal movement."
    return None


def _map_piecewise(value, anchors):
    ordered = sorted(anchors, key=lambda anchor: anchor[0])
    if value <= ordered[0][0]:
        return ordered[0][1]
    if value >= ordered[-1][0]:
        return ordered[-1][1]
    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        if x0 <= value <= x1:
            if abs(x1 - x0) < 1e-8:
                return (y0 + y1) / 2
            portion = (value - x0) / (x1 - x0)
            return y0 + portion * (y1 - y0)
    return ordered[-1][1]


def _validate_servo_config(config):
    if set(config) != set(SERVO_CONFIG):
        raise ValueError("Calibration has an incomplete servo configuration.")
    validated = {}
    for name in SERVO_CONFIG:
        entry = config[name]
        minimum = float(entry["minimum"])
        maximum = float(entry["maximum"])
        neutral = float(entry["neutral"])
        inverted = entry["inverted"]
        if (
            not all(math.isfinite(value) for value in (minimum, maximum, neutral))
            or minimum < 0
            or maximum > 180
            or minimum >= maximum
            or not minimum <= neutral <= maximum
            or not isinstance(inverted, bool)
        ):
            raise ValueError(f"Invalid servo limits for {name}.")
        validated[name] = {
            "minimum": minimum,
            "maximum": maximum,
            "neutral": neutral,
            "inverted": inverted,
        }
    return validated


def _validate_saved_calibration(data, default_config=None):
    if data["version"] != CALIBRATION_VERSION:
        raise ValueError("Calibration file version is not supported.")
    config = _validate_servo_config(
        data.get("servo_config", default_config or SERVO_CONFIG)
    )
    stages = data["stages"]
    expected_stages = {name for name, _ in CALIBRATION_STAGES}
    if set(stages) != expected_stages:
        raise ValueError("Calibration file does not contain all pose stages.")
    for stage_name, stage in stages.items():
        if set(stage["features"]) != set(ARMS):
            raise ValueError(f"Missing arm data in calibration stage {stage_name}.")
        if set(stage["points"]) != {str(index) for index in REQUIRED_LANDMARKS}:
            raise ValueError(f"Missing landmarks in calibration stage {stage_name}.")
        for side in ARMS:
            if set(stage["features"][side]) != {
                "shoulder_vertical",
                "elbow_vertical",
                "elbow_horizontal",
                "upper_length",
                "forearm_length",
            }:
                raise ValueError(f"Invalid arm features in stage {stage_name}.")
            for value in stage["features"][side].values():
                if not math.isfinite(float(value)):
                    raise ValueError(f"Non-finite arm value in stage {stage_name}.")
        for point in stage["points"].values():
            if len(point) != 2 or not all(math.isfinite(float(v)) for v in point):
                raise ValueError(f"Invalid landmark coordinate in stage {stage_name}.")
    for side in ARMS:
        neutral = stages["neutral"]["features"][side]
        if any(
            not 0.08 <= neutral[key] <= 1.5
            for key in ("upper_length", "forearm_length")
        ):
            raise ValueError(f"Invalid calibrated arm length for {side}.")
        shoulder_values = [
            stages[name]["features"][side]["shoulder_vertical"]
            for name in (
                "neutral",
                "shoulder_mid",
                "shoulder_horizontal",
            )
        ]
        if not shoulder_values[0] + 8 < shoulder_values[1] < shoulder_values[2] - 8:
            raise ValueError(f"Invalid shoulder calibration order for {side}.")
        elbow_values = [
            stages["shoulder_horizontal"]["features"][side]["elbow_vertical"],
            stages["elbow_intermediate"]["features"][side]["elbow_vertical"],
            stages["elbow_bent"]["features"][side]["elbow_vertical"],
        ]
        if not elbow_values[0] + 8 < elbow_values[1] < elbow_values[2] - 8:
            raise ValueError(f"Invalid elbow calibration order for {side}.")
        horizontal_neutral = stages["elbow_horizontal_neutral"]["features"][side][
            "elbow_horizontal"
        ]
        horizontal_cross = _unwrap_degrees(
            stages["elbow_cross_body"]["features"][side]["elbow_horizontal"],
            horizontal_neutral,
        )
        horizontal_outward = _unwrap_degrees(
            stages["elbow_outward"]["features"][side]["elbow_horizontal"],
            horizontal_neutral,
        )
        if (
            abs(horizontal_cross - horizontal_neutral) < 10
            or abs(horizontal_outward - horizontal_neutral) < 10
            or (horizontal_cross - horizontal_neutral)
            * (horizontal_outward - horizontal_neutral)
            >= 0
        ):
            raise ValueError(f"Invalid horizontal calibration range for {side}.")
    return {"version": CALIBRATION_VERSION, "stages": stages, "servo_config": config}


class ArmTracker:
    def __init__(self, calibration_path=CALIBRATION_FILE, servo_config=None):
        self.calibration_path = Path(calibration_path)
        self.servo_config = _validate_servo_config(
            servo_config or SERVO_CONFIG
        )
        self.calibration = None
        self.smoothed_outputs = {}
        self.reset_calibration()

    def load(self):
        with self.calibration_path.open(encoding="utf-8") as calibration_file:
            saved = json.load(calibration_file)
        self.calibration = _validate_saved_calibration(
            saved,
            self.servo_config,
        )
        self.servo_config = self.calibration["servo_config"]
        self.stage_index = len(CALIBRATION_STAGES)
        self.calibration_samples = []
        self.capture_active = False
        self.captured_stages = {}
        self.smoothed_outputs.clear()

    def save(self):
        if self.calibration is None:
            raise RuntimeError("Cannot save calibration before all poses are captured.")
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.calibration_path.with_suffix(
            self.calibration_path.suffix + ".tmp"
        )
        temporary_path.write_text(
            json.dumps(self.calibration, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.calibration_path)

    def reset_calibration(self):
        self.calibration = None
        self.stage_index = 0
        self.calibration_samples = []
        self.capture_active = False
        self.captured_stages = {}
        self.smoothed_outputs.clear()

    @property
    def calibrated(self):
        return self.calibration is not None

    @property
    def current_stage(self):
        if self.calibrated or self.stage_index >= len(CALIBRATION_STAGES):
            return None
        return CALIBRATION_STAGES[self.stage_index]

    def begin_capture(self):
        if self.calibrated or self.capture_active:
            return
        self.calibration_samples.clear()
        self.capture_active = True

    def _finish_stage(self):
        if not _capture_is_stable(self.calibration_samples):
            self.calibration_samples.clear()
            self.capture_active = False
            return "Pose moved during capture; hold still and press Space again."

        stage_name = CALIBRATION_STAGES[self.stage_index][0]
        stage = _mean_stage(self.calibration_samples)
        error = _validate_stage(stage_name, stage)
        if error:
            self.calibration_samples.clear()
            self.capture_active = False
            return f"{error} Press Space to recapture this pose."

        self.captured_stages[stage_name] = stage
        self.stage_index += 1
        self.calibration_samples.clear()
        self.capture_active = False
        if self.stage_index == len(CALIBRATION_STAGES):
            try:
                self._complete_calibration()
            except ValueError as error:
                self.reset_calibration()
                return (
                    f"Calibration rejected: {error} Start again from neutral; "
                    "press Space to capture."
                )
            return "Calibration complete and saved."
        return f"Saved {stage_name.replace('_', ' ')} pose."

    def _complete_calibration(self):
        for side in ARMS:
            stages = self.captured_stages
            shoulder_values = [
                stages[name]["features"][side]["shoulder_vertical"]
                for name in (
                    "neutral",
                    "shoulder_mid",
                    "shoulder_horizontal",
                )
            ]
            if not (
                shoulder_values[0] + 8
                < shoulder_values[1]
                < shoulder_values[2] - 8
            ):
                raise ValueError(
                    f"{side.title()} shoulder calibration positions are not "
                    "in increasing lift order. Restart with R."
                )

            straight_flexion = stages["shoulder_horizontal"]["features"][side][
                "elbow_vertical"
            ]
            elbow_values = [
                straight_flexion,
                stages["elbow_intermediate"]["features"][side][
                    "elbow_vertical"
                ],
                stages["elbow_bent"]["features"][side]["elbow_vertical"],
            ]
            if not (
                elbow_values[0] + 8
                < elbow_values[1]
                < elbow_values[2] - 8
            ):
                raise ValueError(
                    f"{side.title()} elbow calibration bends must increase "
                    "from straight to fully bent. Restart with R."
                )

            horizontal_neutral = stages["elbow_horizontal_neutral"][
                "features"
            ][side]["elbow_horizontal"]
            horizontal_cross = _unwrap_degrees(
                stages["elbow_cross_body"]["features"][side][
                    "elbow_horizontal"
                ],
                horizontal_neutral,
            )
            horizontal_outward = _unwrap_degrees(
                stages["elbow_outward"]["features"][side][
                    "elbow_horizontal"
                ],
                horizontal_neutral,
            )
            if (
                abs(horizontal_cross - horizontal_neutral) < 10
                or abs(horizontal_outward - horizontal_neutral) < 10
                or (horizontal_cross - horizontal_neutral)
                * (horizontal_outward - horizontal_neutral)
                >= 0
            ):
                raise ValueError(
                    f"{side.title()} horizontal calibration needs distinct "
                    "cross-body and outward positions. Restart with R."
                )

        neutral = self.captured_stages["neutral"]
        if any(
            not 0.08 <= neutral["features"][side][segment] <= 1.5
            for side in ARMS
            for segment in ("upper_length", "forearm_length")
        ):
            raise ValueError("Calibrated arm segment lengths are invalid.")
        self.calibration = {
            "version": CALIBRATION_VERSION,
            "stages": self.captured_stages.copy(),
            "servo_config": self.servo_config,
        }
        self.save()

    def process(self, landmarks, frame_width, frame_height):
        pose = extract_pose(landmarks, frame_width, frame_height)
        if not self.calibrated:
            if self.capture_active:
                if pose is None:
                    self.calibration_samples.clear()
                    self.capture_active = False
                    status = (
                        "Landmarks lost; capture stopped. Show both arms and hips, "
                        "then press Space again."
                    )
                else:
                    self.calibration_samples.append(pose)
                    status = (
                        f"Capturing {self.current_stage[0].replace('_', ' ')}: "
                        f"{len(self.calibration_samples)}/{CALIBRATION_FRAME_COUNT}"
                    )
                    if len(self.calibration_samples) >= CALIBRATION_FRAME_COUNT:
                        status = self._finish_stage()
                return None, pose, status

            if pose is None:
                return None, None, "Show both shoulders, elbows, wrists, and hips."
            if self.current_stage is None:
                return None, pose, "Calibration state is invalid; press R to restart."
            return (
                None,
                pose,
                f"Calibration {self.stage_index + 1}/{len(CALIBRATION_STAGES)}: "
                f"{self.current_stage[1]} Press Space to capture.",
            )

        if pose is None:
            return None, None, "Tracking lost: hold last servo targets."
        outputs = self._map_pose(pose)
        for field, value in outputs.items():
            previous = self.smoothed_outputs.get(field, value)
            self.smoothed_outputs[field] = (
                SMOOTHING_ALPHA * value
                + (1.0 - SMOOTHING_ALPHA) * previous
            )
        return self.smoothed_outputs.copy(), pose, "Tracking"

    def _map_pose(self, pose):
        stages = self.calibration["stages"]
        mapped = {}
        for side in ARMS:
            features = pose["features"][side]
            neutral = stages["neutral"]["features"][side]
            shoulder_mid = stages["shoulder_mid"]["features"][side]
            shoulder_horizontal = stages["shoulder_horizontal"]["features"][side]
            elbow_mid = stages["elbow_intermediate"]["features"][side]
            elbow_bent = stages["elbow_bent"]["features"][side]
            horizontal_neutral = stages["elbow_horizontal_neutral"]["features"][side]
            horizontal_cross = stages["elbow_cross_body"]["features"][side]
            horizontal_out = stages["elbow_outward"]["features"][side]

            shoulder_config = self.servo_config[f"{side}_shoulder_vertical"]
            shoulder_end = (
                shoulder_config["minimum"]
                if shoulder_config["inverted"]
                else shoulder_config["maximum"]
            )
            mapped[f"{side}_shoulder_vertical"] = _map_piecewise(
                features["shoulder_vertical"],
                (
                    (neutral["shoulder_vertical"], shoulder_config["neutral"]),
                    (
                        shoulder_mid["shoulder_vertical"],
                        (shoulder_config["neutral"] + shoulder_end) / 2,
                    ),
                    (shoulder_horizontal["shoulder_vertical"], shoulder_end),
                ),
            )

            elbow_config = self.servo_config[f"{side}_elbow_vertical"]
            elbow_end = (
                elbow_config["minimum"]
                if elbow_config["inverted"]
                else elbow_config["maximum"]
            )
            mapped[f"{side}_elbow_vertical"] = _map_piecewise(
                features["elbow_vertical"],
                (
                    (stages["shoulder_horizontal"]["features"][side]["elbow_vertical"],
                     elbow_config["neutral"]),
                    (
                        elbow_mid["elbow_vertical"],
                        (elbow_config["neutral"] + elbow_end) / 2,
                    ),
                    (elbow_bent["elbow_vertical"], elbow_end),
                ),
            )

            horizontal_config = self.servo_config[f"{side}_elbow_horizontal"]
            horizontal_reference = horizontal_neutral["elbow_horizontal"]
            cross_value = _unwrap_degrees(
                horizontal_cross["elbow_horizontal"],
                horizontal_reference,
            )
            outward_value = _unwrap_degrees(
                horizontal_out["elbow_horizontal"],
                horizontal_reference,
            )
            current_value = _unwrap_degrees(
                features["elbow_horizontal"],
                horizontal_reference,
            )
            cross_target = (
                horizontal_config["maximum"]
                if horizontal_config["inverted"]
                else horizontal_config["minimum"]
            )
            outward_target = (
                horizontal_config["minimum"]
                if horizontal_config["inverted"]
                else horizontal_config["maximum"]
            )
            mapped[f"{side}_elbow_horizontal"] = _map_piecewise(
                current_value,
                (
                    (cross_value, cross_target),
                    (horizontal_reference, horizontal_config["neutral"]),
                    (outward_value, outward_target),
                ),
            )

        for name, value in mapped.items():
            config = self.servo_config[name]
            mapped[name] = max(config["minimum"], min(config["maximum"], value))
        return mapped


def servo_json_fields(outputs):
    """Map six arm targets to the legacy ESP32 receiver field names."""
    result = {field: None for field in (
        "left_shoulder_vertical_deg",
        "left_wrist_horizontal_deg",
        "left_wrist_vertical_deg",
        "right_shoulder_vertical_deg",
        "right_wrist_horizontal_deg",
        "right_wrist_vertical_deg",
        "left_hip_deg",
        "right_hip_deg",
    )}
    if outputs is not None:
        for servo_name, field in SERVO_FIELDS.items():
            result[field] = outputs[servo_name]
    return result
