import math
import tempfile
import unittest
from pathlib import Path

from arm_tracking import (
    ARMS,
    CALIBRATION_STAGES,
    CALIBRATION_FRAME_COUNT,
    ArmTracker,
    _map_piecewise,
    extract_pose,
    servo_json_fields,
)
from main import SERIAL_SEND_INTERVAL_SECONDS, _build_message, _hip_outputs


class Landmark:
    def __init__(self, x=0.5, y=0.5, visibility=1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


def make_pose():
    landmarks = [Landmark() for _ in range(33)]
    landmarks[11] = Landmark(0.4, 0.3)
    landmarks[12] = Landmark(0.6, 0.3)
    landmarks[13] = Landmark(0.4, 0.4)
    landmarks[14] = Landmark(0.6, 0.4)
    landmarks[15] = Landmark(0.4, 0.5)
    landmarks[16] = Landmark(0.6, 0.5)
    landmarks[23] = Landmark(0.44, 0.6)
    landmarks[24] = Landmark(0.56, 0.6)
    return landmarks


def make_calibration_stage(shoulder, flexion, horizontal):
    features = {
        side: {
            "shoulder_vertical": shoulder,
            "elbow_vertical": flexion,
            "elbow_horizontal": horizontal[side],
            "upper_length": 0.34,
            "forearm_length": 0.34,
        }
        for side in ARMS
    }
    return {
        "points": {
            str(index): [0.0, 0.0]
            for index in (11, 12, 13, 14, 15, 16, 23, 24)
        },
        "features": features,
        "torso_length_px": 100.0,
    }


def make_calibration():
    stages = {
        "neutral": make_calibration_stage(
            0, 0, {"left": 180, "right": -180}
        ),
        "shoulder_mid": make_calibration_stage(
            45, 0, {"left": 180, "right": -180}
        ),
        "shoulder_horizontal": make_calibration_stage(
            90, 0, {"left": 180, "right": -180}
        ),
        "elbow_intermediate": make_calibration_stage(
            90, 45, {"left": 90, "right": -90}
        ),
        "elbow_bent": make_calibration_stage(
            90, 90, {"left": 90, "right": -90}
        ),
        "elbow_horizontal_neutral": make_calibration_stage(
            90, 90, {"left": 90, "right": -90}
        ),
        "elbow_cross_body": make_calibration_stage(
            90, 90, {"left": 60, "right": -60}
        ),
        "elbow_outward": make_calibration_stage(
            90, 90, {"left": 120, "right": -120}
        ),
    }
    return {
        "version": 1,
        "stages": stages,
        "servo_config": ArmTracker().servo_config,
    }


class ArmTrackingTests(unittest.TestCase):
    def test_extract_pose_uses_2d_landmarks_and_normalizes_to_torso(self):
        pose = extract_pose(make_pose(), 640, 480)

        self.assertIsNotNone(pose)
        self.assertAlmostEqual(pose["features"]["left"]["shoulder_vertical"], 0)
        self.assertAlmostEqual(pose["features"]["right"]["shoulder_vertical"], 0)
        self.assertAlmostEqual(pose["features"]["left"]["elbow_vertical"], 0)
        self.assertAlmostEqual(pose["features"]["right"]["elbow_vertical"], 0)
        self.assertAlmostEqual(
            pose["features"]["left"]["upper_length"],
            pose["features"]["right"]["upper_length"],
        )
        self.assertEqual(set(pose["points"]), {"11", "12", "13", "14", "15", "16", "23", "24"})

    def test_pose_is_rejected_if_a_required_landmark_is_not_visible(self):
        landmarks = make_pose()
        landmarks[15].visibility = 0.1
        self.assertIsNone(extract_pose(landmarks, 640, 480))

    def test_calibrated_mapping_hits_neutral_midpoints_and_endpoints(self):
        tracker = ArmTracker()
        tracker.calibration = make_calibration()

        neutral = {
            "left": {
                "shoulder_vertical": 0,
                "elbow_vertical": 0,
                "elbow_horizontal": 90,
            },
            "right": {
                "shoulder_vertical": 0,
                "elbow_vertical": 0,
                "elbow_horizontal": -90,
            },
        }
        output = tracker._map_pose({"features": neutral})
        for name in output:
            self.assertAlmostEqual(output[name], 90)

        for side in ARMS:
            neutral[side]["shoulder_vertical"] = 45
            neutral[side]["elbow_vertical"] = 45
        output = tracker._map_pose({"features": neutral})
        self.assertAlmostEqual(output["left_shoulder_vertical"], 135)
        self.assertAlmostEqual(output["right_shoulder_vertical"], 135)
        self.assertAlmostEqual(output["left_elbow_vertical"], 135)
        self.assertAlmostEqual(output["right_elbow_vertical"], 135)

        for side in ARMS:
            neutral[side]["shoulder_vertical"] = 90
            neutral[side]["elbow_vertical"] = 90
            neutral[side]["elbow_horizontal"] = (
                60 if side == "left" else -60
            )
        output = tracker._map_pose({"features": neutral})
        self.assertEqual(output["left_shoulder_vertical"], 180)
        self.assertEqual(output["right_shoulder_vertical"], 180)
        self.assertEqual(output["left_elbow_vertical"], 180)
        self.assertEqual(output["right_elbow_vertical"], 180)
        self.assertEqual(output["left_elbow_horizontal"], 0)
        self.assertEqual(output["right_elbow_horizontal"], 0)

    def test_piecewise_mapping_clips_and_interpolates(self):
        anchors = ((0, 90), (45, 135), (90, 180))
        self.assertEqual(_map_piecewise(-10, anchors), 90)
        self.assertEqual(_map_piecewise(22.5, anchors), 112.5)
        self.assertEqual(_map_piecewise(100, anchors), 180)
        self.assertTrue(math.isfinite(_map_piecewise(45, anchors)))

    def test_legacy_receiver_json_shape_is_unchanged(self):
        fields = servo_json_fields(
            {
                "left_shoulder_vertical": 95.26,
                "left_elbow_horizontal": 80,
                "left_elbow_vertical": 120,
                "right_shoulder_vertical": 90,
                "right_elbow_horizontal": 105,
                "right_elbow_vertical": 115,
            }
        )
        self.assertEqual(
            set(fields),
            {
                "left_shoulder_vertical_deg",
                "left_wrist_horizontal_deg",
                "left_wrist_vertical_deg",
                "right_shoulder_vertical_deg",
                "right_wrist_horizontal_deg",
                "right_wrist_vertical_deg",
                "left_hip_deg",
                "right_hip_deg",
            },
        )
        self.assertEqual(fields["left_shoulder_vertical_deg"], 95.26)
        self.assertEqual(fields["left_wrist_vertical_deg"], 120)
        self.assertEqual(fields["right_shoulder_vertical_deg"], 90)
        self.assertEqual(fields["right_wrist_horizontal_deg"], 105)
        self.assertIsNone(fields["left_hip_deg"])

    def test_existing_hip_values_and_motor_fields_are_preserved(self):
        hip_outputs = _hip_outputs(
            {"left": (0, 1, 0), "right": (0, 0, 1)},
            {"left": (0, 1, 0), "right": (0, 1, 0)},
        )
        message = _build_message(
            None,
            hip_outputs,
            {
                "right_turn": "HIGH",
                "left_turn": "LOW",
                "forward": "LOW",
                "backward": "HIGH",
            },
        )

        self.assertEqual(message["left_hip_deg"], 90)
        self.assertEqual(message["right_hip_deg"], 180)
        self.assertEqual(message["right_turn"], "HIGH")
        self.assertEqual(message["backward"], "HIGH")
        self.assertEqual(len(message), 12)

    def test_existing_serial_send_frequency_is_preserved(self):
        self.assertEqual(SERIAL_SEND_INTERVAL_SECONDS, 0.1)

    def test_calibration_stage_list_matches_saved_calibration(self):
        self.assertEqual(
            {name for name, _ in CALIBRATION_STAGES},
            set(make_calibration()["stages"]),
        )

    def test_invalid_final_calibration_restarts_without_raising(self):
        tracker = ArmTracker()
        stages = make_calibration()["stages"]
        stages["elbow_outward"]["features"]["left"]["elbow_horizontal"] = 90
        tracker.captured_stages = {
            name: stages[name]
            for name, _ in CALIBRATION_STAGES[:-1]
        }
        tracker.stage_index = len(CALIBRATION_STAGES) - 1
        tracker.capture_active = True
        tracker.calibration_samples = [stages["elbow_outward"]] * 20

        status = tracker._finish_stage()

        self.assertIn("Calibration rejected", status)
        self.assertFalse(tracker.calibrated)
        self.assertEqual(tracker.stage_index, 0)
        self.assertFalse(tracker.capture_active)
        self.assertEqual(tracker.captured_stages, {})

    def test_unstable_capture_requires_explicit_retry(self):
        tracker = ArmTracker()
        tracker.begin_capture()
        tracker.calibration_samples = [
            make_calibration_stage(0, 0, {"left": 180, "right": -180})
            for _ in range(CALIBRATION_FRAME_COUNT)
        ]
        tracker.calibration_samples[-1]["features"]["left"]["shoulder_vertical"] = 60

        status = tracker._finish_stage()

        self.assertIn("hold still", status)
        self.assertFalse(tracker.capture_active)

    def test_completed_calibration_saves_and_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            tracker = ArmTracker(calibration_path=path)
            tracker.captured_stages = make_calibration()["stages"]

            tracker._complete_calibration()

            loaded = ArmTracker(calibration_path=path)
            loaded.load()
            self.assertTrue(loaded.calibrated)
            self.assertEqual(
                set(loaded.calibration["stages"]),
                {name for name, _ in CALIBRATION_STAGES},
            )


if __name__ == "__main__":
    unittest.main()
