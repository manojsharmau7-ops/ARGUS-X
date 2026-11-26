from __future__ import annotations

from collections import Counter, defaultdict, deque
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

Feature = NDArray[np.float32]
MotionSample = tuple[float, float, float, float]


class TrackingAnalytics:
    def __init__(
        self,
        history_limit: int = 300,
        cross_camera_threshold: float = 0.88,
        count_line_position: float = 0.5,
        in_direction: str = "top_to_bottom",
    ) -> None:
        self.history: dict[tuple[str, int], deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=history_limit)
        )
        self.camera_tracks: dict[str, dict[int, str]] = defaultdict(dict)
        self.local_to_global: dict[tuple[str, int], int] = {}
        self.local_match_similarity: dict[tuple[str, int], float | None] = {}
        self.global_tracks: dict[int, dict[str, Any]] = {}
        self.cross_camera_matches: list[dict[str, Any]] = []
        self.last_stable_sides: dict[tuple[str, int], int] = {}
        self.flow_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"in": 0, "out": 0})
        )
        self.flow_events: list[dict[str, Any]] = []
        self.motion_samples: dict[tuple[str, int], deque[MotionSample]] = defaultdict(
            deque
        )
        self.movement_alerts: list[dict[str, Any]] = []
        self.last_movement_alerts: dict[tuple[str, int, str], float] = {}
        self.cross_camera_threshold = cross_camera_threshold
        self.count_line_position = count_line_position
        self.in_direction = in_direction
        self.next_global_id = 1

    @staticmethod
    def _normalize(feature: Feature) -> Feature:
        norm = float(np.linalg.norm(feature))
        return feature / norm if norm > 0 else feature

    def record_track(
        self,
        camera_id: str,
        track_id: int,
        class_name: str,
        box: list[float],
        feature: Feature,
        frame_no: int,
        fps: float,
        frame_height: int,
        record_history: bool = True,
        time_offset_seconds: float = 0.0,
    ) -> tuple[int, float | None]:
        key = (camera_id, track_id)
        normalized_feature = self._normalize(
            np.asarray(feature, dtype=np.float32)
        )
        if key in self.local_to_global:
            global_id = self.local_to_global[key]
            record = self.global_tracks[global_id]
            record["feature"] = self._normalize(
                0.9 * record["feature"] + 0.1 * normalized_feature
            )
            match_similarity = self.local_match_similarity[key]
        else:
            global_id, match_similarity = self._assign_global_id(
                camera_id, class_name, normalized_feature
            )
            self.local_to_global[key] = global_id
            self.local_match_similarity[key] = match_similarity
            self.camera_tracks[camera_id][track_id] = class_name

        if record_history:
            x1, y1, x2, y2 = box
            self.history[key].append({
                "frame": frame_no,
                "time_seconds": round(
                    time_offset_seconds + (frame_no - 1) / max(fps, 1e-6),
                    2,
                ),
                "center": [
                    round((x1 + x2) / 2, 1),
                    round((y1 + y2) / 2, 1),
                ],
                "box": [round(float(value), 1) for value in box],
            })
        self._record_line_crossing(
            camera_id,
            track_id,
            global_id,
            class_name,
            box,
            frame_no,
            fps,
            time_offset_seconds,
            frame_height,
        )
        if class_name.casefold() == "person":
            self._record_movement_alerts(
                camera_id,
                track_id,
                global_id,
                class_name,
                box,
                frame_no,
                fps,
                time_offset_seconds,
            )
        return global_id, match_similarity

    def _record_movement_alerts(
        self,
        camera_id: str,
        track_id: int,
        global_id: int,
        class_name: str,
        box: list[float],
        frame_no: int,
        fps: float,
        time_offset_seconds: float,
    ) -> None:
        now = time_offset_seconds + (frame_no - 1) / max(fps, 1e-6)
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        object_scale = math.sqrt(max(1.0, (x2 - x1) * (y2 - y1)))
        key = (camera_id, track_id)
        samples = self.motion_samples[key]

        if samples and now - samples[-1][0] < 0.2 - 1e-6:
            return
        if samples and now - samples[-1][0] > 1.0:
            samples.clear()

        samples.append((now, center_x, center_y, object_scale))
        while samples and now - samples[0][0] > 31.0:
            samples.popleft()

        previous_samples = list(samples)[:-1]
        rapid_anchor = next(
            (
                sample
                for sample in reversed(previous_samples)
                if 0.4 <= now - sample[0] <= 1.0
            ),
            None,
        )
        if rapid_anchor is not None:
            elapsed = now - rapid_anchor[0]
            distance = math.hypot(center_x - rapid_anchor[1], center_y - rapid_anchor[2])
            scale = (object_scale + rapid_anchor[3]) / 2
            speed = distance / max(elapsed * scale, 1e-6)
            if speed >= 2.5:
                self._emit_movement_alert(
                    camera_id,
                    track_id,
                    global_id,
                    class_name,
                    frame_no,
                    now,
                    center_x,
                    center_y,
                    "rapid_movement",
                    "Rapid movement detected",
                    speed,
                    "object_lengths_per_second",
                )

        older = min(
            previous_samples,
            key=lambda sample: abs((now - sample[0]) - 1.0),
            default=None,
        )
        middle = min(
            previous_samples,
            key=lambda sample: abs((now - sample[0]) - 0.5),
            default=None,
        )
        if (
            older is not None
            and middle is not None
            and abs((now - older[0]) - 1.0) <= 0.25
            and abs((now - middle[0]) - 0.5) <= 0.25
            and middle[0] > older[0]
        ):
            first_x = middle[1] - older[1]
            first_y = middle[2] - older[2]
            second_x = center_x - middle[1]
            second_y = center_y - middle[2]
            first_distance = math.hypot(first_x, first_y)
            second_distance = math.hypot(second_x, second_y)
            scale = (older[3] + middle[3] + object_scale) / 3
            if first_distance >= scale * 0.5 and second_distance >= scale * 0.5:
                cosine = (
                    first_x * second_x + first_y * second_y
                ) / (first_distance * second_distance)
                if cosine <= -0.7:
                    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
                    self._emit_movement_alert(
                        camera_id,
                        track_id,
                        global_id,
                        class_name,
                        frame_no,
                        now,
                        center_x,
                        center_y,
                        "direction_reversal",
                        "Abrupt direction reversal detected",
                        angle,
                        "degrees",
                    )

        loiter_window = [
            sample for sample in samples if sample[0] >= now - 30.0
        ]
        if (
            len(loiter_window) >= 2
            and loiter_window[-1][0] - loiter_window[0][0] >= 29.5
        ):
            center_x_mean = sum(sample[1] for sample in loiter_window) / len(loiter_window)
            center_y_mean = sum(sample[2] for sample in loiter_window) / len(loiter_window)
            max_radius = max(
                math.hypot(sample[1] - center_x_mean, sample[2] - center_y_mean)
                for sample in loiter_window
            )
            mean_scale = sum(sample[3] for sample in loiter_window) / len(loiter_window)
            if max_radius <= mean_scale * 0.45:
                self._emit_movement_alert(
                    camera_id,
                    track_id,
                    global_id,
                    class_name,
                    frame_no,
                    now,
                    center_x,
                    center_y,
                    "loitering",
                    "Person remained in a small area for 30 seconds",
                    30.0,
                    "seconds",
                )

    def _emit_movement_alert(
        self,
        camera_id: str,
        track_id: int,
        global_id: int,
        class_name: str,
        frame_no: int,
        time_seconds: float,
        center_x: float,
        center_y: float,
        alert_type: str,
        message: str,
        value: float,
        unit: str,
    ) -> None:
        alert_key = (camera_id, track_id, alert_type)
        cooldown = 30.0 if alert_type == "loitering" else 15.0
        if time_seconds - self.last_movement_alerts.get(alert_key, -cooldown) < cooldown:
            return

        self.last_movement_alerts[alert_key] = time_seconds
        self.movement_alerts.append({
            "camera_id": camera_id,
            "track_id": track_id,
            "global_id": global_id,
            "class_name": class_name,
            "alert_type": alert_type,
            "message": message,
            "frame": frame_no,
            "time_seconds": round(time_seconds, 2),
            "location": [round(center_x, 1), round(center_y, 1)],
            "value": round(value, 2),
            "unit": unit,
        })

    def _record_line_crossing(
        self,
        camera_id: str,
        track_id: int,
        global_id: int,
        class_name: str,
        box: list[float],
        frame_no: int,
        fps: float,
        time_offset_seconds: float,
        frame_height: int,
    ) -> None:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        line_y = frame_height * self.count_line_position
        hysteresis = max(4, frame_height * 0.01)
        side = (
            -1 if center_y < line_y - hysteresis
            else 1 if center_y > line_y + hysteresis
            else 0
        )
        key = (camera_id, track_id)
        previous_side = self.last_stable_sides.get(key)
        if side == 0:
            return
        if previous_side is None:
            self.last_stable_sides[key] = side
            return
        if side == previous_side:
            return

        top_to_bottom = previous_side == -1 and side == 1
        direction = (
            "in"
            if top_to_bottom == (self.in_direction == "top_to_bottom")
            else "out"
        )
        time_seconds = round(
            time_offset_seconds + (frame_no - 1) / max(fps, 1e-6), 2
        )
        self.flow_counts[camera_id][class_name][direction] += 1
        self.flow_events.append({
            "camera_id": camera_id,
            "track_id": track_id,
            "global_id": global_id,
            "class_name": class_name,
            "direction": direction,
            "frame": frame_no,
            "time_seconds": time_seconds,
            "center": [round(center_x, 1), round(center_y, 1)],
        })
        self.last_stable_sides[key] = side

    def _assign_global_id(
        self,
        camera_id: str,
        class_name: str,
        feature: Feature,
    ) -> tuple[int, float | None]:
        candidates = [
            (global_id, record)
            for global_id, record in self.global_tracks.items()
            if (
                record["class_name"] == class_name
                and camera_id not in record["camera_ids"]
            )
        ]
        best_id: int | None = None
        best_similarity = -1.0
        if candidates:
            features = np.stack([
                record["feature"] for _, record in candidates
            ])
            similarities = features @ feature
            best_index = int(np.argmax(similarities))
            best_id = candidates[best_index][0]
            best_similarity = float(similarities[best_index])

        if best_id is not None and best_similarity >= self.cross_camera_threshold:
            record = self.global_tracks[best_id]
            source_camera = next(iter(record["camera_ids"]))
            record["camera_ids"].add(camera_id)
            self.cross_camera_matches.append({
                "global_id": best_id,
                "from_camera": source_camera,
                "to_camera": camera_id,
                "appearance_similarity": round(best_similarity, 3),
            })
            return best_id, round(best_similarity, 3)

        global_id = self.next_global_id
        self.next_global_id += 1
        self.global_tracks[global_id] = {
            "class_name": class_name,
            "feature": feature.copy(),
            "camera_ids": {camera_id},
        }
        return global_id, None

    def get_trajectory(
        self, camera_id: str, track_id: int
    ) -> list[dict[str, Any]]:
        return list(self.history[(camera_id, track_id)])

    def get_camera_counts(self, camera_id: str) -> dict[str, Any]:
        by_class = Counter(self.camera_tracks[camera_id].values())
        return {
            "total_unique": sum(by_class.values()),
            "by_class": dict(sorted(by_class.items())),
        }

    def get_global_counts(self) -> dict[str, Any]:
        by_class = Counter(
            record["class_name"] for record in self.global_tracks.values()
        )
        return {
            "total_unique": len(self.global_tracks),
            "by_class": dict(sorted(by_class.items())),
        }

    def get_camera_flow_counts(self, camera_id: str) -> dict[str, Any]:
        by_class = {
            class_name: dict(counts)
            for class_name, counts in sorted(self.flow_counts[camera_id].items())
        }
        return {
            "by_class": by_class,
            "total_in": sum(counts["in"] for counts in by_class.values()),
            "total_out": sum(counts["out"] for counts in by_class.values()),
        }

    def get_camera_flow_events(self, camera_id: str) -> list[dict[str, Any]]:
        return [
            event for event in self.flow_events
            if event["camera_id"] == camera_id
        ]

    def get_camera_movement_alerts(self, camera_id: str) -> list[dict[str, Any]]:
        return [
            alert for alert in self.movement_alerts
            if alert["camera_id"] == camera_id
        ]

    def get_all_flow_counts(self) -> dict[str, Any]:
        by_camera = {
            camera_id: self.get_camera_flow_counts(camera_id)
            for camera_id in sorted(self.camera_tracks)
        }
        return {
            "by_camera": by_camera,
            "total_in": sum(counts["total_in"] for counts in by_camera.values()),
            "total_out": sum(counts["total_out"] for counts in by_camera.values()),
        }
