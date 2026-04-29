# tracker.py

from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30)

    def update(self, detections, frame):
        """
        detections format:
        [
          {"bbox": [x1,y1,x2,y2], "confidence": float, "class": int}
        ]
        """

        formatted_detections = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1

            formatted_detections.append(
                ([x1, y1, w, h], det["confidence"], det["class"])
            )

        tracks = self.tracker.update_tracks(
            formatted_detections,
            frame=frame
        )

        output_tracks = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            output_tracks.append({
                "id": track_id,
                "bbox": [l, t, r, b]
            })

        return output_tracks