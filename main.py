from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

TARGET_WIDTH = 5
TARGET_HEIGHT = 20
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


if __name__ == "__main__":
    # Configuration
    source_video_path = 'vehicles.mp4'
    confidence_threshold = 0.5
    iou_threshold = 0.8
    target_video_path = 'vehicles_with_plates.mp4'

    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    w, h = video_info.resolution_wh
    SOURCE = np.array([[0, h/3], [w, h/3], [w, h], [0, h]])

    model = YOLO("yolov8x.pt")
    plate_model = YOLO('car_plate/weights/best.pt')
    ocr_model = YOLO('train/weights/best.pt')

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=confidence_threshold
    )

    # Initialize annotators
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    plate_box_annotator = sv.BoxAnnotator(color=sv.Color.RED, thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    plate_saved_cars = set()

    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            result = model(frame, classes=[2, 3, 5, 7], conf=0.5)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > confidence_threshold]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=iou_threshold)
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points = view_transformer.transform_points(points=points).astype(int)
            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            speeds = {}
            for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) >= video_info.fps / 2:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)  # in meters
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = (distance / time * 3.6)  # km/h
                    speeds[tracker_id] = speed
                else:
                    speeds[tracker_id] = None

            # Generate labels
            labels = [
                f"#{tracker_id} {int(speeds[tracker_id])} km/h" if speeds[tracker_id] is not None else f"#{tracker_id}"
                for tracker_id in detections.tracker_id
            ]

            # Detect plates and perform OCR for speeding cars
            plate_detections_list = []
            for i, tracker_id in enumerate(detections.tracker_id):
                bbox = detections.xyxy[i]
                x1, y1, x2, y2 = map(int, bbox)
                cropped_frame = frame[y1:y2, x1:x2]

                # Detect plate
                plate_result = plate_model(cropped_frame, conf=0.5)[0]
                plate_detections = sv.Detections.from_ultralytics(plate_result)
                plate_detections = plate_detections[plate_detections.confidence > confidence_threshold]

                if len(plate_detections) > 0:
                    # Adjust coordinates to original frame
                    plate_detections.xyxy += np.array([x1, y1, x1, y1])

                    # Save plate and perform OCR once per car
                    if tracker_id not in plate_saved_cars:
                        plate_bbox = plate_detections.xyxy[0]
                        px1, py1, px2, py2 = map(int, plate_bbox)
                        plate_crop = frame[py1:py2, px1:px2]
                        # cv2.imwrite(f'{tracker_id}.jpg', plate_crop)

                        # Perform OCR on the plate crop
                        ocr_result = ocr_model(plate_crop)[0]
                        ocr_detections = sv.Detections.from_ultralytics(ocr_result)

                        if len(ocr_detections) > 0:
                            class_names = ocr_model.names
                            # Filter out 'car' and 'plate' classes
                            valid_detections = [(box, class_id) for box, class_id in
                                                zip(ocr_detections.xyxy, ocr_detections.class_id)
                                                if class_names[class_id] not in ['Car', 'License Plate']]
                            if valid_detections:
                                # Sort characters by x-coordinate and construct plate number
                                characters = [(box[0], class_names[class_id]) for box, class_id in valid_detections]
                                characters.sort(key=lambda x: x[0])  # Sort by x-coordinate (left to right)
                                plate_number = ' '.join([char for _, char in characters])
                            else:
                                plate_number = None
                        else:
                            plate_number = None

                        # # Save the plate number to a file
                        # with open("plates.txt", "a") as f:
                        #     if plate_number:
                        #         f.write(f"Car {tracker_id}: {plate_number}\n")
                        #         plate_saved_cars.add(tracker_id)

                    plate_detections_list.append(plate_detections)

            # Annotate frame
            annotated_frame = frame.copy()
            if plate_detections_list:
                all_plate_detections = sv.Detections.merge(plate_detections_list)
                annotated_frame = plate_box_annotator.annotate(
                    scene=annotated_frame, detections=all_plate_detections
                )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            # Write and display
            sink.write_frame(annotated_frame)

        cv2.destroyAllWindows()
