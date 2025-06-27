import cv2
import numpy as np

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IDEAL_VIDEO_SIZE = (1300, 2500)
COLORS = {
    0: (255, 20, 147),
    1: (255, 20, 147),
    2: (0, 255, 255),
    3: (0, 0, 255),
    5: (255, 0, 0),
    7: (0, 255, 0)
}

def load_yolo_model(weights_path, config_path):
    try:
        return cv2.dnn.readNet(weights_path, config_path)
    except cv2.error as e:
        print(f"Error loading YOLO model: {e}")
        exit(1)

def load_classes(names_path):
    try:
        with open(names_path, "r") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Error: coco.names file not found.")
        exit(1)

def get_bounding_boxes(frame, outs):
    boxes, confidences, class_ids = [], [], []
    height, width = frame.shape[:2]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids

def draw_boxes(frame, boxes, confidences, class_ids):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    zoomed_vehicle = None

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = COLORS.get(class_ids[i], (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            roi = frame[y:y + h, x:x + w]
            if roi.size == 0:
                continue

            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            red_mask = cv2.inRange(hsv_roi, lower_red1, upper_red1) | cv2.inRange(hsv_roi, lower_red2, upper_red2)

            if np.sum(red_mask) > 0:
                zoomed_vehicle = cv2.resize(roi, (150, 150))

    return zoomed_vehicle

def resize_frame(frame, target_size):
    h, w = frame.shape[:2]
    target_w, target_h = target_size
    aspect_ratio = w / h
    target_aspect_ratio = target_w / target_h

    if aspect_ratio > target_aspect_ratio:
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        new_h = target_h
        new_w = int(target_h * aspect_ratio)

    return cv2.resize(frame, (new_w, new_h))

def main():
    net = load_yolo_model("yolov3.weights", "yolov3.cfg")
    classes = load_classes("coco.names")
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]

    video = cv2.VideoCapture("video3.mp4")
    if not video.isOpened():
        print("Error: Could not open video.")
        exit(1)

    fps = video.get(cv2.CAP_PROP_FPS)
    speed_factor = 8
    delay = int(1000 / (fps * speed_factor))
    frame_skip = speed_factor
    previous_zoomed_vehicle = None

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame = resize_frame(frame, IDEAL_VIDEO_SIZE)

        for _ in range(frame_skip - 1):
            video.grab()

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes, confidences, class_ids = get_bounding_boxes(frame, outs)
        zoomed_vehicle = draw_boxes(frame, boxes, confidences, class_ids)

        if zoomed_vehicle is not None:
            previous_zoomed_vehicle = zoomed_vehicle
        else:
            zoomed_vehicle = previous_zoomed_vehicle

        if zoomed_vehicle is not None:
            zoomed_height, zoomed_width = zoomed_vehicle.shape[:2]
            top_left_x = frame.shape[1] - zoomed_width - 10
            top_left_y = 10
            frame[top_left_y:top_left_y + zoomed_height, top_left_x:top_left_x + zoomed_width] = zoomed_vehicle

        cv2.imshow("Vehicle Detection", frame)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
