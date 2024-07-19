import cv2
import csv
import time
from ultralytics import YOLO

model = YOLO('C:/Users/Alyona/Documents/detectr/runs/detect/custom_model6/weights/best.pt')

def count_objects(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка открытия видео.")
        return

    object_counter = 0
    tracked_objects = {}
    object_id = 0
    center_line_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
    start_time = time.time()
    interval = 15
    results_data = []
    current_time_window_start = start_time

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        results = model(frame)
        detected_centers = []
        for result in results:
            for detection in result.boxes:
                x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                detected_centers.append((center_x, center_y))

        new_tracked_objects = {}
        for obj_id, (last_x, last_y) in tracked_objects.items():
            if len(detected_centers) > 0:
                distances = [((cx - last_x) ** 2 + (cy - last_y) ** 2) for cx, cy in detected_centers]
                min_dist_index = distances.index(min(distances))
                new_x, new_y = detected_centers[min_dist_index]

                if last_x < center_line_x and new_x >= center_line_x:
                    object_counter += 1

                new_tracked_objects[obj_id] = (new_x, new_y)
                detected_centers.pop(min_dist_index)

        for center_x, center_y in detected_centers:
            new_tracked_objects[object_id] = (center_x, center_y)
            object_id += 1

        tracked_objects = new_tracked_objects

        cv2.line(frame, (center_line_x, 0), (center_line_x, frame.shape[0]), (0, 255, 0), 2)
        for detection in results[0].boxes:
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

        cv2.imshow('Frame', frame)
        if time.time()-2 - current_time_window_start >= interval:
            results_data.append([time.strftime('%H:%M:%S', time.gmtime(current_time_window_start - start_time)), object_counter])
            current_time_window_start = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Objects passed'])
        writer.writerows(results_data)

    print(f"Общее число прошедших изделий: {sum([row[1] for row in results_data])}")

video_path = 'test.mp4'
output_csv = 'products_passed.csv'
count_objects(video_path, output_csv)
