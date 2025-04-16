# -- coding: utf-8 --

from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    # Load a model
    model = YOLO(r'E:\yolov11\runs\train\ultimately\weights\best.pt')

    # Open the video file
    video_path = r'E:\杂七杂八二\source.mp4'  # Replace with your video file path
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Run YOLO prediction on the frame
            results = model.predict(frame, save=False, show=False)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow('YOLOv11 Detection', annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # If the video ends, break the loop
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
