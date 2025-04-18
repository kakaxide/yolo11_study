
from ultralytics import YOLO
import cv2

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'E:\yolov11\runs\train\exp6\weights\best.pt')

    # Open a video source (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the video source
        ret, frame = cap.read()

        if not ret:
            break

        # Perform prediction on the frame
        results = model.predict(source=frame, save=False, show=False)

        # Display the results on the frame
        annotated_frame = results[0].plot()

        # Show the annotated frame
        cv2.imshow('YOLOv11 Detection', annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
