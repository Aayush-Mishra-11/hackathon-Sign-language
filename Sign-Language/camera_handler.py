import cv2

def capture_video(process_frame_callback):
    """Capture video input and process frames using a callback function."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video. Please check your camera connection.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame. Please ensure the camera is functioning correctly.")
            break

        try:
            # Process the frame using the callback function
            process_frame_callback(frame)
        except Exception as e:
            print(f"Error during frame processing: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()