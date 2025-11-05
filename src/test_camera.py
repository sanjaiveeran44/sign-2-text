import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend (fixes blinking/no window issue)

if not cap.isOpened():
    print("❌ Error: Cannot access webcam")
else:
    print("✅ Webcam detected. Opening window...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    cv2.imshow("Simple Camera Test - Press 'q' to exit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
