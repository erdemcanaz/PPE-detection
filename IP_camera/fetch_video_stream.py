import cv2

def display_single_ip_camera_stream(username = None, password= None, ip_address = None, stream_path = None):

    if username is None:
        username = input("Enter username: ")
    if password is None:
        password = input("Enter password: ")
    if ip_address is None:
        ip_address = input("Enter IP address: ")
    if stream_path is None:
        stream_path = input("Enter stream path: ")

    # Replace with your camera's URL and credentials
    url = f'rtsp://{username}:{password}@{ip_address}/{stream_path}'

    # Set up a video capture object
    cap = cv2.VideoCapture(url)

    # Read from the video capture in a loop
    while True:
        ret, frame = cap.read()
        if ret:
            # Process the frame (e.g., display it)
            cv2.imshow('IP Camera Stream', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the capture object and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
