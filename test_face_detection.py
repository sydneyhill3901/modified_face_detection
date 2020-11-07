import cv2
import face_detection_modified
import video
def test(vid_filename):

    video_capture = video.Video(vid_filename)
    video_capture.start()
    detector = face_detection_modified.FaceDetection()
    facecount = 0
    framecount = 0

    while True:

        frame = video_capture.get_frame()
        framecount += 1

        if frame is None:
            print("facecount:" + str(facecount))
            print("framecount:" + str(framecount))
            break

        frame, face_frame, ROI1, ROI2, status, mask = detector.face_detect(frame)
        if status:
            facecount += 1

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    video_capture.stop()
    cv2.destroyAllWindows()

test("cropped_test_vid.mp4")