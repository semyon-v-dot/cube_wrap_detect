import torch
import cv2 as cv
import os

if __name__ == '__main__':
    VID_NAME = 'test.mp4'
    SKIP_SEC_BEGINNING = 90

    model = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path='cube_wrap_detect.pt',
        _verbose=False
    )
    vid = cv.VideoCapture(VID_NAME)
    vid_fps = vid.get(cv.CAP_PROP_FPS)
    vid.set(1, vid_fps * SKIP_SEC_BEGINNING)
    ret, frame1 = vid.read()
    ret, frame2 = vid.read()
    fr_counter = 2
    while ret and vid.isOpened():
        res = model(cv.cvtColor(frame1, cv.COLOR_BGR2RGB), size=640)
        df = res.pandas().xyxy[0]
        for i in range(len(df.index)):
            cv.rectangle(
                frame1, 
                (int(df['xmin'][i]), int(df['ymin'][i])), 
                (int(df['xmax'][i]), int(df['ymax'][i])), 
                (255),
                thickness=3
                )
            
        cv.imshow("", frame1)
        waitkey = cv.waitKey(1)
        if waitkey == ord('q'):
            break

        frame1 = frame2
        ret, frame2 = vid.read()
        fr_counter += 1
        sec = fr_counter / int(vid_fps)

    vid.release()
    cv.destroyAllWindows()
