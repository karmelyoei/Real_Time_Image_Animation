import imageio
import torch
from tqdm import tqdm
from animate import normalize_kp
from demo import load_checkpoints
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage import img_as_ubyte
from skimage.transform import resize
import cv2
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_image", required=True,help="Path to image to animate")
ap.add_argument("-c", "--checkpoint", required=True,help="Path to checkpoint")
ap.add_argument("-v","--input_video", required=False, help="Path to video input")

args = vars(ap.parse_args())

print("[INFO] loading source image and checkpoint...")
source_path = args['input_image']
checkpoint_path = args['checkpoint']
if args['input_video']:
    video_path = args['input_video']
else:
    video_path = None
source_image = imageio.imread(source_path)
source_image = resize(source_image,(256,256))[..., :3]

generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml', checkpoint_path=checkpoint_path)

if not os.path.exists('output'):
    os.mkdir('output')


relative=True
adapt_movement_scale=True
cpu=False

if video_path:
    cap = cv2.VideoCapture(video_path) 
    print("[INFO] Loading video from the given path")
else:
    # cap = cv2.VideoCapture(0)
    # JavaScript to properly create our live video stream using our webcam as input
    def video_stream():
        js = Javascript('''
            var video;
            var div = null;
            var stream;
            var captureCanvas;
            var imgElement;
            var labelElement;
            
            var pendingResolve = null;
            var shutdown = false;
            
            function removeDom() {
            stream.getVideoTracks()[0].stop();
            video.remove();
            div.remove();
            video = null;
            div = null;
            stream = null;
            imgElement = null;
            captureCanvas = null;
            labelElement = null;
            }
            
            function onAnimationFrame() {
            if (!shutdown) {
                window.requestAnimationFrame(onAnimationFrame);
            }
            if (pendingResolve) {
                var result = "";
                if (!shutdown) {
                captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
                result = captureCanvas.toDataURL('image/jpeg', 0.8)
                }
                var lp = pendingResolve;
                pendingResolve = null;
                lp(result);
            }
            }
            
            async function createDom() {
            if (div !== null) {
                return stream;
            }

            div = document.createElement('div');
            div.style.border = '2px solid black';
            div.style.padding = '3px';
            div.style.width = '100%';
            div.style.maxWidth = '600px';
            document.body.appendChild(div);
            
            const modelOut = document.createElement('div');
            modelOut.innerHTML = "<span>Status:</span>";
            labelElement = document.createElement('span');
            labelElement.innerText = 'No data';
            labelElement.style.fontWeight = 'bold';
            modelOut.appendChild(labelElement);
            div.appendChild(modelOut);
                
            video = document.createElement('video');
            video.style.display = 'block';
            video.width = div.clientWidth - 6;
            video.setAttribute('playsinline', '');
            video.onclick = () => { shutdown = true; };
            stream = await navigator.mediaDevices.getUserMedia(
                {video: { facingMode: "environment"}});
            div.appendChild(video);

            imgElement = document.createElement('img');
            imgElement.style.position = 'absolute';
            imgElement.style.zIndex = 1;
            imgElement.onclick = () => { shutdown = true; };
            div.appendChild(imgElement);
            
            const instruction = document.createElement('div');
            instruction.innerHTML = 
                '<span style="color: red; font-weight: bold;">' +
                'When finished, click here or on the video to stop this demo</span>';
            div.appendChild(instruction);
            instruction.onclick = () => { shutdown = true; };
            
            video.srcObject = stream;
            await video.play();

            captureCanvas = document.createElement('canvas');
            captureCanvas.width = 640; //video.videoWidth;
            captureCanvas.height = 480; //video.videoHeight;
            window.requestAnimationFrame(onAnimationFrame);
            
            return stream;
            }
            async function stream_frame(label, imgData) {
            if (shutdown) {
                removeDom();
                shutdown = false;
                return '';
            }

            var preCreate = Date.now();
            stream = await createDom();
            
            var preShow = Date.now();
            if (label != "") {
                labelElement.innerHTML = label;
            }
                    
            if (imgData != "") {
                var videoRect = video.getClientRects()[0];
                imgElement.style.top = videoRect.top + "px";
                imgElement.style.left = videoRect.left + "px";
                imgElement.style.width = videoRect.width + "px";
                imgElement.style.height = videoRect.height + "px";
                imgElement.src = imgData;
            }
            
            var preCapture = Date.now();
            var result = await new Promise(function(resolve, reject) {
                pendingResolve = resolve;
            });
            shutdown = false;
            
            return {'create': preShow - preCreate, 
                    'show': preCapture - preShow, 
                    'capture': Date.now() - preCapture,
                    'img': result};
            }
            ''')

    display(js)
    
    def video_frame(label, bbox):
    data = eval_js('stream_frame("{}", "{}")'.format(label, bbox))
    return data
    # start streaming video from webcam
    video_stream()
    # label for video
    label_html = 'Capturing...'
    # initialze bounding box to empty
    bbox = ''
    count = 0 
    while True:
        js_reply = video_frame(label_html, bbox)
        if not js_reply:
            break

        # convert JS response to OpenCV Image
        img = js_to_image(js_reply["img"])

        # create transparent overlay for bounding box
        bbox_array = np.zeros([480,640,4], dtype=np.uint8)

        # grayscale image for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # get face region coordinates
        faces = face_cascade.detectMultiScale(gray)
        # get face bounding box for overlay
        for (x,y,w,h) in faces:
        bbox_array = cv2.rectangle(bbox_array,(x,y),(x+w,y+h),(255,0,0),2)

        bbox_array[:,:,3] = (bbox_array.max(axis = 2) > 0 ).astype(int) * 255
        # convert overlay of bbox into bytes
        bbox_bytes = bbox_to_bytes(bbox_array)
        # update bbox so next frame gets new overlay
        bbox = bbox_bytes

    print("[INFO] Initializing front camera...")

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out1 = cv2.VideoWriter('output/test.avi', fourcc, 12, (256*3 , 256), True)

cv2_source = cv2.cvtColor(source_image.astype('float32'),cv2.COLOR_BGR2RGB)
with torch.no_grad() :
    predictions = []
    source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    if not cpu:
        source = source.cuda()
    kp_source = kp_detector(source)
    count = 0
    while(True):
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        if ret == True:
            
            if not video_path:
                x = 143
                y = 87
                w = 322
                h = 322 
                frame = frame[y:y+h,x:x+w]
            frame1 = resize(frame,(256,256))[..., :3]
            
            if count == 0:
                source_image1 = frame1
                source1 = torch.tensor(source_image1[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                kp_driving_initial = kp_detector(source1)
            
            frame_test = torch.tensor(frame1[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

            driving_frame = frame_test
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source,
                                kp_driving=kp_driving,
                                kp_driving_initial=kp_driving_initial, 
                                use_relative_movement=relative,
                                use_relative_jacobian=relative, 
                                adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
            im = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
            joinedFrame = np.concatenate((cv2_source,im,frame1),axis=1)
            
            cv2.imshow('Test',joinedFrame)
            out1.write(img_as_ubyte(joinedFrame))
            count += 1
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        else:
            break
        
    cap.release()
    out1.release()
    cv2.destroyAllWindows()