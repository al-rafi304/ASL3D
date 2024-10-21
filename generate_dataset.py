import cv2
import os
import json
import mediapipe as mp
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO, WARNING, and ERROR logs

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

def genSnap(vid_src, num_frames, out_location):
    cap = cv2.VideoCapture(vid_src)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // num_frames, 1)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            cv2.imwrite(f'{out_location}/{frame_count}.jpg', frame)

        frame_count += 1
    cap.release()

def getCoords(IMAGE_FILES, out_loc):
    data = []
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        for idx, file in enumerate(IMAGE_FILES):
            print(file)
            image = cv2.flip(cv2.imread(file), 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print handedness and draw hand landmarks on the image.
            print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
                print(f"\n{file}:\nNo hands detected!\n")
                continue

            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            frame_data = {'frame': file.split('/')[-1].split('.')[0]}
            for hand_landmarks in results.multi_hand_landmarks:
                hand = 'hand_left' if results.multi_hand_landmarks.index(hand_landmarks) == 0 else 'hand_right'
                
                # Extracting Hand Data from current frame
                for i in range(len(hand_landmarks.landmark)):
                    xyz = {
                        'x': hand_landmarks.landmark[i].x,
                        'y': hand_landmarks.landmark[i].y,
                        'z': hand_landmarks.landmark[i].z,
                    }

                    if frame_data.get(hand):
                        frame_data[hand].append(xyz)
                    else:
                        frame_data[hand] = []
                        frame_data[hand].append(xyz)

                # Annotating Image with coordinates
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            
            # Pose Landmark
            with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.7) as holistic:
                pos_results = holistic.process(annotated_image)
                if pos_results.pose_landmarks:
                    mp_drawing.draw_landmarks(annotated_image, pos_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    
                    # Extracting Pose Data from current frame
                    for i in [0, 11, 12]:
                        xyz = {
                            'x': pos_results.pose_landmarks.landmark[i].x,
                            'y': pos_results.pose_landmarks.landmark[i].y,
                            'z': pos_results.pose_landmarks.landmark[i].z,
                        }
                        if frame_data.get('pose'):
                            frame_data['pose'].append(xyz)
                        else:
                            frame_data['pose'] = []
                            frame_data['pose'].append(xyz)

            # Output Annotated Image
            cv2.imwrite(f'{out_loc}/{file.split('/')[-1].split('.')[0]}.jpg', cv2.flip(annotated_image, 1))

            data.append(frame_data)
    
    return data


GENERATE_COUNT = 200
wlasl = json.load(open('Dataset/WLASL_v0.3.json', 'r'))
dataset = {}

start_time = time.time()

for word in wlasl[:GENERATE_COUNT]:

    gloss = word['gloss']
    instances = word['instances']
    print(gloss)

    i = 0
    found_vid = False
    while i < len(instances) and not found_vid:
        video = instances[i]
        vid_name = video['video_id']
        vid_src = f'Dataset/videos/{vid_name}.mp4'
        coords_dir = f'Dataset/coordinates/{gloss} - {vid_name}'
        frames_dir = f'Dataset/frames/{gloss} - {vid_name}'
        num_frames = 20

        if not os.path.isfile(vid_src):
            i += 1
            continue
        else:
            found_vid = True

        print(' -', vid_name)

        os.makedirs(f'{coords_dir}', exist_ok=True)
        os.makedirs(f'{frames_dir}', exist_ok=True)

        genSnap(vid_src, num_frames, frames_dir)


        files = [f'{frames_dir}/{f}' for f in os.listdir(f'{frames_dir}') if os.path.isfile(os.path.join(f'{frames_dir}', f))]
        print(files)

        data = getCoords(files, coords_dir)
        dataset[gloss] = data
        with open(f'{coords_dir}/data.json', 'w') as fd:
            fd.write(json.dumps(data))
        
        i += 1

with open(f'dataset.json', 'w') as file:
    json.dump(dataset, file, indent=4)

end_time = time.time()
minute = int((end_time - start_time) // 60)
seconds = int((end_time - start_time) % 60)
print(f'\n\nTotal Time: {minute}min {seconds}s')