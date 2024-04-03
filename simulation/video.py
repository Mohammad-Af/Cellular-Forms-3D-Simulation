import cv2
import os

for i in range(2, 7):
    iterations = 300
    image_folder = f'res/images{i}/'
    video_name = f'video{i}.avi'

    images = [image_folder + str(i) + '.ppm' for i in range(300)]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 15, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
