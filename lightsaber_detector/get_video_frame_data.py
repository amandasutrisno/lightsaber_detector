import cv2
import pickle as pk


class VideoConverter:
    def __init__(self, path_to_video, path_to_output, scaling_factor):
        self.path_to_video = path_to_video
        self.path_to_output = path_to_output
        self.scaling_factor = scaling_factor

    def get_frame(self, cap, scaling_factor):
        _, frame = cap.read()
        frame = cv2.resize(frame, None, fx=scaling_factor,
                       fy=scaling_factor,
                       interpolation=cv2.INTER_AREA)
        return frame

    def export_frames(self, n_min, n_max):
        n_count = 0
        cap = cv2.VideoCapture(self.path_to_video)

        all_frames = []

        while True:
            n_count += 1
            # Grab the current frame
            frame = self.get_frame(cap, self.scaling_factor)
            #print(n_count)

            if n_count < n_max and n_count >= n_min:
                all_frames.append(frame)

            if n_count >= n_max:
                break

        cap.release()
        # Close all the windows
        cv2.destroyAllWindows()

        #print(self.path_to_output)
        with open(self.path_to_output, 'wb') as f:
            pk.dump(all_frames, f)

