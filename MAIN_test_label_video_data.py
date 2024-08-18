from lightsaber_detector.get_video_frame_data import *

if __name__ == "__main__":
    path_to_video = 'data/raw_video_training_test_data/test_footage.mp4'
    path_to_output = 'data/training_test_data/duel_frames_only.pkl'
    scaling_factor = 0.5
    n_min = 1100
    n_max = 1500

    videoconvert = VideoConverter(path_to_video, path_to_output, scaling_factor)
    videoconvert.export_frames(n_min, n_max)