from lightsaber_detector.label_video_frame_data import *

if __name__ == "__main__":
    path_to_framedata = 'data/training_test_data/duel_frames_only.pkl'
    path_to_labeldata = 'data/raw_video_labels/duel_frames_labels_0_to_60.csv'
    path_to_outputdata = 'data/training_test_data/duel_frames_pixel_labels.pkl'

    video_frame_label = VideoFrameLabel(path_to_framedata, path_to_labeldata, path_to_outputdata)
    video_frame_label.label_data()