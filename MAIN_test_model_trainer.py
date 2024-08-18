from lightsaber_detector.lightsaber_detector import *

if __name__ == "__main__":
    crop_params = {}
    crop_params['crop_y'] = 348
    crop_params['crop_x'] = 636
    crop_params['original_y'] = 360
    crop_params['original_x'] = 640
    crop_params['final_label_y'] = 164
    crop_params['final_label_x'] = 452

    model_params = {}
    model_params['batch_size'] = 2
    model_params['num_epochs'] = 20
    model_params['gamma'] = 0.95
    model_params['weights'] = [0.01, 0.99]
    model_params['lr'] = 5e-5
    model_params['path_to_model_savefile'] = 'output/best_lightsaber_model.pkl'

    path_to_frames = 'data/training_test_data/duel_frames_only.pkl'
    path_to_frame_labels = 'data/training_test_data/duel_frames_pixel_labels.pkl'

    lightsaber_model = LightsaberModel(path_to_frames, path_to_frame_labels, crop_params, model_params)
    lightsaber_model.train_model()
    lightsaber_model.view_predictions(10)