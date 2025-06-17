# lightsaber_detector
Convolutional neural network to detect lightsabers during lightsaber duels.

This is an in progress hobby project of mine to train a neural network to analyze live video feed of a lightsaber duel, and accurately detect and label pixels corresponding to lightsabers. It is ideally intended to be paired with human body segmentation models and pose detection models(e.g. OpenPose) to automatically detect hits between lightsabers and humans to do auto-refereeing.

![example pixel labeling](photos/image1.png)

To Run Example:
- run "MAIN_test_label_video_data.py" to extract only the part of the video that shows a lightsaber duel from data/raw_video_training_test_data/test_footage.mp4
- run "MAIN_test_label_video_frame_data.py" to take the hand-labeled bounding boxes of lightsabers in the data/raw_video_labels/duel_frames_labels_0_to_60.csv to label pixels that fit in these bounding boxes
- run "MAIN_run_trained_model.py" to visualize model predictions on training data video of lightsaber duel. White pixels show correctly predicted lightsaber pixels, black pixels show correctly predicted non-lightsaber pixels, blue pixels show lightsaber pixels incorrectly predicted to be non-lightsaber pixels, and green pixels show non-lightsaber pixels incorrectly predicted to be lightsaber pixels.


# Neural network architecture
The model architecture used to label pixels is U-net(https://arxiv.org/abs/1505.04597), which was originally used to do biomedical image segmentation to cells under a microscope, but was adapted to detect lightsaber pixels instead. This model does not preserve image size between input image and output classified pixel labels, instead cropping the photo from 636x348 to 452x164 in size.

# Training data
Selected frames of lightsaber duel are hand labeled by defining 4 corners per lightsaber that define a quadrilateral that bounds the lightsaber pixels per frame. Not every frame is labeled, only every 3rd frame, and interpolation is used to label intermediate frames to reduce time to handlabel data.

# Training process
We use an ADAM optimizer and exponential learning rate scheduler to train the uNet model to detect lightsabers. The loss function used is weighted cross-entropy loss, where lightsaber pixels are weighted 99x more than non lightsaber pixels due to there being far more non lightsaber pixels relative to lightsaber pixels per frame.
