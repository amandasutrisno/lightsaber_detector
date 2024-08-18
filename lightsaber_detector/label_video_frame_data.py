import pandas as pd
import numpy as np
import pickle as pk
import copy
import cv2
from scipy.interpolate import CubicSpline

class VideoFrameLabel:
    def __init__(self, path_to_framedata, path_to_labeldata, path_to_outputdata):
        #path to frame data (.pkl file)
        self.path_to_framedata = path_to_framedata
        #path to label data (.csv file)
        self.path_to_labeldata = path_to_labeldata
        #path to output labeled frames (.pkl file)
        self.path_to_outputdata = path_to_outputdata

        with open(path_to_framedata, 'rb') as f:
            self.frame_data = pk.load(f)

    def label_data(self):
        print('getting x,y lightsaber labels...')
        self.data_dict = self.get_xy_lightsaber_labels()
        #interpolating frame labels in csv file
        print('interpolating x,y labels across unlabeled frames...')
        self.data_dict = self.interpolate_box_coords(self.data_dict)
        #get labeled frames
        print("labeling frames...")
        self.labeled_frames = self.create_labeled_frames()
        #save labeled frames
        print("saving labeled frame data...")
        self.save_labeled_frames()
    
    def save_labeled_frames(self):
        with open(self.path_to_outputdata, 'wb') as f:
            pk.dump(self.labeled_frames, f)

    def create_labeled_frames(self):
        labeled_frames = []
        for row in range(0, len(self.data_dict['frame_idx'])):
            frame_idx = int(self.data_dict['frame_idx'][row])
            rgb_frame = cv2.cvtColor(self.frame_data[frame_idx], cv2.COLOR_BGR2RGB)
            highlight_frame, labeled_frame = self.label_pixels_frame(row, self.data_dict, rgb_frame)

            labeled_frames.append(labeled_frame)
        
        return labeled_frames

    def get_xy_lightsaber_labels(self):
        # Step 1: Read the CSV file into a DataFrame
        df = pd.read_csv(self.path_to_labeldata)

        # Step 2: Initialize an empty dictionary to store columns
        columns_dict = {}

        # Step 3: Extract columns into the dictionary
        for column in df.columns:
            columns_dict[column] = df[column].tolist()
        
        return columns_dict
    
    def check_in_box(self, x_coord, y_coord, box):
    
        coeffs = []
        in_box = True
        for idx in range(0, 4):
            idx_p1 = (idx + 1) % 4
            dx = box[idx_p1][0] - box[idx][0]
            dy = - (box[idx_p1][1] - box[idx][1])

            rot_vec_dx = dy
            rot_vec_dy = -dx

            coeff_rot_vec = rot_vec_dx*(x_coord - box[idx][0]) + rot_vec_dy*( - y_coord + box[idx][1])

            coeffs.append(coeff_rot_vec)
            #print(coeff_rot_vec)

            if coeff_rot_vec < 0:
                in_box = False

        return in_box


    def label_pixels_frame(self, row, data_dict, frame):
        box1 = [ [data_dict['box1_x' + str(n)][row], data_dict['box1_y' + str(n)][row]]for n in range(1, 5) ]
        box2 = [ [data_dict['box2_x' + str(n)][row], data_dict['box2_y' + str(n)][row]] for n in range(1, 5) ]

        #scan pixels
        highlight_frame = copy.deepcopy(frame)
        labeled_frame = np.zeros((frame.shape[0], frame.shape[1], 1))
        
        #return frame_label, labeled_frame_for_viewing
        for y_coord, y_slice in enumerate(frame):
            for x_coord, colors in enumerate(y_slice):
                box1_flag = self.check_in_box(x_coord, y_coord, box1)
                box2_flag = self.check_in_box(x_coord, y_coord, box2)

                if box1_flag:
                    highlight_frame[y_coord][x_coord][0] = 255
                    highlight_frame[y_coord][x_coord][1] = 255
                    highlight_frame[y_coord][x_coord][2] = 0.0

                    
                    labeled_frame[y_coord][x_coord][0] = 1
                
                elif box2_flag:
                    highlight_frame[y_coord][x_coord][0] = 0.0
                    highlight_frame[y_coord][x_coord][1] = 255
                    highlight_frame[y_coord][x_coord][2] = 255

                    
                    labeled_frame[y_coord][x_coord][0] = 1

                else:
                    
                    labeled_frame[y_coord][x_coord][0] = 0

                #break
            #break
        return highlight_frame, labeled_frame

    def get_interp_coords(self, x, y, x_interp):
        cs = CubicSpline(x, y)
        y_interp = cs(x_interp)
        return y_interp

    def interpolate_box_coords(self, data_dict):
        frame_idx_arr = data_dict['frame_idx']

        box1_x_coords = [data_dict['box1_x' + str(n)] for n in range(1,5)]
        box1_y_coords = [data_dict['box1_y' + str(n)] for n in range(1,5)]
        box2_x_coords = [data_dict['box2_x' + str(n)] for n in range(1,5)]
        box2_y_coords = [data_dict['box2_y' + str(n)] for n in range(1,5)]
        
        frame_idx_interp = np.linspace(frame_idx_arr[0], frame_idx_arr[-1], frame_idx_arr[-1]+1)

        new_data_dict = {}
        new_data_dict['frame_idx'] = frame_idx_interp

        for n_idx in range(0, 4):
            #interpolate for other frames
            box1_x_n_interp = self.get_interp_coords(frame_idx_arr, box1_x_coords[n_idx], frame_idx_interp)
            box1_y_n_interp = self.get_interp_coords(frame_idx_arr, box1_y_coords[n_idx], frame_idx_interp)

            box2_x_n_interp = self.get_interp_coords(frame_idx_arr, box2_x_coords[n_idx], frame_idx_interp)
            box2_y_n_interp = self.get_interp_coords(frame_idx_arr, box2_y_coords[n_idx], frame_idx_interp)

            new_data_dict['box1_x' + str(n_idx+1)] = box1_x_n_interp
            new_data_dict['box1_y' + str(n_idx+1)] = box1_y_n_interp
            new_data_dict['box2_x' + str(n_idx+1)] = box2_x_n_interp
            new_data_dict['box2_y' + str(n_idx+1)] = box2_y_n_interp
            
        return new_data_dict
