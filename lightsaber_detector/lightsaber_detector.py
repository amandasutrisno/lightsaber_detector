
import torch
import torch.nn as nn
import pickle as pk
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

class uNet(nn.Module):
    def __init__(self, crop_y, crop_x):
        super(uNet, self).__init__()
        conv2d_layer1 = nn.Conv2d(3, 64, kernel_size=3)
        self.relu_layer = nn.ReLU()
        conv2d_layer2 = nn.Conv2d(64, 64, kernel_size=3)
        self.maxpool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv2d_layers_p1 = [conv2d_layer1, conv2d_layer2]
        start_channel = 64

        #keep track of sizes for skip connections
        skip_conn_sizes = [[crop_y-4, crop_x-4]]
        start_upsample_size = []
        for n_layer in range(0, 4):
            #print(start_channel)
            #print(start_channel*2)
            if n_layer < 3:
                skip_conn_sizes.append([skip_conn_sizes[-1][0]/2-4, skip_conn_sizes[-1][1]/2-4])
            else:
                start_upsample_size = [skip_conn_sizes[-1][0]/2-4, skip_conn_sizes[-1][1]/2-4]

            self.conv2d_layers_p1.append(nn.Conv2d(start_channel, start_channel*2, kernel_size=3))
            self.conv2d_layers_p1.append(nn.Conv2d(start_channel*2, start_channel*2, kernel_size=3))
            start_channel = start_channel*2
        
        self.skip_conn_sizes = skip_conn_sizes
        self.conv2d_layers_p2 = []
        upsample_sizes = [[start_upsample_size[0]*2-2, start_upsample_size[1]*2-2]]
        #print('upsample sizes:', upsample_sizes)

        for n_layer in range(0, 4):
            upsample_sizes.append( [(upsample_sizes[-1][0]-2)*2 -2 , (upsample_sizes[-1][1]-2)*2 -2 ] )
            self.conv2d_layers_p2.append(nn.Conv2d(start_channel, int(start_channel/2), kernel_size=3))
            self.conv2d_layers_p2.append(nn.Conv2d(int(start_channel/2), int(start_channel/2), kernel_size=3))
            start_channel = int(start_channel/2)

        self.upsample_sizes = upsample_sizes
        self.conv2d_lastlayer = nn.Conv2d(64, 2, kernel_size=1)


        #wrap as learnable parameters
        self.conv2d_layers_p1 = torch.nn.ModuleList(self.conv2d_layers_p1)
        self.conv2d_layers_p2 = torch.nn.ModuleList(self.conv2d_layers_p2)
        
    def forward(self, s):

        s_skip_states = []
        #PART 1: Downsampling
        for n_layer in range(0, 5):
            #run convolutions
            #print(s.shape)
            s = self.conv2d_layers_p1[n_layer*2](s)
            s = self.relu_layer(s)
            s = self.conv2d_layers_p1[n_layer*2+1](s)
            #print(s.shape)
            
            #downsample with maxpooling
            if n_layer < 4:
                #save skip connection before downsample
                s_skip_states.append(s)
                s = self.maxpool_layer(s)

        #PART 2: Upsampling
        for n_layer in range(0, 4):
            #upsample
            s = self.upsample_layer(s)
            #print(s.shape)

            #get skip connection
            s_skip = s_skip_states[-1-n_layer]
            #crop the skip connection
            skip_conn_size = self.skip_conn_sizes[-1-n_layer]
            upsample_size = self.upsample_sizes[n_layer]

            start_crop_y = int(np.floor((skip_conn_size[0] - upsample_size[0])/2))
            fin_crop_y = start_crop_y + int(upsample_size[0])
            start_crop_x = int(np.floor((skip_conn_size[1] - upsample_size[1])/2))
            fin_crop_x = start_crop_x + int(upsample_size[1])

            s_skip_crop = s_skip[:,:, start_crop_y:fin_crop_y, start_crop_x:fin_crop_x]
            #print(s_skip_crop.shape)

            #reduce number of channels
            s = self.conv2d_layers_p2[n_layer*2](s)
            #print(s.shape)
            s = self.conv2d_layers_p2[n_layer*2+1](s + s_skip_crop)
            #print(s.shape)

        #apply final layer, outputs segmentation map
        s = self.conv2d_lastlayer(s)

        return s

class LightsaberModel:
    def __init__(self, path_to_frames, path_to_frame_labels, crop_params, model_params):
        with open(path_to_frames, 'rb') as f:
            self.all_frames = pk.load(f)

        with open(path_to_frame_labels, 'rb') as f:
            self.all_frames_labels = pk.load(f)

        self.crop_y = crop_params['crop_y']
        self.crop_x = crop_params['crop_x']
        self.original_y = crop_params['original_y']
        self.original_x = crop_params['original_x']
        self.final_label_y = crop_params['final_label_y']
        self.final_label_x = crop_params['final_label_x']

        self.batch_size = model_params['batch_size']
        self.num_epochs = model_params['num_epochs']
        self.gamma = model_params['gamma']
        self.weights = model_params['weights']
        self.lr = model_params['lr']
        self.path_to_model_savefile = model_params['path_to_model_savefile']

        self.all_dataset, self.all_dataset_labels = convert_dataset_to_torch_format(self.all_frames, \
                                                                                    self.all_frames_labels, \
                                                                      self.crop_y, self.crop_x, \
                                                                        self.original_y, self.original_x, \
                                                                           self.final_label_y, \
                                                                             self.final_label_x )
       

        self.lightsaber_detector = LightsaberDetector(self.crop_y, self.crop_x, self.original_y, \
                                            self.original_x, self.final_label_y, self.final_label_x, \
                                            self.all_dataset, self.all_dataset_labels, self.weights, \
                                                self.batch_size, self.lr, self.gamma, self.path_to_model_savefile)
        self.lightsaber_detector.split_dataset_make_dataloader()

    def train_model(self):
        self.lightsaber_detector.train_loop(self.num_epochs)

    def view_predictions(self, frames_to_view):
        if frames_to_view > len(self.all_dataset):
            frames_to_view = len(self.all_dataset)
        for n in range(0, frames_to_view):
            self.lightsaber_detector.view_predictions(self.all_dataset[n], self.all_dataset_labels[n])
        
class LightsaberDetector:
    def __init__(self, crop_y, crop_x, original_y, original_x, final_label_y, final_label_x, \
                 all_dataset, all_dataset_labels, weights, batch_size, lr, gamma, path_to_model_savefile):
        self.crop_y = crop_y
        self.crop_x = crop_x
        self.original_y = original_y
        self.original_x = original_x
        self.final_label_y = final_label_y
        self.final_label_x = final_label_x
        self.path_to_model_savefile = path_to_model_savefile

        self.start_crop_y = int(np.floor( (original_y - crop_y)/2))
        self.fin_crop_y = self.start_crop_y + crop_y
        self.start_crop_x = int(np.floor((original_x - crop_x)/2))
        self.fin_crop_x = self.start_crop_x + crop_x

        self.start_crop_y_label = int(np.floor( (original_y - final_label_y)/2))
        self.fin_crop_y_label = self.start_crop_y_label + final_label_y
        self.start_crop_x_label = int(np.floor((original_x - final_label_x)/2))
        self.fin_crop_x_label = self.start_crop_x_label + final_label_x


        self.all_dataset = all_dataset
        self.all_dataset_labels = all_dataset_labels
        self.model = uNet(crop_y, crop_x)
        self.device = torch.device("cuda")
        self.model = self.model.to(self.device)

        self.all_dataset_train = []
        self.all_dataset_test = []
        self.all_dataset_labels_train = []
        self.all_dataset_labels_test = []

        self.dataloader_train = []
        self.dataloader_test = []
        self.batch_size = batch_size
        self.weights = torch.tensor(weights)
        self.weights = self.weights.to(self.device)

        self.loss_fn = torch.nn.CrossEntropyLoss(weight = self.weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        
    def split_dataset_make_dataloader(self):
        #split into train/test
        self.all_dataset_train, self.all_dataset_test, \
            self.all_dataset_labels_train,  self.all_dataset_labels_test = train_test_split(self.all_dataset, 
                                                                                   self.all_dataset_labels,
                                                                                   test_size=0.33)
        
        joint_dataset_train = TensorDataset(torch.tensor(np.array(self.all_dataset_train, dtype=np.float32)), \
                                            torch.tensor(np.array(self.all_dataset_labels_train, dtype=np.longlong)))
        joint_dataset_test = TensorDataset(torch.tensor(np.array(self.all_dataset_test, dtype=np.float32)), \
                                            torch.tensor(np.array(self.all_dataset_labels_test, dtype=np.longlong)))

        self.dataloader_train = DataLoader(dataset=joint_dataset_train, batch_size =self.batch_size, shuffle=True)
        self.dataloader_test = DataLoader(dataset=joint_dataset_test, batch_size =self.batch_size, shuffle=True)
        
    def view_predictions(self, frame_input, frame_true_label):
        frame_data = torch.tensor(np.array(frame_input, dtype=np.float32))
        frame_data = frame_data.to(self.device)
        pred = self.model(frame_data.unsqueeze(0))
        
        final_pred = torch.argmax(pred, dim=1)
        final_pred = final_pred.squeeze(0)
        final_pred = final_pred.unsqueeze(2)
        final_pred_plt = final_pred.cpu().numpy()
        final_pred_show = np.zeros((self.final_label_y, self.final_label_x, 3))

        #here we view the results using matplotlib
        start_crop_y = int(np.floor( (self.crop_y - self.final_label_y)/2))
        fin_crop_y = start_crop_y + self.final_label_y
        start_crop_x = int(np.floor((self.crop_x - self.final_label_x)/2))
        fin_crop_x = start_crop_x + self.final_label_x

        frame_input_plt = torch.tensor(frame_input)
        #print(frame_input_plt.shape)
        frame_input_plt = frame_input_plt[:,start_crop_y:fin_crop_y, start_crop_x:fin_crop_x]
        #print(frame_input_plt.shape)
        frame_input_plt = frame_input_plt.permute(1,2,0)
        #print(frame_input_plt.shape)
        frame_input_plt = frame_input_plt.cpu().numpy()

        frame_true_label_plt = torch.tensor(frame_true_label)
        frame_true_label_plt = frame_true_label_plt.permute(1,2,0)
        frame_true_label_plt = frame_true_label_plt.cpu().numpy()

        
        for y in range(0, self.final_label_y):
            for x in range(0, self.final_label_x):
                if frame_true_label_plt[y, x][0] > 0.99 and final_pred_plt[y, x][0] > 0.99:
                    #true positive set to be white
                    final_pred_show[y, x, 0] =  1.0
                    final_pred_show[y, x, 1] = 1.0
                    final_pred_show[y, x, 2] = 1.0
                elif frame_true_label_plt[y, x][0] > 0.99 and final_pred_plt[y,x][0] <= 0.99:
                    #false negative set to be blue
                    final_pred_show[y, x, 0] =  0.0
                    final_pred_show[y, x, 1] = 0.0
                    final_pred_show[y, x, 2] = 1.0
                elif frame_true_label_plt[y, x][0] <= 0.99 and final_pred_plt[y,x][0] <= 0.99:
                    #true negative set to be black
                    final_pred_show[y, x, 0] = 0.0
                    final_pred_show[y, x, 1] = 0.0
                    final_pred_show[y, x, 2] = 0.0
                elif frame_true_label_plt[y, x][0] <= 0.99 and final_pred_plt[y,x][0] > 0.00:
                    #false positive set to be green
                    final_pred_show[y, x, 0] = 0.0
                    final_pred_show[y, x, 1] = 1.0
                    final_pred_show[y, x, 2] = 0.0
        
        fig, axs = plt.subplots(2, 1)  # 2x2 grid of subplots
        axs[0].imshow(frame_input_plt)
        axs[0].title.set_text('Original photo')
        #axs[1].imshow(frame_true_label_plt, cmap='gray')
        axs[1].imshow(final_pred_show)
        axs[1].title.set_text('Predicted pixel labels(white=lightsaber, black=not lightsaber)')
        plt.tight_layout()
        plt.show()
        
        
    def train_loop(self, num_epochs):
        accuracy_hist_vec = []
        accur_weight = [0.9, 0.1]
        best_avg_weighted_accur = 0.0
        for epoch in range(0, num_epochs):
            print("Epoch:", epoch)
            accuracy_train_by_label = {0:0, 1:0}
            n_pixels_train_by_label = {0:0, 1:0}
            n_batch_checked = 0
            
            #training
            for frame_data, frame_labels in self.dataloader_train:
                n_batch_checked += 1
                #print('n batch checked:', n_batch_checked)
                frame_data2 = frame_data.to(self.device)
                frame_labels2 = frame_labels.to(self.device)
                pred = self.model(frame_data2)
                
                pred_flatten = pred.view(pred.shape[0], pred.shape[1], pred.shape[2]*pred.shape[3])
                frame_labels_flatten = frame_labels2.view(pred.shape[0], 1, pred.shape[2]*pred.shape[3]).squeeze(1)
                frame_labels_np = frame_labels2.cpu().numpy()
                #print(pred_flatten.shape)
                #print(frame_labels_flatten.shape)
                loss = self.loss_fn(pred_flatten, frame_labels_flatten)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                #compute number correct
                pred_after_argmax = torch.argmax(pred, dim=1)
                is_correct_0 = (pred_after_argmax==frame_labels2.squeeze(1)) & (frame_labels2.squeeze(1)==0)
                is_correct_1 = (pred_after_argmax==frame_labels2.squeeze(1)) & (frame_labels2.squeeze(1)==1)
                pred_after_argmax = torch.argmax(pred, dim=1)
                
                #print(is_correct_np.shape)
                #print(is_correct_0.shape)
                n_pixels_1 = frame_labels2.sum()
                n_pixels_0 = pred.shape[0]*pred.shape[2]*pred.shape[3] - n_pixels_1

                n_pixels_train_by_label[0] += n_pixels_0.cpu().numpy().item()
                n_pixels_train_by_label[1] += n_pixels_1.cpu().numpy().item()
                accuracy_train_by_label[0] += is_correct_0.sum().cpu().numpy().item()
                accuracy_train_by_label[1] += is_correct_1.sum().cpu().numpy().item()
                
            #print('accuracy by label:', accuracy_train_by_label)
            for label in accuracy_train_by_label.keys():
                accuracy_train_by_label[label] = (accuracy_train_by_label[label] + 0.0) / (n_pixels_train_by_label[label] + 0.0)
            print('n pixels train by label:', n_pixels_train_by_label)
            print('accuracy by label:', accuracy_train_by_label)

            accuracy_test_by_label = {0:0, 1:0}
            n_pixels_test_by_label = {0:0, 1:0}

            #get test accuracy:
            for frame_data, frame_labels in self.dataloader_test:
                n_batch_checked += 1
                #print('n batch checked:', n_batch_checked)
                frame_data2 = frame_data.to(self.device)
                frame_labels2 = frame_labels.to(self.device)
                pred = self.model(frame_data2)
                
                pred_flatten = pred.view(pred.shape[0], pred.shape[1], pred.shape[2]*pred.shape[3])
                frame_labels_flatten = frame_labels2.view(pred.shape[0], 1, pred.shape[2]*pred.shape[3]).squeeze(1)
                frame_labels_np = frame_labels2.cpu().numpy()

                #compute number correct
                pred_after_argmax = torch.argmax(pred, dim=1)
                is_correct_0 = (pred_after_argmax==frame_labels2.squeeze(1)) & (frame_labels2.squeeze(1)==0)
                is_correct_1 = (pred_after_argmax==frame_labels2.squeeze(1)) & (frame_labels2.squeeze(1)==1)
                pred_after_argmax = torch.argmax(pred, dim=1)
                
                #print(is_correct_np.shape)
                #print(is_correct_0.shape)
                n_pixels_1 = frame_labels2.sum()
                n_pixels_0 = pred.shape[0]*pred.shape[2]*pred.shape[3] - n_pixels_1

                n_pixels_test_by_label[0] += n_pixels_0.cpu().numpy().item()
                n_pixels_test_by_label[1] += n_pixels_1.cpu().numpy().item()
                accuracy_test_by_label[0] += is_correct_0.sum().cpu().numpy().item()
                accuracy_test_by_label[1] += is_correct_1.sum().cpu().numpy().item()
            
            for label in accuracy_test_by_label.keys():
                accuracy_test_by_label[label] = (accuracy_test_by_label[label] + 0.0) / (n_pixels_test_by_label[label] + 0.0)
            print('n pixels test by label:', n_pixels_test_by_label)
            print('accuracy by label:', accuracy_test_by_label)

            #compute best accuracy
            avg_weighted_accur = 0.5*accur_weight[0]*(accuracy_test_by_label[0] + accuracy_train_by_label[0])\
                                + 0.5*accur_weight[1]*(accuracy_test_by_label[1] + accuracy_train_by_label[1])
            
            if avg_weighted_accur > best_avg_weighted_accur:
                print('saving best model...')
                best_avg_weighted_accur = avg_weighted_accur
                #save best model
                torch.save(self.model.state_dict(), self.path_to_model_savefile)
        #loading best model
        self.model.load_state_dict(torch.load(self.path_to_model_savefile))
            
def flip_duplicate_dataset(all_dataset, all_dataset_labels):
    #we duplicate the number of training/test examples by flipping each data horizontally
    new_all_dataset = []
    new_all_dataset_labels = []
    for idx in range(0, len(all_dataset)):
        new_all_dataset.append(all_dataset[idx])
        new_all_dataset_labels.append(all_dataset_labels[idx])

        #flip operation
        new_all_dataset.append(np.flip(all_dataset[idx], axis=2))
        new_all_dataset_labels.append(np.flip(all_dataset_labels[idx], axis=2))
    
    return np.array(new_all_dataset), np.array(new_all_dataset_labels)

def convert_dataset_to_torch_format(all_frames, all_frames_labels, crop_y, crop_x, original_y, original_x, final_label_y, final_label_x):
    #change frame format to (color, height, width)
    all_dataset = []
    all_dataset_labels = []

    start_crop_y = int(np.floor( (original_y - crop_y)/2))
    fin_crop_y = start_crop_y + crop_y

    #print(start_crop_y)
    #print(fin_crop_y)

    start_crop_x = int(np.floor((original_x - crop_x)/2))
    fin_crop_x = start_crop_x + crop_x
    #print(start_crop_x)
    #print(fin_crop_x)

    #compute label crop x and y
    start_crop_y_label = int(np.floor( (original_y - final_label_y)/2))
    fin_crop_y_label = start_crop_y_label + final_label_y

    start_crop_x_label = int(np.floor((original_x - final_label_x)/2))
    fin_crop_x_label = start_crop_x_label + final_label_x

    for idx in range(0, len(all_frames_labels)):
        #process frame
        frame = all_frames[idx]
        frame_rgb =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tens = torch.tensor(frame_rgb)
        frame_tens = frame_tens.transpose(0,2)
        frame_tens = frame_tens.transpose(1,2)

        #crop operation:
        frame_crop = frame_tens[:,start_crop_y:fin_crop_y, start_crop_x:fin_crop_x]

        frame_cpu = frame_crop.cpu().numpy()
        all_dataset.append(frame_cpu)

        #process label frame
        label_frame = all_frames_labels[idx]
        label_frame_tens = torch.tensor(label_frame)
        label_frame_tens = label_frame_tens.transpose(0,2)
        label_frame_tens = label_frame_tens.transpose(1,2)

        #crop operation on final prediction
        label_frame_crop = label_frame_tens[:,start_crop_y_label:fin_crop_y_label, start_crop_x_label:fin_crop_x_label]
        label_frame_cpu = label_frame_crop.cpu().numpy()
        #print(label_frame_tens.shape)
        all_dataset_labels.append(label_frame_cpu)


    return np.array(all_dataset), np.array(all_dataset_labels)