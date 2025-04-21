import cv2 
import numpy as np

class single_mask:
    def __init__(self, img_raw_pth, int_range_convert, img_savpth):
        img_raw = cv2.imread(img_raw_pth, cv2.IMREAD_GRAYSCALE)
        self.img_raw = img_raw
        self.int_info = int_range_convert
        self.pth_savbase = img_savpth
        
    def int_loc(self):
        min_int_list, max_int_list = self.int_info[:,0], self.int_info[:,1]
        
        assert np.all(min_int_list < max_int_list), 'Maximum intensity should be larger than minimum intensity'
        
        self.filter_zone = [np.where((min_int <= self.img_raw) & (max_int >= self.img_raw))
                            for min_int, max_int in zip(min_int_list, max_int_list)]
    
    def mask_set(self):
        mask_buffer = np.zeros((len(self.int_info), self.img_raw.shape[0], self.img_raw.shape[1]))

        
        desire_int = self.int_info[:,2]
        

        for i in range(len(self.int_info)):
            mask = mask_buffer[i,:,:]
            mask[self.filter_zone[i]] = desire_int[i]
            #* if keep the original intensity
            # mask[self.filter_zone[i]] = self.img_raw[self.filter_zone[i]]

            mask_buffer[i,:,:] = mask
            
        self.mask_buffer = mask_buffer
        
    def main_img(self):
        
        self.int_loc()
        self.mask_set()
        for i in range(len(self.int_info)):
            min_int, max_int, target_int = self.int_info[i]
            cv2.imwrite(self.pth_savbase+f'/{min_int}_{max_int}_{target_int}.bmp',
                self.mask_buffer[i])

            print(f'Image {i} saved:', self.pth_savbase+f'/{min_int}_{max_int}_{target_int}.bmp')
            
        return self.mask_buffer 
