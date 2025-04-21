import numpy as np 
import multiprocessing as mp

#* algo is that for binary STEM image there exists a grid size s = NxN
#* for any grid smaller than size s there are at most two areas with different intensity degrees
#* and each area must be simply connected

class grid_signal_search:
    def __init__(self, mp_core, img, grid_size):
        self.mp_core = mp_core
        self.img = img
        self.grid_size = grid_size
        
    def center_def(self, img_grid):
        ''' 
        img must be binary
        '''
        w_loc = np.where(img_grid == 255)
        w_coord = np.array(list(zip(w_loc[0], w_loc[1])))
        
        if len(w_coord) > 0:
            return True, np.mean(w_coord, axis=0)
        return False, np.array([])

    def grid_sampling(self):
        ''' 
        grid size should be no bigger than half of lattice constant in pixels
        typical lattice constant is 18 pixels for experimental STEM image
        '''
        grid_buffer = []
        for i in range(0, self.img.shape[0], self.grid_size):
            for j in range(0, self.img.shape[1], self.grid_size):
                i_min = np.clip(i, 0, self.img.shape[0]-1)
                j_min = np.clip(j, 0, self.img.shape[1]-1)
                i_max = np.clip(i+self.grid_size, 0, self.img.shape[0]-1)
                j_max = np.clip(j+self.grid_size, 0, self.img.shape[1]-1)
                grid = [i_min, i_max, j_min, j_max]
                
                grid_buffer.append(grid)
        
        self.grid_buffer = np.array(grid_buffer)   
        return

    def grid_centre(self, i):
        grid_ = self.grid_buffer[i]
        i_min, i_max, j_min, j_max = grid_
        img_grid = self.img[i_min:i_max, j_min:j_max]
        centre_exist, centre_ = self.center_def(img_grid)
        if not centre_exist:
            return np.empty((0,2))
        centre_ = centre_.astype(int)
        centre_ = np.array([i_min+centre_[0], j_min+centre_[1]])
        return centre_.reshape(1,-1)
    
    def main(self):
        self.grid_sampling()
        centre_res = mp.Pool(self.mp_core).map(self.grid_centre, range(len(self.grid_buffer)))
        centre_list = np.concatenate(centre_res, axis=0)
        
        return centre_list