import cv2
import numpy as np
import multiprocessing as mp
from srosignal import recog_grid
from srosignal import utils
from sklearn.cluster import KMeans

from itertools import combinations, product

class extract:
    def __init__(self, 
                img_raw_pth = None,
                grid_size = 6,
                mp_core = 1,
                centre_maxdis = 0,
                
                dot_radius = 2,
                dot_color = (0,0,255),
                dot_thickness = 1,
                alpha = 0.5,
                img_savpth = ''):
        
        if img_raw_pth is None:
            raise ValueError("Please provide the path to the raw image.")
        
        self.img_raw = cv2.imread(img_raw_pth, cv2.IMREAD_GRAYSCALE)
        self.img_color = cv2.cvtColor(self.img_raw, cv2.COLOR_GRAY2BGR)
        self.overlay = self.img_color.copy()
        
        #* step 1 to roughly determine the centroids
        self.centre_list = recog_grid.grid_signal_search(
            mp_core, self.img_raw, grid_size
        ).main()
        
        self.threshold_centre = centre_maxdis
        self.mp_core = mp_core  
        
        #* plotting parameters
        self.dot_radius = dot_radius
        self.dot_color = dot_color
        self.dot_thickness = dot_thickness
        self.alpha = alpha
        self.img_savpth = img_savpth

    def dis_mat_centre(self, i):
        i1, i2 = self.combinatorial_ind[i]
        dis = np.linalg.norm(self.centre_list[i1]-self.centre_list[i2])
        return dis
    
    def dis_mat_centpair(self, i):
        i1, i2 = self.combinatorial_ind[i]
        dis = np.linalg.norm(self.centre_list_pair[i1]-self.centre_list_pair[i2])
        return dis 
         
    def similar_coord(self, dismat_func, centlist_to_use):
        
        buffer_similarcentre = []
        ind_allsimilar = np.empty(0)

        #* Use multi-processing to generate distance matrix
        self.combinatorial_ind = np.array(list(combinations(range(len(centlist_to_use)), 2)))
        with mp.Pool(self.mp_core) as p:
            dis_list_raw = np.array(p.map(dismat_func, range(self.combinatorial_ind.shape[0])))

        dis_mat = np.zeros((len(self.centre_list), len(self.centre_list)))
        dis_mat[self.combinatorial_ind[:,0], self.combinatorial_ind[:,1]] = dis_list_raw
        dis_mat[self.combinatorial_ind[:,1], self.combinatorial_ind[:,0]] = dis_list_raw

        for comb_i in range(len(self.combinatorial_ind)):
            
            i, j = self.combinatorial_ind[comb_i]
            dis = dis_mat[i,j]
            if dis > self.threshold_centre:
                continue
            
            ind_allsimilar = np.unique(np.concatenate((ind_allsimilar, np.array([i,j])), axis=0))
                
            if len(buffer_similarcentre) == 0:
                buffer_similarcentre.append(np.array([i,j]))
                continue
            
            unic_ij = False
            for k in range(len(buffer_similarcentre)):
                buffer_ = buffer_similarcentre[k]
                if np.isin(np.array([i,j]), buffer_).any():
                    buffer_similarcentre[k] = np.unique(np.concatenate((buffer_, np.array([i,j])), axis=0))
                    unic_ij = True
                    break
            
            if not unic_ij:
                buffer_similarcentre.append(np.array([i,j]))
                
        return buffer_similarcentre, ind_allsimilar
    
    def max_search(self, raw_p, img_):
        
        img = img_.copy()
        img = (img - np.mean(img))/np.std(img)
        raw_p = np.round(raw_p, 0).astype(int)
        #* Record the trajectory and return the best one
        best_int = -1
        global_x_min, global_x_max, global_y_min, global_y_max = \
            img.shape[0], 0, img.shape[1], 0
        for _ in range(100): 
            p_searchmat = np.clip(
                #* define the search area
                np.array([raw_p+np.array([i,j]) for i in range(-1,2) for j in range(-1,2)]),
                0, np.array(img.shape)-1 #* what's the optimized search area
            )
            
            p_searchintensity = img[p_searchmat[:,0], p_searchmat[:,1]]
            p_searchintensity_raw = img_[p_searchmat[:,0], p_searchmat[:,1]]
            zero_ind = np.where(p_searchintensity_raw == 0)[0]
            prob_ = utils.softmax(p_searchintensity)
            prob_[zero_ind] = 0
            if np.sum(prob_) == 0:
                prob_ = np.ones(len(p_searchmat))/len(p_searchmat)
            prob_ = prob_/np.sum(prob_)

            max_intensity_ind = np.random.choice(range(len(p_searchmat)), p=prob_)
            raw_p = p_searchmat[max_intensity_ind]
            
            int_ = np.sum(p_searchintensity_raw)
            if int_ > best_int:
                best_int = int_
                best_p = raw_p
            
            #* Store the x_min, x_max, y_min, y_max to draw rectangle
            valid_ind = np.where(p_searchintensity_raw != 0)[0]
            if len(valid_ind) == 0:
                continue
            
            x_min, x_max = np.min(p_searchmat[valid_ind][:,0]), np.max(p_searchmat[valid_ind][:,0])
            y_min, y_max = np.min(p_searchmat[valid_ind][:,1]), np.max(p_searchmat[valid_ind][:,1])
            
            if x_min < global_x_min:
                global_x_min = x_min
            if x_max > global_x_max:
                global_x_max = x_max
            if y_min < global_y_min:
                global_y_min = y_min
            if y_max > global_y_max:
                global_y_max = y_max
            
        return np.array([global_x_min, global_y_min, global_x_max, global_y_max,
                    (global_x_max+global_x_min)/2, (global_y_max+global_y_min)/2]).reshape(1,-1)
        
    def centre_detect(self):
        
        # buffer_similarcentre, _ = self.similar_coord(dismat_func, centlist_to_use)
        
        centre_list_asind = np.round(self.centre_list, 0).astype(int)
        # intensity_list = np.array([
        #     self.img_raw[centre_list_asind[i][0], centre_list_asind[i][1]] for i in range(len(self.centre_list))
        # ])
        
        #* do without prior clustering?
        old_centre_uniq = self.centre_list.copy()
        #* Also calibrate the unic centroids.
        old_centre_uniq_ = np.empty((0,6))
        for old_centre_ in old_centre_uniq:
            k_old_centre_ = self.max_search(old_centre_, self.img_raw)
            old_centre_uniq_ = np.concatenate((old_centre_uniq_, k_old_centre_), axis=0)
        new_centre_list = old_centre_uniq_.copy()
        
        return new_centre_list
    
    def centre_extract_main(self):
        
        #* step 2 calculate similar centroids
        new_centre_list = self.centre_detect()
        self.centre_list_pair = new_centre_list[:,-2:]
        buffer_similarcentre, ind_all_similar = self.similar_coord(self.dis_mat_centpair, new_centre_list)
        
        ind_raw_tot = np.arange(len(new_centre_list))
        ind_raw = np.delete(ind_raw_tot, ind_all_similar.astype(int))
        unic_ind = np.array([i[0] for i in buffer_similarcentre])
        ind_new = np.concatenate((ind_raw, unic_ind), axis=0).astype(int)
        new_centre_list = new_centre_list[ind_new]
        
        return new_centre_list
        
    def plotting(self):
        
        new_centre_list = self.centre_extract_main() 
        #* Draw rectangle from the `max_search` function
        for (x_min, y_min, x_max, y_max) in new_centre_list[:,:4]:
            x_min, y_min, x_max, y_max = int(round(x_min, 0)), int(round(y_min, 0)), int(round(x_max, 0)), int(round(y_max, 0))
            cv2.rectangle(self.img_color, (y_min, x_min), (y_max, x_max), self.dot_color, self.dot_thickness)
            cv2.line(self.img_color, (y_min, x_min), (y_max, x_max), self.dot_color, self.dot_thickness)
            cv2.line(self.img_color, (y_min, x_max), (y_max, x_min), self.dot_color, self.dot_thickness)
            
        cv2.addWeighted(self.overlay, self.alpha, self.img_color, 1-self.alpha, 0, self.img_color)
        cv2.imwrite(self.img_savpth + '/dotted.bmp', self.img_color)
        np.save(self.img_savpth + '/centre_list.npy', new_centre_list) 
        
        return    
    