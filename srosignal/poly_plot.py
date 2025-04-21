from srosignal import utils 
import numpy as np 
import cv2
import copy
import os

class poly_plot:
    def __init__(self,
                 img_raw,
                 poly_info_dict,
                 new_centre_list,
                 polygon_def_dict,
                 inv_filter = False,
                 allow_dot_in_pattern = False,
                 line_thickness_t = 2,
                 line_thickness_v = 2,
                 atol = 4,
                 allow_virtual_vertex = True,
                 alpha = 0.1,
                 img_savpth = ''):
        
        #* load image as canvas
        self.img_raw = img_raw 
        self.img_color = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
        self.overlay = self.img_color.copy()
        self.alpha = alpha
        self.poly_info_dict = poly_info_dict
        self.allow_virtual_vertex = allow_virtual_vertex    
        self.new_centre_list = new_centre_list
        self.poly_def = polygon_def_dict
        
        #* painting parameters
        self.line_thickness_t = line_thickness_t
        self.line_thickness_v = line_thickness_v
        self.atol = atol
        
        #* sav options
        if not inv_filter:
            self.img_type = '_real'
        else:
            self.img_type='_inv'
            
        self.sav_denote = ''
        if not allow_dot_in_pattern:
            self.sav_denote = '_pure_polygon'
            
        self.img_savpth = [os.path.join(img_savpth, self.img_type + self.sav_denote + '_raw.bmp'),
                           os.path.join(img_savpth, self.img_type + self.sav_denote + '.bmp')]
        
    def trip_make(self, para_ind):
        tri_list = []
        for i in range(len(para_ind//2)):
            ind_para_ = np.unique(np.concatenate((
                para_ind[2*i], para_ind[2*i+1]
            )).reshape(-1,2), axis=0)
            tri_list.append(ind_para_)
        
        return tri_list

    def virt_buffer_make(self, para_v, count_v):
        return para_v[[2*i for i in range(count_v)]][:,0,:]

    def tri_mat_make(self, para_t, para_v):
        len_para = len(para_t)
        tri_mat = []
        for i in range(len_para//2):
            p1t, p2t, p1v, p2v = \
                para_t[2*i], \
                para_t[2*i+1], \
                para_v[2*i], \
                para_v[2*i+1]
                
            tri_t = np.unique(np.concatenate(np.array([p1t, p2t]), axis=0), axis=0) #* 3x2 matrix
            tri_v = np.unique(np.concatenate(np.array([p1v, p2v]), axis=0), axis=0) #* 3x2 matrix
            
            tri_mat.append([tri_t, tri_v])
        
        return tri_mat

    def para_renorm(self, para_t, para_v, atol):
        len_paraind = len(para_t)
        mat_validbuffer = []
        para_validind = []
        for i in range(len_paraind//2):
            p1t, p2t, p1v, p2v = para_t[2*i], \
                para_t[2*i+1], \
                para_v[2*i], \
                para_v[2*i+1]
                
            p = np.concatenate((p1t, p2t, p1v, p2v), axis=0)
            p = np.unique(p, axis=0) #* 4-dot
            
            sim_mat = False
            for mat in mat_validbuffer:
                if utils.mat_identical(mat, p, atol=atol):
                    sim_mat = True 
                    break
            
            if not sim_mat:
                mat_validbuffer.append(p)
                para_validind.append(i)
        
        para_validind = np.array(para_validind).astype(int)
        num_poly = len(para_validind)
        para_validind = np.array([[2*i, 2*i+1] for i in para_validind]).flatten()
        para_t_valid = para_t[para_validind]
        para_v_valid = para_v[para_validind]
        
        return para_t_valid, para_v_valid, num_poly
    
    def valid_poly(self, para_t, para_v, virt_vert_buffer_all, allow_virtual=True):
        
        if allow_virtual:
            return para_t, para_v
        
        else:
            tri_mat = self.tri_mat_make(para_t, para_v)
            ind_array = np.arange(len(tri_mat))

            invalid_poly_ind = []
            tol_angle = 1 #TODO important to modify
            tol_dis = -1e-2
            for tri_i, bi_tri in enumerate(tri_mat):
                tri1, tri2 = bi_tri #* two tri. form a quad.
                
                for virt in virt_vert_buffer_all:
                    if ((utils.dot_in_tri(tri1, virt, tol_angle=tol_angle, tol_dis=tol_dis)) 
                        or (utils.dot_in_tri(tri2, virt, tol_angle=tol_angle, tol_dis=tol_dis))):
                        invalid_poly_ind.append(tri_i)
                        break
                    
            valid_poly_ind = np.setdiff1d(ind_array, np.array(invalid_poly_ind))
            valid_poly_ind = np.array([[2*i, 2*i+1] for i in valid_poly_ind]).flatten()
            
            para_t_valid = para_t[valid_poly_ind]
            para_v_valid = para_v[valid_poly_ind]
            
            return para_t_valid, para_v_valid
    
    def draw_poly(self, bond_t, bond_v, c_bond):
        bond_buffer = []
        for bond_t, bond_v in zip(bond_t, bond_v):
            x1, y1, x2, y2 = bond_t
            x1_v, y1_v, x2_v, y2_v = bond_v
            
            mat_t = np.array([[x1, y1],[x2, y2]])
            mat_v = np.array([[x1_v, y1_v],[x2_v, y2_v]])
                
            sim_bond_t, sim_bond_v = False, False
            for bond in bond_buffer:
                if utils.mat_identical(bond, mat_t, atol=self.atol):
                    sim_bond_t = True
                
                if utils.mat_identical(bond, mat_v, atol=self.atol):
                    sim_bond_v = True
                
            if not sim_bond_t:
                bond_buffer.append(mat_t)
                cv2.line(self.img_color, (y1, x1), (y2, x2), c_bond, self.line_thickness_t)
                
            if not sim_bond_v:
                bond_buffer.append(mat_v)
                cv2.line(self.img_color, (y1_v, x1_v), (y2_v, x2_v), c_bond, self.line_thickness_v)
        
        return

    def draw_dot(self,):
        self.img_color_raw = copy.deepcopy(self.img_color)
        #* Draw dots 
        radius = 2
        thickness = 1
        for (x, y) in self.new_centre_list:
            x, y =  int(round(x, 0)), int(round(y, 0))
            cv2.circle(self.img_color, (y,x), radius, (112,233,255), thickness)
            
        cv2.addWeighted(self.overlay, self.alpha, self.img_color_raw, 1-self.alpha, 0, self.img_color_raw)
        cv2.addWeighted(self.overlay, self.alpha, self.img_color, 1-self.alpha, 0, self.img_color)
        
        return 
   
    def poly_coord_extract(self):
        
        virt_vert_buffer_all = []
        poly_vertex_dict = {}
        poly_count = {}
        for key, val in self.poly_info_dict.items():
            poly_ind_true = val['para_ind_real']
            poly_ind_virtual = val['para_ind_virtual']
            poly_orient = val['orient']
            poly_ind_true_valid, poly_ind_virtual_valid, count_poly = self.para_renorm(poly_ind_true, poly_ind_virtual, atol=4)
            
            virt_vert_buffer_all.append(
                self.virt_buffer_make(poly_ind_virtual_valid, count_poly)
            )
            
            poly_vertex_dict[key] = {
                'poly_ind_true': poly_ind_true_valid,
                'poly_ind_virtual': poly_ind_virtual_valid,
                'poly_orient': poly_orient,
                'count_poly': count_poly
            }
        
        virt_vert_buffer_all = np.concatenate(virt_vert_buffer_all, axis=0)
        
        for key, val in poly_vertex_dict.items():
            poly_ind_true, poly_ind_virtual = self.valid_poly(
                val['poly_ind_true'], val['poly_ind_virtual'], virt_vert_buffer_all, self.allow_virtual_vertex
            )
            
            poly_ind_true = np.unique(poly_ind_true.reshape(-1,2,4), axis=0).reshape(-1,4).astype(int)
            poly_ind_virtual = np.unique(poly_ind_virtual.reshape(-1,2,4), axis=0).reshape(-1,4).astype(int)
            
            poly_vertex_dict[key]['poly_ind_true'] = poly_ind_true
            poly_vertex_dict[key]['poly_ind_virtual'] = poly_ind_virtual

            bond_v, bond_t = [], []
            for i in range(len(poly_ind_true)):
                bond_t.append(tuple(poly_ind_true[i]))
                bond_v.append(tuple(poly_ind_virtual[i]))
                
            self.draw_poly(bond_t, bond_v, self.poly_def[key][0][-1])
            
            poly_count[key] = len(bond_t)/2
            
        self.draw_dot()
        cv2.imwrite(self.img_savpth[0], self.img_color_raw)
        cv2.imwrite(self.img_savpth[1], self.img_color)
        
        print('Image saved at: ', self.img_savpth[1])
        
        return poly_count, self.img_color, self.img_color_raw