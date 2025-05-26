from srosignal import utils 
import cv2
import numpy as np
from itertools import product, combinations
import multiprocessing as mp

class polygon_recog:
    def __init__(self,
                img_raw_pth = None, 
                stem_coords_threshold_pth = None,
                stem_coords_all_pth = None,
                polygon_type_list = ['tri37', 'tri53', 'tri127', 'tri143', 'square'],
                inv_filter = False,
                dup_remove_maxdis = 7,
                mp_core = 1,
                l_range = [10, 80],
                max_lengthdiff = 3,
                max_anglediff = 3,
                #* polygon definition
                polygon_def_dict = None, 
                allow_dot_in_pattern = False,):

        ''' 
        img_raw_pth: path to the raw image 
        stem_coords_threshold_pth: path to the atom coordinates that are carried out with binary thresholding
        stem_coords_all_pth: path to all the atomic coordinates before thresholding
        polygon_type_list: list of polygon types to be recognized
        inv_filter: if True, turn on inverse point (full coord set - thresholded coord set) recognition
        mp_core: number of cores for multiprocessing
        l_range: lower and upper limit for bound length of the polygon
        max_lengthdiff: maximum length difference for the polygon recognition (in pixels)
        max_anglediff: maximum angle difference for the polygon recognition (in degrees)
        allow_dot_in_pattern: if True, allow the dot (atom) in the recognized polygon pattern
        '''
        
        if img_raw_pth is None:
            raise ValueError("Please provide the path to the raw image.")
        if stem_coords_threshold_pth is None:
            raise ValueError("Please provide the path to the atom coordinates that are carried out with binary thresholding.")
        if inv_filter and stem_coords_all_pth is None:
            raise ValueError("If turn on inverse point recognition, please provide the full atomic coordinates pluts thresholded coordinates.")
        if polygon_type_list is None and polygon_def_dict is None:
            raise ValueError("Please provide the polygon type list or polygon definition dictionary.")
        
        self.new_centre_list = utils.centre_list_gen(inv_filter, stem_coords_threshold_pth, stem_coords_all_pth, dup_remove_maxdis)
        self.dis_matrix = utils.dis_mat_calc(mp_core, self.new_centre_list).mat_gen()
        self.img_raw = cv2.imread(img_raw_pth, cv2.IMREAD_GRAYSCALE)
        self.l_range = l_range
        self.centre_grids = utils.grid_prepare(self.l_range, self.img_raw, self.new_centre_list)
        
        self.max_lengthdiff = max_lengthdiff    
        self.max_anglediff = max_anglediff
        
        self.polygon_type_list = polygon_type_list 
        self.polygon_def_dict = polygon_def_dict
        self.allow_dot_in_pattern = allow_dot_in_pattern
        
        self.ind_map = {
            (0,1): [[1,0],[2,0]],
            (0,2): [[0,1],[2,1]],
            (1,2): [[0,2],[1,2]],
        }

        self.diff_bond_map = {
            0: [0,1],
            1: [0,2],
            2: [1,2],
        }
        
        self.mp_core = mp_core
        
    def recog_grid(self, count):
        
        c_grid = self.centre_grids[count]

        poly_recog_dict = {}
        for polygon_type in self.polygon_type_list:
            poly_recog_dict[polygon_type] = {
                'para_ind_real': np.empty((0,2,2)),
                'para_ind_virtual': np.empty((0,2,2)),
                'orient': [],
            }
        
        for i, j, k in combinations(c_grid, 3):
            
            ijk_ind = np.array([i,j,k])
            dis_list = [self.dis_matrix[i,j], self.dis_matrix[i,k], self.dis_matrix[j,k]]
            bond_discrimitive = np.array([
                np.prod(np.sign(dis_list[_]-self.l_range)) for _ in range(len(dis_list))
            ])
            
            ind_validbond = np.where(np.abs(bond_discrimitive+1)<1e-2)[0]
            #* Filter for range
            if len(ind_validbond) < 2:
                continue 
            
            #* Filter for equal length 
            dis_diff = np.abs(np.array([dis_list[d_i]-dis_list[d_j] for d_i, d_j in combinations(range(len(dis_list)), 2)]))
            ind_validlength = np.where(dis_diff < self.max_lengthdiff)[0] #* Only one bond pair can be identical in length
            
            if len(ind_validlength) < 1:
                continue

            ri, rj, rk = self.new_centre_list[i], self.new_centre_list[j], self.new_centre_list[k]
                
            angle_list_temp = np.rad2deg([
                utils.angle_calc(rj-ri, rk-ri),
                utils.angle_calc(ri-rj, rk-rj),
                utils.angle_calc(ri-rk, rj-rk),
            ])
            
            rji = (rj-ri)/np.linalg.norm(rj-ri)
            rki = (rk-ri)/np.linalg.norm(rk-ri)
            rij = (ri-rj)/np.linalg.norm(ri-rj)
            rkj = (rk-rj)/np.linalg.norm(rk-rj)
            rik = (ri-rk)/np.linalg.norm(ri-rk)
            rjk = (rj-rk)/np.linalg.norm(rj-rk)
            
            if np.max(angle_list_temp) < 80:
            
                ind_validlength = [np.argmax([
                    rji@rki, rij@rkj, rik@rjk
                ])]
                
            else:
                ind_validlength = [np.argmin([
                    rji@rki, rij@rkj, rik@rjk
                ])]

            # if len(ind_validlength) > 1:
            #     ind_validlength = [np.argmin(dis_diff)]

            #* Filter for angle
            ind_map_ = self.ind_map[tuple(self.diff_bond_map[ind_validlength[0]])]
            bond_1, bond_2 = ind_map_[0], ind_map_[1]
            
            angle = utils.angle_calc(self.new_centre_list[ijk_ind[bond_1[0]]]-self.new_centre_list[ijk_ind[bond_1[1]]], 
                        self.new_centre_list[ijk_ind[bond_2[0]]]-self.new_centre_list[ijk_ind[bond_2[1]]])
            
            for angle in [angle%np.pi, np.pi-angle%np.pi]:

                vertex_ind = self.new_centre_list[ijk_ind[bond_1[1]]] #* Vertex for equal-length bond
                bot_ind = self.new_centre_list[np.array([ijk_ind[bond_1[0]], ijk_ind[bond_2[0]]])]
                vertex_2nd = utils.c2_dot(vertex_ind, bot_ind)
                
                tri_coord_1, tri_coord_2 = \
                    np.concatenate((vertex_ind.reshape(1,-1), bot_ind), axis=0), \
                    np.concatenate((vertex_2nd.reshape(1,-1), bot_ind), axis=0)
                    
                sav_coord = np.array([
                    [vertex_ind, bot_ind[0]],
                    [vertex_ind, bot_ind[1]],
                    [vertex_2nd, bot_ind[0]],
                    [vertex_2nd, bot_ind[1]],
                ])
                
                vec_coord = 1/2*(bot_ind[0]+bot_ind[1])-vertex_ind

                for polygon_type in self.polygon_type_list:
                    poly_valid = self.polygon_filter(
                        angle, ijk_ind, dis_list,
                        tri_coord_1, tri_coord_2,
                        polygon_type = polygon_type,
                    )
                    if poly_valid:
                        poly_recog_dict[polygon_type]['para_ind_real'] = np.concatenate((poly_recog_dict[polygon_type]['para_ind_real'], sav_coord[:2]), axis=0)
                        poly_recog_dict[polygon_type]['para_ind_virtual'] = np.concatenate((poly_recog_dict[polygon_type]['para_ind_virtual'], sav_coord[2:]), axis=0)
                        poly_recog_dict[polygon_type]['orient'].append(vec_coord)
                        
        return poly_recog_dict

    def polygon_filter(self, angle, ijk_ind, dis_list,
                       tri_coord_1, tri_coord_2,
                       polygon_type = None,):

        
        for polygon_def in self.polygon_def_dict[polygon_type]:
            angle_def, upper_max, lower_max, upper_min, lower_min, _ = polygon_def
            
            if (np.rad2deg(np.abs(angle-angle_def)) < self.max_anglediff 
                    and np.max(dis_list) < upper_max
                    and np.max(dis_list) > lower_max
                    and np.min(dis_list) > lower_min
                    and np.min(dis_list) < upper_min):
                
                #* Check if any dot in the triangle, if so, remove it
                dot_in_tri_list = []
                
                if not self.allow_dot_in_pattern:
                    for i_dot in range(len(self.new_centre_list)):
                        
                        if i_dot in ijk_ind:
                            continue
                        
                        for tri_coords in [tri_coord_1, tri_coord_2]:
                            dot_in_tri_list.append(utils.dot_in_tri(
                                                    tri_coords, 
                                                    self.new_centre_list[i_dot]))
                    
                    if np.sum(dot_in_tri_list) <= 0:
                        return True
        
        return False 
                
    def recog_main(self):
        
        poly_info_dict = {}
        
        with mp.Pool(self.mp_core) as p:
            result_dictbuffer = p.map(self.recog_grid, range(len(self.centre_grids)))
        
        for poly in self.polygon_type_list:
            poly_info_dict.setdefault(poly, {})
            poly_info_dict[poly]['para_ind_real'] = np.empty((0,2,2))
            poly_info_dict[poly]['para_ind_virtual'] = np.empty((0,2,2))
            poly_info_dict[poly]['orient'] = np.empty((0,2))
            
            for res_dict in result_dictbuffer:
                para_ind_real = np.array(res_dict[poly]['para_ind_real'])
                para_ind_virtual = np.array(res_dict[poly]['para_ind_virtual'])
                orient = np.array(res_dict[poly]['orient'])
                
                if len(para_ind_real) > 0:
                    poly_info_dict[poly]['para_ind_real'] = np.concatenate((poly_info_dict[poly]['para_ind_real'], para_ind_real), axis=0)
                    poly_info_dict[poly]['para_ind_virtual'] = np.concatenate((poly_info_dict[poly]['para_ind_virtual'], para_ind_virtual), axis=0)
                    poly_info_dict[poly]['orient'] = np.concatenate((poly_info_dict[poly]['orient'], orient), axis=0)
                
            # result_realind = [res_dict[poly]['para_ind_real'] for res_dict in result_dictbuffer]
            # result_virtualind = [res_dict[poly]['para_ind_virtual'] for res_dict in result_dictbuffer]
            # result_orient = [res_dict[poly]['orient'] for res_dict in result_dictbuffer]
            
            # print(result_realind)
            
            # poly_info_dict[poly]['para_ind_real'] = np.concatenate(result_realind, axis=0)
            # poly_info_dict[poly]['para_ind_virtual'] = np.concatenate(result_virtualind, axis=0)
            # poly_info_dict[poly]['orient'] = np.concatenate(result_orient, axis=0)
            
        return poly_info_dict