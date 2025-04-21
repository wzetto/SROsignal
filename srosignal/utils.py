import numpy as np 
import copy
from itertools import product, combinations, permutations
import multiprocessing as mp

def dup_remove(centre_list_buffer, max_dis=9):
    
    ''' 
    remove duplicated points that represents the same atom in STEM images
    '''
    
    x_res = centre_list_buffer[:,0]-centre_list_buffer[:,0].reshape(-1,1)
    y_res = centre_list_buffer[:,1]-centre_list_buffer[:,1].reshape(-1,1)
    d_res = np.tril(np.sqrt(x_res**2+y_res**2))
    delete_ind = np.where((d_res > 1e-5) & (d_res < max_dis)) #* threshold be set to 4
    centre_list_buffer = np.delete(centre_list_buffer, delete_ind[0], axis=0)
    
    return centre_list_buffer

def centre_list_gen(inv_filter, coord_threshold, coord_all=None):

    if inv_filter:
        #* Special treatment: get the null space of the centroids matrix.
        cent_all = np.load(coord_all)[:,-2:]
        cent_filt = np.load(coord_threshold)[:,-2:]

        delete_ind = []
        #! need vectorized
        for i in range(len(cent_all)):
            dis_min = np.min(np.linalg.norm(cent_filt-cent_all[i], axis=1))
            if dis_min < 8:
                delete_ind.append(i)
                
        new_centre_list = dup_remove(np.delete(cent_all, np.array(delete_ind), axis=0), max_dis = 7)
        
    elif not inv_filter:
        cent_filt = np.load(coord_threshold)[:,-2:]
        
        new_centre_list = dup_remove(cent_filt, max_dis = 7)
        
    return new_centre_list

def dot_in_tri(tri_coord, dot_, tol_angle=-1e-2, tol_dis=1e-3):
    
    ''' 
    determine whether a point is inside a triangle or not
    '''
    
    dot = copy.deepcopy(dot_)
    # dot = np.concatenate((dot, dot_virtual), axis=0)
    t1, t2, t3 = tri_coord
    t1x, t1y, t2x, t2y, t3x, t3y = t1[0], t1[1], t2[0], t2[1], t3[0], t3[1]
    tri_bond_vector = np.array([t2-t1, t1-t3, t3-t2]) #* means the vertex is t3, t2, t1
    tri_bond_vector = tri_bond_vector/np.linalg.norm(tri_bond_vector, axis=1).reshape(-1,1)

    #* Projection of dot on each bond
    proj_coord_t1 = np.dot(dot-t1, tri_bond_vector[0])*tri_bond_vector[0] + t1
    inner_dot_t3 = np.dot(proj_coord_t1-dot, proj_coord_t1-t3)
    proj_len_1 = np.linalg.norm(dot-proj_coord_t1) #* t1-t2
    if proj_len_1 < tol_dis and np.min([t1x, t2x]) < dot[0] < np.max([t1x, t2x]) and np.min([t1y, t2y]) < dot[1] < np.max([t1y, t2y]):
        return True

    proj_coord_t3 = np.dot(dot-t3, tri_bond_vector[1])*tri_bond_vector[1] + t3
    inner_dot_t2 = np.dot(proj_coord_t3-dot, proj_coord_t3-t2)
    proj_len_2 = np.linalg.norm(dot-proj_coord_t3) #* t1-t3
    if proj_len_2 < tol_dis and np.min([t1x, t3x]) < dot[0] < np.max([t1x, t3x]) and np.min([t1y, t3y]) < dot[1] < np.max([t1y, t3y]):
        return True

    proj_coord_t2 = np.dot(dot-t2, tri_bond_vector[2])*tri_bond_vector[2] + t2
    inner_dot_t1 = np.dot(proj_coord_t2-dot, proj_coord_t2-t1)
    proj_len_3 = np.linalg.norm(dot-proj_coord_t2) #* t2-t3

    if proj_len_3 < tol_dis and np.min([t2x, t3x]) < dot[0] < np.max([t2x, t3x]) and np.min([t2y, t3y]) < dot[1] < np.max([t2y, t3y]):
        return True
    
    #TODO determine the threshold for inner product
    elif np.min([inner_dot_t1, inner_dot_t2, inner_dot_t3]) > tol_angle:
        return True
    else:
        return False

def c2_dot(dot, vec_vertices):
    ''' 
    return v' which is the dot after C2 operation
    '''
    v1, v2 = vec_vertices
    dot_c2 = v2 - (dot-v1)
    return dot_c2

def angle_calc(a, b):
    return np.arccos(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))
 
def rot_mat(theta):
    ''' 
    rotation matrix
    '''
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])

class dis_mat_calc:
    def __init__(self, mp_core, new_centre_list):
        self.mp_core = mp_core
        self.new_centre_list = new_centre_list
    
    def dis_calc(self, i):
        i1, i2 = self.combinatorial_ind[i]
        dis = np.linalg.norm(self.new_centre_list[i1]-self.new_centre_list[i2])
        return dis
    
    def mat_gen(self):
        self.combinatorial_ind = np.array(list(product(range(len(self.new_centre_list)), repeat=2)))
        with mp.Pool(self.mp_core) as p:
            dis_list_raw = np.array(p.map(self.dis_calc, range(self.combinatorial_ind.shape[0])))
        
        dis_matrix = np.zeros((len(self.new_centre_list), len(self.new_centre_list)))
        dis_matrix[self.combinatorial_ind[:,0], self.combinatorial_ind[:,1]] = dis_list_raw
        
        return dis_matrix
        
# def dis_mat_calc(mp_core, new_centre_list):

#     def main(i):
#         i1, i2 = combinatorial_ind[i]
#         dis = np.linalg.norm(new_centre_list[i1]-new_centre_list[i2])
#         return dis
    
#     global combinatorial_ind
#     combinatorial_ind = np.array(list(product(range(len(new_centre_list)), repeat=2)))

#     with mp.Pool(mp_core) as p:
#         dis_list_raw = np.array(p.map(main, range(combinatorial_ind.shape[0])))

#     dis_matrix = np.zeros((len(new_centre_list), len(new_centre_list)))
#     dis_matrix[combinatorial_ind[:,0], combinatorial_ind[:,1]] = dis_list_raw

#     return dis_matrix

def grid_prepare(l_range, img, new_centre_list):
    
    ''' 
    prepare grid of single image to save computational time
    '''
    
    grid_interval = np.max(l_range)
    grid_num = int(np.ceil(np.max(img.shape)/grid_interval))
    # grid_num = 3 #* prefer not to specify it unless accuracy issue.
    grid_x = np.linspace(0, img.shape[0], grid_num)
    grid_y = np.linspace(0, img.shape[1], grid_num)
    grid = np.array(list(product(grid_x, grid_y)))

    mega_grid_interval = grid_interval*2
    mega_grid_num = int(np.ceil(np.max(img.shape)/mega_grid_interval))
    # mega_grid_num = 2 #* prefer not to specify it unless accuracy issue.
    mega_grid_x = np.linspace(0, img.shape[0], mega_grid_num)
    mega_grid_y = np.linspace(0, img.shape[1], mega_grid_num)
    mega_grid = np.array(list(product(mega_grid_x, mega_grid_y)))

    #* find the nearest megagrid for each grid and make a map
    map_grid2mega = [None]*len(grid)
    for i in range(len(grid)):
        dis = np.linalg.norm(mega_grid-grid[i], axis=1)
        ind = np.argsort(dis)[:4]
        map_grid2mega[i] = ind
    
    #* Seperate the centre_list into NxN parts
    centre_grids = [[] for _ in range(len(grid))]
    #! turn on megagrids?
    # centre_megagrids = [[] for _ in range(len(mega_grid))]
    for i in range(len(new_centre_list)):
        centre_ = new_centre_list[i]
        dis = np.linalg.norm(grid-centre_, axis=1)
        valid_zone = np.where(dis <= np.max(l_range)*np.sqrt(2))[0]
        for j in valid_zone:
            centre_grids[j].append(i)
            # for k in range(4):
            #     centre_megagrids[map_grid2mega[j][k]].append(i)
            
    return centre_grids

def mat_identical(mat_gt, mat, atol=1e-3):
   mat_buffer = mat_permute(mat)
   similar = False
   for mat_ in mat_buffer:
       if np.allclose(mat_gt, mat_, atol=atol):
           similar = True 
           break 
   
   return similar

def mat_permute(mat):
   mat_buffer = []
   for comb_ind in permutations(range(len(mat)), len(mat)):
       mat_buffer.append([mat[i] for i in comb_ind])
       
   return np.array(mat_buffer)

def bivariate_gs(rho, std_x, std_y, mu_x, mu_y, x, y):
    norm_term = 1/(2*np.pi*std_x*std_y*np.sqrt(1-rho**2))
    exp_term = np.exp(-1/(2*(1-rho**2))*(
        ((x-mu_x)/std_x)**2 + ((y-mu_y)/std_y)**2 - 2*rho*(x-mu_x)*(y-mu_y)/(std_x*std_y)
    ))
    return norm_term*exp_term

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))