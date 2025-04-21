from srosignal import poly_plot
from srosignal import utils 
from srosignal import poly_connect 

def poly_extract(
    img_raw_pth = None,
    stem_coords_threshold_pth = None,
    stem_coords_all_pth = None,
    polygon_type_list = ['tri37', 'tri53', 'tri127', 'tri143', 'square'],
    inv_filter = False,
    mp_core = 1,
    l_range = [10, 80],
    max_lengthdiff = 3,
    max_anglediff = 3,
    #! append the supplementary polygon definition
    poly_def = None,
    allow_dot_in_pattern = False,
    
    #* painting options
    line_thickness_t = 2,
    line_thickness_v = 2,
    atol = 4,
    allow_virtual_vertex = True,
    alpha = 0.1,
    img_savpth = ''             
):
    
    polygon_recog = poly_connect.polygon_recog(
        img_raw_pth, stem_coords_threshold_pth, stem_coords_all_pth,
        polygon_type_list, inv_filter, mp_core, l_range, max_lengthdiff,
        max_anglediff, poly_def, allow_dot_in_pattern)
    
    polygon_info_dict = polygon_recog.recog_main()
    
    poly_count = poly_plot.poly_plot(polygon_recog.img_raw,
                                         polygon_info_dict,
                                        polygon_recog.new_centre_list,
                                        poly_def,
                                        inv_filter,
                                        allow_dot_in_pattern,
                                        line_thickness_t,
                                        line_thickness_v,
                                        atol,
                                        allow_virtual_vertex,
                                        alpha,
                                        img_savpth)
    
    poly_count_dict, img_polyconnect, img_raw = poly_count.poly_coord_extract() 
    
    return poly_count_dict, img_polyconnect, polygon_recog.img_raw