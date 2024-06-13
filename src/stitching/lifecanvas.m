stitch_voxel_size_um = [2, 2, 2];
zero_num_sec = 0;
zero_last_section_Q = true;
medfilt_Q = false;
%% Load all the tiles 
mip_str = struct;
[mip_str.yx, mip_str.zx, mip_str.zy] = deal(cell(1, 2));
t_tic = tic;
