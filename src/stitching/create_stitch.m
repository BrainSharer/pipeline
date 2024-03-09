% Stitching with static lens deformation correction 
clc;clear;close all;
DataManager = WBIMFileManager;
%%
exp_group = 'LifeCanvas';
exp_name = '003_20240209';
tile_str = DataManager.load_tile_in_experiment(exp_group, exp_name);
% vis_folder = fullfile(DataManager.fp_experiment(exp_group, exp_name), 'visualization', 'Stitched');
vis_folder = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK20230126-003/preps/C1/registration/';
write_stitched_data_Q = false;
%% Parameters
stitch_voxel_size_um = [10, 10, 10];
zero_num_sec = 0;
zero_last_section_Q = true;
medfilt_Q = false;
%% Load all the tiles 
mip_str = struct;
[mip_str.yx, mip_str.zx, mip_str.zy] = deal(cell(1, 2));
t_tic = tic;

stitch_set = WBIMMicroscopeMode.Scan;
stitch_tiles = tile_str.(char(stitch_set));
layer_list = 1:21;
% layer_list = 1 : numel(stitch_tiles);
stitch_tiles = cat(1, stitch_tiles{layer_list});
% ch_list = stitch_tiles(1).channel;
ch_list = 1
num_ch = numel(ch_list);
num_tiles = numel(stitch_tiles);
% Compute overall bounding box
bbox_mmxx_um = cat(1, stitch_tiles.tile_mmxx_um);
layer_z_um = [stitch_tiles.layer_z_um];
stack_size_um = stitch_tiles(1).stack_size_um;

stack_size = stitch_tiles(1).stack_size;
ds_stack_size = round(stack_size_um ./ stitch_voxel_size_um);

bbox_mmxx_pxl = cat(1, stitch_tiles.tile_mmxx_pxl);

vol_bbox_z_mx_um = [min(layer_z_um), max(layer_z_um) + stack_size_um(3) - 1];
vol_bbox_mm_um = [min(bbox_mmxx_um(:, 1:2), [], 1), vol_bbox_z_mx_um(1)];
vol_bbox_xx_um = [max(bbox_mmxx_um(:, 3:4), [], 1), vol_bbox_z_mx_um(2)];
vol_bbox_ll_um = vol_bbox_xx_um - vol_bbox_mm_um + 1;

ds_bbox_ll = round(vol_bbox_ll_um ./ stitch_voxel_size_um);

tile_data = cell(1, num_ch);
% This process is dominated by reading H5 file (0.3 second per tile)
for i_ch = 1 : num_ch
    tmp_ch = ch_list(i_ch);
    tmp_stitch_data = zeros(ds_bbox_ll, 'uint16');
    for i = 1 : num_tiles
        try
            tmp_tile = stitch_tiles(i);
            tmp_tile_data = tmp_tile.load_tile(tmp_ch);
            tmp_tile_data = tmp_tile_data{1};
            % Apply lens deformation correction 
            if medfilt_Q
                tmp_tile_data = medfilt3(tmp_tile_data);
            end
            if zero_num_sec && (tmp_tile.layer > 1)
                tmp_tile_data(:, :, 1:zero_num_sec) = 0;
            end
            if zero_last_section_Q
                tmp_tile_data(:, :, end) = 0;
            end
            tmp_tile_bbox_mm_um = [tmp_tile.tile_mmxx_um(1:2), tmp_tile.layer_z_um];
            tmp_tile_bbox_ll_um = [tmp_tile.tile_mmll_um(3:4), tmp_tile.stack_size_um(3)];
            tmp_tile_ll_ds_pxl = round(tmp_tile_bbox_ll_um ./ stitch_voxel_size_um);
            % Downsample image stack - need smoothing? 
            tmp_tile_data = imresize3(tmp_tile_data, tmp_tile_ll_ds_pxl);
            tmp_tile.clear_buffer();
            % Local bounding box 
            tmp_local_bbox_um = tmp_tile_bbox_mm_um - vol_bbox_mm_um;
            tmp_local_bbox_mm_ds_pxl = round(tmp_local_bbox_um ./ stitch_voxel_size_um);
            % Deal with edge: 
            tmp_local_bbox_mm_ds_pxl = max(tmp_local_bbox_mm_ds_pxl, 1);
            tmp_local_bbox_xx_ds_pxl = tmp_local_bbox_mm_ds_pxl + tmp_tile_ll_ds_pxl - 1;
            % Max - rendering
            tmp_stitch_data(tmp_local_bbox_mm_ds_pxl(1) : tmp_local_bbox_xx_ds_pxl(1), ...
                tmp_local_bbox_mm_ds_pxl(2) : tmp_local_bbox_xx_ds_pxl(2), ...
                tmp_local_bbox_mm_ds_pxl(3) : tmp_local_bbox_xx_ds_pxl(3)) = max(tmp_stitch_data(...
                tmp_local_bbox_mm_ds_pxl(1) : tmp_local_bbox_xx_ds_pxl(1), ...
                tmp_local_bbox_mm_ds_pxl(2) : tmp_local_bbox_xx_ds_pxl(2), ...
                tmp_local_bbox_mm_ds_pxl(3) : tmp_local_bbox_xx_ds_pxl(3)), tmp_tile_data);
                
            fprintf('Finish adding tile %d (%.3f %%)\n', i, (i/num_tiles) * 100);
        catch ME
            fprintf('Failed to add tile %d (%.3f %%)\n', i, (i/num_tiles) * 100);
        end
    end
    tile_data{i_ch} = tmp_stitch_data;
    % Write to tiff stack? 
    stack_fp = fullfile(vis_folder, sprintf('%s_%s_stitched_stack_CH_%d.tif', ...
        exp_group, exp_name, i_ch));
    DataManager.write_tiff_stack(tile_data{i_ch}, stack_fp);
    fprintf('Finish processing channel %d. Elapsed time is %.2f seconds\n', ...
        i_ch, toc(t_tic));
end
