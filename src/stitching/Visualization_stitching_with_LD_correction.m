% Stitching with static lens deformation correction 
clc;clear;close all;
DataManager = WBIMFileManager;
%%
exp_group = 'LifeCanvas';
exp_name = '003_20240209';
tile_str = DataManager.load_tile_in_experiment(exp_group, exp_name);
vis_folder = fullfile(DataManager.fp_experiment(exp_group, exp_name), 'visualization', 'Stitched');
write_stitched_data_Q = false;
%% Parameters
stitch_voxel_size_um = [2, 2, 2];
zero_num_sec = 0;
zero_last_section_Q = true;
medfilt_Q = false;
%% Load all the tiles 
mip_str = struct;
[mip_str.yx, mip_str.zx, mip_str.zy] = deal(cell(1, 2));
t_tic = tic;

stitch_set = WBIMMicroscopeMode.Scan;
stitch_tiles = tile_str.(char(stitch_set));
layer_list = 1:4;
% layer_list = 1 : numel(stitch_tiles);
stitch_tiles = cat(1, stitch_tiles{layer_list});
ch_list = stitch_tiles(1).channel;
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
    if write_stitched_data_Q
        stack_fp = fullfile(vis_folder, sprintf('%s_%s_stitched_stack_CH_%d.tif', ...
            exp_group, exp_name, i_ch));
        DataManager.write_tiff_stack(tile_data{i_ch}, stack_fp);
    end
    fprintf('Finish processing channel %d. Elapsed time is %.2f seconds\n', ...
        i_ch, toc(t_tic));
end
%% Merge channel 
rgb_im = zeros([ds_bbox_ll, 3], 'uint8');
rgb_im(:, :, :, 1) = im2uint8(fun_stretch_contrast(tile_data{1}));
rgb_im(:, :, :, 2) = im2uint8(fun_stretch_contrast(tile_data{2}));
% rgb_im = permute(rgb_im, [1,2,4,3]);
% avi_fp = fullfile(vis_folder, sprintf('%s_%s_stitched_stack_merged.avi', ...
%     exp_group, exp_name));
% fun_vis_write_stack_to_avi(rgb_im, avi_fp);
%%
rgb_im_zyx = permute(rgb_im, [3, 1, 4, 2]);
% implay(rgb_im_zyx);
avi_fp = fullfile(vis_folder, sprintf('%s_%s_stitched_stack_merged_zyx.avi', ...
    exp_group, exp_name));
fun_vis_write_stack_to_avi(rgb_im_zyx, avi_fp);
%% Downsample 2x 
% Convert to log
% rgb_im = fun_merge_image_stacks(tile_data, 'method', 'rgb', ...
%     'stretch_contrast_Q', true);

tile_data_sc = cell(num_ch, 1);
% Increase by 1 before taking the log 
% tile_data_sc{1} = im2uint8(fun_stretch_contrast(tile_data{1}));
tile_data_sc{1} = im2uint8(rescale(single(tile_data{1})).^(1/2));
tile_data_sc{2} = im2uint8(rescale(single(tile_data{2})).^(1/3));
tile_data_sc{3} = im2uint8(rescale(single(tile_data{3})).^(1/3));
% for i = 1 : num_ch
%     tile_data_sc{i} = im2uint8(rescale(single(tile_data{i})).^(1/2));
% end
rgb_im = fun_merge_image_stacks(tile_data_sc, 'method', 'GraySkull', 'stretch_contrast_Q', false);
% rgb_im = cat(4, tile_data_sc{[1,3,2]});
% rgb_im = permute(rgb_im, [1,2,4,3]);


% tile_data_sc = cellfun(@fun_stretch_contrast, tile_data, 'UniformOutput', false);
% tmp = fun_stretch_contrast(tile_data{3});
% tmp_zyx = permute(tmp, [3, 1, 2]);
% implay(tmp_zyz);
% ds_ds_bbox_ll = round(ds_bbox_ll / 2);
% im_ch1_ds2 = im2uint8(fun_stretch_contrast(imresize3(tile_data{1}, ds_ds_bbox_ll)));
% im_ch2_ds2 = im2uint8(fun_stretch_contrast(imresize3(tile_data{2}, ds_ds_bbox_ll)));
% im_ch3_ds2 = im2uint8(fun_stretch_contrast(imresize3(tile_data{3}, ds_ds_bbox_ll)));
% rbg_im = repmat(im_ch2_ds2, 1, 1, 1, 3) * 0.6;
% rbg_im(:, :, :, 1) = max(im_ch1_ds2, rbg_im(:, :, :, 1));
% rbg_im(:, :, :, 2) = max(im_ch3_ds2, rbg_im(:, :, :, 2));
% 
% rbg_im = permute(rbg_im, [1,2,4,3]);

rgb_im_zyx = permute(rgb_im, [4,1,3,2]);
rgb_im_zxy = permute(rgb_im, [4,2,3,1]);

avi_fp = fullfile(vis_folder, sprintf('%s_%s_stitched_stack_merged_ds2_yxz_sc.avi', ...
    exp_group, exp_name));
fun_vis_write_stack_to_avi(rgb_im, avi_fp);
% Conver to MP4

avi_fp = fullfile(vis_folder, sprintf('%s_%s_stitched_stack_merged_ds2_zyx_sc.avi', ...
    exp_group, exp_name));
fun_vis_write_stack_to_avi(rgb_im_zyx, avi_fp);
DataManager.write_tiff_stack(rgb_im_zyx, strrep(avi_fp, 'avi', 'tif'), 'color');

avi_fp = fullfile(vis_folder, sprintf('%s_%s_stitched_stack_merged_ds2_zxy_sc.avi', ...
    exp_group, exp_name));
fun_vis_write_stack_to_avi(rgb_im_zxy, avi_fp);
DataManager.write_tiff_stack(rgb_im_zxy, strrep(avi_fp, 'avi', 'tif'), 'color');
