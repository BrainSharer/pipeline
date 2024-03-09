classdef WBIMFileManager < handle
    
    %% Properties
    properties
        HOSTNAME
        SCRIPT_PATH;
        SERVER_SCRIPT_PATH;
        SCRATCH_ROOT_PATH;
        DATA_ROOT_PATH;
        SERVER_ROOT_PATH;
        PYTHON_EXE_PATH = [];
        PYTHON_SCRIPT_PATH = [];
        CONDA_PATH = [];
    end
    
    properties(Dependent)
        fp_acquisition_scratch
        fp_acquisition_disk
        fp_processing_disk
        fp_processing_scratch
    end

    properties(Hidden)
%         HOST_SYSTEM
    end
    %% Setup
    methods
        function obj = WBIMFileManager(host_name)
            persistent host_name_
            if nargin < 1 || isempty(host_name)
                if isempty(host_name_)
                    [~, host_name_] = system('hostname');
                    host_name_ = host_name_(1:end-1);
                end
                host_name = host_name_;
            end
            obj.HOSTNAME = host_name;
            switch host_name
                case {'tobor', 'tobor.eddyod.com'}
                    obj.DATA_ROOT_PATH = '/net/birdstore/Vessel/WBIM';
                    obj.SCRIPT_PATH = '/home/eddyod/programming/pipeline/src/stitching';
                    obj.SCRATCH_ROOT_PATH = '/scratch/Vessel/WBIM';
                    obj.SERVER_ROOT_PATH = '/net/birdstore/Vessel/WBIM';
                    obj.SERVER_SCRIPT_PATH = fullfile(obj.SERVER_ROOT_PATH, 'Script', 'WBIM');                    
                case {'muralis', 'muralis.dk.ucsd.edu'}
                    obj.DATA_ROOT_PATH = '/net/birdstore/Vessel/WBIM';
                    obj.SCRIPT_PATH = '/home/eodonnell/programming/pipeline/src/stitching';
                    obj.SCRATCH_ROOT_PATH = '/scratch/Vessel/WBIM';
                    obj.SERVER_ROOT_PATH = '/net/birdstore/Vessel/WBIM';
                    obj.SERVER_SCRIPT_PATH = fullfile(obj.SERVER_ROOT_PATH, 'Script', 'WBIM');                    
                otherwise
                    error('Unrecognized machine.');
            end
        end
    end
    %% Gets
    methods
        function fp = get.fp_acquisition_scratch(obj)
            fp = fullfile(obj.SCRATCH_ROOT_PATH, 'Acquisition');
        end
        
        function fp = get.fp_acquisition_disk(obj)
            fp = fullfile(obj.DATA_ROOT_PATH, 'Acquisition');
        end

        function fp = get.fp_processing_disk(obj)
            fp = fullfile(obj.DATA_ROOT_PATH, 'Processing');
        end

        function fp = get.fp_processing_scratch(obj)
            fp = fullfile(obj.SCRATCH_ROOT_PATH, 'Processing');
        end
    end
    %% Fileanme and folder name formating
    methods(Static)
        %% Scan image
        function fnb = fn_si_tile_name(file_basename, file_count)
            if ~isempty(file_basename)
                fnb = sprintf('%s_%05d.tif', file_basename, file_count);
            else
                fnb = sprintf('%05d.tif', file_count);
            end
        end
        
        function fnb = fn_si_tile_name_fix_num_frames(file_basename, file_count)
            if ~isempty(file_basename)
                fnb = sprintf('%s_%05d_00001.tif', file_basename, file_count);
            else
                fnb = sprintf('%05d_00001.tif', file_count);
            end
        end
        
        %% Post-processing - step 1
        function fn = fn_tile_folder(grid_sub1, grid_sub2)
            fn = sprintf('%05d_%05d', grid_sub1, grid_sub2);
        end
        
        function fn = fn_layer(layer_id)
            fn = sprintf('%05d', layer_id);
        end
        
        function fn = fn_acq_count(acq_count)
            fn = sprintf('%05d', acq_count);
        end
        
        function fnb = fn_acq_raw_tile_basename(file_count)
            fnb = sprintf('%05d', file_count);
        end
        
        function fn = fn_acq_hdf5_filename()
            fn = sprintf('tile.h5');
        end
        
        function fn = fn_acq_si_info()
            fn = sprintf('si_info.txt');
        end
        
        function fn = fn_acq_si_raw()
            fn = sprintf('raw.tif');
        end
        
        function fn = fn_acq_info()
            % To be determine - the format of the metadata
            fn = sprintf('info.mat');
        end
        
        function fn = fn_acq_info_json()
            % To be determine - the format of the metadata
            fn = sprintf('info.json');
        end
        
        function fpr = fpr_experiment(exp_group, exp_name)
            fpr = fullfile(exp_group, exp_name);
        end
        
        function h5_ds = fn_acq_hdf5_ds(channel_id)
            h5_ds = sprintf('/CH%d', channel_id);
        end
        
        function fn = fn_acq_mip(channel_id)
            fn = sprintf('mip_CH%d.tif', channel_id);
        end
        
        function fn = fn_post_processing_log()
            fn = sprintf('log.txt');
        end
        
        function fn = fn_post_processing_error_log()
            fn = sprintf('err.txt');
        end
        
        
        function fn = fn_tile_manager(exp_group, exp_name, acq_mode)
            fn = sprintf('%s_%s_%s_tile_manager.mat', exp_group, exp_name, acq_mode);
        end
        
        function fn = fn_tcp_server_log_file(exp_group, exp_name)
            fn = sprintf('%s_%s_tcp_server.txt', exp_group, exp_name);
        end

        %% Processing
        function fn = fn_descriptor(version)
            arguments
                version = 0
            end
            if isnumeric(version)
                version = num2str(version);
            end
            fn = sprintf('descriptor_%s.mat', version);
        end
        
        function fn = fn_matched_descriptor(match_dir, version)
            arguments
                match_dir (1,1) {mustBeInteger}
                version (1,1) {mustBeInteger} = 0
            end
            fn = sprintf('matched_desc_dir_%d_v%d.mat',...
                match_dir, version);
        end
    end
    methods        
        function fp = fp_layer(obj, exp_group, exp_name, layer)
            fp = fullfile(obj.fp_acquisition_disk, obj.fpr_layer(exp_group, ...
                exp_name, layer));
        end
        
        function fp = fp_layer_acq(obj, exp_group, exp_name, layer, acq_mode)
            fp = fullfile(obj.fp_layer(exp_group, exp_name, layer), string(acq_mode));
        end
        
        function fp = fp_experiment(obj, exp_group, exp_name)
           fp = fullfile(obj.fp_acquisition_disk, obj.fpr_experiment(exp_group, ...
               exp_name));
        end
        
        function fp = fp_tile_folder(obj, exp_group, exp_name, layer, acq_mode, ...
                grid_sub_1, grid_sub_2, acq_count)
            if nargin < 9
                acq_count = 0;
            end
            fp = fullfile(obj.fp_layer_acq(exp_group, exp_name, layer, acq_mode), ...
                obj.fn_tile_folder(grid_sub_1, grid_sub_2), obj.fn_acq_count(acq_count));            
        end


        %% Stitching 
        function fp = fp_processing_tile_info(obj, exp_group, exp_name)
            arguments
                obj (1,1) WBIMFileManager
                exp_group
                exp_name
            end
            fp = fullfile(obj.fp_processing_disk, exp_group, exp_name, 'info.mat');
        end
    end
    
    %% Post-processing - step 1 - utility - scanning
    methods
        function file_str = util_pp_file_info(obj, layer, acq_mode, ...
                grid_sub_2D, acq_count, channel_list)
            file_str = struct;
            file_str.acq_mode = acq_mode;
            file_str.layer = layer;
            file_str.grid_sub = grid_sub_2D;
            file_str.acq_count = acq_count;
            file_str.channel = channel_list;
            % layer/acq_mode/grid_sub_2D/acq_count/...
            file_str.fprr_tile_folder = obj.fprr_tile_folder(...
                layer, acq_mode, grid_sub_2D(1), grid_sub_2D(2), acq_count);
            file_str.fprr_tile = fullfile(file_str.fprr_tile_folder, ...
                obj.fn_acq_hdf5_filename);
            file_str.fprr_info = fullfile(file_str.fprr_tile_folder, ...
                obj.fn_acq_info);
            file_str.fprr_info_json = fullfile(file_str.fprr_tile_folder, ...
                obj.fn_acq_info_json);
            file_str.fprr_si_info = fullfile(file_str.fprr_tile_folder, ...
                obj.fn_acq_si_info);
            file_str.fprr_raw = fullfile(file_str.fprr_tile_folder, ...
                obj.fn_acq_si_raw);
            file_str.fprr_pplog = fullfile(file_str.fprr_tile_folder, ...
                obj.fn_post_processing_log);
            file_str.fprr_pperr = fullfile(file_str.fprr_tile_folder, ...
                obj.fn_post_processing_error_log);
            
            file_str.fprr_mip = arrayfun(@(x) fullfile(file_str.fprr_tile_folder, ...
                obj.fn_acq_mip(x)), channel_list, 'UniformOutput', false);
            file_str.h5_dataset = arrayfun(@(x) obj.fn_acq_hdf5_ds(x), ...
                channel_list, 'UniformOutput', false);
        end
        
    end
    
    methods(Static)
        function tile_info_vec = util_acq_load_tile_info_from_folder(folder_path)
            tile_dir = dir(fullfile(folder_path, '**', obj.fn_acq_info));
            tile_info_vec = WBIMTileMetadata.empty();
            for i = numel(tile_dir) : -1 : 1
                tmp_dir = tile_dir(i);
                tile_info_vec(i) = obj.load_data(fullfile(tmp_dir.folder, ...
                    tmp_dir.name));
            end
        end        
    end
    %% Post-processing - step 1 utility - overview
    %% Directory
    methods
        function fpr = fpr_layer(obj, exp_group, exp_name, layer)
            fpr = fullfile(obj.fpr_experiment(exp_group, exp_name), ...
                obj.fn_layer(layer));
        end
        
        function fpr = fpr_tile_folder(obj, exp_group, exp_name, layer, ...
                acq_mode, grid_sub_1, grid_sub_2)
            % Relative directory w.r.t. the root folder
            fpr = fullfile(obj.fpr_layer(exp_group, exp_name, layer), ...
                string(acq_mode), obj.fn_tile_folder(grid_sub_1, grid_sub_2));
        end
        
        function fprr = fprr_tile_folder(obj, layer, acq_mode, grid_sub_1, ...
                grid_sub_2, acq_count)
            % Relative directory w.r.t. the experiment folder
            fprr = fullfile(obj.fn_layer(layer), string(acq_mode), obj.fn_tile_folder(grid_sub_1, ...
                grid_sub_2), obj.fn_acq_count(acq_count));
        end

    end
    %% Acquisition
    methods(Static)
        %% Control
        function fn = fn_acq_control_parameter(version)
            arguments
                version (1,1) double = 0
            end
            fn = sprintf('WBIMControlParameters_%d.mat', version);
        end
        
        function fn = fn_acq_ctrl_im(exp_group, exp_name, postfix)
            if nargin < 3
                postfix = [];
            end
            if ~isempty(postfix)
                fn = sprintf('WBIMAcqIm_%s_%s_%s.xml', exp_group, exp_name, postfix);
            else
                fn = sprintf('WBIMAcqIm_%s_%s.xml', exp_group, exp_name);
            end
        end
    end
    
    methods
        function fp = fp_acq_control_parameter(obj, exp_group, exp_name, version)
            arguments
                obj WBIMFileManager
                exp_group char
                exp_name char 
                version (1,1) double = 0
            end
            fp = fullfile(obj.fp_acquisition_disk, obj.fpr_experiment(exp_group, exp_name), ...
                obj.fn_acq_control_parameter(version));            
        end
        
        
        function fp = fp_acq_ctrl(obj, exp_group, exp_name, postfix)
            if nargin < 4
                postfix = [];
            end
            fp = fullfile(obj.fp_acquisition_disk, obj.fpr_experiment(exp_group, exp_name), ...
                obj.fn_acq_ctrl_im(exp_group, exp_name, postfix));
        end
    end
    %% Acquisition post processing
    %%
    methods(Static)
        function fn_prefix_si = fn_acq_tile_fn_prefix_si(file_count, channel_id)
            if nargin < 2
                %                channel_id = []; % place holder at the moment.
            end
            fn_prefix_si = sprintf('%05d', file_count);
        end
    end
    methods
        function fp = fp_tcp_server_log_file(obj, exp_group, exp_name)
            fp = fullfile(obj.fp_acquisition_scratch, obj.fpr_experiment(exp_group, exp_name), ...
                obj.fn_tcp_server_log_file(exp_group, exp_name));
        end
        
        function fp = fp_tile_manager_folder(obj, exp_group, exp_name)
            fp = fullfile(obj.fp_acquisition_disk, obj.fpr_experiment(exp_group, ...
                exp_name), 'tile_manager');
        end
        
        function fp = fp_tile_manager(obj, exp_group, exp_name, acq_mode)
            fp = fullfile(obj.fp_tile_manager_folder(exp_group, exp_name), ...
                obj.fn_tile_manager(exp_group, exp_name, acq_mode));
        end
        
        function tile_list = load_tile_in_layer(obj, exp_group, exp_name, layer, acq_mode)
            if isa(acq_mode, 'WBIMMicroscopeMode')
                acq_mode = string(acq_mode);
            end
            layer_acq_folder = obj.fp_layer_acq(exp_group, exp_name, layer, acq_mode);
            tile_dir = dir(fullfile(layer_acq_folder, '**', obj.fn_acq_info));
            num_tiles = numel(tile_dir);
            tile_list = cell(num_tiles);
            if num_tiles > 0
                for i = 1 : num_tiles
                    tile_list{i} = WBIMTileMetadata.load(fullfile(tile_dir(i).folder,...
                        tile_dir(i).name));
                end
                tile_list = cat(1, tile_list{:});
            end            
        end
        
        function tile_str = load_tile_in_experiment(obj, exp_group, exp_name, acq_mode_list)
            arguments
                obj
                exp_group
                exp_name
                acq_mode_list (1, :) WBIMMicroscopeMode = [WBIMMicroscopeMode.Scan]
            end
            exp_root_folder = obj.fp_experiment(exp_group, exp_name);
            layer_folder_list = dir(exp_root_folder);
            layer_folder_list = layer_folder_list([layer_folder_list.isdir]);
            match_idx = ~cellfun(@isempty, regexpi({layer_folder_list.name}, '[0-9]{5}', 'match'));
            layer_list = arrayfun(@(x) str2double(x.name), layer_folder_list(match_idx));
            layer_list = layer_list(isfinite(layer_list));
            num_layer = numel(layer_list);
            tile_str = struct;
            num_mode = numel(acq_mode_list);
            for i = 1 : num_mode
                acq_mode = acq_mode_list(i);
                tmp_data = cell(1, max(layer_list));
                wb_hdl = waitbar(0, 'Loading...', 'Name', sprintf('Loading %s tiles', acq_mode));
                for iter_layer = 1 : num_layer
                    tmp_layer_idx = layer_list(iter_layer);
                    tmp_data{tmp_layer_idx} = obj.load_tile_in_layer(exp_group, exp_name, ...
                        tmp_layer_idx, acq_mode);
                    waitbar(iter_layer / num_layer, wb_hdl, sprintf('Loading layer %d', ...
                        tmp_layer_idx));
                end
                delete(wb_hdl);
                if num_mode > 1
                    tile_str.(string(acq_mode)) = tmp_data;
                else
                    tile_str = tmp_data;
                end
            end
        end        
    end
    %% Logging
    methods
        % Log folder directory for each layer
        function fp = fp_log_folder(obj, exp_group, exp_name, layer)
            fp = fullfile(obj.fp_acquisition_scratch, obj.fpr_layer(exp_group, ...
                exp_name, layer), 'log');
        end
        
        function fp = fp_log_archive_folder(obj, exp_group, exp_name, layer)
            fp = fullfile(obj.fp_acquisition_disk, obj.fpr_layer(exp_group, ...
                exp_name, layer), 'log');
        end
        
        function fp = fp_imaging_laser_log_file(obj, exp_group, exp_name, layer)
            fp = fullfile(obj.fp_log_folder(exp_group, exp_name, layer), ...
                obj.fn_imaging_laser_log_file(exp_group, exp_name, layer));
        end
        
        function fp = fp_microscope_log_file(obj, exp_group, exp_name, layer)
            fp = fullfile(obj.fp_log_folder(exp_group, exp_name, layer), ...
                obj.fn_microscope_log_file(exp_group, exp_name, layer));
        end

        function fp = fp_ablation_record_folder(obj, exp_group, exp_name, layer)
            fp = fullfile(obj.fp_acquisition_disk, obj.fpr_layer(exp_group, ...
                exp_name, layer), 'Ablation');
        end

        function fp = fp_ablation_instruction(obj, exp_group, exp_name, layer)
            fp = fullfile(obj.fp_ablation_record_folder(exp_group, exp_name, layer), ...
                obj.fn_ablation_instruction(exp_group, exp_name, layer));
        end
        
        function fp = fp_ablation_volume_visualization(obj, exp_group, exp_name, ...
                layer, acq_mode)
            fp = fullfile(obj.fp_ablation_record_folder(exp_group, exp_name, layer), ...
                sprintf('Ablation_after_%s_%s', char(acq_mode),  datestr(now, 'yyyymmdd_HHMMSS')));
        end
        
        function fp = fp_visualization_folder(obj, exp_group, exp_name)
            fp = fullfile(obj.fp_experiment(exp_group, exp_name), 'visualization');
        end

        function fp = fp_stitched_merged_mip(obj, exp_group, exp_name, ...
                layer, acq_mode)
            fp = fullfile(obj.fp_visualization_folder(exp_group, exp_name), ...
                sprintf('%s_%s_layer_%d_%s_stitched_merged_mip_%s.png', ...
                exp_group, exp_name, layer, char(acq_mode), datestr(now, 'yyyymmdd_HHMMSS')));
        end
    end
    
    methods(Static)
        function fn = fn_imaging_laser_log_file(exp_group, exp_name, layer)
            fn = sprintf('%s_%s_layer_%05d_imaging_laser_log.txt', exp_group, exp_name, layer);
        end
        
        function fn = fn_microscope_log_file(exp_group, exp_name, layer)
            fn = sprintf('%s_%s_layer_%05d_microscope_log.txt', exp_group, exp_name, layer);
        end

        function fn = fn_ablation_instruction(exp_group, exp_name, layer)
            fn = sprintf('%s_%s_layer_%05d_ablation_at_%s.mat', exp_group, ...
                exp_name, layer, datestr(now, 'yyyymmdd_HHMMSS'));
        end
    end
    %% Utility
    methods

    end
    methods(Static)
        function output = load_data(fp)
            [~, ~, ext] = fileparts(fp);
            switch ext
                case '.mat'
                    output = load(fp);
                    field_name = fieldnames(output);
                    if numel(field_name) == 1
                        output = getfield(output, field_name{1}); %#ok<GFLD>
                    end
                    
                case {'.tiff', '.tif'}
                    output = WBIMFileManager.load_single_tiff(fp);
                case {'.nii', '.gz'}
                    if strcmp(ext, '.gz')
                        assert(endsWith(fp, '.nii.gz'));
                    end
                    output = niftiread(fp);
                case {'.xml'}
                    output = xml_read(fp);
                case {'.nrrd'}
                    output = nrrdread(fp);
                case {'dcm'}
                    output = dicomread(fp);
                otherwise
                    error('Cannot recoginize file format %s', ext);
            end
        end        

        function write_data(output_fp, data)
           [tmp_f, tmp_n, tmp_ext] = fileparts(output_fp);
           
           if ~isfolder(tmp_f)
               mkdir(tmp_f);
           end
           switch tmp_ext
               case {'.mat'}
                   if isstruct(data)
                       save(output_fp, '-struct', 'data', '-v7.3');
                   else
                       save(output_fp, 'data', '-v7.3');
                   end
               case {'.tiff', '.tif'}
                   write_tiff_stack(data, output_fp);
               otherwise
                   error('Unrecognized output file format');
           end
           assert(isfile(output_fp), sprintf('Fail to write file %s for unknown reason', ...
               output_fp));
        end

        %% Read / write Tiff
        function output = load_single_tiff(data_fp, section_list)
            warning('off', 'MATLAB:imagesci:tiffmexutils:libtiffWarning');
            warning('off', 'imageio:tiffmexutils:libtiffWarning');
            % load_single_tiff loads tiff (stack) at data_fp. If
            % section_list is not given, loads all section in the stack.
            if nargin < 2
                section_list = [];
            end
            if ~isfile(data_fp)
                error('File %s does not exist', data_fp);
            end
            image_info = imfinfo(data_fp);
            try
                tifLink = Tiff(data_fp, 'r');
                num_slice = size(image_info,1);
                image_size_1 = image_info(1).Height;
                image_size_2 = image_info(1).Width;
                image_bit_depth = image_info(1).BitDepth;
                
                if isempty(section_list)
                    section_list = 1 : num_slice;
                end
                n_specified_slice = length(section_list);
                
                switch image_bit_depth
                    case 32
                        image_type = 'single';
                    case 64
                        image_type = 'double';
                    case 16
                        switch tifLink.getTag('SampleFormat')
                            case 1
                                image_type = 'uint16';
                            case 2
                                image_type = 'int16';
                        end
                    case {8, 24}
                        switch tifLink.getTag('SampleFormat')
                            case 1
                                image_type = 'uint8';
                            case 2
                                image_type = 'int8';
                        end
                    case 1
                        image_type = 'logical'; % In matlab, logical is actually uint8
                    otherwise
                        error('Unrecongnized image bit depth %d', image_bit_depth);
                end
                
                if isscalar(section_list)
                    output = tifLink.read();
                else
                    if image_bit_depth == 24
                        output = zeros(image_size_1, image_size_2, 3, num_slice, image_type);
                        for iSection = 1 : n_specified_slice
                            tifLink.setDirectory(section_list(iSection));
                            output(:,:,:,iSection) = tifLink.read();
                        end
                    else
                        output = zeros(image_size_1, image_size_2, num_slice, image_type);
                        for iSection  = 1 : n_specified_slice
                            tifLink.setDirectory(section_list(iSection));
                            output(:,:,iSection) = tifLink.read();
                        end
                    end
                end
            catch ME
                tifLink.close();
                rethrow(ME.message)
            end
            tifLink.close();
        end
        
        function write_tiff_stack(inputArray, fp, image_type,output_mode)
            % Adapt from YoonOh Tak's scrip saveastiff
            % This function also support output single section tiff.
            [folder_path, ~, ~] = fileparts(fp);
            if ~isfolder(folder_path)
                warning('Folder does not exist. Create folder.');
                mkdir(folder_path)
            end
            if nargin < 3
                image_type = 'grayscale';
                output_mode = 'overwrite';
            elseif nargin < 4
                output_mode = 'overwrite';
            end
            
            switch image_type
                case 'grayscale'
                    tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
                    [im_height, im_width, im_depth] = size(inputArray);
                    tagstruct.SamplesPerPixel = 1;
                case {'color', 'RGB'}
                    tagstruct.Photometric = Tiff.Photometric.RGB;
                    [im_height, im_width, im_cc, im_depth] = size(inputArray);
                    tagstruct.SamplesPerPixel = im_cc;
                    if im_cc == 4
                        tagstruct.ExtraSamples = Tiff.ExtraSamples.AssociatedAlpha;
                    end
            end
            tagstruct.ImageLength = im_height;
            tagstruct.ImageWidth = im_width;
            tagstruct.Compression = Tiff.Compression.None;
            % (RGB RGB,RGB RGB,RGB RGB),
            % http://www.awaresystems.be/imaging/tiff/tifftags/planarconfiguration.html
            tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
            switch class(inputArray)
                case {'uint8', 'uint16', 'uint32', 'logical'}
                    tagstruct.SampleFormat = Tiff.SampleFormat.UInt;
                case {'int8', 'int16', 'int32'}
                    tagstruct.SampleFormat = Tiff.SampleFormat.Int;
                case {'single', 'double', 'uint64', 'int64'}
                    tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
                otherwise
                    error('Unsupported format');
            end
            % Bits per sample
            switch class(inputArray)
                case {'logical'}
                    tagstruct.BitsPerSample = 1;
                case {'uint8', 'int8'}
                    tagstruct.BitsPerSample = 8;
                case {'uint16', 'int16'}
                    tagstruct.BitsPerSample = 16;
                case {'uint32', 'int32'}
                    tagstruct.BitsPerSample = 32;
                case {'single'}
                    tagstruct.BitsPerSample = 32;
                case {'double', 'uint64', 'int64'}
                    tagstruct.BitsPerSample = 64;
                otherwise
                    error('Unsupported format');
            end
            % Rows per strip
            maxstripsize = 8192;
            byte_depth = tagstruct.BitsPerSample/8;
            tagstruct.RowsPerStrip = ceil(maxstripsize/(im_width*(byte_depth)*tagstruct.SamplesPerPixel)); % http://www.awaresystems.be/imaging/tiff/tifftags/rowsperstrip.html
            inputArray = reshape(inputArray, im_height, im_width, tagstruct.SamplesPerPixel, im_depth);
            % Open file
            switch output_mode
                case {'overwrite', 'w'}
                    if (im_height * im_width * tagstruct.SamplesPerPixel * im_depth * byte_depth) > 2^32 - 1
                        tfile = Tiff(fp, 'w8');
                    else
                        tfile = Tiff(fp, 'w');
                    end
                case {'append', 'r+'}
                    tfile = Tiff(fp, 'r+');
                    while ~tfile.lastDirectory() % Append a new image to the last directory of an exiting file
                        tfile.nextDirectory();
                    end
                    tfile.writeDirectory();
            end
            for secID = 1 : im_depth
                tfile.setTag(tagstruct);
                tfile.write(inputArray(:,:,:,secID));
                if secID ~= im_depth
                    tfile.writeDirectory();
                end
            end
            tfile.close();
        end


        %% JSON file
        function json_str = load_json_file(data_fp)
            assert(isfile(data_fp), 'File does not exist.')
            fid = fopen(data_fp, 'r');
            try 
                json_str = jsondecode(char(fread(fid, inf)'));
                fclose(fid);
            catch ME
                fclose(fid);
                rethrow(ME);
            end            
        end
        %% Move files        
        function exit_code = move_file_local_w(source_fp, dest_fp, asynchronous_Q)
            if nargin < 3
                asynchronous_Q = false;
            end
            cmd_str = sprintf('move "%s" "%s"', source_fp, dest_fp);
            if asynchronous_Q
                % /b starts the application without creating a new window
                cmd_str = sprintf('start /b %s', cmd_str);
            end
            exit_code = system(cmd_str);
        end
        
        function copy_file_locally_w(source_folder_path, dest_folder_path, ...
                source_file_name, asynchronous_Q)
            if nargin < 4
                asynchronous_Q = false;
            end
            % /MOV deletes files after copy
            cmd_str = sprintf('robocopy "%s" "%s" "%s" /Copy:DATSO', ...
                source_folder_path, dest_folder_path, source_file_name);
            if asynchronous_Q
                cmd_str = sprintf('start /b %s', cmd_str);
            end
            system(cmd_str);
        end
        
        function run_command_in_wsl(linux_cmd_str)
            cmd_str = sprintf('wsl %s', linux_cmd_str);
            system(cmd_str);
        end
        
        function fp_str_wsl = fp_convert_w2wsl(fp_str_w)
            fp_str_w = fp_str_w{1};
            disk_name = fp_str_w(1);
            assert(strcmp(fp_str_w(2:3), ':\'));
            fp_str_wsl = strrep(fp_str_w(4:end), filesep, '/');
            disk_path_wsl = sprintf('/mnt/%s', lower(disk_name));
            fp_str_wsl = sprintf('%s/%s', disk_path_wsl, fp_str_wsl);
        end
        %% XML
        %% H5
        function output = load_and_parse_h5_file(filepath)
            % To be implemented
            if ~isfile(filepath)
                error('%s does not exist', filepath);
            else
                tmp_info = h5info(filepath);
            end
            output = struct;
            num_ds = numel(tmp_info.Datasets);
            for i = 1 : num_ds
                tmp_ds_str = tmp_info.Datasets(i);
                output.(tmp_ds_str.Name) = struct;
                output.(tmp_ds_str.Name).data = h5read(tmp_info.Filename, ...
                    sprintf('%s%s', tmp_info.Name, tmp_ds_str.Name));
                num_att = numel(tmp_ds_str.Attributes);
                for j = 1 : num_att
                    output.(tmp_ds_str.Name).(tmp_ds_str.Attributes(j).Name) = ...
                        tmp_ds_str.Attributes(j).Value;
                end
            end
        end
    end
    %% Copy files
    methods
        function copy_file_local_wsl(obj, source_fp_w, target_fp_w, asynchronous_Q)
            % WSL is not for production propose and might have network
            % issue (not verified)
            if nargin < 4
                asynchronous_Q = false;
            end
            [disk_folder, ~, ~] = fileparts(target_fp_w);
            if ~isfolder(disk_folder)
                mkdir(disk_folder);
            end            
            source_fp_w = obj.fp_convert_w2wsl(source_fp_w);
            target_fp_w = obj.fp_convert_w2wsl(target_fp_w);
            cmd_str = sprintf('rsync -ra --checksum "%s" "%s"', source_fp_w, target_fp_w);
            if asynchronous_Q
                cmd_str = sprintf('%s &', cmd_str);
            end
            obj.run_command_in_wsl(cmd_str);
        end
        
        function fp = fp_download_if_not_exist(obj, fp)
            if ~isfile(fp)
                switch obj.HOSTNAME
                    case {'bird.dk.ucsd.edu', 'muralis', 'bird'}
%                         fp = strrep(fp, obj.ROOT_PATH, obj.SERVER_ROOT_PATH);
                        if ~isfile(fp)
                            warning('File exist on neither local disk nor server');
                        end
                    case 'Precision7530'
                        % 1. Windows linux subsystem is required to run
                        % wsl. See the following link for reference:
                        % https://docs.microsoft.com/en-us/windows/wsl/interop
                        
                        % 2. A public key must be set up for ssh before rsync.
                        % to copy files from the server. To set up the
                        % public key, see: https://www.tecmint.com/ssh-passwordless-login-using-ssh-keygen-in-5-easy-steps/
                        % Notice that when creating the ras file, do NOT
                        % rename the file, otherwise ssh will still ask for
                        % password
                        disp('File does not exist locally. Try to download from the server');
                        fp_wsl = strrep(fp, filesep, '/');
                        fp_wsl = strrep(fp_wsl, 'C:', '/mnt/c');
                        [folder_path, ~, ~] = fileparts(fp);
                        if ~isfolder(folder_path)
                            mkdir(folder_path);
                        end
                        fm_remote = WBIMFileManager('bird');                        
                        fp_remote = strrep(fp, obj.DATA_ROOT_PATH, fm_remote.DATA_ROOT_PATH);
                        fp_remote = strrep(fp_remote, filesep, '/');
                        fp_remote = sprintf('bird:%s', fp_remote);
                        command_str = sprintf('wsl rsync -arv %s %s', fp_remote, fp_wsl);
                        system(command_str);
                        if ~isfile(fp)
                            warning('File does not exist on local machien nor remote workstation');
                        end
                    case 'XiangJi-PC'
                        disp('File does not exist locally. Try to download from the server');
                        fp_wsl = strrep(fp, filesep, '/');
                        fp_wsl = strrep(fp_wsl, 'D:', '/mnt/d');
                        [folder_path, ~, ~] = fileparts(fp);
                        if ~isfolder(folder_path)
                            mkdir(folder_path);
                        end
                        fp_remote = strrep(fp, obj.ROOT_PATH, obj.SERVER_ROOT_PATH);
                        fp_remote = strrep(fp_remote, filesep, '/');
                        fp_remote = sprintf('bird:%s', fp_remote);
                        command_str = sprintf('wsl rsync -arv %s %s', fp_remote, fp_wsl);
                        system(command_str);
                        if ~isfile(fp)
                            warning('File does not exist on local machien nor remote workstation');
                        end
                    otherwise
                        error('Unrecognized machine');
                        %                         file_path_parts = strsplit(pwd, filesep);
                        %                         output = fullfile(file_path_parts{1}, 'Vessel');
                end
            end
        end
        
        function download_tile_info_in_experiment(obj, exp_group_name, exp_name)
            root_data_folder = obj.fp_experiment(exp_group_name, exp_name);
            
            bird_fm = WBIMFileManager('bird');
            bird_exp_folder = strrep(sprintf('bird:%s', bird_fm.fp_experiment(exp_group_name, exp_name)), filesep, '/');
            
            ubuntu_fp = strrep(strrep(root_data_folder, 'C:', '/mnt/c'), filesep, '/');
            cmd = sprintf("rsync -rav --include='info.*' --include='*/' --include='WBIMControlParameters*.mat' --exclude='*' %s/ %s", ...
                bird_exp_folder, ubuntu_fp);
            obj.run_command_in_wsl(cmd)
        end
        
        
        function sync_script_to_server(obj)
            sync_source = obj.SCRIPT_PATH;
            sync_target = obj.SERVER_SCRIPT_PATH;
            % Do not sync the hidden files (e.g. under .git/)
            sync_str = sprintf('rsync -rav --exclude=".*" %s/ %s', ...
                sync_source, sync_target);
            system(sync_str);
            fprintf('Finish synchronizing local script to the data folder.\n');
        end
    end
    
    methods(Static)
        function fp_wsl = convert_win_filepath_to_wsl_filepath(fp_win)
            assert(ischar(fp_win) || isstring(fp_win));
            assert(fp_win(2) == ':');
            disk_id = fp_win(1);
            fp_in_disk = strrep(fp_win(4:end), '\', '/');
            disk_wsl = sprintf('/mnt/%s', lower(disk_id));
            fp_wsl = sprintf('%s/%s', disk_wsl, fp_in_disk);
        end
    end
end
