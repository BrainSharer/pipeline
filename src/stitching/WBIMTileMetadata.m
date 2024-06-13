classdef WBIMTileMetadata < matlab.mixin.Copyable & dynamicprops
    properties
        experiment_group char
        experiment char
        acq_mode  WBIMMicroscopeMode
        acq_count (1,1) double
        channel (1, :) double
        % Grid coordinates
        layer (1,1) double
        layer_z_um (1,1) double
        
        grid_size (1, 2) double
        grid_ind (1,1) double
        grid_sub (1,2) double
        stack_size (1, 3) double
        stack_size_um (1, 3) double
        pixel_size_um (1, 3) double
        overlap_size_um (1, 3) double
        overlap_size (1, 3) double
        
        center_sub (1,2) double
        center_xy_um (1,2) double
        
        tile_mmxx_pxl (1, 4) double
        tile_mmll_pxl (1, 4) double
        tile_mmxx_um (1, 4) double
        tile_mmll_um (1, 4) double
        
        neighbor_grid_ind (1, :) double
        
        % directory
        fprr_tile_folder char
        fprr_tile char
        fprr_info char
        fprr_info_json char
        fprr_si_info char
        fprr_raw char
        fprr_pplog char
        fprr_pperr char
        fprr_mip (1, :) cell
        h5_dataset (1, :) cell
        
        SI_filepath char
        % Acquisition record
        t_init char
        t_done char
        stage_xyz_um_done (1, 3) double
        sample_xyz_um_done (1, 3) double
        piezo_z0_r_um  (1,1) double
        normal_acq_Q (1,1) logical
%         channel_offset (1, 4) int16
    end
    
    properties(Transient, Hidden, NonCopyable, SetAccess=protected)
        tile (1, :) cell
        mip (1, :) cell
        mip_stat (1, :) cell
        mip_step (1, :) cell
        stat struct
    end
    
    properties(Transient, NonCopyable)
        processing_state (1,1) WBIMProcessingState = WBIMProcessingState.Unprocessed
        sync_state (1,1) WBIMProcessingState = WBIMProcessingState.Unprocessed
    end
    
    properties(Constant, Hidden)
        ImageJVisVariableName = 'WBIMTileMetadataVisTile';
    end
    
    properties(Transient, Access=private)
        data_manager
        last_known_processing_state (1,1) logical = false
    end
    %%
    properties(Dependent)
        save_root_folder char
        fp_tile char
        fp_info char
        fp_info_json char
        init_SI_file_exist_Q (1,1) logical
        raw_file_exist_Q (1,1) logical
        h5_file_exist_Q (1,1) logical
        mip_file_exist_Q (1,1) logical
        stat_file_exist_Q (1,1) logical
        file_processed_Q (1,1) logical
    end
    
    properties(Dependent, Hidden)
        fp_stat char
        fp_birdstore_folder
    end
    
    properties(Access=private, NonCopyable, Transient)
        save_root_folder_
        fp_tile_
        fp_info_
        fp_info_json_
    end
    %% Main functions
    methods
        function obj = WBIMTileMetadata(exp_group, exp_name, layer_idx, acq_mode, ...
                grid_ind, grid_hdl, acq_count, channel, overwriteQ)
            if nargin < 9
                overwriteQ = true;
            end
            obj.init_handles();
            obj.experiment_group = exp_group;
            obj.experiment = exp_name;
            obj.layer = layer_idx;
            obj.acq_mode = acq_mode;
            obj.grid_ind = grid_ind;
            obj.acq_count = acq_count;
            obj.channel = channel;
            [obj.mip, obj.tile] = deal(cell(1, numel(channel)));
            
            obj.add_grid_info(grid_hdl);
            obj.add_dir_info();
            if ~overwriteQ
                tile_fp = fullfile(obj.save_root_folder, obj.fprr_tile);
                while isfile(tile_fp)
                    obj.acq_count = obj.acq_count + 1;
                    obj.add_dir_info();
                    tile_fp = fullfile(obj.save_root_folder, obj.fprr_tile);
                    fprintf('%s already exist. Increment acquisition count number\n', ...
                        tile_fp);
                end
            end
        end
        
        function exit_code = save(obj)
            exit_code = -1;
            [folder, ~] = fileparts(obj.fp_info);
            if ~isfolder(folder)
                mkdir(folder);
            end
            save(obj.fp_info, 'obj');
            obj.write_json_file();
        end
        
        function mip = fetch_process_output(obj)
            if ~isempty(obj.processing_job)
                output = fetchOutputs(obj.processing_job);
                mip = output.mip;
            end
        end
        
        function write_json_file(obj, json_fp)
            if nargin < 2
                json_fp = obj.fp_info_json;
            end
            info_json = jsonencode(obj, 'PrettyPrint', true);
            if isfile(json_fp)
                warning('%s already exist. Overwrite.', json_fp);
            end
            fid = fopen(json_fp, 'w');
            fwrite(fid, info_json);
            fclose(fid);
        end
        
        function clear_buffer(obj)
            obj.mip = [];
            obj.tile = [];
            obj.mip_stat = [];
            obj.mip_step = [];
        end
        
        function vis_tile(obj, ch_id)
            validateattributes(ch_id, {'numeric'}, {'scalar'});
            if evalin('base', "exist('IJM')")
                tile_im = obj.load_tile(ch_id);
                tile_im = permute(tile_im{1}, [2,1,3]);
                tile_im(tile_im<0) = 0;
                tile_im = cast(tile_im, 'uint16');
                assignin('base', obj.ImageJVisVariableName, tile_im);
                evalin('base', sprintf("IJM.show('%s')", ...
                    obj.ImageJVisVariableName));
            else
                error('ImageJ has not been initialized yet');
            end
        end
    end
    %% Helper functions
    methods
        %% Gets
        function val = get.save_root_folder(obj)
            if isempty(obj.save_root_folder_)
                obj.save_root_folder_ = fullfile(obj.data_manager.fp_acquisition_disk, ...
                    obj.data_manager.fpr_experiment(obj.experiment_group, obj.experiment));
            end
            val = obj.save_root_folder_;
        end
        
        function val = get.fp_tile(obj)
            if isempty(obj.fp_tile_)
                obj.fp_tile_ = fullfile(obj.save_root_folder, obj.fprr_tile);
            end
            val = obj.fp_tile_;
        end
        
        function val = get.fp_info(obj)
            if isempty(obj.fp_info_)
                obj.fp_info_ = fullfile(obj.save_root_folder, obj.fprr_info);
            end
            val = obj.fp_info_;
        end
        
        function val = get.fp_info_json(obj)
            if isempty(obj.fp_info_json_)
                obj.fp_info_json_ = strrep(obj.fp_info, '.mat', '.json');
            end
            val = obj.fp_info_json_;
        end
        
        function val = get.init_SI_file_exist_Q(obj)
            val = isfile(obj.SI_filepath);
        end
        
        function val = get.raw_file_exist_Q(obj)
            val = isfile(fullfile(obj.save_root_folder, obj.fprr_raw));
        end
        
        function val = get.h5_file_exist_Q(obj)
            val = isfile(fullfile(obj.save_root_folder, obj.fprr_tile));
        end
        
        function val = get.mip_file_exist_Q(obj)
            val = false;
            for i = 1 : numel(obj.fprr_mip)
                tmp_fn = obj.fprr_mip{i};
                if ~isfile(fullfile(obj.save_root_folder, ...
                        tmp_fn))
                    return;
                end
            end
            val = true;
        end
        
        function val = get.stat_file_exist_Q(obj)
            val = isfile(obj.fp_stat);
        end
        
        function val = get.file_processed_Q(obj)
            if obj.last_known_processing_state
                val = true;
            else
                if obj.h5_file_exist_Q && obj.mip_file_exist_Q && ...
                        obj.stat_file_exist_Q
                    obj.last_known_processing_state = true;
                    val = true;
                else
                    val = false;
                end
            end
        end
        
        function val = get.fp_stat(obj)
            val = strrep(obj.fp_info, 'info', 'stat');
        end
        
        function val = get.fp_birdstore_folder(obj)
            persistent pmb
            if isempty(pmb)
               pmb = WBIMFileManager('birdstore');
            end
            local_folder = fullfile(obj.save_root_folder, obj.fprr_tile_folder);
            if pmb.DATA_ROOT_PATH(1) == '/'
                remote_file_sep = '/';
            else
                remote_file_sep = '\';
            end
            val = strrep(local_folder, obj.data_manager.DATA_ROOT_PATH, ...
                pmb.DATA_ROOT_PATH);
            val = sprintf('birdstore:%s', strrep(val, filesep, remote_file_sep));
        end
        
        function val = get_tile_folder(obj, wslQ)
            if nargin < 2
                wslQ = false;
            end
            val = sprintf('%s\\', fullfile(obj.save_root_folder, obj.fprr_tile_folder));
            % Convert to filepath in WSL
            if wslQ && ispc
               [~, val] = system(sprintf('wsl wslpath "%s"', val)); 
            end            
        end
        %% Processing filepaths
        function val = process_tile_folder(obj)
            val = fullfile(obj.data_manager.fp_processing_disk, ...
                obj.data_manager.fpr_experiment(obj.experiment_group, obj.experiment), ...
                obj.fprr_tile_folder);
        end
        
        function val = fp_descriptor(obj, version)
            arguments
                obj WBIMTileMetadata
                version = 0
            end
            val = fullfile(obj.process_tile_folder, obj.data_manager.fn_descriptor(version));          
        end
        
        function val = fp_matched_descriptor(obj, search_dir, version)
            arguments
                obj WBIMTileMetadata
                search_dir (1,1) {mustBeMember(search_dir, [1,2,3])}
                version (1,1) {mustBeInteger} = 0
            end
            val = fullfile(obj.process_tile_folder, obj.data_manager.fn_matched_descriptor(...
                search_dir, version));
        end
        %% Online processing        
        function [successQ, done_time] = check_processing_state(obj)
            log_fp = fullfile(obj.save_root_folder, obj.fprr_pplog);
            successQ = false;
            done_time = "";
            if isfile(log_fp)
                log_txt = readlines(log_fp);
                % Remove empty lines
                log_txt = log_txt(log_txt ~= "");
                last_line = log_txt{end};
                success_idx = strfind(last_line, "Successfully finish processing SI Tiff file");
                if ~isempty(success_idx)
                    successQ = true;
                    done_time = last_line(1:success_idx-2);
                end
            end            
        end
        
        
    end
    
    methods(Hidden)
        function obj = init_handles(obj)
            persistent data_manager
            if isempty(data_manager)
                data_manager = WBIMFileManager();
            end
            obj.data_manager = data_manager;
            obj.add_dir_info();
        end
        
        function add_grid_info(obj, grid_hdl)
            info = grid_hdl.get_stack_info(obj.grid_ind);
            obj.add_str_to_exist_prop(info);
        end
        
        function add_dir_info(obj)
            dir_info = obj.data_manager.util_pp_file_info(obj.layer, string(obj.acq_mode), ...
                obj.grid_sub, obj.acq_count, obj.channel);
            
            obj.add_str_to_exist_prop(dir_info);
        end
        
        function add_str_to_exist_prop(obj, add_str)
            str_fn = fieldnames(add_str);
            for iter_fn = 1 : numel(str_fn)
                tmp_fn = str_fn{iter_fn};
                if isprop(obj, tmp_fn)
                    obj.(tmp_fn) = add_str.(tmp_fn);
                    %                 else
                    %                     fprintf('%s is not among the properties of this class\n', tmp_fn);
                end
            end
        end
        
    end
    %% Loading
    methods
        function mip_cell = load_mip(obj, channel_id)
            if nargin < 2
                channel_id = obj.channel;
            end
            if isempty(obj.mip)
                obj.mip = cell(1, max(obj.channel));
            end
            num_channel_to_load = numel(channel_id);
            mip_cell = cell(num_channel_to_load, 1);
            for i_c = 1 : num_channel_to_load
                tmp_ch = channel_id(i_c);
                tmp_ch_idx = find(tmp_ch == obj.channel);
                if isempty(tmp_ch_idx)
                    continue;
                else
                    assert(isscalar(tmp_ch_idx));
                end
                % TODO: change to read from the stat file? Seems to be
                % faster 
                if isempty(obj.mip{tmp_ch}) || ...
                        all(obj.mip{tmp_ch} == intmin(WBIMConfig.SI_CONVERTED_IM_TYPE), 'all')
                    mip_fp = fullfile(obj.save_root_folder, obj.fprr_mip{tmp_ch_idx});
                    obj.data_manager.fp_download_if_not_exist(mip_fp);
                    try
                        obj.mip{tmp_ch} = obj.data_manager.load_data(mip_fp);
                    catch ME
                        obj.mip{tmp_ch} = repelem(intmin(WBIMConfig.SI_CONVERTED_IM_TYPE), ...
                            obj.stack_size(1), obj.stack_size(2));
                        warning('Fail to read %s\n', mip_fp);
                    end
                end
                mip_cell{i_c} = obj.mip{tmp_ch};
            end
        end
        
        function raw_stack = load_raw(obj)
            raw_fp = fullfile(obj.save_root_folder, obj.fprr_raw);
            obj.data_manager.fp_download_if_not_exist(raw_fp);
            raw_tiff_hdl = ScanImageTiffReader.ScanImageTiffReader(raw_fp);
            raw_stack = permute(raw_tiff_hdl.data(), [2,1,3]);
        end
        
        function tile_cell = load_tile(obj, channel_id)
            if nargin < 2
                channel_id = obj.channel;
            end
            if isempty(obj.tile)
                obj.tile = cell(1, max(obj.channel));
            end
            hdf5_fp = fullfile(obj.save_root_folder, obj.fprr_tile);
            obj.data_manager.fp_download_if_not_exist(hdf5_fp);
            num_channel = numel(channel_id);
            tile_cell = cell(num_channel, 1);
            for i_c = 1 : num_channel
                tmp_ch = channel_id(i_c);
                tmp_ch_idx = find(tmp_ch == obj.channel);
                if isempty(obj.tile{tmp_ch})
                    obj.tile{tmp_ch} = permute(h5read(hdf5_fp, ...
                        sprintf('%s/%s', obj.h5_dataset{tmp_ch_idx}, 'raw')), [2,1,3]);
                end
                tile_cell{i_c} = obj.tile{tmp_ch};
            end
        end
        
        function mip_step_cell = load_step_mip(obj, channel_id)
            arguments
                obj
                channel_id (1, :) double = obj.channel
            end
            
            ch_name = arrayfun(@(x) sprintf('CH%d', x), channel_id, 'UniformOutput', false);
            num_ch = numel(ch_name);
            if isempty(obj.stat)
                obj.stat = struct;
            end
            if isempty(obj.mip_step)
                obj.mip_step = cell(1, max(channel_id));
            end
            mip_step_cell = cell(1, num_ch);
            for i = 1 : num_ch
                tmp_ch = channel_id(i);
                fn = ch_name{i};
                if isempty(obj.mip_step{tmp_ch})
                    if isfield(obj.stat, fn)        
                        obj.data_manager.fp_download_if_not_exist(obj.fp_stat);
                        tmp_data = load(obj.fp_stat, fn);
                        tmp_data = tmp_data.(fn);
                        % TODO: merge with load_stat
                        % Wired ordering... The stat file was saved in
                        % Python. It seems that a 2D matrix is saved in
                        % correct order, but the third dimension remains in
                        % as the first dimension... 
                        obj.mip_step{tmp_ch} = permute(tmp_data.step_mip, [2,3,1]);
                    end
                end
                mip_step_cell{i} = obj.mip_step{tmp_ch};
            end            
        end
        
        function data = load_data(obj, data_name, channel_id)
            if nargin < 3
                channel_id = obj.channel;
            end
            switch data_name
                case 'mip'
                    data = obj.load_mip(channel_id);
                case 'raw'
                    data = obj.load_raw(channel_id);
                case 'tile'
                    data = obj.load_tile(channel_id);
                otherwise
                    error('To be implemented');
            end
        end
        
        function data = load_stat(obj, channel_id)
            if nargin < 2
                channel_id = [];
            end
            if isempty(channel_id)
                ch_name = arrayfun(@(x) sprintf('CH%d', x), obj.channel, 'UniformOutput', false);
            else
                ch_name = arrayfun(@(x) sprintf('CH%d', x), channel_id, 'UniformOutput', false);
            end
            if isempty(obj.stat)
                obj.stat = struct;
            end
            data = struct;
            for i = 1 : numel(ch_name)
                fn = ch_name{i};
                if ~isfield(obj.stat, fn)
                    obj.data_manager.fp_download_if_not_exist(obj.fp_stat);
                    tmp_data = load(obj.fp_stat, fn);
                    tmp_data = tmp_data.(fn);
                    tmp_data.step_mip_pixel_yxz_um = double(tmp_data.step_mip_pixel_yxz_um');
                    tmp_data.step_mip = permute(tmp_data.step_mip, [2,3,1]);
                    % Add other analysis here later
                    %
                    obj.stat.(fn) = tmp_data;
                end
                data.(fn) = obj.stat.(fn);
            end
        end
    end
    
    methods(Static)
        function output = load(filepath)
            % Loading the Classobj
            persistent DM
            if isempty(DM)
                DM = WBIMFileManager();
            end
            if isfolder(filepath)
                filepath = fullfile(filepath, DM.fn_acq_info);
            end
            if isfile(filepath)
                output = DM.load_data(filepath);
            end
            output = output.init_handles();
        end
        
        function output = load_tile_info(exp_group, exp_name, layer, acq_mode,...
                grid_sub_1, grid_sub_2, acq_count)
            persistent DM
            if isempty(DM)
                DM = WBIMFileManager();
            end
            if nargin < 7
                acq_count = 0;
            end
            fp = fullfile(DM.fp_experiment(exp_group, exp_name), ...
                DM.fprr_tile_folder(layer, acq_mode, grid_sub_1,...
                grid_sub_2, acq_count), DM.fn_acq_info);
            output = WBIMTileMetadata.load(fp);
        end
    end
    %% Utilities
    methods
        function open_tile_folder(obj)
            tile_folder = fileparts(obj.fp_tile);
            if ispc
                winopen(tile_folder);
            else
                error('Unrecognized system');
            end
        end
        
    end
    
    methods(Static)
        function c_info = combine_tile_info(tile_info)
            c_info = WBIMTileMetadata.collect_tiles_info(tile_info);
            c_info = WBIMImagingGrid2D.add_combined_region_info(c_info);            
        end
        
        function c_info = combine_tile_info_3D(tile_info)
            c_info = WBIMTileMetadata.collect_tiles_info(tile_info);
            c_info = WBIMImagingGrid2D.add_combined_region_info_3D(c_info);            
        end
        
        function c_info = collect_tiles_info(tile_info)
            c_info = struct;
            if isempty(tile_info)
                return;
            end
            %             class_prop = propert(tile_info(1));
            copy_prop = {'grid_size', 'grid_ind', 'grid_sub', 'layer', 'channel', 'stack_size', ...
                'stack_size_um', 'pixel_size_um', 'center_sub', 'center_xy_um', ...
                'tile_mmxx_pxl', 'tile_mmll_pxl', 'tile_mmxx_um', 'tile_mmll_um', 'stage_xyz_um_done', 'sample_xyz_um_done', ...
                'piezo_z0_r_um', 'overlap_size_um', 'overlap_size'};
            
            num_prop = numel(copy_prop);
            for iter_prop = 1 : num_prop
                tmp_prop_name = copy_prop{iter_prop};
                switch class(tile_info(1).(tmp_prop_name))
                    case {'char', 'string'}
                        merge_data = {tile_info.(tmp_prop_name)};
                    otherwise
                        merge_data = cat(1, tile_info.(tmp_prop_name));
                end
                [merge_data_unique, selected_idx, ~] = unique(merge_data, 'rows', 'stable');
                if numel(selected_idx) == 1
                    merge_data = merge_data_unique;
                end
                c_info.(tmp_prop_name) = merge_data;
            end
        end
        
        
        function dt = logger_time_string_to_datetime(date_string)
           dt = datetime(date_string,'InputFormat', 'yyyy-MM-dd HH:mm:ss,SSS');                     
        end
        
        
    end
end