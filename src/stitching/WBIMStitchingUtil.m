classdef WBIMStitchingUtil < handle
    
    
    
    
    methods(Static)
        %% Generate interpolation grid
        function [start_xyz_pxl, end_xyz_pxl] = get_contol_pix_locations(im_size_xyz, ...
                pixel_size_xyz_um, tile_overlap_xyz_um, num_ptrl_div, target_overlap_xyz_um)
            arguments
                im_size_xyz (1, 3) double
                pixel_size_xyz_um (1, 3) double
                tile_overlap_xyz_um (1, 3) double
                num_ptrl_div (1,1) double
                target_overlap_xyz_um (1, 3) double = [5, 5, 5]
            end
            % Number of overlapping pixels in each direction
            target_overlap_xyz_pxl = round(target_overlap_xyz_um ./ pixel_size_xyz_um); %in um
            overlap_px = round(tile_overlap_xyz_um ./ pixel_size_xyz_um);
            
            start_xyz_pxl = round(overlap_px/2 - target_overlap_xyz_pxl/2);
            % find nearest integer that makes ed-st divisible by number of segments (N)
            end_xyz_pxl = start_xyz_pxl + floor((im_size_xyz - 2 * start_xyz_pxl)/num_ptrl_div) * num_ptrl_div;
        end
        
        function div_ind = get_1d_ctrl_sub(int_size, dist_2_edge, num_div)
            % dist_2_edge: number of pixel to the edge of the interval
            % (starting at 1), so dist_2_edge = 0 means start from 1
            arguments
                int_size (1,1) double
                dist_2_edge (1,1) double {mustBeNonnegative}
                num_div (1,1) double {mustBeInteger, mustBePositive} % number of sub-intervals between points
            end
            assert(int_size > dist_2_edge * 2);
            start_ind = dist_2_edge + 1;
            end_ind = int_size - dist_2_edge;
            
            step_size = round((end_ind - start_ind) / num_div);
            div_ind = start_ind : step_size : int_size;
        end
        
        function [ctrl_sub, varargout] = get_nd_ctrl_sub(im_size, overlap, ...
                num_div, overlap_offset)
            arguments
                im_size (1, :) double
                overlap (1, :) double
                num_div (1,:) double {mustBePositive, mustBeInteger}
                overlap_offset (1, :) double = 0; % number of pixels
            end
            num_dim = numel(im_size(im_size ~= 1));
            if num_dim > 1
                if isscalar(overlap)
                    overlap = repelem(overlap, num_dim, 1);
                end
                if isscalar(num_div)
                    num_div = repelem(num_div, num_dim, 1);
                end
                if isscalar(overlap_offset)
                    overlap_offset = repelem(overlap_offset, num_dim, 1);
                end
            end
            single_axis_sub = cell(num_dim, 1);
            for i = 1 : num_dim
                single_axis_sub{i} = WBIMStitchingUtil.get_1d_ctrl_sub(im_size(i), ...
                    round((overlap(i) + overlap_offset(i)) / 2), num_div(i));
            end
            
            switch num_dim
                case 1
                    ctrl_sub = single_axis_sub{1};
                case 2
                    [sub1, sub2] = ndgrid(single_axis_sub{:});
                    ctrl_sub = cat(2, sub1(:), sub2(:));
                case 3
                    [sub1, sub2, sub3] = ndgrid(single_axis_sub{:});
                    ctrl_sub = cat(2, sub1(:), sub2(:), sub3);
            end
            if nargout > 1
                varargout = single_axis_sub;
            end
        end        
        
        %%
        function [box, ori, sz] = bboxFromCorners(pts)
            ori = min(pts);
            sz = max(pts)-min(pts);
            
            lims = [ori; ori + sz];
            box = zeros(8,3);
            % eight corners of the bounding box
            for i = 0:7
                inds = round([1 + mod(i,2), 1 + mod(floor(i/2), 2), 1 + mod(floor(i/4), 2)]);
                box(i+1,:) = [lims(inds(1), 1), lims(inds(2), 2), lims(inds(3), 3)];
            end
        end
        %%
        function exit_code = writeYML(vecfield, output_fp)
            
            output_folder = fileparts(output_fp);
            if ~isfolder(output_folder)
                mkdir(output_folder);
            end            
            size_xyzc_pxl = vecfield.stack_size_xyzc;
            del = '  ';
            
            fid = fopen(output_fp, 'w+');
            fprintf(fid, '%s\n', ['path: ' vecfield.root]);
            fprintf(fid, 'tiles:\n');
            pts = vecfield.control;
            for i = 1 : numel(vecfield.path)
                % make sure path starts with seperator
                if vecfield.path{i}(1)~='/'
                    fprintf(fid, '- path: /%s\n', vecfield.path{i});
                else
                    fprintf(fid, '- path: %s\n', vecfield.path{i});
                end
                
                tform = vecfield.tform(:,:,i);
                
                fprintf(fid, '%saabb:\n', del);
                fprintf(fid, '%s%sori: [%d, %d, %d]\n', del, del, round(vecfield.origin(i,:)));
                fprintf(fid, '%s%sshape: [%d, %d, %d]\n', del, del, round(vecfield.sz(i,:)));
                
                fprintf(fid, '%sshape:\n', del);
                fprintf(fid, '%s%stype: u16\n', del, del);
                fprintf(fid, '%s%sdims: [%d, %d, %d, %d]\n', del, del, round(size_xyzc_pxl));
                
                fprintf(fid, '%shomography: [%17.16f, %17.16f, %17.16f, %3.1f, %17.8f,\n', del, tform(1,:));
                fprintf(fid, '%s%s%17.16f, %17.16f, %17.16f, %3.1f, %17.8f,\n', del, del,  tform(2,:));
                fprintf(fid, '%s%s%17.16f, %17.16f, %17.16f, %3.1f, %17.8f,\n', del, del,  tform(3,:));
                fprintf(fid, '%s%s%3.1f, %3.1f, %3.1f, %3.1f, %3.1f, ', del, del, tform(4,:));
                fprintf(fid, '%3.1f, %3.1f, %3.1f, %3.1f, %3.1f]\n', tform(5,:));
                
                fprintf(fid, '%sgrid:\n', del);
                fprintf(fid, '%s%sxlims: [%d', del, del, round(vecfield.xlim_cntrl(1)));
                for j = 2 : numel(vecfield.xlim_cntrl)
                    fprintf(fid, ', %d', round(vecfield.xlim_cntrl(j)));
                end
                fprintf(fid, ']\n');
                fprintf(fid, '%s%sylims: [%d', del, del, round(vecfield.ylim_cntrl(1)));
                for j = 2 : numel(vecfield.ylim_cntrl)
                    fprintf(fid, ', %d', round(vecfield.ylim_cntrl(j)));
                end
                fprintf(fid, ']\n');
                fprintf(fid, '%s%szlims: [%d', del, del, round(vecfield.zlim_cntrl(1,i)));
                for j = 2 : size(vecfield.zlim_cntrl, 1)
                    fprintf(fid, ', %d', round(vecfield.zlim_cntrl(j,i)));
                end
                fprintf(fid, ']\n');
                
                fprintf(fid, '%s%scoordinates: [%d, %d, %d,\n', del, del, round(pts(1,:,i)));
                for j = 2 : size(pts, 1) - 1
                    fprintf(fid, '%s%s%s%d, %d, %d,\n', del, del, del, round(pts(j,:,i)));
                end
                fprintf(fid, '%s%s%s%d, %d, %d]\n', del, del, del, round(pts(end,:,i)));
            end
            exit_code = fclose(fid);
        end
        
    end
end