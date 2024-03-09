classdef SLURMJobGenerator < handle
    
   properties
              
   end
   
   methods
       
       
   end
   %% 
   methods(Static)
       
       
       
       
   end   
   %% General
   methods(Static)       
       function exit_code = submit_sbatch_job(sbatch_fp)
          exit_code = system(sprintf('sbatch %s', sbatch_fp));           
       end
       
       function exit_code = write_sbatch_file(fp, cmd, opt)
           arguments
               fp 
               cmd
               opt.job_name {char, string} = 'WBIMJob'
               opt.partition {char, string} = 'xiang'
               opt.nodelist (1, :) cell = {'bird'}
               opt.nodes (1,1) double = 1
               opt.ntasks (1,1) double = 1
               opt.cpus_per_task (1,1) double = 1
               opt.mem (1,1) double = 16; % GB
               opt.time (1,1) double = 60; % minutes
               opt.output = '/net/birdstore/slurm/logs/xij072/slurm_%%j_xj.log';% log filepath 
           end
           [folder, fn, ext] = fileparts(fp);
           if ~isfolder(folder)
               mkdir(folder)
           end
           assert(strcmp(ext, '.sbatch'));          
           if ~isempty(opt.output)
              [output_f, output_fn] = fileparts(opt.output);
              if ~isfolder(output_f)
                  mkdir(output_f);
              end              
           end          
           
           fid = fopen(fp, 'w'); % overwrite
           try
               fprintf(fid, '#!/bin/bash\n');
               % Write settings 
               field_name = fieldnames(opt);
               for i = 1 : numel(field_name)
                  tmp_fn = field_name{i};
                  tmp_val = opt.(tmp_fn);
                  switch tmp_fn
                      case 'nodelist'
                          if iscell(tmp_val)
                              if numel(tmp_val) > 1
                                  tmp_val = strjoin(tmp_val, ',');
                              else
                                  tmp_val = tmp_val{1};
                              end
                          else
                              assert(isa(tmp_val, 'char') || isa(tmp_val, 'string'), 'nodelist must be char or string');
                          end
                      case 'mem'
                          tmp_val = sprintf('%dgb', tmp_val);
                      case 'time'
                          tmp_hour = floor(tmp_val / 60);
                          tmp_min = tmp_val - tmp_hour * 60;
                          tmp_min_r = floor(tmp_min);
                          tmp_sec = ceil((tmp_min - tmp_min_r) * 60);
                          tmp_val = sprintf('%02d:%02d:%02d', tmp_hour, tmp_min_r, tmp_sec);
                      otherwise
                          tmp_fn = strrep(tmp_fn, '_', '-');
                          if isnumeric(tmp_val)
                              tmp_val = num2str(tmp_val);
                          end
                  end
                  fprintf(fid, sprintf('#SBATCH --%s=%s\n', tmp_fn, tmp_val));
               end
%                fprintf(fid, '\n# The following is the command\n');
               fprintf(fid, '\n');
               fprintf(fid, 'echo "Job submitted by $(whoami) on $(date)"\n');
               % Write commands
               if isstring(cmd) || ischar(cmd)
                   cmd = {cmd};
               end               
               for i = 1 : numel(cmd)
                   fprintf(fid, sprintf('%s\n', cmd{i}));               
               end
               
               exit_code = fclose(fid);
           catch ME
              fclose(fid);
              rethrow(ME)
           end           
       end
   end    
end