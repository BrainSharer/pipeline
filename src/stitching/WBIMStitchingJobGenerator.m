classdef WBIMStitchingJobGenerator < SLURMJobGenerator
    properties(Constant)
        SCRIPT_FOLDER = '/net/birdstore/Vessel/WBIM/Script/WBIMStitching';
        
    end
    
    methods(Static)
        function exit_code = write_and_submit_matlab_job(fp, matlab_cmd, opt)
            arguments
                fp % sbatch file path
                matlab_cmd
                opt.job_name {char, string} = 'WBIMJob'
                opt.partition {char, string} = 'xiang'
                opt.nodelist (1, :) cell = {'bird'}
                opt.nodes (1,1) double = 1
                opt.ntasks (1,1) double = 1
                opt.cpus_per_task (1,1) double = 4
                opt.mem (1,1) double = 32; % GB
                opt.time (1,1) double = 60; % minutes
                opt.output = '/net/birdstore/slurm/logs/xij072/slurm_%%j_xj.log';% log filepath
            end
            [folder, fn, ext] = fileparts(fp);
            error_fp = fullfile(folder, [fn, '.err']);
            log_fp = fullfile(folder, [fn, '.log']);
            cmd_str = {};
            % Delete error file if exist
            cmd_str{end+1} = sprintf('FILE=%s; [ -f "$FILE" ] && rm "$FILE"', error_fp);
            % Run MATLAB
            cmd_str{end+1} = sprintf('cd %s\n', WBIMStitchingJobGenerator.SCRIPT_FOLDER);
            cmd_str{end+1} = matlab_cmd;
            SLURMJobGenerator.write_sbatch_file(fp, cmd_str, 'job_name', opt.job_name, ...
                'partition', opt.partition, 'nodelist', opt.nodelist, 'nodes', opt.nodes, ...
                'ntasks', opt.ntasks, 'cpus_per_task', opt.cpus_per_task, ...
                'mem', opt.mem, 'time', opt.time, 'output', log_fp);
            exit_code = SLURMJobGenerator.submit_sbatch_job(fp);
        end
        
        
    end
end