classdef ComputeTaskGenerator < handle

properties(Constant)
    Name = 'ComputeTaskGenerator';
end

properties
    cmd_format = 'cd %s; matlab -nodisplay -nosplash -r "addpath(genpath(''./'')); try, %s(''%s''); catch ME, %s quit; end; quit;" &';
    error_handle_format = 'system_write(sprintf(''Error message: %%s\\n'', ME.message), ''%s'', ''text'');';
    script_fp 
    fun_name
    fun_arg
end


methods
    function obj = ComputeTaskGenerator(task_script_fp, fun_name, fun_arg)
        task_folder = fileparts(task_script_fp);
        if ~isfolder(task_folder)
            mkdir(task_folder);
        end
        obj.script_fp = task_script_fp;
        obj.fun_name = fun_name;
        obj.fun_arg = fun_arg;
    end
end


methods(Static)
    function exit_code = write_task_bash(file_fp, cmd_cell)
        fid = fopen(file_fp, 'w');
        try
            fprintf(fid, '#!/bin/bash\n');
            for i = 1 : numel(cmd_cell)
                fprintf(fid, '%s\n', cmd_cell{i});
            end
            exit_code = fclose(fid);
            system(sprintf('chmod g+rx %s', file_fp));
        catch ME
            fclose(fid);
            rethrow(ME);
        end
    end

    function exit_code = write_text(fp, text)
        fd = fileparts(fp);
        fid = fopen(fd, 'w+');
        try
            assert(ischar(text) || isstring(text), 'The input should be a string or char')
            fprintf(fid, text);
            exit_code = fclose(fid);
        catch ME
            exit_code = fclose(fid);
        end
    end
    
    function exit_code = sync_stitching_script_to_server()
        DataManager = WBIMFileManager;
        github_folder = fileparts(DataManager.SCRIPT_PATH);
        repo_folder = fullfile(github_folder, 'WBIMStitching');
        server_folder = fullfile(fileparts(DataManager.SERVER_SCRIPT_PATH), 'WBIMStitching');
        if ~isfolder(server_folder)
            mkdir(server_folder);
        end
        sync_cmd = sprintf('rsync -rav --exclude=".*" --exclude="*.asv"  %s/ %s', repo_folder, server_folder);
        exit_code = system(sync_cmd);        
    end
    
    function run_command_on_machine(machine_name, cmd_str, no_wait_Q)
        if nargin < 3
            no_wait_Q = true;
        end
        % Use the following information to setup the machine first:
        % 1. https://www.ostechnix.com/how-to-create-ssh-alias-in-linux/
        % 2. https://askubuntu.com/questions/8653/how-to-keep-processes-running-after-ending-ssh-session
        ssh_cmd_str = strjoin({'"', cmd_str, '"'}, ' ');
        if no_wait_Q
            ssh_str = strjoin({'ssh', machine_name, ssh_cmd_str, '&'}, ' ');
        else
            ssh_str = strjoin({'ssh', machine_name, ssh_cmd_str}, ' ');
        end
        system(ssh_str);
    end
    
    
    
    
end

end