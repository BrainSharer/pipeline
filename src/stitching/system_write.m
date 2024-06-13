function system_write(text, fp, input_type)
% Input: if input_type = 'text', append text to fp; if input_type =
% 'command', run the command and save the output to fp. 
%        fp(string, specify the path to the log file)
switch input_type
    case 'text'
        system(sprintf('echo "%s" >> %s', text, fp));
    case 'command'
        system(sprintf('%s >> %s', text, fp));
    otherwise
        error('Unrecognized input type');
end
end