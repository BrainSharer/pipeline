INPUT = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/MD635/preps/jp2';
OUTPUT = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/MD635/preps/tif';
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isfolder(INPUT)
  fprintf(1, 'Directory missing %.s\n', INPUT);
  return;
end
if ~isfolder(OUTPUT)
  fprintf(1, 'Directory missing %.s\n', OUTPUT);
  return;
end
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(INPUT, '*.jp2'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
for k = 1 : length(theFiles)
  baseFileName = theFiles(k).name;
  fullFileName = fullfile(INPUT, baseFileName);
  [~, basename, ~] = fileparts(fullFileName);
  filepath = fullfile(OUTPUT, strcat(basename,'.tif'));
  if isfile(filepath)
    % File exists so do not write.
    fprintf(1, 'File exists %s\n', filepath);  
  else
    % File does not exist so write
    fprintf(1, 'Writing to %s\n', filepath);  
    img = imread(fullFileName);
    try
      imwrite(img, filepath );
    catch e1
      fprintf(1, 'Error writing %s failed\n', filepath);  
      fprintf(1,'Error is 1st catch: %s\n',e1.message);  
      fprintf(1,'1st identifier was: %s\n',e1.identifier);
      t = Tiff(filepath, 'w');
      tagstruct.ImageLength = size(img, 1);
      tagstruct.ImageWidth = size(img, 2);
      tagstruct.Photometric = Tiff.Photometric.RGB;
      tagstruct.BitsPerSample = 16;
      tagstruct.SamplesPerPixel = 3;
      tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
      tagstruct.Software = 'MATLAB';
      setTag(t, tagstruct);
      try
        write(t, img);
      catch e2
        fprintf(1, 'Error writing tif %s failed\n', filepath);  
        fprintf(1,'Error is 2nd catch: %s\n',e2.message);  
        fprintf(1,'2nd identifier was: %s\n',e2.identifier);
      end % end nested catch
      close(t);
    end
  end % end if file exists

end % end loop