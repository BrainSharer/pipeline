INPUT = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/MD591/preps/jp2';
OUTPUT = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/MD591/preps/tif';
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
    % File exists.
    fprintf(1, 'File exists %s\n', filepath);  
  else
  % File does not exist.
  fprintf(1, 'Writing to %s\n', filepath);  
  img = imread(fullFileName);
  % imwrite(img, filepath );
  t = Tiff(filepath, 'w');
  tagstruct.ImageLength = size(img, 1);
  tagstruct.ImageWidth = size(img, 2);
  tagstruct.Photometric = Tiff.Photometric.RGB;
  tagstruct.BitsPerSample = 8;
  tagstruct.SamplesPerPixel = 3;
  tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
  tagstruct.Software = 'MATLAB';
  setTag(t, tagstruct);
  try
    write(t, img);
  catch
    fprintf(1, 'Error writing %s failed\n', filepath);  
  end
  close(t);
  end
end