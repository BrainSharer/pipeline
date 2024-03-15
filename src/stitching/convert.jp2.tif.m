clc;clear;close all;
cd('/home/eodonnell/programming/pipeline/src/stitching');

% nohup matlab -nodisplay -nosplash -r "run('convert.jp2.tif.m'); exit" > output.log 2>&1 &
% Specify the folder where the files live.
INPUT = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/MD590/jp2';
OUTPUT = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/MD590/tif';
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isdir(INPUT)
  fprintf('Directory missing %.s\n', INPUT);
  return;
end
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(INPUT, '*.jp2'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
for k = 1 : length(theFiles)
  baseFileName = theFiles(k).name;
  fullFileName = fullfile(INPUT, baseFileName);
  [folder, basename, extension] = fileparts(fullFileName);
  img = imread(fullFileName);
  filepath = fullfile(OUTPUT, strcat(basename,'.tif'));
  fprintf(1, 'Writing to %s\n', filepath);  
  imwrite(img, filepath );
end