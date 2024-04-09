INPUT = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/MD635/preps/jp2';
OUTPUT = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/MD635/preps/tif';
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isfolder(INPUT)
  fprintf(1, 'Input directory missing %.s\n', INPUT);
  return;
end
if ~isfolder(OUTPUT)
  fprintf(1, 'Output directory missing %.s\n', OUTPUT);
  return;
end
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(INPUT, '*.jp2'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
for k = 1 : length(theFiles)
  baseFileName = theFiles(k).name;
  inpath = fullfile(INPUT, baseFileName);
  [~, basename, ~] = fileparts(inpath);
  outpath = fullfile(OUTPUT, strcat(basename,'.tif'));
  if isfile(outpath)
    % File exists so do not write.
    fprintf(1, 'File exists: %s\n', outpath);  
  else
    % File does not exist so write
    fprintf(1, 'Reading %s\t', baseFileName); 
    try 
      img = imread(inpath);
    catch read_exception
      fprintf(1, 'Error reading %s\n', inpath);  
      fprintf(1,'Error in catch: %s\n',read_exception.message);  
      fprintf(1,'Error identifier: %s\n',read_exception.identifier);
      return;
    end % end read catch
    write3Dtiff(img, outpath);
  end % end if file exists

end % end loop

function write3Dtiff(img, outpath)
  % make sure the vector returned by size() is of length 4

  dims = size(img,1:4);
  % disp(dims);
  % info = imfinfo(inpath);
  % disp(info);
  


  % Set data type specific fields
  if isa(img, 'single')
      bitsPerSample = 32;
      sampleFormat = Tiff.SampleFormat.IEEEFP;
  elseif isa(img, 'uint16')
      bitsPerSample = 16;
      sampleFormat = Tiff.SampleFormat.UInt;
  elseif isa(img, 'uint8')
      bitsPerSample = 8;
      sampleFormat = Tiff.SampleFormat.UInt;
  else
      % if you want to handle other numeric classes, add them yourself
      disp('Unsupported data type');
      return;
  end
  
  % Open TIFF file in write mode
  t = Tiff(outpath,'w');
  
  % Loop through frames
  % Set tag structure for each frame
  tagstruct.ImageLength = dims(1);
  tagstruct.ImageWidth = dims(2);
  tagstruct.SamplesPerPixel = dims(3);
  tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
  tagstruct.BitsPerSample = bitsPerSample;
  tagstruct.SampleFormat = sampleFormat;
  % tagstruct.RowsPerStrip = 32;
  tagstruct.TileLength = 256;
  tagstruct.TileWidth = 256;  
  if any(dims(3) == [3 4]) % assume these are RGB/RGBA
      tagstruct.Photometric = Tiff.Photometric.RGB;
  else % otherwise assume it's I/IA or volumetric
      tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
  end
  
  if any(dims(3) == [2 4]) % assume these are IA/RGBA
      tagstruct.ExtraSamples = Tiff.ExtraSamples.AssociatedAlpha;
  end
  
  % set LZW compression
  tagstruct.Compression = Tiff.Compression.LZW;
  % Set the tag for the current frame
  t.setTag(tagstruct);
  try
    write(t, img);
    fprintf(1, 'Wrote %s\n', outpath);  
  catch write_exception
    fprintf(1, 'tiff write error %s\n', outpath);  
    fprintf(1,'Error in catch: %s\n', write_exception.message);  
    fprintf(1,'Error identifier: %s\n', write_exception.identifier);
  end % end write catch
  close(t);
end