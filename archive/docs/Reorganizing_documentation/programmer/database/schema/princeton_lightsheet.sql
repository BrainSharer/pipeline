#### princeton_lightsheet.sql

### Background:

## The below schema represents the backbone of the Flask application 
## that is used for management of the 
## brain clearing, light-sheet imaging and image processing pipeline
## at the Princeton Neuroscience Institute.
## Researchers submit requests to use the facility via this Flask application
## Each request consists of up to 50 tissue samples that need to be imaged
## Technicians carry out the requests by clearing and imaging the requests
## Once the tissue samples reach a certain point in the pipeline,
## we convert the image data to precomputed format and generate neuroglancer links
## for the user.
## Currently, we do not store those links or the neuroglancer state in this database

### List of tables

## User management table

CREATE TABLE `#user` (
  `username` varchar(20) NOT NULL COMMENT 'user in the lab',
  `princeton_email` varchar(50) NOT NULL,
  PRIMARY KEY (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='Users of the light sheet microscope';

## Table to keeping track of user requests to use the microscopy facility

CREATE TABLE `request` (
  `username` varchar(20) NOT NULL COMMENT 'user in the lab',
  `request_name` varchar(64) NOT NULL,
  `requested_by` varchar(20) NOT NULL COMMENT 'user in the lab',
  `auditor` varchar(20) DEFAULT NULL COMMENT 'user in the lab',
  `date_submitted` date NOT NULL COMMENT 'The date it was submitted as a request',
  `time_submitted` time NOT NULL COMMENT 'The time it was submitted as a request',
  `labname` varchar(50) NOT NULL,
  `correspondence_email` varchar(100) NOT NULL DEFAULT '',
  `description` varchar(250) NOT NULL,
  `species` varchar(50) NOT NULL,
  `number_of_samples` tinyint(4) NOT NULL,
  `testing` tinyint(1) NOT NULL DEFAULT 0,
  `is_archival` tinyint(1) NOT NULL DEFAULT 0,
  `raw_data_retention_preference` enum('important','kind of important','not important','not sure') DEFAULT NULL,
  `sent_processing_email` tinyint(1) DEFAULT 0,
  PRIMARY KEY (`username`,`request_name`),
  KEY `requested_by` (`requested_by`),
  KEY `auditor` (`auditor`),
  CONSTRAINT `request_ibfk_1` FOREIGN KEY (`username`) REFERENCES `#user` (`username`) ON UPDATE CASCADE,
  CONSTRAINT `request_ibfk_2` FOREIGN KEY (`requested_by`) REFERENCES `#user` (`username`) ON UPDATE CASCADE,
  CONSTRAINT `request_ibfk_3` FOREIGN KEY (`auditor`) REFERENCES `#user` (`username`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='The highest level table for handling user requests to the Core Facility';

## Table to keep track of tissue samples in a request

CREATE TABLE `request__sample` (
  `username` varchar(20) NOT NULL COMMENT 'tissue sample',
  `request_name` varchar(64) NOT NULL,
  `sample_name` varchar(64) NOT NULL,
  `subject_fullname` varchar(64) NOT NULL DEFAULT '',
  PRIMARY KEY (`username`,`request_name`,`sample_name`),
  CONSTRAINT `request__sample_ibfk_1` FOREIGN KEY (`username`, `request_name`) REFERENCES `request` (`username`, `request_name`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='Samples from a request, belonging to a clearing batch or imaging batch';

### Clearing related tables

## Samples are often cleared in batches of several at a time 

CREATE TABLE `request__clearing_batch` (
  `username` varchar(20) NOT NULL COMMENT 'user in the lab',
  `request_name` varchar(64) NOT NULL,
  `clearing_batch_number` tinyint(4) NOT NULL,
  `clearing_protocol` enum('iDISCO+_immuno','iDISCO abbreviated clearing','iDISCO abbreviated clearing (rat)','uDISCO','uDISCO (rat)','iDISCO_EdU','experimental') NOT NULL,
  `antibody1` varchar(100) NOT NULL DEFAULT '',
  `antibody2` varchar(100) NOT NULL DEFAULT '',
  `antibody1_lot` varchar(64) DEFAULT NULL,
  `antibody2_lot` varchar(64) DEFAULT NULL,
  `clearing_progress` enum('incomplete','in progress','complete') NOT NULL,
  `number_in_batch` tinyint(4) NOT NULL,
  `perfusion_date` date DEFAULT NULL,
  `expected_handoff_date` date DEFAULT NULL,
  `clearer` varchar(20) DEFAULT NULL COMMENT 'user in the lab',
  `notes_for_clearer` varchar(8192) NOT NULL DEFAULT '',
  `link_to_clearing_spreadsheet` varchar(256) DEFAULT NULL,
  PRIMARY KEY (`username`,`request_name`,`clearing_batch_number`),
  KEY `clearer` (`clearer`),
  CONSTRAINT `request__clearing_batch_ibfk_1` FOREIGN KEY (`username`, `request_name`) REFERENCES `request` (`username`, `request_name`) ON UPDATE CASCADE,
  CONSTRAINT `request__clearing_batch_ibfk_2` FOREIGN KEY (`clearer`) REFERENCES `#user` (`username`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='Samples from a particular request';

## An individual sample in a clearing batch
CREATE TABLE `request__clearing_batch_sample` (
  `username` varchar(20) NOT NULL COMMENT 'user in the lab',
  `request_name` varchar(64) NOT NULL,
  `sample_name` varchar(64) NOT NULL,
  `clearing_batch_number` tinyint(4) NOT NULL,
  `clearing_protocol` enum('iDISCO+_immuno','iDISCO abbreviated clearing','iDISCO abbreviated clearing (rat)','uDISCO','uDISCO (rat)','iDISCO_EdU','experimental') NOT NULL,
  `antibody1` varchar(100) NOT NULL DEFAULT '',
  `antibody2` varchar(100) NOT NULL DEFAULT '',
  PRIMARY KEY (`username`,`request_name`,`sample_name`,`clearing_batch_number`),
  KEY `username` (`username`,`request_name`,`clearing_batch_number`),
  CONSTRAINT `request__clearing_batch_sample_ibfk_1` FOREIGN KEY (`username`, `request_name`, `sample_name`) REFERENCES `request__sample` (`username`, `request_name`, `sample_name`) ON UPDATE CASCADE,
  CONSTRAINT `request__clearing_batch_sample_ibfk_2` FOREIGN KEY (`username`, `request_name`, `clearing_batch_number`) REFERENCES `request__clearing_batch` (`username`, `request_name`, `clearing_batch_number`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='Samples in a ClearingBatch';

### Imaging related tables

## Samples are often imaged with the same parameters (wavelengths,exposure time, etc.) as others in the same request
## We call this an imaging batch
CREATE TABLE `request__imaging_batch` (
  `username` varchar(20) NOT NULL COMMENT 'user in the lab',
  `request_name` varchar(64) NOT NULL,
  `clearing_batch_number` tinyint(4) NOT NULL DEFAULT 1,
  `imaging_batch_number` tinyint(4) NOT NULL,
  `imaging_request_number` tinyint(4) NOT NULL,
  `imager` varchar(20) DEFAULT NULL COMMENT 'user in the lab',
  `number_in_imaging_batch` tinyint(4) NOT NULL COMMENT 'date that the imaging form was submitted by the imager',
  `imaging_request_date_submitted` date NOT NULL COMMENT 'date that the user submitted the request for imaging',
  `imaging_request_time_submitted` time NOT NULL COMMENT 'time that the user submitted the request for imaging',
  `imaging_performed_date` date DEFAULT NULL COMMENT 'date that the imaging form was submitted by the imager',
  `imaging_progress` enum('incomplete','in progress','complete') NOT NULL,
  `imaging_dict` blob NOT NULL,
  PRIMARY KEY (`username`,`request_name`,`clearing_batch_number`,`imaging_batch_number`,`imaging_request_number`),
  KEY `imager` (`imager`),
  CONSTRAINT `request__imaging_batch_ibfk_1` FOREIGN KEY (`username`, `request_name`, `clearing_batch_number`) REFERENCES `request__clearing_batch` (`username`, `request_name`, `clearing_batch_number`) ON UPDATE CASCADE,
  CONSTRAINT `request__imaging_batch_ibfk_2` FOREIGN KEY (`imager`) REFERENCES `#user` (`username`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='Batch of Samples to image the same way';

## An individual sample in an imaging batch
CREATE TABLE `request__imaging_batch_sample` (
  `username` varchar(20) NOT NULL COMMENT 'user in the lab',
  `request_name` varchar(64) NOT NULL,
  `sample_name` varchar(64) NOT NULL,
  `clearing_batch_number` tinyint(4) NOT NULL DEFAULT 1,
  `imaging_batch_number` tinyint(4) NOT NULL,
  `imaging_request_number` tinyint(4) NOT NULL,
  PRIMARY KEY (`username`,`request_name`,`sample_name`,`clearing_batch_number`,`imaging_batch_number`,`imaging_request_number`),
  KEY `username` (`username`,`request_name`,`clearing_batch_number`,`imaging_batch_number`,`imaging_request_number`),
  CONSTRAINT `request__imaging_batch_sample_ibfk_1` FOREIGN KEY (`username`, `request_name`, `sample_name`) REFERENCES `request__sample` (`username`, `request_name`, `sample_name`) ON UPDATE CASCADE,
  CONSTRAINT `request__imaging_batch_sample_ibfk_2` FOREIGN KEY (`username`, `request_name`, `clearing_batch_number`, `imaging_batch_number`, `imaging_request_number`) REFERENCES `request__imaging_batch` (`username`, `request_name`, `clearing_batch_number`, `imaging_batch_number`, `imaging_request_number`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='Samples in an ImagingBatch';

## Users can request to have imaging re-done for a single sample or set of samples
## This table keeps track of the number of times each sample has been imaged
CREATE TABLE `request__imaging_request` (
  `username` varchar(20) NOT NULL COMMENT 'user in the lab',
  `request_name` varchar(64) NOT NULL,
  `sample_name` varchar(64) NOT NULL,
  `imaging_request_number` tinyint(4) NOT NULL,
  `imager` varchar(20) DEFAULT NULL COMMENT 'user in the lab',
  `imaging_request_date_submitted` date NOT NULL COMMENT 'date that the user submitted the request for imaging',
  `imaging_request_time_submitted` time NOT NULL COMMENT 'time that the user submitted the request for imaging',
  `imaging_performed_date` date DEFAULT NULL COMMENT 'date that the imaging form was submitted by the imager',
  `imaging_progress` enum('incomplete','in progress','complete') NOT NULL,
  `imaging_skipped` tinyint(1) DEFAULT NULL COMMENT '1 if this sample skipped, 0 or NULL if not skipped',
  PRIMARY KEY (`username`,`request_name`,`sample_name`,`imaging_request_number`),
  KEY `imager` (`imager`),
  CONSTRAINT `request__imaging_request_ibfk_1` FOREIGN KEY (`username`, `request_name`, `sample_name`) REFERENCES `request__sample` (`username`, `request_name`, `sample_name`) ON UPDATE CASCADE,
  CONSTRAINT `request__imaging_request_ibfk_2` FOREIGN KEY (`imager`) REFERENCES `#user` (`username`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='Imaging request';

## Each sample in a given imaging request 
# might require multiple image resolutions (objectives) 
## This table keeps track of which image resolutions were used for each sample
CREATE TABLE `request__imaging_resolution_request` (
  `username` varchar(20) NOT NULL COMMENT 'user in the lab',
  `request_name` varchar(64) NOT NULL,
  `sample_name` varchar(64) NOT NULL,
  `imaging_request_number` tinyint(4) NOT NULL,
  `image_resolution` enum('1.3x','4x','1.1x','2x','3.6x','15x') NOT NULL,
  `microscope` enum('LaVision','SmartSPIM') DEFAULT NULL,
  `notes_for_imager` varchar(1024) NOT NULL DEFAULT '',
  `notes_from_imaging` varchar(1024) NOT NULL DEFAULT '',
  PRIMARY KEY (`username`,`request_name`,`sample_name`,`imaging_request_number`,`image_resolution`),
  CONSTRAINT `request__imaging_resolution_request_ibfk_1` FOREIGN KEY (`username`, `request_name`, `sample_name`, `imaging_request_number`) REFERENCES `request__imaging_request` (`username`, `request_name`, `sample_name`, `imaging_request_number`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='Imaging parameters for a channel, belonging to a sample';

## Channels (aka wavelengths) used to image a sample in a given request
CREATE TABLE `request__imaging_channel` (
  `username` varchar(20) NOT NULL COMMENT 'user in the lab',
  `request_name` varchar(64) NOT NULL,
  `sample_name` varchar(64) NOT NULL,
  `imaging_request_number` tinyint(4) NOT NULL,
  `image_resolution` enum('1.3x','4x','1.1x','2x','3.6x','15x') NOT NULL,
  `channel_name` varchar(64) NOT NULL,
  `ventral_up` tinyint(1) NOT NULL DEFAULT 0 COMMENT 'whether brain was flipped upside down to be imaged',
  `imaging_date` date DEFAULT NULL,
  `zoom_body_magnification` float DEFAULT NULL COMMENT 'only applicable for 2x',
  `left_lightsheet_used` tinyint(1) NOT NULL DEFAULT 1,
  `right_lightsheet_used` tinyint(1) NOT NULL DEFAULT 1,
  `registration` tinyint(1) NOT NULL DEFAULT 0,
  `injection_detection` tinyint(1) NOT NULL DEFAULT 0,
  `probe_detection` tinyint(1) NOT NULL DEFAULT 0,
  `cell_detection` tinyint(1) NOT NULL DEFAULT 0,
  `generic_imaging` tinyint(1) NOT NULL DEFAULT 0,
  `pixel_type` varchar(32) DEFAULT NULL,
  `image_orientation` enum('sagittal','coronal','horizontal') NOT NULL COMMENT 'how the imager imaged the sample. Most of the time will be horizontal',
  `numerical_aperture` float DEFAULT NULL COMMENT 'it is not always recorded in metadata so those times it will be NULL',
  `tiling_scheme` char(3) NOT NULL DEFAULT '1x1',
  `tiling_overlap` float NOT NULL DEFAULT 0,
  `z_step` float NOT NULL DEFAULT 10 COMMENT 'distance between z planes in microns',
  `number_of_z_planes` smallint(5) unsigned DEFAULT NULL,
  `rawdata_subfolder` varchar(512) DEFAULT NULL,
  `imspector_channel_index` tinyint(4) DEFAULT NULL COMMENT 'refers to multi-channel imaging - 0 if first (or only) channel in rawdata_subfolder, 1 if second, 2 if third, ...',
  `left_lightsheet_precomputed_spock_jobid` varchar(32) DEFAULT NULL,
  `left_lightsheet_precomputed_spock_job_progress` enum('NOT_SUBMITTED','SUBMITTED','COMPLETED','FAILED','RUNNING','PENDING','BOOT_FAIL','CANCELLED','DEADLINE','OUT_OF_MEMORY','REQUEUED',' RESIZING','REVOKED','SUSPENDED','TIMEOUT') DEFAULT NULL,
  `right_lightsheet_precomputed_spock_jobid` varchar(32) DEFAULT NULL,
  `right_lightsheet_precomputed_spock_job_progress` enum('NOT_SUBMITTED','SUBMITTED','COMPLETED','FAILED','RUNNING','PENDING','BOOT_FAIL','CANCELLED','DEADLINE','OUT_OF_MEMORY','REQUEUED',' RESIZING','REVOKED','SUSPENDED','TIMEOUT') DEFAULT NULL,
  PRIMARY KEY (`username`,`request_name`,`sample_name`,`imaging_request_number`,`image_resolution`,`channel_name`,`ventral_up`),
  CONSTRAINT `request__imaging_channel_ibfk_1` FOREIGN KEY (`username`, `request_name`, `sample_name`, `imaging_request_number`, `image_resolution`) REFERENCES `request__imaging_resolution_request` (`username`, `request_name`, `sample_name`, `imaging_request_number`, `image_resolution`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='Imaging parameters for a channel, belonging to a sample';

### Image-processing related tables
## Our pipeline consists of image stitching (in the case of tiled datasets),
## brain registration and converting the data to precomputed format
## so that it can be viewed in Neuroglancer

## Users can request to have their samples processed multiple times
## this table keeps track of the number of times each sample has been processed
CREATE TABLE `request__processing_request` (
  `username` varchar(20) NOT NULL COMMENT 'user in the lab',
  `request_name` varchar(64) NOT NULL,
  `sample_name` varchar(64) NOT NULL,
  `imaging_request_number` tinyint(4) NOT NULL,
  `processing_request_number` tinyint(4) NOT NULL,
  `processor` varchar(20) DEFAULT NULL COMMENT 'user in the lab',
  `processing_request_date_submitted` date NOT NULL COMMENT 'date that the user submitted the request for processing',
  `processing_request_time_submitted` time NOT NULL COMMENT 'time that the user submitted the request for processing',
  `processing_performed_date` date DEFAULT NULL COMMENT 'date that the processing form was submitted by the processor',
  `processing_progress` enum('incomplete','running','failed','complete') NOT NULL,
  PRIMARY KEY (`username`,`request_name`,`sample_name`,`imaging_request_number`,`processing_request_number`),
  KEY `processor` (`processor`),
  CONSTRAINT `request__processing_request_ibfk_1` FOREIGN KEY (`username`, `request_name`, `sample_name`, `imaging_request_number`) REFERENCES `request__imaging_request` (`username`, `request_name`, `sample_name`, `imaging_request_number`) ON UPDATE CASCADE,
  CONSTRAINT `request__processing_request_ibfk_2` FOREIGN KEY (`processor`) REFERENCES `#user` (`username`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='Processing request - this needs to exist because for each imaging request there can be multiple processing requests';

## For each image resolution, the processing pipeline needs to be
## run for a slightly different set of parameters
CREATE TABLE `request__processing_resolution_request` (
  `username` varchar(20) NOT NULL COMMENT 'user in the lab',
  `request_name` varchar(64) NOT NULL,
  `sample_name` varchar(64) NOT NULL,
  `imaging_request_number` tinyint(4) NOT NULL,
  `processing_request_number` tinyint(4) NOT NULL,
  `image_resolution` enum('1.3x','4x','1.1x','2x','3.6x','15x') NOT NULL,
  `ventral_up` tinyint(1) NOT NULL DEFAULT 0 COMMENT 'whether brain was flipped upside down to be imaged',
  `atlas_name` enum('allen_2017','allen_2011','princeton_mouse_atlas','paxinos') NOT NULL,
  `final_orientation` enum('sagittal','coronal','horizontal') NOT NULL,
  `notes_for_processor` varchar(1024) NOT NULL DEFAULT '',
  `notes_from_processing` varchar(1024) NOT NULL DEFAULT '',
  `lightsheet_pipeline_spock_jobid` varchar(16) DEFAULT NULL COMMENT 'the jobid from the final step in the light sheet processing pipeline',
  `lightsheet_pipeline_spock_job_progress` enum('NOT_SUBMITTED','SUBMITTED','COMPLETED','FAILED','RUNNING','PENDING','BOOT_FAIL','CANCELLED','DEADLINE','OUT_OF_MEMORY','REQUEUED',' RESIZING','REVOKED','SUSPENDED','TIMEOUT') DEFAULT NULL COMMENT 'the spock job status code for the final step in the light sheet processing pipeline',
  `brainpipe_commit` char(7) DEFAULT NULL COMMENT 'the commit that is checked out on the machine at the time the job was submitted',
  PRIMARY KEY (`username`,`request_name`,`sample_name`,`imaging_request_number`,`processing_request_number`,`image_resolution`,`ventral_up`),
  CONSTRAINT `request__processing_resolution_request_ibfk_1` FOREIGN KEY (`username`, `request_name`, `sample_name`, `imaging_request_number`, `processing_request_number`) REFERENCES `request__processing_request` (`username`, `request_name`, `sample_name`, `imaging_request_number`, `processing_request_number`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='Processing parameters at the image resolution level for a given ProcessingRequest(). These represent spock jobs';

## The individual channels are processed and the resulting metadata is stored in this table
## We launch jobs off to a computing cluster to do the processing
## and this table contains the information on those jobs
## when jobs are done this triggers the precomputed pipelines
CREATE TABLE `request__processing_channel` (
  `username` varchar(20) NOT NULL COMMENT 'user in the lab',
  `request_name` varchar(64) NOT NULL,
  `sample_name` varchar(64) NOT NULL,
  `imaging_request_number` tinyint(4) NOT NULL,
  `image_resolution` enum('1.3x','4x','1.1x','2x','3.6x') NOT NULL,
  `channel_name` varchar(64) NOT NULL,
  `ventral_up` tinyint(4) NOT NULL DEFAULT 0 COMMENT 'whether brain was flipped upside down to be imaged',
  `processing_request_number` tinyint(4) NOT NULL,
  `lightsheet_channel_str` enum('regch','injch','cellch','gench') NOT NULL,
  `imspector_version` varchar(128) NOT NULL DEFAULT '',
  `datetime_processing_started` datetime NOT NULL,
  `datetime_processing_completed` datetime DEFAULT NULL,
  `intensity_correction` tinyint(1) NOT NULL DEFAULT 1,
  `metadata_xml_string` mediumblob DEFAULT NULL COMMENT 'The entire metadata xml string. Sometimes it is not available so those times it will be NULL',
  `left_lightsheet_stitched_precomputed_spock_jobid` varchar(32) DEFAULT NULL,
  `left_lightsheet_stitched_precomputed_spock_job_progress` enum('NOT_SUBMITTED','SUBMITTED','COMPLETED','FAILED','RUNNING','PENDING','BOOT_FAIL','CANCELLED','DEADLINE','OUT_OF_MEMORY','REQUEUED',' RESIZING','REVOKED','SUSPENDED','TIMEOUT') DEFAULT NULL,
  `right_lightsheet_stitched_precomputed_spock_jobid` varchar(32) DEFAULT NULL,
  `right_lightsheet_stitched_precomputed_spock_job_progress` enum('NOT_SUBMITTED','SUBMITTED','COMPLETED','FAILED','RUNNING','PENDING','BOOT_FAIL','CANCELLED','DEADLINE','OUT_OF_MEMORY','REQUEUED',' RESIZING','REVOKED','SUSPENDED','TIMEOUT') DEFAULT NULL,
  `blended_precomputed_spock_jobid` varchar(32) DEFAULT NULL,
  `blended_precomputed_spock_job_progress` enum('NOT_SUBMITTED','SUBMITTED','COMPLETED','FAILED','RUNNING','PENDING','BOOT_FAIL','CANCELLED','DEADLINE','OUT_OF_MEMORY','REQUEUED',' RESIZING','REVOKED','SUSPENDED','TIMEOUT') DEFAULT NULL,
  `downsized_precomputed_spock_jobid` varchar(32) DEFAULT NULL,
  `downsized_precomputed_spock_job_progress` enum('NOT_SUBMITTED','SUBMITTED','COMPLETED','FAILED','RUNNING','PENDING','BOOT_FAIL','CANCELLED','DEADLINE','OUT_OF_MEMORY','REQUEUED',' RESIZING','REVOKED','SUSPENDED','TIMEOUT') DEFAULT NULL,
  `registered_precomputed_spock_jobid` varchar(32) DEFAULT NULL,
  `registered_precomputed_spock_job_progress` enum('NOT_SUBMITTED','SUBMITTED','COMPLETED','FAILED','RUNNING','PENDING','BOOT_FAIL','CANCELLED','DEADLINE','OUT_OF_MEMORY','REQUEUED',' RESIZING','REVOKED','SUSPENDED','TIMEOUT') DEFAULT NULL,
  PRIMARY KEY (`username`,`request_name`,`sample_name`,`imaging_request_number`,`image_resolution`,`channel_name`,`ventral_up`,`processing_request_number`,`lightsheet_channel_str`),
  KEY `username` (`username`,`request_name`,`sample_name`,`imaging_request_number`,`processing_request_number`,`image_resolution`,`ventral_up`),
  CONSTRAINT `request__processing_channel_ibfk_1` FOREIGN KEY (`username`, `request_name`, `sample_name`, `imaging_request_number`, `image_resolution`, `channel_name`, `ventral_up`) REFERENCES `request__imaging_channel` (`username`, `request_name`, `sample_name`, `imaging_request_number`, `image_resolution`, `channel_name`, `ventral_up`) ON UPDATE CASCADE,
  CONSTRAINT `request__processing_channel_ibfk_2` FOREIGN KEY (`username`, `request_name`, `sample_name`, `imaging_request_number`, `processing_request_number`, `image_resolution`, `ventral_up`) REFERENCES `request__processing_resolution_request` (`username`, `request_name`, `sample_name`, `imaging_request_number`, `processing_request_number`, `image_resolution`, `ventral_up`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='Processing parameters for a channel. There can be more than one purpose for a single channel, hence why lightsheet_channel_str is a primary key';
