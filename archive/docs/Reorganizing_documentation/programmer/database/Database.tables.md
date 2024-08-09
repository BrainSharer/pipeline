## Active brain atlas specific tables initially created by Yoav and David and later modified by Ed
This is a listing of all tables used by the Active Brain Atlas project.
Each table has all the columns listed along with the column type and some
additional description. Columns that have never been used are marked by *NOTUSED*. 

For a graphical (ERD) view of the database, click this [diagram](schema/database.erd.png).

To view the datajoint Princeton lightsheet schema, 
[click here](schema/princeton_lightsheet.py).

For a full listing of all tables in the active_atlas_production database sorted by category, 
[see here](schema/table_names_reorganized.txt)

The following prefixes are used to mark the appropriate key:
1. Foreign keys =  `FK__`
1. Index keys = `K__`
1. Unique keys = `UK__`

### animal
* `prep_id` varchar(20) PRIMARY KEY NOT NULL COMMENT 'Name for lab mouse/rat, max 20 chars'
* `performance_center` enum('CSHL','Salk','UCSD','HHMI','Duke') DEFAULT NULL
* `date_of_birth` date DEFAULT NULL COMMENT 'the mouse''s date of birth'
* `species` enum('mouse','rat') DEFAULT NULL
* `strain` varchar(50) DEFAULT NULL
* `sex` enum('M','F') DEFAULT NULL COMMENT '(M/F) either ''M'' for male, ''F'' for female'
* `genotype` varchar(100) DEFAULT NULL COMMENT 'transgenic description, usually "C57"; We will need a genotype table'
* `breeder_line` varchar(100) DEFAULT NULL COMMENT 'We will need a local breeding table'
* `vendor` enum('Jackson','Charles River','Harlan','NIH','Taconic') DEFAULT NULL
* `stock_number` varchar(100) DEFAULT NULL COMMENT 'if not from a performance center'
* `tissue_source` enum('animal','brain','slides') DEFAULT NULL
* `ship_date` date DEFAULT NULL
* `shipper` enum('FedEx','UPS') DEFAULT NULL
* `tracking_number` varchar(100) DEFAULT NULL
* `aliases_1` varchar(100) DEFAULT NULL COMMENT 'names given by others'
* `aliases_2` varchar(100) DEFAULT NULL
* `aliases_3` varchar(100) DEFAULT NULL *NOTUSED*
* `aliases_4` varchar(100) DEFAULT NULL *NOTUSED*
* `aliases_5` varchar(100) DEFAULT NULL *NOTUSED*
* `comments` varchar(2001) DEFAULT NULL COMMENT 'assessment'
* `active` tinyint(4) NOT NULL DEFAULT 1
* `created` timestamp NULL DEFAULT current_timestamp()

### scan_run 
* `id` int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT
* `prep_id` varchar(200) NOT NULL
* `performance_center` enum('CSHL','Salk','UCSD','HHMI') DEFAULT NULL COMMENT 'default population is from Histology'
* `machine` enum('Zeiss','Axioscan','Nanozoomer','Olympus VA') DEFAULT NULL
* `objective` enum('60X','40X','20X','10X') DEFAULT NULL
* `resolution` float NOT NULL DEFAULT 0 COMMENT '(µm) lateral resolution if available'
* `zresolution` float NOT NULL DEFAULT 20
* `number_of_slides` int(11) NOT NULL DEFAULT 0
* `scan_date` date DEFAULT NULL
* `file_type` enum('CZI','JPEG2000','NDPI','NGR') DEFAULT NULL
* `scenes_per_slide` enum('1','2','3','4','5','6') DEFAULT NULL
* `section_schema` enum('L to R','R to L') DEFAULT NULL COMMENT 'agreement is one row'
* `channels_per_scene` enum('1','2','3','4') DEFAULT NULL
* `slide_folder_path` varchar(200) DEFAULT NULL COMMENT 'the path to the slides folder on birdstore (files to be converted)'
* `converted_folder_path` varchar(200) DEFAULT NULL COMMENT 'the path to the slides folder on birdstore after convertion'
* `converted_status` enum('not started','converted','converting','error') DEFAULT NULL
* `ch_1_filter_set` enum('68','47','38','46','63','64','50') DEFAULT NULL COMMENT 'This is counterstain Channel'
* `ch_2_filter_set` enum('68','47','38','46','63','64','50') DEFAULT NULL
* `ch_3_filter_set` enum('68','47','38','46','63','64','50') DEFAULT NULL
* `ch_4_filter_set` enum('68','47','38','46','63','64','50') DEFAULT NULL
* `width` int(11) NOT NULL DEFAULT 0
* `height` int(11) NOT NULL DEFAULT 0
* `rotation` int(11) NOT NULL DEFAULT 0
* `flip` enum('none','flip','flop') NOT NULL DEFAULT 'none'
* `comments` varchar(2001) DEFAULT NULL COMMENT 'assessment'
* `active` tinyint(4) NOT NULL DEFAULT 1
* `created` timestamp NULL DEFAULT current_timestamp()
    * KEY `FK__scan_run_prep_id` (`prep_id`)
    * CONSTRAINT `FK__scan_run_prep_id` FOREIGN KEY (`prep_id`) REFERENCES `animal` (`prep_id`)


### slide
* `id` int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT
* `scan_run_id` int(11) NOT NULL
* `slide_physical_id` int(11) NOT NULL COMMENT 'one per slide'
* `rescan_number` enum('','1','2','3') NOT NULL DEFAULT ''
* `slide_status` enum('Bad','Good') NOT NULL DEFAULT 'Good'
* `scenes` int(11) DEFAULT NULL
* `insert_before_one` tinyint(4) NOT NULL DEFAULT 0
* `insert_between_one_two` tinyint(4) NOT NULL DEFAULT 0
* `insert_between_two_three` tinyint(4) NOT NULL DEFAULT 0
* `insert_between_three_four` tinyint(4) NOT NULL DEFAULT 0
* `insert_between_four_five` tinyint(4) NOT NULL DEFAULT 0
* `insert_between_five_six` tinyint(4) NOT NULL DEFAULT 0
* `file_name` varchar(200) NOT NULL
* `comments` varchar(2001) DEFAULT NULL COMMENT 'assessment'
* `active` tinyint(4) NOT NULL DEFAULT 1
* `created` timestamp NULL DEFAULT current_timestamp()
* `file_size` float NOT NULL DEFAULT 0
* `processing_duration` float NOT NULL DEFAULT 0
* `processed` tinyint(4) NOT NULL DEFAULT 0
* `scene_qc_1` tinyint(4) NOT NULL DEFAULT 0
* `scene_qc_2` tinyint(4) NOT NULL DEFAULT 0
* `scene_qc_3` tinyint(4) NOT NULL DEFAULT 0
* `scene_qc_4` tinyint(4) NOT NULL DEFAULT 0
* `scene_qc_5` tinyint(4) NOT NULL DEFAULT 0
* `scene_qc_6` tinyint(4) NOT NULL DEFAULT 0
    * KEY `K__slide_scan_run_id` (`scan_run_id`)
    * CONSTRAINT `FK__slide_scan_run_id` FOREIGN KEY (`scan_run_id`) REFERENCES `scan_run` (`id`) ON UPDATE CASCADE


### slide_czi_to_tif 
* `id` int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT
* `slide_id` int(11) NOT NULL
* `file_name` varchar(200) NOT NULL
* `scene_number` tinyint(4) NOT NULL
* `channel` tinyint(4) NOT NULL
* `width` int(11) NOT NULL DEFAULT 0
* `height` int(11) NOT NULL DEFAULT 0
* `comments` varchar(2000) DEFAULT NULL COMMENT 'assessment'
* `active` tinyint(4) NOT NULL DEFAULT 1
* `created` timestamp NULL DEFAULT current_timestamp()
* `file_size` float NOT NULL DEFAULT 0
* `scene_index` int(11) NOT NULL DEFAULT 0
* `processing_duration` float NOT NULL DEFAULT 0
    * KEY `K__slide_id` (`slide_id`)
    * CONSTRAINT `FK__slide_id` FOREIGN KEY (`slide_id`) REFERENCES `slide` (`id`) ON DELETE CASCADE ON UPDATE CASCADE

### histology
* `id` int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT
* `prep_id` varchar(20) NOT NULL
* `virus_id` int(11) DEFAULT NULL
* `label_id` int(11) DEFAULT NULL
* `performance_center` enum('CSHL','Salk','UCSD','HHMI') DEFAULT NULL COMMENT 'default population is from Injection'
* `anesthesia` enum('ketamine','isoflurane','pentobarbital','fatal plus') DEFAULT NULL
* `perfusion_age_in_days` tinyint(3) unsigned NOT NULL DEFAULT 0
* `perfusion_date` date DEFAULT NULL
* `exsangination_method` enum('PBS','aCSF','Ringers') DEFAULT NULL
* `fixative_method` enum('Para','Glut','Post fix') DEFAULT NULL
* `special_perfusion_notes` varchar(200) DEFAULT NULL
* `post_fixation_period` tinyint(3) unsigned NOT NULL DEFAULT 0 COMMENT '(days)'
* `whole_brain` enum('Y','N') DEFAULT NULL
* `block` varchar(200) DEFAULT NULL COMMENT 'if applicable'
* `date_sectioned` date DEFAULT NULL
* `side_sectioned_first` enum('ASC','DESC') NOT NULL DEFAULT 'ASC'
* `sectioning_method` enum('cryoJane','cryostat','vibratome','optical','sliding microtiome') DEFAULT NULL
* `section_thickness` tinyint(3) unsigned NOT NULL DEFAULT 20 COMMENT '(µm)'
* `orientation` enum('coronal','horizontal','sagittal','oblique') DEFAULT NULL
* `oblique_notes` varchar(200) DEFAULT NULL
* `mounting` enum('every section','2nd','3rd','4th','5ft','6th') DEFAULT NULL COMMENT 'used to automatically populate Placeholder'
* `counterstain` enum('thionin','NtB','NtFR','DAPI','Giemsa','Syto41') DEFAULT NULL
* `comments` varchar(2001) DEFAULT NULL COMMENT 'assessment'
* `created` timestamp NOT NULL DEFAULT current_timestamp()
* `active` tinyint(4) NOT NULL DEFAULT 1
    * KEY `K__histology_virus_id` (`virus_id`)
    * KEY `K__histology_label_id` (`label_id`)
    * KEY `K__histology_prep_id` (`prep_id`)
    * CONSTRAINT `FK__histology_label_id` FOREIGN KEY (`label_id`) REFERENCES `organic_label` (`id`) ON UPDATE CASCADE
    * CONSTRAINT `FK__histology_prep_id` FOREIGN KEY (`prep_id`) REFERENCES `animal` (`prep_id`) ON UPDATE CASCADE
    * CONSTRAINT `FK__histology_virus_id` FOREIGN KEY (`virus_id`) REFERENCES `virus` (`id`) ON UPDATE CASCADE

### organic_label *this table is not being used at all*
* `id` int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT
* `label_id` varchar(20) NOT NULL
* `label_type` enum('Cascade Blue','Chicago Blue','Alexa405','Alexa488','Alexa647','Cy2','Cy3','Cy5','Cy5.5','Cy7','Fluorescein','Rhodamine B','Rhodamine 6G','Texas Red','TMR') DEFAULT NULL
* `type_lot_number` varchar(20) DEFAULT NULL
* `type_tracer` enum('BDA','Dextran','FluoroGold','DiI','DiO') DEFAULT NULL
* `type_details` varchar(500) DEFAULT NULL
* `concentration` float NOT NULL DEFAULT 0 COMMENT '(µM) if applicable'
* `excitation_1p_wavelength` int(11) NOT NULL DEFAULT 0 COMMENT '(nm)'
* `excitation_1p_range` int(11) NOT NULL DEFAULT 0 COMMENT '(nm)'
* `excitation_2p_wavelength` int(11) NOT NULL DEFAULT 0 COMMENT '(nm)'
* `excitation_2p_range` int(11) NOT NULL DEFAULT 0 COMMENT '(nm)'
* `lp_dichroic_cut` int(11) NOT NULL DEFAULT 0 COMMENT '(nm)'
* `emission_wavelength` int(11) NOT NULL DEFAULT 0 COMMENT '(nm)'
* `emission_range` int(11) NOT NULL DEFAULT 0 COMMENT '(nm)'
* `label_source` enum('','Invitrogen','Sigma','Thermo-Fisher') DEFAULT NULL
* `source_details` varchar(100) DEFAULT NULL
* `comments` varchar(2000) DEFAULT NULL COMMENT 'assessment'
* `created` timestamp NOT NULL DEFAULT current_timestamp()
* `active` tinyint(4) NOT NULL DEFAULT 1

### injection
* `id` int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT
* `prep_id` varchar(200) NOT NULL
* `label_id` int(11) DEFAULT NULL
* `performance_center` enum('CSHL','Salk','UCSD','HHMI','Duke') DEFAULT NULL
* `anesthesia` enum('ketamine','isoflurane') DEFAULT NULL
* `method` enum('iontophoresis','pressure','volume') DEFAULT NULL
* `injection_volume` float NOT NULL DEFAULT 0 COMMENT '(nL)'
* `pipet` enum('glass','quartz','Hamilton','syringe needle') DEFAULT NULL
* `location` varchar(20) DEFAULT NULL COMMENT 'examples: muscle, brain region'
* `angle` varchar(20) DEFAULT NULL
* `brain_location_dv` float NOT NULL DEFAULT 0 COMMENT '(mm) dorsal-ventral relative to Bregma'
* `brain_location_ml` float NOT NULL DEFAULT 0 COMMENT '(mm) medial-lateral relative to Bregma; check if positive'
* `brain_location_ap` float NOT NULL DEFAULT 0 COMMENT '(mm) anterior-posterior relative to Bregma'
* `injection_date` date DEFAULT NULL
* `transport_days` int(11) NOT NULL DEFAULT 0
* `virus_count` int(11) NOT NULL DEFAULT 0
* `comments` varchar(2001) DEFAULT NULL COMMENT 'assessment'
* `created` timestamp NOT NULL DEFAULT current_timestamp()
* `active` tinyint(4) NOT NULL DEFAULT 1
    * KEY `K__label_id` (`label_id`)
    * KEY `FK__injection_prep_id` (`prep_id`)
    * CONSTRAINT `FK__injection_label_id` FOREIGN KEY (`label_id`) REFERENCES `organic_label` (`id`) ON UPDATE CASCADE
    * CONSTRAINT `FK__injection_prep_id` FOREIGN KEY (`prep_id`) REFERENCES `animal` (`prep_id`)

### injection_virus
* `id` int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT
* `injection_id` int(11) NOT NULL
* `virus_id` int(11) NOT NULL
* `created` timestamp NOT NULL DEFAULT current_timestamp()
* `active` tinyint(4) NOT NULL DEFAULT 1
    * KEY `K__IV_injection_id` (`injection_id`)
    * KEY `K__IV_virus_id` (`virus_id`)
    * CONSTRAINT `FK__IV_injection_id` FOREIGN KEY (`injection_id`) REFERENCES `injection` (`id`) ON UPDATE CASCADE
    * CONSTRAINT `FK__IV_virus_id` FOREIGN KEY (`virus_id`) REFERENCES `virus` (`id`) ON UPDATE CASCADE

### virus
* `id` int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT
* `virus_name` varchar(50) NOT NULL
* `virus_type` enum('Adenovirus','AAV','CAV','DG rabies','G-pseudo-Lenti','Herpes','Lenti','N2C rabies','Sinbis') DEFAULT NULL
* `virus_active` enum('yes','no') DEFAULT NULL
* `type_details` varchar(500) DEFAULT NULL
* `titer` float NOT NULL DEFAULT 0 COMMENT '(particles/ml) if applicable'
* `lot_number` varchar(20) DEFAULT NULL
* `label` enum('YFP','GFP','RFP','histo-tag') DEFAULT NULL
* `label2` varchar(200) DEFAULT NULL
* `excitation_1p_wavelength` int(11) NOT NULL DEFAULT 0 COMMENT '(nm) if applicable'
* `excitation_1p_range` int(11) NOT NULL DEFAULT 0 COMMENT '(nm) if applicable'
* `excitation_2p_wavelength` int(11) NOT NULL DEFAULT 0 COMMENT '(nm) if applicable'
* `excitation_2p_range` int(11) NOT NULL DEFAULT 0 COMMENT '(nm) if applicable'
* `lp_dichroic_cut` int(11) NOT NULL DEFAULT 0 COMMENT '(nm) if applicable'
* `emission_wavelength` int(11) NOT NULL DEFAULT 0 COMMENT '(nm) if applicable'
* `emission_range` int(11) NOT NULL DEFAULT 0 COMMENT '(nm) if applicable0'
* `virus_source` enum('Adgene','Salk','Penn','UNC') DEFAULT NULL
* `source_details` varchar(100) DEFAULT NULL
* `comments` varchar(2000) DEFAULT NULL COMMENT 'assessment'
* `created` timestamp NOT NULL DEFAULT current_timestamp()
* `active` tinyint(4) NOT NULL DEFAULT 1

## Tables specifically related to Neuroglancer metadata
### layer_data 
* `id` int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT
* `prep_id` varchar(20) NOT NULL
* `structure_id` int(11) NOT NULL
* `person_id` int(11) NOT NULL
* `updated_by` int(11) DEFAULT NULL
* `input_type_id` int(11) NOT NULL DEFAULT 1
* `vetted` enum('yes','no') DEFAULT NULL
* `layer` varchar(255) DEFAULT NULL,
* `x` float DEFAULT NULL
* `y` float DEFAULT NULL
* `section` float NOT NULL DEFAULT 0
* `active` tinyint(1) DEFAULT NULL
* `created` datetime(6) NOT NULL
* `updated` timestamp NOT NULL DEFAULT current_timestamp()
    * KEY `K__LDA_AID` (`prep_id`)
    * KEY `K__LDA_SID` (`structure_id`)
    * KEY `K__LDA_PID` (`person_id`)
    * KEY `K__LDA_ITID` (`input_type_id`)
    * CONSTRAINT `FK__LDA_AID` FOREIGN KEY (`prep_id`) REFERENCES `animal` (`prep_id`) ON UPDATE CASCADE
    * CONSTRAINT `FK__LDA_ITID` FOREIGN KEY (`input_type_id`) REFERENCES `com_type` (`id`)
    * CONSTRAINT `FK__LDA_PID` FOREIGN KEY (`person_id`) REFERENCES `auth_user` (`id`)
    * CONSTRAINT `FK__LDA_STRID` FOREIGN KEY (`structure_id`) REFERENCES `structure` (`id`) ON UPDATE CASCADE

### structure 
* `id` int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT
* `abbreviation` varchar(25) COLLATE utf8_bin NOT NULL
* `description` longtext COLLATE utf8_bin NOT NULL
* `color` int(11) NOT NULL DEFAULT 100
* `hexadecimal` char(7) COLLATE utf8_bin DEFAULT NULL
* `active` tinyint(1) NOT NULL
* `created` datetime(6) NOT NULL
    * KEY `K__S_ABBREV` (`abbreviation`)
### com_type 
* `id` int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT
* `input_type` varchar(50) NOT NULL
* `description` varchar(255) DEFAULT NULL
* `active` tinyint(1) NOT NULL DEFAULT 1
* `created` datetime(6) NOT NULL
* `updated` timestamp NOT NULL DEFAULT current_timestamp()
    * KEY `K__CT_INID` (`input_type`)
### neuroglancer_urls
* `id` int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT
* `person_id` int(11) DEFAULT NULL
* `url` longtext NOT NULL
* `active` tinyint(1) DEFAULT NULL
* `vetted` tinyint(1) DEFAULT NULL
* `created` datetime(6) NOT NULL
* `user_date` varchar(25) DEFAULT NULL
* `comments` varchar(255) DEFAULT NULL
* `updated` timestamp NOT NULL DEFAULT current_timestamp()

### `elastix_transformation`
* `id` int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT
* `prep_id`  varchar(20) NOT NULL
* `section`  char(3)
* `rotation` float
* `xshift`   float
* `yshift`   float
* `created`  timestamp NOT NULL current_timestamp()
* `active`   tinyint(4)
    * UNIQUE KEY `UK__ETR_prep_id_section` (`prep_id`,`section`)
    * KEY `K__ETR_prepid` (`prep_id`)
    * CONSTRAINT `FK__ETR_prepid` FOREIGN KEY (`prep_id`) REFERENCES `animal` (`prep_id`)

### `file_log`
* `id` int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT
* `prep_id`      varchar(20)                            
* `progress_id`  int(11)                                
* `filename`     varchar(255)                               
* `active`       tinyint(1)                                
* `created`      timestamp NOT NULL current_timestamp()                 
    * UNIQUE KEY `UK__AID_PID_C_S` (`prep_id`,`progress_id`,`filename`)
    * KEY `K__FILE_LOG_AID` (`prep_id`)
    * KEY `K__FILE_LOG_PID` (`progress_id`)
    * CONSTRAINT `FK__FILE_LOG_AID` FOREIGN KEY (`prep_id`) REFERENCES `animal` (`prep_id`) ON UPDATE CASCADE,
    * CONSTRAINT `FK__FILE_LOG_PID` FOREIGN KEY (`progress_id`) REFERENCES `progress_lookup` (`id`) ON UPDATE CASCADE

