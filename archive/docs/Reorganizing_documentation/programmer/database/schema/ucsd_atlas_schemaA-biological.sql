/* TABLE OF CONTENTS - OVERALL ORGANIZATION STRUCTURE (BIOLOGICAL):

   TOTAL TABLES (12): alias, animal, structure, breeder_line, genotype, histology, injection, injection_virus, organic_label, scan_run, virus, vendor_strain

FOR REVIEW (27-JAN-2022):
-animal [NORMALIZATION]
-breeder_line [NEW - NORMALIZATION]
-genotype [NEW - NORMALIZATION]
-vendor_strain

   UPDATE (6-APR-2022):
   renamed 'brain_region' to 'structure'

   TABLES MODIFIED FROM schema2.sql [WITH JUSTIFICATION]
   alias [NORMALIZATION] *APPROVED 26-JAN-2022
   animal [NORMALIZATION]
   brain_region [NORMALIZATION AND NEW NAME BETTER DESCRIBES ROLE OF TABLE] *APPROVED 26-JAN-2022
   breeder_line [NEW - NORMALIZATION]
   genotype [NEW - NORMALIZATION]
   histology [NORMALIZATION] *APPROVED 26-JAN-2022
   injection [NORMALIZATION] *APPROVED 26-JAN-2022
   scan_run [NORMALIZATION] *APPROVED 26-JAN-2022
   vendor_strain [NORMALIZATION & ADDED STRAIN (PREV stock_number FIELD IN animal TABLE)]

   TABLES NOT MODIFIED FROM schema2.sql
   injection_virus
   organic_label
   virus
*/

/*
   COMMENTS RELATED TO TABLE: animal
   DR - performance_center FIELD REMOVED; DATA IN SEPARATE TABLE (performance_center) [JUSTIFICATION: NORMALIZATION]
   DR - aliases_1, aliases_2, aliases_3, aliases_4, aliases_5 FIELDS REMOVED; DATA IN SEPARATE TABLE alias [JUSTIFICATION: NORMALIZATION]
   DR - FK_vendor_strain_id FIELD MODIFIED TO REFERENCE vendor_strain TABLE (INCLUDES stock_number, vendor_name)
*/

DROP TABLE IF EXISTS `alias`;
CREATE TABLE `alias` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `FK_animal_id` int(11),
  `name` varchar(100) DEFAULT NULL,
  FOREIGN KEY (`FK_animal_id`) REFERENCES animal(`id`) ON DELETE CASCADE,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE `animal` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `prep_id` varchar(20) DEFAULT NULL COMMENT 'LEGACY: Name for lab animal, max 20 chars',
  `date_of_birth` date DEFAULT NULL COMMENT 'organism date of birth',
  `sex` enum('M', 'F', 'Hermaphrodite', 'DoesNotApply') DEFAULT NULL,
  `tissue_source` enum('animal','brain','slides') DEFAULT NULL,
  `ship_date` date DEFAULT NULL,
  `shipper` enum('FedEx','UPS') DEFAULT NULL,
  `tracking_number` varchar(100) DEFAULT NULL,
  `comments` varchar(2001) DEFAULT NULL COMMENT 'assessment',
  `active` tinyint(1) NOT NULL DEFAULT 1,
  `created` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `species` enum('mouse','rat') DEFAULT NULL,
  `FK_performance_center_id` int(11),
  `FK_alias_id` int(11),
  `FK_vendor_strain_id` int(11),
  `FK_breeder_line_id` int(11),
  `FK_genotype_id` int(11),
  FOREIGN KEY (`FK_performance_center_id`) REFERENCES performance_center(`id`),
  FOREIGN KEY (`FK_alias_id`) REFERENCES alias(`id`),
  FOREIGN KEY (`FK_vendor_strain_id`) REFERENCES vendor_strain(`id`),
  FOREIGN KEY (`FK_breeder_line_id`) REFERENCES breeder_line(`id`),
  FOREIGN KEY (`FK_genotype_id`) REFERENCES genotype(`id`),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `structure`;
CREATE TABLE `structure` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `active` tinyint(1) NOT NULL DEFAULT 1,
  `created` datetime DEFAULT current_timestamp(),
  `abbreviation` varchar(200) NOT NULL,
  `description` longtext NOT NULL,
  `color` int(11) NOT NULL DEFAULT 100,
  `hexadecimal` char(7) COLLATE utf8_bin DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `breeder_line`;
CREATE TABLE `breeder_line` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `breeder_line` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `genotype`;
CREATE TABLE `genotype` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `parent1` varchar(100) DEFAULT NULL COMMENT 'father',  
  `parent2` varchar(100) DEFAULT NULL COMMENT 'mother',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `histology`;
CREATE TABLE `histology` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `prep_id` varchar(20) DEFAULT NULL COMMENT 'LEGACY: Name for lab animal, max 20 chars',
  `anesthesia` enum('ketamine','isoflurane','pentobarbital','fatal plus') DEFAULT NULL,
  `perfusion_age_in_days` tinyint(3) unsigned NOT NULL DEFAULT 0,
  `perfusion_date` date DEFAULT NULL,
  `exsangination_method` enum('PBS','aCSF','Ringers') DEFAULT NULL,
  `fixative_method` enum('Para','Glut','Post fix') DEFAULT NULL,
  `special_perfusion_notes` varchar(200) DEFAULT NULL,
  `post_fixation_period` tinyint(3) unsigned NOT NULL DEFAULT 0 COMMENT '(days)',
  `whole_brain` enum('Y','N') DEFAULT NULL,
  `block` varchar(200) DEFAULT NULL COMMENT 'if applicable',
  `date_sectioned` date DEFAULT NULL,
  `side_sectioned_first` enum('ASC','DESC') NOT NULL DEFAULT 'ASC',
  `sectioning_method` enum('cryoJane','cryostat','vibratome','optical','sliding microtiome') DEFAULT NULL,
  `section_thickness` tinyint(3) unsigned NOT NULL DEFAULT 20 COMMENT '(µm)',
  `orientation` enum('coronal','horizontal','sagittal','oblique') DEFAULT NULL,
  `oblique_notes` varchar(200) DEFAULT NULL,
  `mounting` enum('every section','2nd','3rd','4th','5ft','6th') DEFAULT NULL COMMENT 'used to automatically populate Placeholder',
  `counterstain` enum('thionin','NtB','NtFR','DAPI','Giemsa','Syto41') DEFAULT NULL,
  `comments` longtext DEFAULT NULL COMMENT 'assessment',
  `active` tinyint(1) NOT NULL DEFAULT 1,
  `created` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `FK_animal_id` int(11),
  `FK_virus_id` int(11),
  `FK_performance_center_id` int(11),
  `FK_organic_label_id` int(11),
  FOREIGN KEY (`FK_animal_id`) REFERENCES animal(`id`) ON UPDATE CASCADE,
  FOREIGN KEY (`FK_virus_id`) REFERENCES virus(id),
  FOREIGN KEY (`FK_performance_center_id`) REFERENCES performance_center(`id`),
  FOREIGN KEY (`FK_organic_label_id`) REFERENCES organic_label(id),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/*
   COMMENTS RELATED TO TABLE: injection
   DR - FIELDS 'label_id' - related to Neuroglancer?
   DR - FIELD performance_center NOW INCORPORATED INTO performance_center TABLE
   DR - FIELD injection_volume PREVIOUSLY DID NOT HAVE UNIT (COMMENT WAS nL); NEW FIELD ADDED injection_volume_ul
*/

DROP TABLE IF EXISTS `injection`;
CREATE TABLE `injection` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `prep_id` varchar(20) NOT NULL COMMENT 'LEGACY: Name for lab animal, max 20 chars',
  `label_id` int(11) DEFAULT NULL,
  `active` tinyint(1) NOT NULL DEFAULT 1,
  `created` datetime DEFAULT current_timestamp(),
  `anesthesia` enum('ketamine','isoflurane') DEFAULT NULL,
  `method` enum('iontophoresis','pressure','volume') DEFAULT NULL,
  `injection_volume_ul` varchar(20) DEFAULT NULL,
  `pipet` enum('glass','quartz','Hamilton','syringe needle') DEFAULT NULL,
  `location` varchar(20) DEFAULT NULL COMMENT 'examples: muscle, brain region',
  `angle` varchar(20) DEFAULT NULL,
  `brain_location_dv` double NOT NULL DEFAULT 0 COMMENT '(mm) dorsal-ventral relative to Bregma',
  `brain_location_ml` double NOT NULL DEFAULT 0 COMMENT '(mm) medial-lateral relative to Bregma; check if positive',
  `brain_location_ap` double NOT NULL DEFAULT 0 COMMENT '(mm) anterior-posterior relative to Bregma',
  `injection_date` date DEFAULT NULL,
  `transport_days` int(11) NOT NULL DEFAULT 0,
  `virus_count` int(11) NOT NULL DEFAULT 0,
  `comments` longtext DEFAULT NULL,
  `FK_performance_center_id` int(11),
  `FK_animal_id` int(11),
  `FK_ref_atlas_id` int(11),
  FOREIGN KEY (`FK_performance_center_id`) REFERENCES performance_center(`id`),
  FOREIGN KEY (`FK_animal_id`) REFERENCES animal(`id`),
  FOREIGN KEY (`FK_ref_atlas_id`) REFERENCES biosource(`id`),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `injection_virus`;
CREATE TABLE `injection_virus` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `active` tinyint(1) NOT NULL DEFAULT 1,
  `created` datetime DEFAULT current_timestamp(),
  `FK_injection_id` int(11) NOT NULL,
  `FK_virus_id` int(11) NOT NULL,
  FOREIGN KEY (`FK_injection_id`) REFERENCES injection(`id`),
  FOREIGN KEY (`FK_virus_id`) REFERENCES virus(`id`),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/*
   COMMENTS RELATED TO TABLE: organic_label
   Unknown contrib -  Oraganics are labels that can be injected
*/

DROP TABLE IF EXISTS `organic_label`;
CREATE TABLE `organic_label` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `label_id` varchar(20) NOT NULL,
  `label_type` enum('Cascade Blue','Chicago Blue','Alexa405','Alexa488','Alexa647','Cy2','Cy3','Cy5','Cy5.5','Cy7','Fluorescein','Rhodamine B','Rhodamine 6G','Texas Red','TMR') DEFAULT NULL,
  `type_lot_number` varchar(20) DEFAULT NULL,
  `type_tracer` enum('BDA','Dextran','FluoroGold','DiI','DiO') DEFAULT NULL,
  `type_details` varchar(500) DEFAULT NULL,
  `concentration` float NOT NULL DEFAULT 0 COMMENT '(µM) if applicable',
  `excitation_1p_wavelength` int(11) NOT NULL DEFAULT 0 COMMENT '(nm)',
  `excitation_1p_range` int(11) NOT NULL DEFAULT 0 COMMENT '(nm)',
  `excitation_2p_wavelength` int(11) NOT NULL DEFAULT 0 COMMENT '(nm)',
  `excitation_2p_range` int(11) NOT NULL DEFAULT 0 COMMENT '(nm)',
  `lp_dichroic_cut` int(11) NOT NULL DEFAULT 0 COMMENT '(nm)',
  `emission_wavelength` int(11) NOT NULL DEFAULT 0 COMMENT '(nm)',
  `emission_range` int(11) NOT NULL DEFAULT 0 COMMENT '(nm)',
  `label_source` enum('','Invitrogen','Sigma','Thermo-Fisher') DEFAULT NULL,
  `source_details` varchar(100) DEFAULT NULL,
  `comments` longtext DEFAULT NULL COMMENT 'assessment',
  `created` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `active` tinyint(1) NOT NULL DEFAULT 1,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/*
   COMMENTS RELATED TO TABLE: scan_run
   DR - FIELD performance_center NOW INCORPORATED INTO performance_center TABLE
   DR - FIELD performance_center MOVED TO SEPARATE TABLE
*/

DROP TABLE IF EXISTS `scan_run`;
CREATE TABLE `scan_run` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `prep_id` varchar(20) NOT NULL COMMENT 'LEGACY: Name for lab animal, max 20 chars',
  `active` tinyint(1) NOT NULL DEFAULT 1,
  `created` datetime DEFAULT current_timestamp(),
  `machine` enum('Zeiss','Axioscan','Nanozoomer','Olympus VA') DEFAULT NULL,
  `objective` enum('60X','40X','20X','10X') DEFAULT NULL,
  `resolution` double NOT NULL DEFAULT 0 COMMENT '(µm) lateral resolution if available',
  `zresolution` double NOT NULL,
  `number_of_slides` int(11) NOT NULL,
  `scan_date` date DEFAULT NULL,
  `file_type` enum('CZI','JPEG2000','NDPI','NGR') DEFAULT NULL,
  `scenes_per_slide` enum('1','2','3','4','5','6') DEFAULT NULL,
  `section_schema` enum('L to R','R to L') DEFAULT NULL COMMENT 'agreement is one row',
  `channels_per_scene` enum('1','2','3','4') DEFAULT NULL,
  `slide_folder_path` varchar(200) DEFAULT NULL COMMENT 'the path to the slides folder on birdstore (files to be converted)',
  `converted_folder_path` varchar(200) DEFAULT NULL COMMENT 'the path to the slides folder on birdstore after convertion',
  `converted_status` enum('not started','converted','converting','error') DEFAULT NULL,
  `ch_1_filter_set` enum('68','47','38','46','63','64','50') DEFAULT NULL COMMENT 'This is counterstain Channel',
  `ch_2_filter_set` enum('68','47','38','46','63','64','50') DEFAULT NULL,
  `ch_3_filter_set` enum('68','47','38','46','63','64','50') DEFAULT NULL,
  `ch_4_filter_set` enum('68','47','38','46','63','64','50') DEFAULT NULL,
  `rotation` int(11) NOT NULL DEFAULT 0,
  `flip` enum('none','flip','flop') NOT NULL DEFAULT 'none',
  `width` int(11) NOT NULL,
  `height` int(11) NOT NULL,
  `comments` longtext DEFAULT NULL,
  `FK_animal_id` int(11),
  `FK_performance_center_id` int(11),
  FOREIGN KEY (`FK_animal_id`) REFERENCES animal(`id`),
  FOREIGN KEY (`FK_performance_center_id`) REFERENCES performance_center(`id`),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/*
   COMMENTS RELATED TO TABLE: virus
   DR - recommend move virus_type, virus_source to separate tables
*/

DROP TABLE IF EXISTS `virus`;
CREATE TABLE `virus` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `active` tinyint(1) NOT NULL DEFAULT 1,
  `created` datetime DEFAULT current_timestamp(),
  `virus_name` varchar(50) NOT NULL,
  `virus_type` enum('Adenovirus','AAV','CAV','DG rabies','G-pseudo-Lenti','Herpes','Lenti','N2C rabies','Sinbis') DEFAULT NULL,
  `virus_active` enum('yes','no') DEFAULT NULL,
  `type_details` varchar(500) DEFAULT NULL,
  `titer` double NOT NULL DEFAULT 0 COMMENT '(particles/ml) if applicable',
  `lot_number` varchar(20) DEFAULT NULL,
  `label` enum('YFP','GFP','RFP','histo-tag') DEFAULT NULL,
  `label2` varchar(200) DEFAULT NULL,
  `excitation_1p_wavelength` int(11) NOT NULL,
  `excitation_1p_range` int(11) NOT NULL,
  `excitation_2p_wavelength` int(11) NOT NULL,
  `excitation_2p_range` int(11) NOT NULL,
  `lp_dichroic_cut` int(11) NOT NULL,
  `emission_wavelength` int(11) NOT NULL,
  `emission_range` int(11) NOT NULL,
  `virus_source` enum('Adgene','Salk','Penn','UNC') DEFAULT NULL,
  `source_details` varchar(100) DEFAULT NULL,
  `comments` longtext DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `vendor_strain`;
CREATE TABLE `vendor_strain` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `vendor` enum('Jackson','Charles River','Harlan','NIH','Taconic', 'UCSD') DEFAULT NULL,
  `strain` varchar(220) DEFAULT NULL COMMENT 'strain is inclusive of stock_number',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
