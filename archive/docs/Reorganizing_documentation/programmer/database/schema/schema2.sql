-- MySQL dump 10.19  Distrib 10.3.31-MariaDB, for debian-linux-gnu (x86_64)
--
-- Host: localhost    Database: active_atlas_development
-- ------------------------------------------------------
-- Server version	10.3.31-MariaDB-0ubuntu0.20.04.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Main Tables, used by all users of brainsharer.
--

-- Table structure for table `animal`
--

DROP TABLE IF EXISTS `animal`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `animal` (
  `prep_id` varchar(20) NOT NULL COMMENT 'Name for lab mouse/rat, max 20 chars',
  `performance_center` enum('CSHL','Salk','UCSD','HHMI','Duke') DEFAULT NULL,
  `date_of_birth` date DEFAULT NULL COMMENT 'the mouse''s date of birth',
  `species` enum('mouse','rat') DEFAULT NULL,
  `strain` varchar(50) DEFAULT NULL,
  `sex` enum('M','F') DEFAULT NULL COMMENT '(M/F) either ''M'' for male, ''F'' for female',
  `genotype` varchar(100) DEFAULT NULL COMMENT 'transgenic description, usually "C57"; We will need a genotype table',
  `breeder_line` varchar(100) DEFAULT NULL COMMENT 'We will need a local breeding table',
  `vender` enum('Jackson','Charles River','Harlan','NIH','Taconic') DEFAULT NULL,
  `stock_number` varchar(100) DEFAULT NULL COMMENT 'if not from a performance center',
  `tissue_source` enum('animal','brain','slides') DEFAULT NULL,
  `ship_date` date DEFAULT NULL,
  `shipper` enum('FedEx','UPS') DEFAULT NULL,
  `tracking_number` varchar(100) DEFAULT NULL,
  `aliases_1` varchar(100) DEFAULT NULL COMMENT 'names given by others, alternatives to prep-id. Should be refactored to a separate table because often no aliases and sometimes more than 5'
  `aliases_2` varchar(100) DEFAULT NULL,
  `aliases_3` varchar(100) DEFAULT NULL,
  `aliases_4` varchar(100) DEFAULT NULL,
  `aliases_5` varchar(100) DEFAULT NULL,
  `comments` varchar(2001) DEFAULT NULL COMMENT 'assessment',
  `active` tinyint(4) NOT NULL DEFAULT 1,
  `created` timestamp NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`prep_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- A list of 3D regions in the mouse brain
-- Includes the 52 landmarks from the foundation brains.
DROP TABLE IF EXISTS `structure`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `structure` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `abbreviation` varchar(25) COLLATE utf8_bin NOT NULL,
  `description` longtext COLLATE utf8_bin NOT NULL,
  `color` int(11) NOT NULL DEFAULT 100,
  `hexadecimal` char(7) COLLATE utf8_bin DEFAULT NULL,   /** What is this for? */
  `active` tinyint(1) NOT NULL,
  `created` datetime(6) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `K__S_ABBREV` (`abbreviation`)
) ENGINE=InnoDB AUTO_INCREMENT=54 DEFAULT CHARSET=utf8 COLLATE=utf8_bin;

/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `transformation`
--  /* YF: Does this store transformations between stack coordinates and atlas coordinates (both */
-- /* ways)?
--  YF: that we need is a blob that holds a pickle string that
---  defined the type of transformation and the parameters of the transformation
--

DROP TABLE IF EXISTS `transformation`;
/*   What is the role of this table *****/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `transformation` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `prep_id` varchar(20) NOT NULL,
  `person_id` int(11) NOT NULL,
  `input_type_id` int(11) NOT NULL DEFAULT 1,
  `com_name` varchar(50) NOT NULL,
  `active` tinyint(1) NOT NULL DEFAULT 1,
  `created` datetime(6) NOT NULL,
  `updated` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  UNIQUE KEY `UK__T_AID_PID_ITID` (`prep_id`,`person_id`,`input_type_id`),
  KEY `K__T_AID` (`prep_id`),
  KEY `K__T_PID` (`person_id`),
  KEY `K__T_ITID` (`input_type_id`),
  CONSTRAINT `FK__T_ITID` FOREIGN KEY (`input_type_id`) REFERENCES `com_type` (`id`),
  CONSTRAINT `FK__T_PID` FOREIGN KEY (`person_id`) REFERENCES `auth_user` (`id`),
  CONSTRAINT `FK__T_prep_id` FOREIGN KEY (`prep_id`) REFERENCES `animal` (`prep_id`) ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `histology`
--

DROP TABLE IF EXISTS `histology`;
/* this table describes the histological preparation of the animal***/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `histology` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `prep_id` varchar(20) NOT NULL,
  `virus_id` int(11) DEFAULT NULL,
  `label_id` int(11) DEFAULT NULL,
  `performance_center` enum('CSHL','Salk','UCSD','HHMI') DEFAULT NULL COMMENT 'default population is from Injection',
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
  `comments` varchar(2001) DEFAULT NULL COMMENT 'assessment',
  `created` timestamp NOT NULL DEFAULT current_timestamp(),
  `active` tinyint(4) NOT NULL DEFAULT 1,
  PRIMARY KEY (`id`),
  KEY `K__histology_virus_id` (`virus_id`),
  KEY `K__histology_label_id` (`label_id`),
  KEY `K__histology_prep_id` (`prep_id`),
  CONSTRAINT `FK__histology_label_id` FOREIGN KEY (`label_id`) REFERENCES `organic_label` (`id`) ON UPDATE CASCADE,
  CONSTRAINT `FK__histology_prep_id` FOREIGN KEY (`prep_id`) REFERENCES `animal` (`prep_id`) ON UPDATE CASCADE,
  CONSTRAINT `FK__histology_virus_id` FOREIGN KEY (`virus_id`) REFERENCES `virus` (`id`) ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=25 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;


--
--  injection related tables
--

-- Table structure for table `injection`
-- DK: organics can also be injected.

DROP TABLE IF EXISTS `injection`;
/* Describes the location and type of injected dyes and viruses */
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `injection` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `prep_id` varchar(200) NOT NULL,
  `label_id` int(11) DEFAULT NULL,
  `performance_center` enum('CSHL','Salk','UCSD','HHMI','Duke') DEFAULT NULL,
  `anesthesia` enum('ketamine','isoflurane') DEFAULT NULL,
  `method` enum('iontophoresis','pressure','volume') DEFAULT NULL,
  `injection_volume` float NOT NULL DEFAULT 0 COMMENT '(nL)',
  `pipet` enum('glass','quartz','Hamilton','syringe needle') DEFAULT NULL,
  `location` varchar(20) DEFAULT NULL COMMENT 'examples: muscle, brain region',
  `angle` varchar(20) DEFAULT NULL,
  `brain_location_dv` float NOT NULL DEFAULT 0 COMMENT '(mm) dorsal-ventral relative to Bregma',
  `brain_location_ml` float NOT NULL DEFAULT 0 COMMENT '(mm) medial-lateral relative to Bregma; check if positive',
  `brain_location_ap` float NOT NULL DEFAULT 0 COMMENT '(mm) anterior-posterior relative to Bregma',
  `injection_date` date DEFAULT NULL,
  `transport_days` int(11) NOT NULL DEFAULT 0,
  `virus_count` int(11) NOT NULL DEFAULT 0,
  `comments` varchar(2001) DEFAULT NULL COMMENT 'assessment',
  `created` timestamp NOT NULL DEFAULT current_timestamp(),
  `active` tinyint(4) NOT NULL DEFAULT 1,
  PRIMARY KEY (`id`),
  KEY `K__label_id` (`label_id`),  /* What is the meaning of the name  `K__label_id` *********/
  KEY `FK__injection_prep_id` (`prep_id`),
  CONSTRAINT `FK__injection_label_id` FOREIGN KEY (`label_id`) REFERENCES `organic_label` (`id`) ON UPDATE CASCADE,
  CONSTRAINT `FK__injection_prep_id` FOREIGN KEY (`prep_id`) REFERENCES `animal` (`prep_id`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `injection_virus`
--  - many to many join table
--

DROP TABLE IF EXISTS `injection_virus`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `injection_virus` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `injection_id` int(11) NOT NULL,
  `virus_id` int(11) NOT NULL,
  `created` timestamp NOT NULL DEFAULT current_timestamp(),
  `active` tinyint(4) NOT NULL DEFAULT 1,
  PRIMARY KEY (`id`),
  KEY `K__IV_injection_id` (`injection_id`),
  KEY `K__IV_virus_id` (`virus_id`),
  CONSTRAINT `FK__IV_injection_id` FOREIGN KEY (`injection_id`) REFERENCES `injection` (`id`) ON UPDATE CASCADE,
  CONSTRAINT `FK__IV_virus_id` FOREIGN KEY (`virus_id`) REFERENCES `virus` (`id`) ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `virus`
--

DROP TABLE IF EXISTS `virus`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `virus` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `virus_name` varchar(50) NOT NULL,
  `virus_type` enum('Adenovirus','AAV','CAV','DG rabies','G-pseudo-Lenti','Herpes','Lenti','N2C rabies','Sinbis') DEFAULT NULL,
  `virus_active` enum('yes','no') DEFAULT NULL,
  `type_details` varchar(500) DEFAULT NULL,
  `titer` float NOT NULL DEFAULT 0 COMMENT '(particles/ml) if applicable',
  `lot_number` varchar(20) DEFAULT NULL,
  `label` enum('YFP','GFP','RFP','histo-tag') DEFAULT NULL,
  `label2` varchar(200) DEFAULT NULL,
  `excitation_1p_wavelength` int(11) NOT NULL DEFAULT 0 COMMENT '(nm) if applicable',
  `excitation_1p_range` int(11) NOT NULL DEFAULT 0 COMMENT '(nm) if applicable',
  `excitation_2p_wavelength` int(11) NOT NULL DEFAULT 0 COMMENT '(nm) if applicable',
  `excitation_2p_range` int(11) NOT NULL DEFAULT 0 COMMENT '(nm) if applicable',
  `lp_dichroic_cut` int(11) NOT NULL DEFAULT 0 COMMENT '(nm) if applicable',
  `emission_wavelength` int(11) NOT NULL DEFAULT 0 COMMENT '(nm) if applicable',
  `emission_range` int(11) NOT NULL DEFAULT 0 COMMENT '(nm) if applicable0',
  `virus_source` enum('Adgene','Salk','Penn','UNC') DEFAULT NULL,
  `source_details` varchar(100) DEFAULT NULL,
  `comments` varchar(2000) DEFAULT NULL COMMENT 'assessment',
  `created` timestamp NOT NULL DEFAULT current_timestamp(),
  `active` tinyint(4) NOT NULL DEFAULT 1,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `organic_label`
-- Oraganics are labels that can be injected 

DROP TABLE IF EXISTS `organic_label`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
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
  `comments` varchar(2000) DEFAULT NULL COMMENT 'assessment',
  `created` timestamp NOT NULL DEFAULT current_timestamp(),
  `active` tinyint(4) NOT NULL DEFAULT 1,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Annotation related tables   ---------------------------------------------------------
--

--
-- tables of brain locations (x,y,z location data)
--

-- Table structure for table `layer_data`
-- The name "layer_data should be changed"
--  - main annotation table where the x,y,z data is stored
--   Where are the atlas COMs stored? **************************************************

DROP TABLE IF EXISTS `layer_data`;
/*   What is the role of this table *****/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `layer_data` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `prep_id` varchar(20) NOT NULL,
  `structure_id` int(11) NOT NULL comments 'either structure, point, or line   do we really want line here?',
  `person_id` int(11) NOT NULL comments 'person that created the point',
  `updated_by` int(11) DEFAULT NULL comments 'person that updated the row',
  `input_type_id` int(11) NOT NULL DEFAULT 1 comments 'manual person, corrected person, detected computer',
  `vetted` enum('yes','no') DEFAULT NULL comments 'good enough for public',
  `layer` varchar(255) DEFAULT NULL comments 'freeform name/label the layer',
  `x` float DEFAULT NULL,
  `y` float DEFAULT NULL,
  `section` float NOT NULL DEFAULT 0,
  `active` tinyint(1) DEFAULT NULL comments 'old data gets marked inactive from Neuroglancer',
  `created` datetime(6) NOT NULL comments 'when the row was created',
  `updated` timestamp NOT NULL DEFAULT current_timestamp() comments 'auto timestamp',
  PRIMARY KEY (`id`),
  KEY `K__LDA_AID` (`prep_id`),
  KEY `K__LDA_SID` (`structure_id`),
  KEY `K__LDA_PID` (`person_id`),
  KEY `K__LDA_ITID` (`input_type_id`),
  CONSTRAINT `FK__LDA_AID` FOREIGN KEY (`prep_id`) REFERENCES `animal` (`prep_id`) ON UPDATE CASCADE,
  CONSTRAINT `FK__LDA_ITID` FOREIGN KEY (`input_type_id`) REFERENCES `com_type` (`id`),
  CONSTRAINT `FK__LDA_PID` FOREIGN KEY (`person_id`) REFERENCES `auth_user` (`id`),
  CONSTRAINT `FK__LDA_STRID` FOREIGN KEY (`structure_id`) REFERENCES `structure` (`id`) ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=15566 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `com_type`
--  - should be renamed to input_type, lookup table of manual, detected, corrected
--

DROP TABLE IF EXISTS `com_type`;
/* what is the role of this table *********/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `com_type` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `input_type` varchar(50) NOT NULL,
  `description` varchar(255) DEFAULT NULL,
  `active` tinyint(1) NOT NULL DEFAULT 1,
  `created` datetime(6) NOT NULL,
  `updated` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  KEY `K__CT_INID` (`input_type`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;


--
-- tables of boundaries, does not exist yet. Will store 2D and 3D boundaries
--


--
-- tables related to hardware/software --------------------------------------------------
--

--
--  Zeiss Z1 Scanner related tables
--

-- Table structure for table `slide`
--

DROP TABLE IF EXISTS `slide`;
/*   What is the role of this table, did this subsume the tablee "sections"? *****/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `slide` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `scan_run_id` int(11) NOT NULL,
  `slide_physical_id` int(11) NOT NULL COMMENT 'one per slide',
  `rescan_number` enum('','1','2','3') NOT NULL DEFAULT '',
  `slide_status` enum('Bad','Good') NOT NULL DEFAULT 'Good',
  `scenes` int(11) DEFAULT NULL,
  `insert_before_one` tinyint(4) NOT NULL DEFAULT 0,
  `insert_between_one_two` tinyint(4) NOT NULL DEFAULT 0,
  `insert_between_two_three` tinyint(4) NOT NULL DEFAULT 0,
  `insert_between_three_four` tinyint(4) NOT NULL DEFAULT 0,
  `insert_between_four_five` tinyint(4) NOT NULL DEFAULT 0,
  `insert_between_five_six` tinyint(4) NOT NULL DEFAULT 0,
  `file_name` varchar(200) NOT NULL,
  `comments` varchar(2001) DEFAULT NULL COMMENT 'assessment',
  `active` tinyint(4) NOT NULL DEFAULT 1,
  `created` timestamp NULL DEFAULT current_timestamp(),
  `file_size` float NOT NULL DEFAULT 0,
  `processing_duration` float NOT NULL DEFAULT 0,
  `processed` tinyint(4) NOT NULL DEFAULT 0,
  `scene_qc_1` tinyint(4) NOT NULL DEFAULT 0,
  `scene_qc_2` tinyint(4) NOT NULL DEFAULT 0,
  `scene_qc_3` tinyint(4) NOT NULL DEFAULT 0,
  `scene_qc_4` tinyint(4) NOT NULL DEFAULT 0,
  `scene_qc_5` tinyint(4) NOT NULL DEFAULT 0,
  `scene_qc_6` tinyint(4) NOT NULL DEFAULT 0,
  PRIMARY KEY (`id`),
  KEY `K__slide_scan_run_id` (`scan_run_id`),
  CONSTRAINT `FK__slide_scan_run_id` FOREIGN KEY (`scan_run_id`) REFERENCES `scan_run` (`id`) ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=4799 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `slide_czi_to_tif`
--

DROP TABLE IF EXISTS `slide_czi_to_tif`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `slide_czi_to_tif` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `slide_id` int(11) NOT NULL,
  `file_name` varchar(200) NOT NULL,
  `scene_number` tinyint(4) NOT NULL,
  `channel` tinyint(4) NOT NULL,
  `width` int(11) NOT NULL DEFAULT 0,
  `height` int(11) NOT NULL DEFAULT 0,
  `comments` varchar(2000) DEFAULT NULL COMMENT 'assessment',
  `active` tinyint(4) NOT NULL DEFAULT 1,
  `created` timestamp NULL DEFAULT current_timestamp(),
  `file_size` float NOT NULL DEFAULT 0,
  `scene_index` int(11) NOT NULL DEFAULT 0,
  `processing_duration` float NOT NULL DEFAULT 0,
  PRIMARY KEY (`id`),
  KEY `K__slide_id` (`slide_id`),
  CONSTRAINT `FK__slide_id` FOREIGN KEY (`slide_id`) REFERENCES `slide` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=47004 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `scan_run`
--

DROP TABLE IF EXISTS `scan_run`;
/*   This table describes a sequence of slides scanned by the Zeiss Axio scanner  *****/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `scan_run` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `prep_id` varchar(200) NOT NULL,
  `performance_center` enum('CSHL','Salk','UCSD','HHMI') DEFAULT NULL COMMENT 'default population is from Histology',
  `machine` enum('Zeiss','Axioscan','Nanozoomer','Olympus VA') DEFAULT NULL,
  `objective` enum('60X','40X','20X','10X') DEFAULT NULL,
  `resolution` float NOT NULL DEFAULT 0 COMMENT '(µm) lateral resolution if available',
  `number_of_slides` int(11) NOT NULL DEFAULT 0,
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
  `width` int(11) NOT NULL DEFAULT 0,
  `height` int(11) NOT NULL DEFAULT 0,
  `rotation` int(11) NOT NULL DEFAULT 0,
  `flip` enum('none','flip','flop') NOT NULL DEFAULT 'none',
  `comments` varchar(2001) DEFAULT NULL COMMENT 'assessment',
  `active` tinyint(4) NOT NULL DEFAULT 1,
  `created` timestamp NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  KEY `FK__scan_run_prep_id` (`prep_id`),
  CONSTRAINT `FK__scan_run_prep_id` FOREIGN KEY (`prep_id`) REFERENCES `animal` (`prep_id`)
) ENGINE=InnoDB AUTO_INCREMENT=31 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `elastix_transformation`
-- Defines section to section alignment.

DROP TABLE IF EXISTS `elastix_transformation`;
/* What is this table *****/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `elastix_transformation` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `prep_id` varchar(20) NOT NULL,
  `section` char(3) NOT NULL,
  `rotation` float NOT NULL DEFAULT 0,
  `xshift` float NOT NULL DEFAULT 0,
  `yshift` float NOT NULL DEFAULT 0,
  `created` timestamp NULL DEFAULT current_timestamp(),
  `active` tinyint(4) NOT NULL DEFAULT 1,
  PRIMARY KEY (`id`),
  UNIQUE KEY `UK__ETR_prep_id_section` (`prep_id`,`section`),
  KEY `K__ETR_prepid` (`prep_id`),
  CONSTRAINT `FK__ETR_prepid` FOREIGN KEY (`prep_id`) REFERENCES `animal` (`prep_id`)
) ENGINE=InnoDB AUTO_INCREMENT=7396 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;


--
--  Neuroglancer related tables
--

-- Table structure for table `neuroglancer_urls`
--  /* Information in this table should be short lived,a */
-- new URL can be started from the web interface by specifying brain and
-- viewing configuration. Once the user hits "save" the json, 
-- information should be parsed and added to the appropriate
-- tables. com,to coms tables and configuration to configuration tables
-- (for storing the chosen histogram configuration, point of view, etc). */
--

DROP TABLE IF EXISTS `neuroglancer_urls`;
/* YF: This table is for storing information sent to and from neuroglancer. *****/
/* It should be constructed by the user using tables, and stored back in tables when the user hit "save" */
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `neuroglancer_urls` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `person_id` int(11) DEFAULT NULL,
  `url` longtext NOT NULL,
  `active` tinyint(1) DEFAULT NULL,
  `vetted` tinyint(1) DEFAULT NULL,
  `created` datetime(6) NOT NULL,
  `user_date` varchar(25) DEFAULT NULL,
  `comments` varchar(255) DEFAULT NULL,
  `updated` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  KEY `K__NU` (`person_id`),
  CONSTRAINT `FK__NU_user_id` FOREIGN KEY (`person_id`) REFERENCES `auth_user` (`id`) ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=350 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;


--
--  Engine tables - tables related to CVAT
--

-- Table structure for table `engine_attributespec`
--

DROP TABLE IF EXISTS `engine_attributespec`;
/* What is this table *****/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_attributespec` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(64) NOT NULL,
  `mutable` tinyint(1) NOT NULL,
  `input_type` varchar(16) NOT NULL,
  `default_value` varchar(128) NOT NULL,
  `values` varchar(4096) NOT NULL,
  `label_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `engine_attributespec_label_id_name_d85e616c_uniq` (`label_id`,`name`),
  CONSTRAINT `engine_attributespec_label_id_274838ef_fk_engine_label_id` FOREIGN KEY (`label_id`) REFERENCES `engine_label` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_clientfile`
--

DROP TABLE IF EXISTS `engine_clientfile`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_clientfile` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `file` varchar(1024) NOT NULL,
  `data_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=11024 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_data`
--

DROP TABLE IF EXISTS `engine_data`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_data` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `chunk_size` int(10) unsigned DEFAULT NULL,
  `size` int(10) unsigned NOT NULL,
  `image_quality` smallint(5) unsigned NOT NULL,
  `start_frame` int(10) unsigned NOT NULL,
  `stop_frame` int(10) unsigned NOT NULL,
  `frame_filter` varchar(256) NOT NULL,
  `compressed_chunk_type` varchar(32) NOT NULL,
  `original_chunk_type` varchar(32) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=38 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_image`
--

DROP TABLE IF EXISTS `engine_image`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_image` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `path` varchar(1024) NOT NULL,
  `frame` int(10) unsigned NOT NULL,
  `width` int(10) unsigned NOT NULL,
  `height` int(10) unsigned NOT NULL,
  `data_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `engine_image_data_id_e89da547_fk_engine_data_id` (`data_id`),
  CONSTRAINT `engine_image_data_id_e89da547_fk_engine_data_id` FOREIGN KEY (`data_id`) REFERENCES `engine_data` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=11024 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_job`
--

DROP TABLE IF EXISTS `engine_job`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_job` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `status` varchar(32) NOT NULL,
  `assignee_id` int(11) DEFAULT NULL,
  `segment_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `engine_job_segment_id_f615a866_fk_engine_segment_id` (`segment_id`),
  KEY `engine_job_assignee_id_b80bea03_fk_auth_user_id` (`assignee_id`),
  CONSTRAINT `engine_job_assignee_id_b80bea03_fk_auth_user_id` FOREIGN KEY (`assignee_id`) REFERENCES `auth_user` (`id`),
  CONSTRAINT `engine_job_segment_id_f615a866_fk_engine_segment_id` FOREIGN KEY (`segment_id`) REFERENCES `engine_segment` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=33 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_jobcommit`
--

DROP TABLE IF EXISTS `engine_jobcommit`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_jobcommit` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `version` int(10) unsigned NOT NULL,
  `timestamp` datetime(6) NOT NULL,
  `message` varchar(4096) NOT NULL,
  `author_id` int(11) DEFAULT NULL,
  `job_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `engine_jobcommit_author_id_fe2728f3_fk_auth_user_id` (`author_id`),
  KEY `engine_jobcommit_job_id_02b6da1d_fk_engine_job_id` (`job_id`),
  CONSTRAINT `engine_jobcommit_author_id_fe2728f3_fk_auth_user_id` FOREIGN KEY (`author_id`) REFERENCES `auth_user` (`id`),
  CONSTRAINT `engine_jobcommit_job_id_02b6da1d_fk_engine_job_id` FOREIGN KEY (`job_id`) REFERENCES `engine_job` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=11420 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_label`
--

DROP TABLE IF EXISTS `engine_label`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_label` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(64) NOT NULL,
  `task_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `engine_label_task_id_name_00e8779a_uniq` (`task_id`,`name`),
  CONSTRAINT `engine_label_task_id_f11c5c1a_fk_engine_task_id` FOREIGN KEY (`task_id`) REFERENCES `engine_task` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=976 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_labeledimage`
--

DROP TABLE IF EXISTS `engine_labeledimage`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_labeledimage` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `frame` int(10) unsigned NOT NULL,
  `group` int(10) unsigned DEFAULT NULL,
  `job_id` int(11) NOT NULL,
  `label_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `engine_labeledimage_job_id_7406d161_fk_engine_job_id` (`job_id`),
  KEY `engine_labeledimage_label_id_b22eb9f7_fk_engine_label_id` (`label_id`),
  CONSTRAINT `engine_labeledimage_job_id_7406d161_fk_engine_job_id` FOREIGN KEY (`job_id`) REFERENCES `engine_job` (`id`),
  CONSTRAINT `engine_labeledimage_label_id_b22eb9f7_fk_engine_label_id` FOREIGN KEY (`label_id`) REFERENCES `engine_label` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_labeledimageattributeval`
--

DROP TABLE IF EXISTS `engine_labeledimageattributeval`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_labeledimageattributeval` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `value` varchar(4096) NOT NULL,
  `image_id` bigint(20) NOT NULL,
  `spec_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `engine_labeledimagea_image_id_f4c34a7a_fk_engine_la` (`image_id`),
  KEY `engine_labeledimagea_spec_id_911f524c_fk_engine_at` (`spec_id`),
  CONSTRAINT `engine_labeledimagea_image_id_f4c34a7a_fk_engine_la` FOREIGN KEY (`image_id`) REFERENCES `engine_labeledimage` (`id`),
  CONSTRAINT `engine_labeledimagea_spec_id_911f524c_fk_engine_at` FOREIGN KEY (`spec_id`) REFERENCES `engine_attributespec` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_labeledshape`
--

DROP TABLE IF EXISTS `engine_labeledshape`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_labeledshape` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `frame` int(10) unsigned NOT NULL,
  `group` int(10) unsigned DEFAULT NULL,
  `type` varchar(16) NOT NULL,
  `occluded` tinyint(1) NOT NULL,
  `z_order` int(11) NOT NULL,
  `points` longtext NOT NULL,
  `job_id` int(11) NOT NULL,
  `label_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `engine_labeledshape_job_id_b7694c3a_fk_engine_job_id` (`job_id`),
  KEY `engine_labeledshape_label_id_872e4658_fk_engine_label_id` (`label_id`),
  CONSTRAINT `engine_labeledshape_job_id_b7694c3a_fk_engine_job_id` FOREIGN KEY (`job_id`) REFERENCES `engine_job` (`id`),
  CONSTRAINT `engine_labeledshape_label_id_872e4658_fk_engine_label_id` FOREIGN KEY (`label_id`) REFERENCES `engine_label` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=66892 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_labeledshapeattributeval`
--

DROP TABLE IF EXISTS `engine_labeledshapeattributeval`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_labeledshapeattributeval` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `value` varchar(4096) NOT NULL,
  `shape_id` bigint(20) NOT NULL,
  `spec_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `engine_labeledshapea_shape_id_26c4daab_fk_engine_la` (`shape_id`),
  KEY `engine_labeledshapea_spec_id_144b73fa_fk_engine_at` (`spec_id`),
  CONSTRAINT `engine_labeledshapea_shape_id_26c4daab_fk_engine_la` FOREIGN KEY (`shape_id`) REFERENCES `engine_labeledshape` (`id`),
  CONSTRAINT `engine_labeledshapea_spec_id_144b73fa_fk_engine_at` FOREIGN KEY (`spec_id`) REFERENCES `engine_attributespec` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_labeledtrack`
--

DROP TABLE IF EXISTS `engine_labeledtrack`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_labeledtrack` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `frame` int(10) unsigned NOT NULL,
  `group` int(10) unsigned DEFAULT NULL,
  `job_id` int(11) NOT NULL,
  `label_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `engine_labeledtrack_job_id_e00d9f2f_fk_engine_job_id` (`job_id`),
  KEY `engine_labeledtrack_label_id_75d2c39b_fk_engine_label_id` (`label_id`),
  CONSTRAINT `engine_labeledtrack_job_id_e00d9f2f_fk_engine_job_id` FOREIGN KEY (`job_id`) REFERENCES `engine_job` (`id`),
  CONSTRAINT `engine_labeledtrack_label_id_75d2c39b_fk_engine_label_id` FOREIGN KEY (`label_id`) REFERENCES `engine_label` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=44 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_labeledtrackattributeval`
--

DROP TABLE IF EXISTS `engine_labeledtrackattributeval`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_labeledtrackattributeval` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `value` varchar(4096) NOT NULL,
  `spec_id` int(11) NOT NULL,
  `track_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `engine_labeledtracka_spec_id_b7ee6fd2_fk_engine_at` (`spec_id`),
  KEY `engine_labeledtracka_track_id_4ed9e160_fk_engine_la` (`track_id`),
  CONSTRAINT `engine_labeledtracka_spec_id_b7ee6fd2_fk_engine_at` FOREIGN KEY (`spec_id`) REFERENCES `engine_attributespec` (`id`),
  CONSTRAINT `engine_labeledtracka_track_id_4ed9e160_fk_engine_la` FOREIGN KEY (`track_id`) REFERENCES `engine_labeledtrack` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_plugin`
--

DROP TABLE IF EXISTS `engine_plugin`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_plugin` (
  `name` varchar(32) NOT NULL,
  `description` varchar(8192) NOT NULL,
  `created_at` datetime(6) NOT NULL,
  `updated_at` datetime(6) NOT NULL,
  `maintainer_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_pluginoption`
--

DROP TABLE IF EXISTS `engine_pluginoption`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_pluginoption` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(32) NOT NULL,
  `value` varchar(1024) NOT NULL,
  `plugin_id` varchar(32) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_project`
--

DROP TABLE IF EXISTS `engine_project`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_project` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(256) NOT NULL,
  `bug_tracker` varchar(2000) NOT NULL,
  `created_date` datetime(6) NOT NULL,
  `updated_date` datetime(6) NOT NULL,
  `status` varchar(32) NOT NULL,
  `assignee_id` int(11) DEFAULT NULL,
  `owner_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `engine_project_assignee_id_77655de8_fk_auth_user_id` (`assignee_id`),
  KEY `engine_project_owner_id_de2a8424_fk_auth_user_id` (`owner_id`),
  CONSTRAINT `engine_project_assignee_id_77655de8_fk_auth_user_id` FOREIGN KEY (`assignee_id`) REFERENCES `auth_user` (`id`),
  CONSTRAINT `engine_project_owner_id_de2a8424_fk_auth_user_id` FOREIGN KEY (`owner_id`) REFERENCES `auth_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_remotefile`
--

DROP TABLE IF EXISTS `engine_remotefile`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_remotefile` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `file` varchar(1024) NOT NULL,
  `data_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `engine_remotefile_data_id_ff16acda_fk_engine_data_id` (`data_id`),
  CONSTRAINT `engine_remotefile_data_id_ff16acda_fk_engine_data_id` FOREIGN KEY (`data_id`) REFERENCES `engine_data` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_segment`
--

DROP TABLE IF EXISTS `engine_segment`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_segment` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `start_frame` int(11) NOT NULL,
  `stop_frame` int(11) NOT NULL,
  `task_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `engine_segment_task_id_37d935cf_fk_engine_task_id` (`task_id`),
  CONSTRAINT `engine_segment_task_id_37d935cf_fk_engine_task_id` FOREIGN KEY (`task_id`) REFERENCES `engine_task` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=33 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_serverfile`
--

DROP TABLE IF EXISTS `engine_serverfile`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_serverfile` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `file` varchar(1024) NOT NULL,
  `data_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `engine_serverfile_data_id_2364110a_fk_engine_data_id` (`data_id`),
  CONSTRAINT `engine_serverfile_data_id_2364110a_fk_engine_data_id` FOREIGN KEY (`data_id`) REFERENCES `engine_data` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_task`
--

DROP TABLE IF EXISTS `engine_task`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_task` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(256) NOT NULL,
  `mode` varchar(32) NOT NULL,
  `bug_tracker` varchar(2000) NOT NULL,
  `created_date` datetime(6) NOT NULL,
  `updated_date` datetime(6) NOT NULL,
  `overlap` int(10) unsigned DEFAULT NULL,
  `segment_size` int(10) unsigned NOT NULL,
  `z_order` tinyint(1) NOT NULL,
  `status` varchar(32) NOT NULL,
  `assignee_id` int(11) DEFAULT NULL,
  `data_id` int(11) DEFAULT NULL,
  `owner_id` int(11) DEFAULT NULL,
  `project_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `engine_task_assignee_id_51c82720_fk_auth_user_id` (`assignee_id`),
  KEY `engine_task_data_id_e98ffd9b_fk_engine_data_id` (`data_id`),
  KEY `engine_task_owner_id_95de3361_fk_auth_user_id` (`owner_id`),
  KEY `engine_task_project_id_2dced848_fk_engine_project_id` (`project_id`),
  CONSTRAINT `engine_task_assignee_id_51c82720_fk_auth_user_id` FOREIGN KEY (`assignee_id`) REFERENCES `auth_user` (`id`),
  CONSTRAINT `engine_task_data_id_e98ffd9b_fk_engine_data_id` FOREIGN KEY (`data_id`) REFERENCES `engine_data` (`id`),
  CONSTRAINT `engine_task_owner_id_95de3361_fk_auth_user_id` FOREIGN KEY (`owner_id`) REFERENCES `auth_user` (`id`),
  CONSTRAINT `engine_task_project_id_2dced848_fk_engine_project_id` FOREIGN KEY (`project_id`) REFERENCES `engine_project` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=43 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_trackedshape`
--

DROP TABLE IF EXISTS `engine_trackedshape`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_trackedshape` (
  `type` varchar(16) NOT NULL,
  `occluded` tinyint(1) NOT NULL,
  `z_order` int(11) NOT NULL,
  `points` longtext NOT NULL,
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `frame` int(10) unsigned NOT NULL,
  `outside` tinyint(1) NOT NULL,
  `track_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `engine_trackedshape_track_id_a6dc58bd_fk_engine_labeledtrack_id` (`track_id`),
  CONSTRAINT `engine_trackedshape_track_id_a6dc58bd_fk_engine_labeledtrack_id` FOREIGN KEY (`track_id`) REFERENCES `engine_labeledtrack` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2258 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_trackedshapeattributeval`
--

DROP TABLE IF EXISTS `engine_trackedshapeattributeval`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_trackedshapeattributeval` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `value` varchar(4096) NOT NULL,
  `shape_id` bigint(20) NOT NULL,
  `spec_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `engine_trackedshapea_shape_id_361f0e2f_fk_engine_tr` (`shape_id`),
  KEY `engine_trackedshapea_spec_id_a944a532_fk_engine_at` (`spec_id`),
  CONSTRAINT `engine_trackedshapea_shape_id_361f0e2f_fk_engine_tr` FOREIGN KEY (`shape_id`) REFERENCES `engine_trackedshape` (`id`),
  CONSTRAINT `engine_trackedshapea_spec_id_a944a532_fk_engine_at` FOREIGN KEY (`spec_id`) REFERENCES `engine_attributespec` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `engine_video`
--

DROP TABLE IF EXISTS `engine_video`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `engine_video` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `path` varchar(1024) NOT NULL,
  `width` int(10) unsigned NOT NULL,
  `height` int(10) unsigned NOT NULL,
  `data_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `data_id` (`data_id`),
  CONSTRAINT `engine_video_data_id_b37015e9_fk_engine_data_id` FOREIGN KEY (`data_id`) REFERENCES `engine_data` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `git_gitdata`
--

DROP TABLE IF EXISTS `git_gitdata`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `git_gitdata` (
  `task_id` int(11) NOT NULL,
  `url` varchar(2000) NOT NULL,
  `path` varchar(256) NOT NULL,
  `sync_date` datetime(6) NOT NULL,
  `status` varchar(20) NOT NULL,
  `lfs` tinyint(1) NOT NULL,
  PRIMARY KEY (`task_id`),
  CONSTRAINT `git_gitdata_task_id_a6f2ea20_fk_engine_task_id` FOREIGN KEY (`task_id`) REFERENCES `engine_task` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;


--
--   Administrative tables ---------------------------------------------------------------------
-- 
-- 
--  Django administration tables that are used by the portal, used for user information and Django 
--
-- Table structure for table `django_admin_log`
--

DROP TABLE IF EXISTS `django_admin_log`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `django_admin_log` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `action_time` datetime(6) NOT NULL,
  `object_id` longtext DEFAULT NULL,
  `object_repr` varchar(200) NOT NULL,
  `action_flag` smallint(5) unsigned NOT NULL,
  `change_message` longtext NOT NULL,
  `content_type_id` int(11) DEFAULT NULL,
  `user_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `django_admin_log_content_type_id_c4bce8eb_fk_django_co` (`content_type_id`),
  KEY `django_admin_log_user_id_c564eba6_fk_auth_user_id` (`user_id`),
  CONSTRAINT `django_admin_log_content_type_id_c4bce8eb_fk_django_co` FOREIGN KEY (`content_type_id`) REFERENCES `django_content_type` (`id`),
  CONSTRAINT `django_admin_log_user_id_c564eba6_fk_auth_user_id` FOREIGN KEY (`user_id`) REFERENCES `auth_user` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3588 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `django_content_type`
--

DROP TABLE IF EXISTS `django_content_type`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `django_content_type` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `app_label` varchar(100) NOT NULL,
  `model` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `django_content_type_app_label_model_76bd3d3b_uniq` (`app_label`,`model`)
) ENGINE=InnoDB AUTO_INCREMENT=86 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `django_migrations`
--

DROP TABLE IF EXISTS `django_migrations`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `django_migrations` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `app` varchar(255) NOT NULL,
  `name` varchar(255) NOT NULL,
  `applied` datetime(6) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=29 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `django_plotly_dash_dashapp`
--

DROP TABLE IF EXISTS `django_plotly_dash_dashapp`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `django_plotly_dash_dashapp` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `instance_name` varchar(100) NOT NULL,
  `slug` varchar(110) NOT NULL,
  `base_state` longtext NOT NULL,
  `creation` datetime(6) NOT NULL,
  `update` datetime(6) NOT NULL,
  `save_on_change` tinyint(1) NOT NULL,
  `stateless_app_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `instance_name` (`instance_name`),
  UNIQUE KEY `slug` (`slug`),
  KEY `django_plotly_dash_d_stateless_app_id_220444de_fk_django_pl` (`stateless_app_id`),
  CONSTRAINT `django_plotly_dash_d_stateless_app_id_220444de_fk_django_pl` FOREIGN KEY (`stateless_app_id`) REFERENCES `django_plotly_dash_statelessapp` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `django_plotly_dash_statelessapp`
--

DROP TABLE IF EXISTS `django_plotly_dash_statelessapp`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `django_plotly_dash_statelessapp` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `app_name` varchar(100) NOT NULL,
  `slug` varchar(110) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `app_name` (`app_name`),
  UNIQUE KEY `slug` (`slug`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `django_session`
--

DROP TABLE IF EXISTS `django_session`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `django_session` (
  `session_key` varchar(40) NOT NULL,
  `session_data` longtext NOT NULL,
  `expire_date` datetime(6) NOT NULL,
  PRIMARY KEY (`session_key`),
  KEY `django_session_expire_date_a5c62663` (`expire_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `django_site`
--

DROP TABLE IF EXISTS `django_site`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `django_site` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `domain` varchar(100) NOT NULL,
  `name` varchar(50) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `django_site_domain_a2e37b91_uniq` (`domain`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `auth_group`
--

DROP TABLE IF EXISTS `auth_group`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `auth_group` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(150) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `auth_group_permissions`
--

DROP TABLE IF EXISTS `auth_group_permissions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `auth_group_permissions` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `group_id` int(11) NOT NULL,
  `permission_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_group_permissions_group_id_permission_id_0cd325b0_uniq` (`group_id`,`permission_id`),
  KEY `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm` (`permission_id`),
  CONSTRAINT `FK__AGP_group_id` FOREIGN KEY (`group_id`) REFERENCES `auth_group` (`id`) ON UPDATE CASCADE,
  CONSTRAINT `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm` FOREIGN KEY (`permission_id`) REFERENCES `auth_permission` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `auth_permission`
--

DROP TABLE IF EXISTS `auth_permission`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `auth_permission` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `content_type_id` int(11) NOT NULL,
  `codename` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_permission_content_type_id_codename_01ab375a_uniq` (`content_type_id`,`codename`),
  CONSTRAINT `auth_permission_content_type_id_2f476e4b_fk_django_co` FOREIGN KEY (`content_type_id`) REFERENCES `django_content_type` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=149 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `auth_user`
--

DROP TABLE IF EXISTS `auth_user`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `auth_user` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `password` varchar(128) NOT NULL,
  `last_login` datetime(6) DEFAULT NULL,
  `is_superuser` tinyint(1) NOT NULL,
  `username` varchar(150) NOT NULL,
  `first_name` varchar(150) NOT NULL,
  `last_name` varchar(150) NOT NULL,
  `email` varchar(254) NOT NULL,
  `is_staff` tinyint(1) NOT NULL,
  `is_active` tinyint(1) NOT NULL,
  `date_joined` datetime(6) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`)
) ENGINE=InnoDB AUTO_INCREMENT=36 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `auth_user_groups`
--

DROP TABLE IF EXISTS `auth_user_groups`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `auth_user_groups` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `group_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_user_groups_user_id_group_id_94350c0c_uniq` (`user_id`,`group_id`),
  KEY `auth_user_groups_group_id_97559544_fk_auth_group_id` (`group_id`),
  CONSTRAINT `FK__AUG_group_id` FOREIGN KEY (`group_id`) REFERENCES `auth_group` (`id`) ON UPDATE CASCADE,
  CONSTRAINT `auth_user_groups_user_id_6a12ed8b_fk_auth_user_id` FOREIGN KEY (`user_id`) REFERENCES `auth_user` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=49 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `auth_user_user_permissions`
--

DROP TABLE IF EXISTS `auth_user_user_permissions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `auth_user_user_permissions` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `permission_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_user_user_permissions_user_id_permission_id_14a6b632_uniq` (`user_id`,`permission_id`),
  KEY `auth_user_user_permi_permission_id_1fbb5f2c_fk_auth_perm` (`permission_id`),
  CONSTRAINT `auth_user_user_permi_permission_id_1fbb5f2c_fk_auth_perm` FOREIGN KEY (`permission_id`) REFERENCES `auth_permission` (`id`),
  CONSTRAINT `auth_user_user_permissions_user_id_a95ead1b_fk_auth_user_id` FOREIGN KEY (`user_id`) REFERENCES `auth_user` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=125 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `authtoken_token`
--

DROP TABLE IF EXISTS `authtoken_token`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `authtoken_token` (
  `key` varchar(40) NOT NULL,
  `created` datetime(6) NOT NULL,
  `user_id` int(11) NOT NULL,
  PRIMARY KEY (`key`),
  UNIQUE KEY `user_id` (`user_id`),
  CONSTRAINT `authtoken_token_user_id_35299eff_fk_auth_user_id` FOREIGN KEY (`user_id`) REFERENCES `auth_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;


--
--  Social Account tables, oauth tables
--

-- Table structure for table `socialaccount_socialaccount`
--

DROP TABLE IF EXISTS `socialaccount_socialaccount`;
/*   What is the role of this table *****/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `socialaccount_socialaccount` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `provider` varchar(30) NOT NULL,
  `uid` varchar(191) NOT NULL,
  `last_login` datetime(6) NOT NULL,
  `date_joined` datetime(6) NOT NULL,
  `extra_data` longtext NOT NULL,
  `user_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `socialaccount_socialaccount_provider_uid_fc810c6e_uniq` (`provider`,`uid`),
  KEY `socialaccount_socialaccount_user_id_8146e70c_fk_auth_user_id` (`user_id`),
  CONSTRAINT `socialaccount_socialaccount_user_id_8146e70c_fk_auth_user_id` FOREIGN KEY (`user_id`) REFERENCES `auth_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `socialaccount_socialapp`
--

DROP TABLE IF EXISTS `socialaccount_socialapp`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `socialaccount_socialapp` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `provider` varchar(30) NOT NULL,
  `name` varchar(40) NOT NULL,
  `client_id` varchar(191) NOT NULL,
  `secret` varchar(191) NOT NULL,
  `key` varchar(191) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `socialaccount_socialapp_sites`
--

DROP TABLE IF EXISTS `socialaccount_socialapp_sites`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `socialaccount_socialapp_sites` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `socialapp_id` int(11) NOT NULL,
  `site_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `socialaccount_socialapp_sites_socialapp_id_site_id_71a9a768_uniq` (`socialapp_id`,`site_id`),
  KEY `socialaccount_socialapp_sites_site_id_2579dee5_fk_django_site_id` (`site_id`),
  CONSTRAINT `socialaccount_social_socialapp_id_97fb6e7d_fk_socialacc` FOREIGN KEY (`socialapp_id`) REFERENCES `socialaccount_socialapp` (`id`),
  CONSTRAINT `socialaccount_socialapp_sites_site_id_2579dee5_fk_django_site_id` FOREIGN KEY (`site_id`) REFERENCES `django_site` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `socialaccount_socialtoken`
--

DROP TABLE IF EXISTS `socialaccount_socialtoken`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `socialaccount_socialtoken` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `token` longtext NOT NULL,
  `token_secret` longtext NOT NULL,
  `expires_at` datetime(6) DEFAULT NULL,
  `account_id` int(11) NOT NULL,
  `app_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `socialaccount_socialtoken_app_id_account_id_fca4e0ac_uniq` (`app_id`,`account_id`),
  KEY `socialaccount_social_account_id_951f210e_fk_socialacc` (`account_id`),
  CONSTRAINT `socialaccount_social_account_id_951f210e_fk_socialacc` FOREIGN KEY (`account_id`) REFERENCES `socialaccount_socialaccount` (`id`),
  CONSTRAINT `socialaccount_social_app_id_636a42d7_fk_socialacc` FOREIGN KEY (`app_id`) REFERENCES `socialaccount_socialapp` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;


--
--  Schedule Location related tables
--
-- Table structure for table `schedule`
--

DROP TABLE IF EXISTS `schedule`;
/*   What is the role of this table *****/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `schedule` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `active` tinyint(1) NOT NULL,
  `created` datetime(6) NOT NULL,
  `updated` datetime(6) NOT NULL,
  `start_time` datetime(6) NOT NULL,
  `end_time` datetime(6) NOT NULL,
  `location_id` int(11) NOT NULL,
  `person_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `schedule_location_id_d296afa1_fk_location_id` (`location_id`),
  KEY `schedule_person_id_9f59b05d_fk_auth_user_id` (`person_id`),
  CONSTRAINT `schedule_location_id_d296afa1_fk_location_id` FOREIGN KEY (`location_id`) REFERENCES `location` (`id`),
  CONSTRAINT `schedule_person_id_9f59b05d_fk_auth_user_id` FOREIGN KEY (`person_id`) REFERENCES `auth_user` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1273 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Temporary table structure for view `sections`
--

DROP TABLE IF EXISTS `sections`;
/*   What is the role of this table, what happened to the conversion of slides to sections?*****/
/*!50001 DROP VIEW IF EXISTS `sections`*/;
SET @saved_cs_client     = @@character_set_client;
SET character_set_client = utf8;
/*!50001 CREATE TABLE `sections` (
  `id` tinyint NOT NULL,
  `prep_id` tinyint NOT NULL,
  `czi_file` tinyint NOT NULL,
  `slide_physical_id` tinyint NOT NULL,
  `file_name` tinyint NOT NULL,
  `tif_id` tinyint NOT NULL,
  `scene_number` tinyint NOT NULL,
  `scene_index` tinyint NOT NULL,
  `channel` tinyint NOT NULL,
  `channel_index` tinyint NOT NULL,
  `active` tinyint NOT NULL,
  `created` tinyint NOT NULL
) ENGINE=MyISAM */;
SET character_set_client = @saved_cs_client;

-- Table structure for table `location`
--

DROP TABLE IF EXISTS `location`;
/*   What is the role of this table  (Scanner related ?*) *****/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `location` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `active` tinyint(1) NOT NULL,
  `created` datetime(6) NOT NULL,
  `updated` datetime(6) NOT NULL,
  `room` varchar(25) NOT NULL,
  `description` longtext NOT NULL,
  `people_allowed` int(11) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=33 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `location_primary_people`
--

DROP TABLE IF EXISTS `location_primary_people`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `location_primary_people` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `location_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `location_primary_people_location_id_user_id_58be910f_uniq` (`location_id`,`user_id`),
  KEY `location_primary_people_user_id_4125b3f6_fk_auth_user_id` (`user_id`),
  CONSTRAINT `location_primary_people_location_id_bb62bcf7_fk_location_id` FOREIGN KEY (`location_id`) REFERENCES `location` (`id`),
  CONSTRAINT `location_primary_people_user_id_4125b3f6_fk_auth_user_id` FOREIGN KEY (`user_id`) REFERENCES `auth_user` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=61 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;


--
--  Tables related to logging throughout the workflow --------------------------------------
--

-- Table structure for table `file_log`
--

DROP TABLE IF EXISTS `file_log`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `file_log` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `prep_id` varchar(20) NOT NULL,
  `progress_id` int(11) NOT NULL,
  `filename` varchar(255) NOT NULL,
  `active` tinyint(1) NOT NULL DEFAULT 1,
  `created` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  UNIQUE KEY `UK__AID_PID_C_S` (`prep_id`,`progress_id`,`filename`),
  KEY `K__FILE_LOG_AID` (`prep_id`),
  KEY `K__FILE_LOG_PID` (`progress_id`),
  CONSTRAINT `FK__FILE_LOG_AID` FOREIGN KEY (`prep_id`) REFERENCES `animal` (`prep_id`) ON UPDATE CASCADE,
  CONSTRAINT `FK__FILE_LOG_PID` FOREIGN KEY (`progress_id`) REFERENCES `progress_lookup` (`id`) ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=32992 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `file_operation`
--

DROP TABLE IF EXISTS `file_operation`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `file_operation` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `tif_id` int(11) NOT NULL,
  `operation` varchar(200) NOT NULL,
  `created` timestamp NULL DEFAULT current_timestamp(),
  `file_size` float NOT NULL DEFAULT 0,
  `active` tinyint(4) NOT NULL DEFAULT 1,
  PRIMARY KEY (`id`),
  KEY `K__tif_id` (`tif_id`),
  CONSTRAINT `FK__FP_tif_id` FOREIGN KEY (`tif_id`) REFERENCES `slide_czi_to_tif` (`id`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `logs`
--

DROP TABLE IF EXISTS `logs`;
/*   What is the role of this table *****/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `logs` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `logger` varchar(100) NOT NULL,
  `level` varchar(25) NOT NULL,
  `prep_id` varchar(20) NOT NULL,
  `msg` varchar(255) NOT NULL,
  `created` timestamp NOT NULL DEFAULT current_timestamp(),
  `active` tinyint(4) NOT NULL DEFAULT 1,
  PRIMARY KEY (`id`),
  KEY `K__logs_prep_id` (`prep_id`),
  CONSTRAINT `FK__logs_prep_id` FOREIGN KEY (`prep_id`) REFERENCES `animal` (`prep_id`)
) ENGINE=InnoDB AUTO_INCREMENT=24187 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;


--
-- Tables related to slide QC
--
-- Table structure for table `journals`
--

DROP TABLE IF EXISTS `journals`;
/* Ask Beth what this is ******/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `journals` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `prep_id` varchar(20) NOT NULL,
  `entry` longtext NOT NULL,
  `fix` longtext DEFAULT NULL,
  `image` varchar(255) DEFAULT NULL,
  `issue_link` varchar(255) DEFAULT NULL,
  `created` timestamp NOT NULL DEFAULT current_timestamp(),
  `active` tinyint(4) NOT NULL DEFAULT 1,
  `completed` tinyint(4) NOT NULL DEFAULT 0,
  `person_id` int(11) NOT NULL,
  `problem_id` int(11) NOT NULL,
  `url_id` int(11) DEFAULT NULL,
  `section` int(11) DEFAULT NULL,
  `channel` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `K__journals_prep_id` (`prep_id`),
  KEY `K__journals_person_id` (`person_id`),
  KEY `K__journals_problem_id` (`problem_id`),
  KEY `FK__url_id` (`url_id`),
  CONSTRAINT `FK__journals_person_id` FOREIGN KEY (`person_id`) REFERENCES `auth_user` (`id`) ON UPDATE CASCADE,
  CONSTRAINT `FK__journals_prep_id` FOREIGN KEY (`prep_id`) REFERENCES `animal` (`prep_id`),
  CONSTRAINT `FK__journals_problem_id` FOREIGN KEY (`problem_id`) REFERENCES `problem_category` (`id`),
  CONSTRAINT `FK__url_id` FOREIGN KEY (`url_id`) REFERENCES `neuroglancer_urls` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=140 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `problem_category`
--

DROP TABLE IF EXISTS `problem_category`;
/* How does this relate to the other log tables and how are these log tables used? *********/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `problem_category` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `problem_category` varchar(255) NOT NULL,
  `active` tinyint(1) NOT NULL DEFAULT 1,
  `created` datetime(6) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;


--
-- Tables related to the preprocessing workflow
-- 
-- Table structure for table `progress_lookup`
--

DROP TABLE IF EXISTS `progress_lookup`;
/*   What is the role of this table *****/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `progress_lookup` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `description` text DEFAULT NULL,
  `script` varchar(200) DEFAULT NULL,
  `channel` int(11) NOT NULL,
  `action` varchar(25) NOT NULL,
  `downsample` tinyint(4) NOT NULL DEFAULT 1,
  `active` tinyint(4) NOT NULL DEFAULT 1,
  `created` timestamp NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=257 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `resource`
--

DROP TABLE IF EXISTS `resource`;
/*   What is the role of this table *****/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `resource` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `first_name` varchar(30) NOT NULL,
  `last_name` varchar(30) NOT NULL,
  `role_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `K__RESOURCE_role_id` (`role_id`),
  CONSTRAINT `FK__RESOURCE_role_id` FOREIGN KEY (`role_id`) REFERENCES `task_roles` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `schedule`
--

DROP TABLE IF EXISTS `schedule`;
/*   What is the role of this table *****/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `schedule` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `active` tinyint(1) NOT NULL,
  `created` datetime(6) NOT NULL,
  `updated` datetime(6) NOT NULL,
  `start_time` datetime(6) NOT NULL,
  `end_time` datetime(6) NOT NULL,
  `location_id` int(11) NOT NULL,
  `person_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `schedule_location_id_d296afa1_fk_location_id` (`location_id`),
  KEY `schedule_person_id_9f59b05d_fk_auth_user_id` (`person_id`),
  CONSTRAINT `schedule_location_id_d296afa1_fk_location_id` FOREIGN KEY (`location_id`) REFERENCES `location` (`id`),
  CONSTRAINT `schedule_person_id_9f59b05d_fk_auth_user_id` FOREIGN KEY (`person_id`) REFERENCES `auth_user` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1273 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Temporary table structure for view `sections`
--

DROP TABLE IF EXISTS `sections`;
/*   What is the role of this table, what happened to the conversion of slides to sections?*****/
/*!50001 DROP VIEW IF EXISTS `sections`*/;
SET @saved_cs_client     = @@character_set_client;
SET character_set_client = utf8;
/*!50001 CREATE TABLE `sections` (
  `id` tinyint NOT NULL,
  `prep_id` tinyint NOT NULL,
  `czi_file` tinyint NOT NULL,
  `slide_physical_id` tinyint NOT NULL,
  `file_name` tinyint NOT NULL,
  `tif_id` tinyint NOT NULL,
  `scene_number` tinyint NOT NULL,
  `scene_index` tinyint NOT NULL,
  `channel` tinyint NOT NULL,
  `channel_index` tinyint NOT NULL,
  `active` tinyint NOT NULL,
  `created` tinyint NOT NULL
) ENGINE=MyISAM */;
SET character_set_client = @saved_cs_client;


--
-- Task tables
--

-- Table structure for table `task`
--

DROP TABLE IF EXISTS `task`;
/*   What is the role of this table What are tasks and how are they used? *****/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `task` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `lookup_id` int(11) NOT NULL,
  `prep_id` varchar(20) NOT NULL,
  `completed` tinyint(4) NOT NULL DEFAULT 0,
  `start_date` datetime DEFAULT NULL,
  `end_date` datetime DEFAULT NULL,
  `active` tinyint(4) NOT NULL DEFAULT 1,
  `created` timestamp NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  UNIQUE KEY `UK__progress_data_prep_lookup` (`prep_id`,`lookup_id`),
  KEY `K__task_prep_id` (`prep_id`),
  KEY `K__task_data_lookup_id` (`lookup_id`),
  CONSTRAINT `FK__task_lookup_id` FOREIGN KEY (`lookup_id`) REFERENCES `progress_lookup` (`id`),
  CONSTRAINT `FK__task_prep_id` FOREIGN KEY (`prep_id`) REFERENCES `animal` (`prep_id`)
) ENGINE=InnoDB AUTO_INCREMENT=482 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `task_resources`
--

DROP TABLE IF EXISTS `task_resources`;
/*   What is the role of this table *****/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `task_resources` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `task_id` int(11) NOT NULL,
  `resource_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `UK__TR_task_id` (`task_id`,`resource_id`),
  KEY `K__TR_resource_id` (`resource_id`),
  CONSTRAINT `FK__TR_resource_id` FOREIGN KEY (`resource_id`) REFERENCES `resource` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `FK__TR_task_id` FOREIGN KEY (`task_id`) REFERENCES `task` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `task_roles`
--

DROP TABLE IF EXISTS `task_roles`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `task_roles` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(30) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

-- Table structure for table `task_view`
--

DROP TABLE IF EXISTS `task_view`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `task_view` (
  `prep_id` tinyint(4) NOT NULL,
  `percent_complete` tinyint(4) NOT NULL,
  `complete` tinyint(4) NOT NULL,
  `created` tinyint(4) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;



--
-- Final view structure for view `sections`
--

/*!50001 DROP TABLE IF EXISTS `sections`*/;
/*!50001 DROP VIEW IF EXISTS `sections`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_general_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`eddyod`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `sections` AS select `sc`.`id` AS `id`,`a`.`prep_id` AS `prep_id`,`s`.`file_name` AS `czi_file`,`s`.`slide_physical_id` AS `slide_physical_id`,`sc`.`file_name` AS `file_name`,`sc`.`id` AS `tif_id`,`sc`.`scene_number` AS `scene_number`,`sc`.`scene_index` AS `scene_index`,`sc`.`channel` AS `channel`,`sc`.`channel` - 1 AS `channel_index`,`sc`.`active` AS `active`,`sc`.`created` AS `created` from (((`animal` `a` join `scan_run` `sr` on(`a`.`prep_id` = `sr`.`prep_id`)) join `slide` `s` on(`sr`.`id` = `s`.`scan_run_id`)) join `slide_czi_to_tif` `sc` on(`s`.`id` = `sc`.`slide_id`)) where `s`.`slide_status` = 'Good' */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2021-10-21  7:47:30
