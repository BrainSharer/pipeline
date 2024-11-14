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
/*
 * 16-DEC-2021 SUMMARY (DUANE):
 * MODS FOR animal TABLE:
 * ADDED animal_id AS PRIMARY KEY
 * REMOVE prep_id AS PRIMARY KEY
 * [FIELD AND CONSTRAINT STILL EXIST FOR LEGACY COMPATIBILITY]
 *
 * REMOVE aliases_1, aliases_2, aliases_3, aliases_4, aliases_5 FIELDS
 * [NOTE: WILL NEED TO MIGRATE EXISTING DATA TO NEW alias TABLE]
 *
 * REMOVED vender FIELD
 * ADDED FK_vendorid FIELD
 * [CORRECTED SPELLING AND IS FOREIGN KEY TO vendors TABLE]
 *
 * REMOVED tissue_source FIELD
 * ADDED FK_tissue_source_id FIELD
 *
 * REMOVED performance_center FIELD
 * ADDED FK_performance_center_id FIELD
 * ---------------------------------------
 * NEW TABLES ADDED AT END OF DOCUMENT (EXCEPT FOR annotations_points, annotations_points_archive, archive_sets):
 * ADDED vendor TABLE WITH DEFAULT VALUES (LEGACY)
 * ADDED tissue_source TABLE WITH DEFAULT VALUES (LEGACY)
 * ADDED performance_center WITH DEFAULT VALUES (LEGACY)
 *   - OUTSTANDING TABLE CHANGES: scan_run, injection, histology
 * ADDED input_type TABLE (REPLACES com_type)
 * ADDED alias TABLE
 * ADDED animal_alias TABLE
 * ADDED annotations_points TABLE
 * ADDED annotations_points_archive TABLE
 * ADDED archive_sets TABLE
 * ---------------------------------------
 * RENAMED layer_data TABLE TO annotations_points
 
 * 22-DEC-2021 SUMMARY (Edward):
 * MODS FOR animal TABLE:
 * REMOVED date_of_birth FIELD
 * REMOVED species FIELD
 * REMOVED strain FIELD
 * REMOVED sex FIELD
 * REMOVED genotype FIELD
 * REMOVED breeder_line FIELD
 * REMOVED stock_number FIELD
 * REMOVED ship_date FIELD
 * REMOVED shipper FIELD
 * REMOVED tracking_number FIELD 
 * REMOVED FK_vendor_id KEY
 * REMOVED FK_tissue_source_id KEY
 * REMOVED FK_performance_center_id KEY
 * REMOVED FK_alias_id KEY
 * RENAMED animal_id to id KEY
 * MODS FOR animal transformation:
 * DROP TABLE transformation
 * MODS FOR animal histology:
 * DROP TABLE histology
 * MODS FOR TABLE slide:
 * DROP TABLE slide
 * MODS FOR TABLE slide_czi_to_tif:
 * DROP TABLE slide_czi_to_tif
 * MODS FOR TABLE organic_label:
 * DROP TABLE organic_label
 * MODS FOR VIEW sections:
 * DROP VIEW slide_czi_to_tif
 * MODS FOR TABLE elastix_transformation:
 * DROP TABLE elastix_transformation
 * MODS FOR TABLE neuroglancer_urls:
 * RENAMED table neuroglancer_urls to neuroglancer_state
 * RENAMED column url to neuroglancer_state
 * REMOVED all CVAT tables
*/

DROP TABLE IF EXISTS `animal`;
CREATE TABLE `animal` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `active` tinyint(1) NOT NULL,
  `created` datetime(6) NOT NULL,
  `animal` varchar(20) NOT NULL,
  `comments` longtext DEFAULT NULL,
  `lab_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `animal_lab_id_06fb0224_fk_authentication_lab_id` (`lab_id`),
  CONSTRAINT `animal_lab_id_06fb0224_fk_authentication_lab_id` FOREIGN KEY (`lab_id`) REFERENCES `authentication_lab` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- Table structure for table `structure`
--       /* Does not include the 3D shape information? */
--

DROP TABLE IF EXISTS `structure`;
/*   What is the role of this table does it store the 3D shape of the structures? *****/
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `structure` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `abbreviation` varchar(25) COLLATE utf8_bin NOT NULL,
  `description` longtext COLLATE utf8_bin NOT NULL,
  `color` int(11) NOT NULL DEFAULT 100,
  `hexadecimal` char(7) COLLATE utf8_bin DEFAULT NULL,
  `active` tinyint(1) NOT NULL,
  `created` datetime(6) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `K__S_ABBREV` (`abbreviation`)
) ENGINE=InnoDB AUTO_INCREMENT=54 DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
/*!40101 SET character_set_client = @saved_cs_client */;

DROP TABLE IF EXISTS `transformation`;


--
--  injection related tables
--

--
-- Table structure for table `injection`
--

DROP TABLE IF EXISTS `injection`;
CREATE TABLE `injection` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `active` tinyint(1) NOT NULL,
  `created` datetime(6) NOT NULL,
  `performance_center` enum('CSHL','Salk','UCSD','HHMI','Duke') DEFAULT NULL,
  `anesthesia` enum('ketamine','isoflurane') DEFAULT NULL,
  `method` enum('iontophoresis','pressure','volume') DEFAULT NULL,
  `injection_volume` double NOT NULL,
  `pipet` enum('glass','quartz','Hamilton','syringe needle') DEFAULT NULL,
  `location` varchar(20) DEFAULT NULL,
  `angle` varchar(20) DEFAULT NULL,
  `brain_location_dv` double NOT NULL,
  `brain_location_ml` double NOT NULL,
  `brain_location_ap` double NOT NULL,
  `injection_date` date DEFAULT NULL,
  `transport_days` int(11) NOT NULL,
  `virus_count` int(11) NOT NULL,
  `comments` longtext DEFAULT NULL,
  `animal_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `injection_animal_id_f7065f51_fk_animal_id` (`animal_id`),
  CONSTRAINT `injection_animal_id_f7065f51_fk_animal_id` FOREIGN KEY (`animal_id`) REFERENCES `animal` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Table structure for table `injection_virus`
--

DROP TABLE IF EXISTS `injection_virus`;
CREATE TABLE `injection_virus` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `active` tinyint(1) NOT NULL,
  `created` datetime(6) NOT NULL,
  `injection_id` int(11) NOT NULL,
  `virus_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `injection_virus_injection_id_f2e6c29c_fk_injection_id` (`injection_id`),
  KEY `injection_virus_virus_id_95d00eb8_fk_virus_id` (`virus_id`),
  CONSTRAINT `injection_virus_injection_id_f2e6c29c_fk_injection_id` FOREIGN KEY (`injection_id`) REFERENCES `injection` (`id`),
  CONSTRAINT `injection_virus_virus_id_95d00eb8_fk_virus_id` FOREIGN KEY (`virus_id`) REFERENCES `virus` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


--
-- Table structure for table `virus`
--

DROP TABLE IF EXISTS `virus`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `virus` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `active` tinyint(1) NOT NULL,
  `created` datetime(6) NOT NULL,
  `virus_name` varchar(50) NOT NULL,
  `virus_type` enum('Adenovirus','AAV','CAV','DG rabies','G-pseudo-Lenti','Herpes','Lenti','N2C rabies','Sinbis') DEFAULT NULL,
  `virus_active` enum('yes','no') DEFAULT NULL,
  `type_details` varchar(500) DEFAULT NULL,
  `titer` double NOT NULL,
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


--
-- Annotation related tables   ---------------------------------------------------------
--

--
-- tables of brain locations (x,y,z location data)
--

-- Table structure for table `layer_data`
--  - main annotation table where the x,y,z data is stored
--


/*
 * CHANGED TABLE NAME FROM layer_data TO annotations_points
 * RENAMED person_id TO FK_owner_id [ORG ANNOTATIONS CREATOR/OWNER]
 *
 * REMOVED:
 * KEY `K__LDA_AID` (`prep_id`) [REPLACED BY FK_animal_id TO REFERENCE animal TABLE]
 * KEY `K__LDA_PID` (`person_id`) [REPLACED BY FK_owner_id TO REFERENCE auth_user TABLE; NO SEPARATE INDEX]
 * KEY `K__LDA_ITID` (`input_type_id`) [NO SEPARATE INDEX]
 *
 * ANTICIPATED OPERATION:
 * 1) USER SAVES ANNOTATION POINTS IN NEUROGLANCER
 * 2) NEW ENTRY IN archive_set TABLE (PARENT 'archive_id' - 0 IF FIRST; UPDATE USER; TIMESTAMP )
 * 2) ALL CURRENT POINTS FOR USER ARE MOVED TO annotations_points_archive
 * 3) NEW POINTS ARE ADDED TO annotations_points
 *    - CONSIDERATIONS:
 *      A) IF LATENCY -> DB MODIFICATIONS MAY BE QUEUED AND MADE VIA CRON JOB (DURING OFF-PEAK)
 *      B) annotations_points_archive, archive_sets WILL NOT BE STORED ON LIVE DB
 */

--
-- Table structure for table `annotations_point_archive`
--

DROP TABLE IF EXISTS `annotations_point_archive`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `annotations_point_archive` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `layer` varchar(255) NOT NULL,
  `x` double NOT NULL,
  `y` double NOT NULL,
  `section` double NOT NULL,
  `FK_animal_id` int(11) DEFAULT NULL,
  `FK_archive_set_id` bigint(20) NOT NULL,
  `input_type_id` bigint(20) NOT NULL,
  `FK_owner_id` bigint(20) NOT NULL,
  `FK_structure_id` bigint(20) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `annotations_point_archive_FK_animal_id_2482b389_fk_animal_id` (`FK_animal_id`),
  KEY `annotations_point_ar_FK_archive_set_id_b47d1b53_fk_archive_s` (`FK_archive_set_id`),
  KEY `annotations_point_ar_input_type_id_fcd2bb76_fk_input_typ` (`input_type_id`),
  KEY `annotations_point_ar_FK_owner_id_0a5d593e_fk_authentic` (`FK_owner_id`),
  KEY `annotations_point_ar_FK_structure_id_fe8347b2_fk_structure` (`FK_structure_id`),
  CONSTRAINT `annotations_point_ar_FK_archive_set_id_b47d1b53_fk_archive_s` FOREIGN KEY (`FK_archive_set_id`) REFERENCES `archive_set` (`id`),
  CONSTRAINT `annotations_point_ar_FK_owner_id_0a5d593e_fk_authentic` FOREIGN KEY (`FK_owner_id`) REFERENCES `authentication_user` (`id`),
  CONSTRAINT `annotations_point_ar_FK_structure_id_fe8347b2_fk_structure` FOREIGN KEY (`FK_structure_id`) REFERENCES `structure` (`id`),
  CONSTRAINT `annotations_point_ar_input_type_id_fcd2bb76_fk_input_typ` FOREIGN KEY (`input_type_id`) REFERENCES `input_type` (`id`),
  CONSTRAINT `annotations_point_archive_FK_animal_id_2482b389_fk_animal_id` FOREIGN KEY (`FK_animal_id`) REFERENCES `animal` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `annotations_points`
--

DROP TABLE IF EXISTS `annotations_points`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `annotations_points` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `layer` varchar(255) NOT NULL,
  `x` double NOT NULL,
  `y` double NOT NULL,
  `section` double NOT NULL,
  `FK_animal_id` int(11) DEFAULT NULL,
  `input_type_id` bigint(20) NOT NULL,
  `FK_owner_id` bigint(20) NOT NULL,
  `FK_structure_id` bigint(20) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `annotations_points_FK_animal_id_40c5eea5_fk_animal_id` (`FK_animal_id`),
  KEY `annotations_points_input_type_id_4fb7062a_fk_input_type_id` (`input_type_id`),
  KEY `annotations_points_FK_owner_id_e8e2760b_fk_authentic` (`FK_owner_id`),
  KEY `annotations_points_FK_structure_id_4519e64d_fk_structure_id` (`FK_structure_id`),
  CONSTRAINT `annotations_points_FK_animal_id_40c5eea5_fk_animal_id` FOREIGN KEY (`FK_animal_id`) REFERENCES `animal` (`id`),
  CONSTRAINT `annotations_points_FK_owner_id_e8e2760b_fk_authentic` FOREIGN KEY (`FK_owner_id`) REFERENCES `authentication_user` (`id`),
  CONSTRAINT `annotations_points_FK_structure_id_4519e64d_fk_structure_id` FOREIGN KEY (`FK_structure_id`) REFERENCES `structure` (`id`),
  CONSTRAINT `annotations_points_input_type_id_4fb7062a_fk_input_type_id` FOREIGN KEY (`input_type_id`) REFERENCES `input_type` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `archive_set`
--

DROP TABLE IF EXISTS `archive_set`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `archive_set` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `created` datetime(6) NOT NULL,
  `FK_parent` int(11) NOT NULL,
  `FK_update_user_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `archive_set_FK_update_user_id_4a89ce81_fk_authentication_user_id` (`FK_update_user_id`),
  CONSTRAINT `archive_set_FK_update_user_id_4a89ce81_fk_authentication_user_id` FOREIGN KEY (`FK_update_user_id`) REFERENCES `authentication_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

/*
* MODIFIED TIMESTAMP ON input_type TABLE [DEFAULT SPECIFICITY SUFFICIENT - NOT (6)]
*/

DROP TABLE IF EXISTS `input_type`;
CREATE TABLE `input_type` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `input_type` varchar(50) NOT NULL,
  `description` longtext NOT NULL,
  `active` tinyint(1) NOT NULL,
  `created` datetime(6) NOT NULL,
  `updated` datetime(6) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4;

INSERT INTO input_type (id, input_type) VALUES (1, 'manual person');
INSERT INTO input_type (id, input_type) VALUES (2, 'corrected person');
INSERT INTO input_type (id, input_type) VALUES (3, 'detected computer');
--
-- tables related to hardware/software --------------------------------------------------
--


-- Table structure for table `scan_run`
--

--
-- Table structure for table `scan_run`
--

DROP TABLE IF EXISTS `scan_run`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `scan_run` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `active` tinyint(1) NOT NULL,
  `created` datetime(6) NOT NULL,
  `objective` enum('60X','40X','20X','10X') DEFAULT NULL,
  `resolution` double NOT NULL,
  `zresolution` double NOT NULL,
  `number_of_slides` int(11) NOT NULL,
  `scan_date` date DEFAULT NULL,
  `file_type` enum('CZI','JPEG2000','NDPI','NGR') DEFAULT NULL,
  `channels_per_scene` enum('1','2','3','4') DEFAULT NULL,
  `width` int(11) NOT NULL,
  `height` int(11) NOT NULL,
  `comments` longtext DEFAULT NULL,
  `animal_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `scan_run_animal_id_c9a911a2_fk_animal_id` (`animal_id`),
  CONSTRAINT `scan_run_animal_id_c9a911a2_fk_animal_id` FOREIGN KEY (`animal_id`) REFERENCES `animal` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
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


--
-- Table structure for table `neuroglancer_state`
--

DROP TABLE IF EXISTS `neuroglancer_state`;
CREATE TABLE `neuroglancer_state` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `neuroglancer_state` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL CHECK (json_valid(`neuroglancer_state`)),
  `created` datetime(6) NOT NULL,
  `updated` datetime(6) NOT NULL,
  `user_date` varchar(25) NOT NULL,
  `comments` varchar(255) NOT NULL,
  `owner_id` bigint(20) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `neuroglancer_state_owner_id_f8136735_fk_authentication_user_id` (`owner_id`),
  CONSTRAINT `neuroglancer_state_owner_id_f8136735_fk_authentication_user_id` FOREIGN KEY (`owner_id`) REFERENCES `authentication_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
