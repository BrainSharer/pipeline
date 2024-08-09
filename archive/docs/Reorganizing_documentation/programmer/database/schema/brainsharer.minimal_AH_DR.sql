/* TABLE OF CONTENTS - OVERALL ORGANIZATION STRUCTURE:
   1) TABLES RELATED TO BIOLOGICAL DATA SOURCE & SAMPLE PREP: biosource, biocyc, injection, injection_virus, virus, scan_run, brain_region, brain_atlas
   2) TABLES RELATED TO POINT ANNOTATIONS STORAGE: annotations_points, annotations_point_archive, archive_set, input_type, neuroglancer_state
   3) TABLES RELATED TO USER ACCOUNTS: authentication_user, account_emailaddress, account_emailconfirmation, auth_group, auth_group_permissions, auth_permission, authentication_lab, authentication_user_groups, authentication_user_labs, authentication_user_user_permissions, socialaccount_socialaccount, socialaccount_socialapp, socialaccount_socialapp_sites, socialaccount_socialtoken
   4) TABLES RELATED TO PLATFORM ADMINISTRATION/FUNCTIONALITY: django_admin_log, django_content_type, django_migrations, django_session, django_site
*/


/*
   1) TABLES RELATED TO BIOLOGICAL DATA SOURCE: biosource, biocyc
*/

/* AH - What is the animal field for? We need a species column. */
/* DR - I agree but also we could generalize even more to 'biosource', which could be more inclusive.  Previous work on biological pathways related to metabolomics, genomics, proteomics used biocyc databases (https://biocyc.org/).  Using nomenclature (HUMAN, MOUSE, FLY) as foreign key may lead to additional applications in future.

   DR - RENAMED animal TABLE TO biosource [REFERENCES biocyc TABLE FOR ALL SPECIES]
*/

DROP TABLE IF EXISTS `biosource`;
CREATE TABLE `biosource` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `active` tinyint(4) NOT NULL DEFAULT 1,
  `created` datetime DEFAULT current_timestamp(),
  `comments` varchar(2001) DEFAULT NULL,
  `sex` varchar enum('Male','Female', 'Hermaphrodite', 'DoesNotApply') DEFAULT NULL,
  `tissue` varchar(100) DEFAULT NULL COMMENT 'ex. animal, brain, slides',
   FK_ORGID int(11) COMMENT 'organism id',
  `FK_authentication_lab_id` int(11),
   FOREIGN KEY (`FK_authentication_lab_id`) REFERENCES authentication_lab(`id`),
   FOREIGN KEY (`FK_ORGID`) REFERENCES biocyc(`id`),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* UNIQUE NAMES BASED ON REF: http://bioinformatics.ai.sri.com/biowarehouse/repos/schema/doc/BioSource.html */
DROP TABLE IF EXISTS `biocyc`;
CREATE TABLE `biocyc` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `strain` varchar(220) DEFAULT NULL,
  `name` varchar(200) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
INSERT INTO biocyc (name) VALUES ('MOUSE');
INSERT INTO biocyc (name) VALUES ('RAT');
INSERT INTO biocyc (name) VALUES ('FLY');
INSERT INTO biocyc (name) VALUES ('ZFISH');



/* AH - need to specify units of injection volume, e.g. microliters like
injection_volume -> injection_volume_ul
- what is the "location" field for? Is that brain area?
That depends on the reference atlas that is being used,
so also need a column for the name of the atlas being used
   DR - assume location = brain area, added FK_ref_atlas_id
*/

DROP TABLE IF EXISTS `injection`;
CREATE TABLE `injection` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `active` tinyint(1) NOT NULL,
  `created` datetime DEFAULT current_timestamp(),
  `anesthesia` enum('ketamine','isoflurane') DEFAULT NULL,
  `method` enum('iontophoresis','pressure','volume') DEFAULT NULL,
  `injection_volume` double NOT NULL,
  `pipet` enum('glass','quartz','Hamilton','syringe needle') DEFAULT NULL,
  `location` varchar(20) DEFAULT NULL,
  `angle` varchar(20) DEFAULT NULL,
  `brain_location_dv` double NOT NULL DEFAULT 0 COMMENT '(mm) dorsal-ventral relative to Bregma',
  `brain_location_ml` double NOT NULL DEFAULT 0 COMMENT '(mm) medial-lateral relative to Bregma; check if positive',
  `brain_location_ap` double NOT NULL DEFAULT 0 COMMENT '(mm) anterior-posterior relative to Bregma',
  `injection_date` date DEFAULT NULL,
  `transport_days` int(11) NOT NULL DEFAULT 0,
  `virus_count` int(11) NOT NULL DEFAULT 0,
  `comments` longtext DEFAULT NULL,
  `injection_volume_ul` varchar(20) DEFAULT NULL,
  `FK_performance_center_id` int(11),
  `FK_biosource_id` int(11),
  `FK_ref_atlas_id` int(11),
  FOREIGN KEY (`FK_performance_center_id`) REFERENCES performance_center(`performance_center_id`),
  FOREIGN KEY (`FK_biosource_id`) REFERENCES biosource(`id`),
  FOREIGN KEY (`FK_ref_atlas_id`) REFERENCES biosource(`id`),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `injection_virus`;
CREATE TABLE `injection_virus` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `active` tinyint(1) NOT NULL,
  `created` datetime DEFAULT current_timestamp(),
  `FK_injection_id` int(11) NOT NULL,
  `FK_virus_id` int(11) NOT NULL,
  FOREIGN KEY (`FK_injection_id`) REFERENCES injection(`id`),
  FOREIGN KEY (`FK_virus_id`) REFERENCES virus(`id`),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* DR - recommend move virus_type, virus_source to separate tables */

DROP TABLE IF EXISTS `virus`;
CREATE TABLE `virus` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `active` tinyint(1) NOT NULL,
  `created` datetime DEFAULT current_timestamp(),
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

/* DR - ADDED FK_performance_center_id, instrument FIELDS */

DROP TABLE IF EXISTS `scan_run`;
CREATE TABLE `scan_run` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `active` tinyint(1) NOT NULL,
  `created` datetime DEFAULT current_timestamp(),
  `instrument` enum('Zeiss','Axioscan','Nanozoomer','Olympus VA') DEFAULT NULL,
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
  `FK_biosource_id` int(11),
  `FK_performance_center_id` int(11),
  FOREIGN KEY (`FK_biosource_id`) REFERENCES biosource(`id`),
  FOREIGN KEY (`FK_performance_center_id`) REFERENCES performance_center(`performance_center_id`),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* AH - I assume this is for brain regions.
If so, I suggest renaming it to brain_region.
We also need to add a foreign key to the id field of the brain_atlas table.
That is because each structure is part of a specific atlas.  */
/* DR - renamed table from 'structure' to 'brain_region' */

DROP TABLE IF EXISTS `brain_region`;
CREATE TABLE `brain_region` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `active` tinyint(1) NOT NULL,
  `created` datetime DEFAULT current_timestamp(),
  `abbreviation` varchar(200) NOT NULL,
  `description` longtext NOT NULL,
  `FK_ref_atlas_id` int(11),
  FOREIGN KEY (`FK_ref_atlas_id`) REFERENCES biosource(`id`),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* AH - I suggest we add a table for brain atlases like so: */
/* DR - agree - what are default values? */

CREATE TABLE `brain_atlas` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `atlas_name` varchar(64) NOT NULL,
  `description` longtext NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
INSERT INTO brain_atlas (id, atlas_name, description) VALUES (1, 'UCSD', 'UCSD Kleinfeld lab Active Brain Atlas');

/*
     2) TABLES RELATED TO POINT ANNOTATIONS STORAGE: annotations_points, annotations_point_archive, archive_set, input_type
*/

/* AH - replace name of field "section" with "z". The idea of a section is specific to slice histology. */
/* AH - why do we need both annotations_point_archive and annotations_points tables? 
/* AH - What do the following fields mean: "layer", "archive_set_id", "input_type", "owner_id", "structure_id" */

/* DR - modified field "section" to "z" with comment in annotations_points and annotations_point_archive.
   DR - we do not need annotations_point_archive table if archived versions are stored on disk and referenced with archive_set.id (perhaps filename); this is for versioning of annotated Neuroglancer points (org. proposal was to store in file rather than live in database)
   DR - "layer" is related to Neuroglancer layer (user can name layer for superimposition of annotated points), "FK_archive_set_id" is for versioning of Neuroglancer annotated points (i.e., if points are added/removed/edited user can restore from previous version), "FK_input_type_id" is used to store point annotations input source: 'manual person', 'corrected person', 'detected computer', "FK_owner_id" is user who initially created/uploaded/input annotations
   DR - I believe data is stored in "structure" table is for each brain region (table renamed to brain_region)
*/
*/

DROP TABLE IF EXISTS `annotations_points`;
CREATE TABLE `annotations_points` (
  `id` INT(20) NOT NULL AUTO_INCREMENT,
  `layer` varchar(255) NOT NULL COMMENT 'freeform name/label the layer[annotation]',
  `x` double NOT NULL,
  `y` double NOT NULL,
  `z` double NOT NULL COMMENT 'a.k.a. section (slicing)',
  `FK_biosource_id` int(11),
  `FK_input_type_id` INT(11) NOT NULL DEFAULT 1 COMMENT 'manual person, corrected person, detected computer',
  `FK_owner_id` int(11) NOT NULL,
  `FK_brain_region_id` int(11) DEFAULT NULL,
  FOREIGN KEY (`FK_biosource_id`) REFERENCES biosource(id) ON UPDATE CASCADE,
  FOREIGN KEY (`FK_owner_id`) REFERENCES authentication_user(id) ON DELETE CASCADE,
  FOREIGN KEY (`FK_input_type_id`) REFERENCES input_type(id),
  FOREIGN KEY (`FK_brain_region_id`) REFERENCES brain_region(id)
  PRIMARY KEY (`id`),
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `annotations_point_archive`;
CREATE TABLE `annotations_point_archive` (
  `id` int(20) NOT NULL AUTO_INCREMENT,
  `layer` varchar(255) NOT NULL COMMENT 'freeform name/label the layer[annotation]',
  `x` double NOT NULL,
  `y` double NOT NULL,
  `z` double NOT NULL COMMENT 'a.k.a. section (slicing)',
  `FK_biosource_id` int(11),
  `FK_input_type_id` INT(11) NOT NULL DEFAULT 1 COMMENT 'manual person, corrected person, detected computer',
  `FK_owner_id` int(11) NOT NULL,
  `FK_brain_region_id` int(11) DEFAULT NULL,
  `FK_archive_set_id` bigint(20) NOT NULL,
  FOREIGN KEY (`FK_biosource_id`) REFERENCES biosource(id) ON UPDATE CASCADE,
  FOREIGN KEY (`FK_owner_id`) REFERENCES authentication_user(id) ON DELETE CASCADE,
  FOREIGN KEY (`FK_input_type_id`) REFERENCES input_type(id),
  FOREIGN KEY (`FK_brain_region_id`) REFERENCES brain_region(id),
  FOREIGN KEY (`FK_archive_set_id`) REFERENCES archive_sets(id)
  PRIMARY KEY (`id`),
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* What is this table for? */
/* DR - this table (archive_set) is for versioning of Neuroglancer annotated points (i.e., if points are added/removed/edited user can restore from previous version) */

DROP TABLE IF EXISTS `archive_sets`;
CREATE TABLE `archive_sets` (
  `id` int(20) NOT NULL AUTO_INCREMENT,
  `created` datetime DEFAULT current_timestamp(),
  `FK_parent` INT(11) NOT NULL COMMENT 'REFERENCES archive_id IN THIS TABLE',
  `FK_owner_id` int(11) NOT NULL COMMENT 'USER WHO MADE REVISIONS',
  FOREIGN KEY (`FK_owner_id`) REFERENCES authentication_user(id) ON DELETE CASCADE,
  PRIMARY KEY (`id`),
  ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* AH - What does this table represent? */
/* DR - point annotations input source: 'manual person', 'corrected person', 'detected computer' (INSERT statements added after table) */

DROP TABLE IF EXISTS `input_type`;
CREATE TABLE `input_type` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `input_type` varchar(50) NOT NULL,
  `description` varchar(255) DEFAULT NULL,
  `active` tinyint(1) NOT NULL DEFAULT 1,
  `created` datetime DEFAULT current_timestamp(),
  `updated` datetime DEFAULT current_timestamp(),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
/* INSERT DEFAULTS */
INSERT INTO input_type (id, input_type) VALUES (1, 'manual person');
INSERT INTO input_type (id, input_type) VALUES (2, 'corrected person');
INSERT INTO input_type (id, input_type) VALUES (3, 'detected computer');

DROP TABLE IF EXISTS `neuroglancer_state`;
CREATE TABLE `neuroglancer_state` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `neuroglancer_state` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL CHECK (json_valid(`neuroglancer_state`)),
  `created` datetime DEFAULT current_timestamp(),
  `updated` datetime DEFAULT current_timestamp(),
  `user_date` varchar(25) NOT NULL,
  `comments` varchar(255) NOT NULL,
  `FK_owner_id` int(11) NOT NULL,
  FOREIGN KEY (`FK_owner_id`) REFERENCES authentication_user(id) ON DELETE CASCADE,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/*
  3) TABLES RELATED TO USER ACCOUNTS: authentication_user, account_emailaddress, account_emailconfirmation, auth_group, auth_group_permissions, auth_permission, authentication_lab, authentication_user_groups, authentication_user_labs, authentication_user_user_permissions, socialaccount_socialaccount, socialaccount_socialapp, socialaccount_socialapp_sites, socialaccount_socialtoken
*/
DROP TABLE IF EXISTS `authentication_user`;
CREATE TABLE `authentication_user` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
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
  `lab_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`),
  KEY `authentication_user_lab_id_d8a6764c_fk_authentication_lab_id` (`lab_id`),
  CONSTRAINT `authentication_user_lab_id_d8a6764c_fk_authentication_lab_id` FOREIGN KEY (`lab_id`) REFERENCES `authentication_lab` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `account_emailaddress`;
CREATE TABLE `account_emailaddress` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `email` varchar(254) NOT NULL,
  `verified` tinyint(1) NOT NULL,
  `primary` tinyint(1) NOT NULL,
  `user_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `email` (`email`),
  KEY `account_emailaddress_user_id_2c513194_fk_authentication_user_id` (`user_id`),
  CONSTRAINT `account_emailaddress_user_id_2c513194_fk_authentication_user_id` FOREIGN KEY (`user_id`) REFERENCES `authentication_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `account_emailconfirmation`;
CREATE TABLE `account_emailconfirmation` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `created` datetime DEFAULT current_timestamp(),
  `sent` datetime DEFAULT NULL,
  `key` varchar(64) NOT NULL,
  `email_address_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `key` (`key`),
  KEY `account_emailconfirm_email_address_id_5b7f8c58_fk_account_e` (`email_address_id`),
  CONSTRAINT `account_emailconfirm_email_address_id_5b7f8c58_fk_account_e` FOREIGN KEY (`email_address_id`) REFERENCES `account_emailaddress` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `auth_group`;
CREATE TABLE `auth_group` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(150) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `auth_group_permissions`;
CREATE TABLE `auth_group_permissions` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `group_id` int(11) NOT NULL,
  `permission_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_group_permissions_group_id_permission_id_0cd325b0_uniq` (`group_id`,`permission_id`),
  KEY `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm` (`permission_id`),
  CONSTRAINT `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm` FOREIGN KEY (`permission_id`) REFERENCES `auth_permission` (`id`),
  CONSTRAINT `auth_group_permissions_group_id_b120cbf9_fk_auth_group_id` FOREIGN KEY (`group_id`) REFERENCES `auth_group` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `auth_permission`;
CREATE TABLE `auth_permission` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `content_type_id` int(11) NOT NULL,
  `codename` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_permission_content_type_id_codename_01ab375a_uniq` (`content_type_id`,`codename`),
  CONSTRAINT `auth_permission_content_type_id_2f476e4b_fk_django_co` FOREIGN KEY (`content_type_id`) REFERENCES `django_content_type` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=97 DEFAULT CHARSET=utf8mb4;

/* DR - Can we consolidate authentication_lab and performance_center? */

DROP TABLE IF EXISTS `authentication_lab`;
CREATE TABLE `authentication_lab` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `lab_name` varchar(100) NOT NULL,
  `lab_url` varchar(250) NOT NULL,
  `active` tinyint(1) NOT NULL,
  `created` datetime DEFAULT current_timestamp(),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `performance_center`;
CREATE TABLE `performance_center` (
  `performance_center_id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`performance_center_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/* INSERT DEFAULTS */
INSERT INTO performance_center (performance_center_id, name) VALUES (1, 'CSHL');
INSERT INTO performance_center (performance_center_id, name) VALUES (2, 'Salk');
INSERT INTO performance_center (performance_center_id, name) VALUES (3, 'UCSD');
INSERT INTO performance_center (performance_center_id, name) VALUES (4, 'HHMI');
INSERT INTO performance_center (performance_center_id, name) VALUES (5, 'Duke');

DROP TABLE IF EXISTS `authentication_user_groups`;
CREATE TABLE `authentication_user_groups` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `user_id` bigint(20) NOT NULL,
  `group_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `authentication_user_groups_user_id_group_id_8af031ac_uniq` (`user_id`,`group_id`),
  KEY `authentication_user_groups_group_id_6b5c44b7_fk_auth_group_id` (`group_id`),
  CONSTRAINT `authentication_user__user_id_30868577_fk_authentic` FOREIGN KEY (`user_id`) REFERENCES `authentication_user` (`id`),
  CONSTRAINT `authentication_user_groups_group_id_6b5c44b7_fk_auth_group_id` FOREIGN KEY (`group_id`) REFERENCES `auth_group` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `authentication_user_labs`;
CREATE TABLE `authentication_user_labs` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `user_id` bigint(20) NOT NULL,
  `lab_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `authentication_user_labs_user_id_lab_id_85e83707_uniq` (`user_id`,`lab_id`),
  KEY `authentication_user__lab_id_b7c82161_fk_authentic` (`lab_id`),
  CONSTRAINT `authentication_user__lab_id_b7c82161_fk_authentic` FOREIGN KEY (`lab_id`) REFERENCES `authentication_lab` (`id`),
  CONSTRAINT `authentication_user__user_id_459c7a11_fk_authentic` FOREIGN KEY (`user_id`) REFERENCES `authentication_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `authentication_user_user_permissions`;
CREATE TABLE `authentication_user_user_permissions` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `user_id` bigint(20) NOT NULL,
  `permission_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `authentication_user_user_user_id_permission_id_ec51b09f_uniq` (`user_id`,`permission_id`),
  KEY `authentication_user__permission_id_ea6be19a_fk_auth_perm` (`permission_id`),
  CONSTRAINT `authentication_user__permission_id_ea6be19a_fk_auth_perm` FOREIGN KEY (`permission_id`) REFERENCES `auth_permission` (`id`),
  CONSTRAINT `authentication_user__user_id_736ebf7e_fk_authentic` FOREIGN KEY (`user_id`) REFERENCES `authentication_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `socialaccount_socialaccount`;
CREATE TABLE `socialaccount_socialaccount` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `provider` varchar(30) NOT NULL,
  `uid` varchar(191) NOT NULL,
  `last_login` datetime(6) NOT NULL,
  `date_joined` datetime(6) NOT NULL,
  `extra_data` longtext NOT NULL,
  `user_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `socialaccount_socialaccount_provider_uid_fc810c6e_uniq` (`provider`,`uid`),
  KEY `socialaccount_social_user_id_8146e70c_fk_authentic` (`user_id`),
  CONSTRAINT `socialaccount_social_user_id_8146e70c_fk_authentic` FOREIGN KEY (`user_id`) REFERENCES `authentication_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `socialaccount_socialapp`;
CREATE TABLE `socialaccount_socialapp` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `provider` varchar(30) NOT NULL,
  `name` varchar(40) NOT NULL,
  `client_id` varchar(191) NOT NULL,
  `secret` varchar(191) NOT NULL,
  `key` varchar(191) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `socialaccount_socialapp_sites`;
CREATE TABLE `socialaccount_socialapp_sites` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `socialapp_id` int(11) NOT NULL,
  `site_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `socialaccount_socialapp_sites_socialapp_id_site_id_71a9a768_uniq` (`socialapp_id`,`site_id`),
  KEY `socialaccount_socialapp_sites_site_id_2579dee5_fk_django_site_id` (`site_id`),
  CONSTRAINT `socialaccount_social_socialapp_id_97fb6e7d_fk_socialacc` FOREIGN KEY (`socialapp_id`) REFERENCES `socialaccount_socialapp` (`id`),
  CONSTRAINT `socialaccount_socialapp_sites_site_id_2579dee5_fk_django_site_id` FOREIGN KEY (`site_id`) REFERENCES `django_site` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `socialaccount_socialtoken`;
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/*
  4) TABLES RELATED TO PLATFORM ADMINISTRATION/FUNCTIONALITY: django_admin_log, django_content_type, django_migrations, django_session, django_site
*/

DROP TABLE IF EXISTS `django_admin_log`;
CREATE TABLE `django_admin_log` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `action_time` datetime(6) NOT NULL,
  `object_id` longtext DEFAULT NULL,
  `object_repr` varchar(200) NOT NULL,
  `action_flag` smallint(5) unsigned NOT NULL CHECK (`action_flag` >= 0),
  `change_message` longtext NOT NULL,
  `content_type_id` int(11) DEFAULT NULL,
  `user_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `django_admin_log_content_type_id_c4bce8eb_fk_django_co` (`content_type_id`),
  KEY `django_admin_log_user_id_c564eba6_fk_authentication_user_id` (`user_id`),
  CONSTRAINT `django_admin_log_content_type_id_c4bce8eb_fk_django_co` FOREIGN KEY (`content_type_id`) REFERENCES `django_content_type` (`id`),
  CONSTRAINT `django_admin_log_user_id_c564eba6_fk_authentication_user_id` FOREIGN KEY (`user_id`) REFERENCES `authentication_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `django_content_type`;
CREATE TABLE `django_content_type` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `app_label` varchar(100) NOT NULL,
  `model` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `django_content_type_app_label_model_76bd3d3b_uniq` (`app_label`,`model`)
) ENGINE=InnoDB AUTO_INCREMENT=25 DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `django_migrations`;
CREATE TABLE `django_migrations` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `app` varchar(255) NOT NULL,
  `name` varchar(255) NOT NULL,
  `applied` datetime(6) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=29 DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `django_session`;
CREATE TABLE `django_session` (
  `session_key` varchar(40) NOT NULL,
  `session_data` longtext NOT NULL,
  `expire_date` datetime(6) NOT NULL,
  PRIMARY KEY (`session_key`),
  KEY `django_session_expire_date_a5c62663` (`expire_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `django_site`;
CREATE TABLE `django_site` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `domain` varchar(100) NOT NULL,
  `name` varchar(50) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `django_site_domain_a2e37b91_uniq` (`domain`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4;