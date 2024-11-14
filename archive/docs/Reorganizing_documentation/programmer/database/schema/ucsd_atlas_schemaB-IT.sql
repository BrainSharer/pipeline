/* TABLE OF CONTENTS - OVERALL ORGANIZATION STRUCTURE (IT - OR 'NON-BIOLOGICAL'):
   TOTAL TABLES (65): annotation_session, annotations_points,	annotations_points_archive,	archive_sets,	auth_group,	auth_group_permissions,	auth_permission,	auth_user,	auth_user_groups,	auth_user_user_permissions,	authtoken_token, django_admin_log,	django_content_type,	django_migrations,	django_plotly_dash_dashapp,	django_plotly_dash_statelessapp,	django_session,	django_site,	elastix_transformation,	engine_attributespec,	engine_clientfile,	engine_data,	engine_image,	engine_job,	engine_jobcommit,	engine_label,	engine_labeledimage,	engine_labeledimageattributeval,	engine_labeledshape,	engine_labeledshapeattributeval,	engine_labeledtrack,	engine_labeledtrackattributeval,	engine_plugin,	engine_pluginoption,	engine_project,	engine_remotefile,	engine_segment,	engine_serverfile,	engine_task,	engine_trackedshape,	engine_trackedshapeattributeval,	engine_video,	file_log,	file_operation,	git_gitdata,	input_type,	journals,	logs,	neuroglancer_state,	neuroglancer_urls,	performance_center,	problem_category,	polygon_sequences, progress_lookup,	sections,	slide, slide_czi_to_tif, socialaccount_socialaccount,	socialaccount_socialapp,	socialaccount_socialapp_sites,	socialaccount_socialtoken,	task,	task_resources,	task_roles,	task_view, transformation

*/

/* DR - CAN WE CONSOLIDATE WITH archive_sets TABLE (REPLACE)?
   DR - CHARSET MUST MATCH animal TABLE: utf8 */

DROP TABLE IF EXISTS `annotation_session` ;
CREATE TABLE `annotation_session` (
 `id` int(11) NOT NULL AUTO_INCREMENT,
 `created` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP(),
 `annotation_type` ENUM('POLYGON_SEQUENCE', 'MARKED_CELL', 'STRUCTURE_COM'),
 `FK_annotator_id` int(11) NOT NULL,
 `FK_prep_id` varchar(20) NOT NULL,
 `FK_parent` INT(11) NOT NULL COMMENT 'SELF-REFERENCES id [IN THIS TABLE]',
 `FK_structure_id` int(11) NOT NULL COMMENT 'TABLE structure SHOULD ONLY CONTAIN BIOLOGICAL STRUCTURES',
 PRIMARY KEY (`id`),
 FOREIGN KEY (`FK_annotator_id`) REFERENCES auth_user(id),
 FOREIGN KEY (`FK_prep_id`) REFERENCES animal(prep_id),
 FOREIGN KEY (`FK_structure_id`) REFERENCES structure(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;


/* USED FOR ANNOTATION STORAGE
   DR - CHARSET FOR label MUST BE utf8_bin TO PRESERVE CASE */

DROP TABLE IF EXISTS `polygon_sequences` ;
CREATE TABLE `polygon_sequences` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `label` varchar(255) CHARACTER SET utf8 COLLATE utf8_bin,
  `source` ENUM('NA') COMMENT 'PLACEHOLDER FIELD',
  `x` float DEFAULT NULL COMMENT 'UNIT: MICRONS',
  `y` float DEFAULT NULL COMMENT 'UNIT: MICRONS',
  `z` float NOT NULL DEFAULT 0 COMMENT 'UNIT: MICRONS',
  `active` tinyint(1) DEFAULT NULL,
  `polygon_index` int(11) DEFAULT NULL COMMENT 'ORDERING (INDEX) OF POLYGONS ACROSS VOLUMES',
  `point_order` int(11) NOT NULL DEFAULT 0,
  `FK_session_id` INT(11) NOT NULL COMMENT 'CREATOR/EDITOR',
  PRIMARY KEY (`id`),
  FOREIGN KEY (`FK_session_id`) REFERENCES annotation_session(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* USED FOR ANNOTATION STORAGE
   DR - CHARSET FOR label MUST BE utf8_bin TO PRESERVE CASE */

DROP TABLE IF EXISTS `marked_cells` ;
CREATE TABLE `marked_cells` (
 `id` int(11) NOT NULL AUTO_INCREMENT,
 `label` varchar(255) CHARACTER SET utf8 COLLATE utf8_bin,
 `source` ENUM('MACHINE-SURE', 'MACHINE-UNSURE', 'HUMAN-POSITIVE', 'HUMAN-NEGATIVE'),
 `x` float DEFAULT NULL COMMENT 'UNIT: MICRONS',
 `y` float DEFAULT NULL COMMENT 'UNIT: MICRONS',
 `z` float NOT NULL DEFAULT 0 COMMENT 'UNIT: MICRONS',
 `active` tinyint(1) DEFAULT NULL,
 `FK_session_id` INT(11) NOT NULL COMMENT 'CREATOR/EDITOR',
  PRIMARY KEY (`id`),
  FOREIGN KEY (`FK_session_id`) REFERENCES annotation_session(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* USED FOR ANNOTATION STORAGE
   DR - CHARSET FOR label MUST BE utf8_bin TO PRESERVE CASE */

DROP TABLE IF EXISTS `structure_com` ;
CREATE TABLE `structure_com` (
 `id` INT(20) NOT NULL AUTO_INCREMENT,
 `label` varchar(255) CHARACTER SET utf8 COLLATE utf8_bin,
 `source` ENUM('MANUAL', 'COMPUTER'),
 `x` float DEFAULT NULL COMMENT 'UNIT: MICRONS',
 `y` float DEFAULT NULL COMMENT 'UNIT: MICRONS',
 `z` float NOT NULL DEFAULT 0 COMMENT 'UNIT: MICRONS',
 `active` tinyint(1) DEFAULT NULL,
 `FK_session_id` INT(11) NOT NULL COMMENT 'CREATOR/EDITOR',
  PRIMARY KEY (`id`),
  FOREIGN KEY (`FK_session_id`) REFERENCES annotation_session(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* ARCHIVE OF ALL ANNOTATIONS (polygon_sequences, marked_cells, structure_com)
   DR - CHARSET FOR label MUST BE utf8_bin TO PRESERVE CASE
   DR - FOR source FIELD IMPORTS (ENUM TO INT) -> ENUM CAST(CAST(`source` AS CHAR) AS SIGNED)
   DR - polygon_index, point_order WILL BE NULL FOR marked_cells, structure_com
*/

DROP TABLE IF EXISTS `annotations_archive`;
CREATE TABLE `annotations_archivee` (
  `id` int(11) NOT NULL,
  `label` varchar(255) CHARACTER SET utf8 COLLATE utf8_bin,
  `source` int(11),
  `x` FLOAT DEFAULT NULL,
  `y` FLOAT DEFAULT NULL,
  `z` double NOT NULL COMMENT 'a.k.a. section (slicing)',
  `polygon_index` int(11),
  `point_order` int(11),
  `FK_session_id` INT(11) NOT NULL COMMENT 'CREATOR/EDITOR',
  FOREIGN KEY (`FK_session_id`) REFERENCES annotation_session(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* DR - annotations_points_archive TABLE LIKELY NOT NEEDED DUE TO FUNCTIONAL OVERLAP WITH annotations_archive */
DROP TABLE IF EXISTS `annotations_points_archive`;
CREATE TABLE `annotations_points_archive` (
  `id` int(20) NOT NULL,
  `prep_id` VARCHAR(20) NOT NULL COMMENT '*LEGACY COMPATABILITY*',
  `label` varchar(255) CHARACTER SET utf8 COLLATE utf8_bin,
  `x` FLOAT DEFAULT NULL,
  `y` FLOAT DEFAULT NULL,
  `z` double NOT NULL COMMENT 'a.k.a. section (slicing)',  
  `vetted` ENUM('yes','no') DEFAULT NULL COMMENT 'good enough for public',
  `FK_structure_id` INT(11) NOT NULL COMMENT 'either structure, point, or line   do we really want line here?',
  `FK_owner_id` INT(11) NOT NULL COMMENT 'ORG ANNOTATIONS CREATOR/OWNER',
  `FK_animal_id` INT(11) NOT NULL,
  `FK_input_id` INT(11) NOT NULL DEFAULT 1 COMMENT 'manual person, corrected person, detected computer',
  `FK_archive_set_id` INT(11) NOT NULL,
  FOREIGN KEY (`FK_animal_id`) REFERENCES animal(id),
  FOREIGN KEY (`FK_owner_id`) REFERENCES auth_user(id),
  FOREIGN KEY (`FK_input_id`) REFERENCES input_type(id),
  FOREIGN KEY (`FK_structure_id`) REFERENCES structure(id),
  FOREIGN KEY (`FK_archive_set_id`) REFERENCES archive_sets(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* DR - archive_sets TABLE LIKELY NOT NEEDED DUE TO FUNCTIONAL OVERLAP WITH annotation_session */
DROP TABLE IF EXISTS `archive_sets`;
CREATE TABLE `archive_sets` (
 `id` int(20) NOT NULL AUTO_INCREMENT,
 `created` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP(),
 `label` VARCHAR(255) DEFAULT NULL COMMENT 'freeform name/label the layer[annotation]',
 `annotation_type` ENUM('POLYGON_SEQUENCE', 'MARKED_CELL', 'STRUCTURE_COM'),
 `FK_animal_id` INT(11) NOT NULL,
 `FK_parent` INT(11) NOT NULL COMMENT 'REFERENCES archive_id IN THIS TABLE',
 `FK_owner_id` int(11) NOT NULL COMMENT 'USER WHO MADE REVISIONS',
 PRIMARY KEY (`id`),
 FOREIGN KEY (`FK_animal_id`) REFERENCES animal(id),
 FOREIGN KEY (`FK_owner_id`) REFERENCES auth_user(id)
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
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `group_id` int(11) NOT NULL,
  `permission_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_group_permissions_group_id_permission_id_0cd325b0_uniq` (`group_id`,`permission_id`),
  KEY `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm` (`permission_id`),
  CONSTRAINT `FK__AGP_group_id` FOREIGN KEY (`group_id`) REFERENCES `auth_group` (`id`) ON UPDATE CASCADE,
  CONSTRAINT `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm` FOREIGN KEY (`permission_id`) REFERENCES `auth_permission` (`id`)
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/*
   COMMENTS RELATED TO TABLE: auth_user
   Unknown contrib - MOVING auth_user TO SEPARATE DATABASE MAY REQUIRE RESTRUCTURING THE FOREIGN KEYS ON annotation_points, annotations_points_archive, archive_sets TABLES
* ALTERNATIVE MAY BE TO STORE TABLE JUST FOR SYNCHRONIZATION (BUT NOT ACTIVELY USED BY DJANGO) - CRON JOB CAN SYNC ON SCHEDULE
*/

DROP TABLE IF EXISTS `auth_user`;
CREATE TABLE `auth_user` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `password` varchar(128) NOT NULL,
  `last_login` datetime DEFAULT NULL,
  `is_superuser` tinyint(1) NOT NULL,
  `username` varchar(150) NOT NULL,
  `first_name` varchar(150) NOT NULL,
  `last_name` varchar(150) NOT NULL,
  `email` varchar(254) NOT NULL,
  `is_staff` tinyint(1) NOT NULL,
  `is_active` tinyint(1) NOT NULL,
  `date_joined` datetime NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `auth_user_groups`;
CREATE TABLE `auth_user_groups` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `group_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_user_groups_user_id_group_id_94350c0c_uniq` (`user_id`,`group_id`),
  KEY `auth_user_groups_group_id_97559544_fk_auth_group_id` (`group_id`),
  CONSTRAINT `FK__AUG_group_id` FOREIGN KEY (`group_id`) REFERENCES `auth_group` (`id`) ON UPDATE CASCADE,
  CONSTRAINT `auth_user_groups_user_id_6a12ed8b_fk_auth_user_id` FOREIGN KEY (`user_id`) REFERENCES `auth_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `auth_user_user_permissions`;
CREATE TABLE `auth_user_user_permissions` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `permission_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_user_user_permissions_user_id_permission_id_14a6b632_uniq` (`user_id`,`permission_id`),
  KEY `auth_user_user_permi_permission_id_1fbb5f2c_fk_auth_perm` (`permission_id`),
  CONSTRAINT `auth_user_user_permi_permission_id_1fbb5f2c_fk_auth_perm` FOREIGN KEY (`permission_id`) REFERENCES `auth_permission` (`id`),
  CONSTRAINT `auth_user_user_permissions_user_id_a95ead1b_fk_auth_user_id` FOREIGN KEY (`user_id`) REFERENCES `auth_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `authtoken_token`;
CREATE TABLE `authtoken_token` (
  `key` varchar(40) NOT NULL,
  `created` datetime(6) NOT NULL,
  `user_id` int(11) NOT NULL,
  PRIMARY KEY (`key`),
  UNIQUE KEY `user_id` (`user_id`),
  CONSTRAINT `authtoken_token_user_id_35299eff_fk_auth_user_id` FOREIGN KEY (`user_id`) REFERENCES `auth_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `django_admin_log`;
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `django_content_type`;
CREATE TABLE `django_content_type` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `app_label` varchar(100) NOT NULL,
  `model` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `django_content_type_app_label_model_76bd3d3b_uniq` (`app_label`,`model`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `django_migrations`;
CREATE TABLE `django_migrations` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `app` varchar(255) NOT NULL,
  `name` varchar(255) NOT NULL,
  `applied` datetime(6) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `django_plotly_dash_dashapp`;
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


DROP TABLE IF EXISTS `django_plotly_dash_statelessapp`;
CREATE TABLE `django_plotly_dash_statelessapp` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `app_name` varchar(100) NOT NULL,
  `slug` varchar(110) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `app_name` (`app_name`),
  UNIQUE KEY `slug` (`slug`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/*
   COMMENTS RELATED TO TABLE: elastix_transformation
   Unknown contrib - What is this table?
*/

DROP TABLE IF EXISTS `elastix_transformation`;
CREATE TABLE `elastix_transformation` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `FK_prep_id` varchar(20) NOT NULL COMMENT 'LEGACY: Name for lab animal, max 20 chars',
  `section` char(3) NOT NULL,
  `rotation` float NOT NULL DEFAULT 0,
  `xshift` float NOT NULL DEFAULT 0,
  `yshift` float NOT NULL DEFAULT 0,
  `created` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `active` tinyint(1) NOT NULL DEFAULT 1,
  `FK_animal_id` int(11),
  FOREIGN KEY (`FK_animal_id`) REFERENCES animal(`id`) ON UPDATE CASCADE,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_attributespec 
   DR - SUGGESTED RENAME FIELD label_id TO FK_label_id FOR CONSISTENCY
*/

DROP TABLE IF EXISTS `engine_attributespec`;
CREATE TABLE `engine_attributespec` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(64) NOT NULL,
  `mutable` tinyint(1) NOT NULL,
  `input_type` varchar(16) NOT NULL,
  `default_value` varchar(128) NOT NULL,
  `values` varchar(4096) NOT NULL,
  `label_id` int(11) NOT NULL,
  FOREIGN KEY (`label_id`) REFERENCES engine_label(`id`),
  PRIMARY KEY (`id`),
  UNIQUE KEY `engine_attributespec_label_id_name_d85e616c_uniq` (`label_id`,`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_clientfile */

DROP TABLE IF EXISTS `engine_clientfile`;
CREATE TABLE `engine_clientfile` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `file` varchar(1024) NOT NULL,
  `data_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_data */

DROP TABLE IF EXISTS `engine_data`;
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_image */

DROP TABLE IF EXISTS `engine_image`;
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_job */

DROP TABLE IF EXISTS `engine_job`;
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_jobcommit */

DROP TABLE IF EXISTS `engine_jobcommit`;
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_label */

DROP TABLE IF EXISTS `engine_label`;
CREATE TABLE `engine_label` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(64) NOT NULL,
  `task_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `engine_label_task_id_name_00e8779a_uniq` (`task_id`,`name`),
  CONSTRAINT `engine_label_task_id_f11c5c1a_fk_engine_task_id` FOREIGN KEY (`task_id`) REFERENCES `engine_task` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_labeledimage */

DROP TABLE IF EXISTS `engine_labeledimage`;
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_labeledimageattributeval */

DROP TABLE IF EXISTS `engine_labeledimageattributeval`;
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_labeledshape */

DROP TABLE IF EXISTS `engine_labeledshape`;
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_labeledshapeattributeval */

DROP TABLE IF EXISTS `engine_labeledshapeattributeval`;
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_labeledtrack */

DROP TABLE IF EXISTS `engine_labeledtrack`;
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_labeledtrackattributeval */

DROP TABLE IF EXISTS `engine_labeledtrackattributeval`;
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_engine_plugin */

DROP TABLE IF EXISTS `engine_plugin`;
CREATE TABLE `engine_plugin` (
  `name` varchar(32) NOT NULL,
  `description` varchar(8192) NOT NULL,
  `created_at` datetime(6) NOT NULL,
  `updated_at` datetime(6) NOT NULL,
  `maintainer_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_pluginoption */

DROP TABLE IF EXISTS `engine_pluginoption`;
CREATE TABLE `engine_pluginoption` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(32) NOT NULL,
  `value` varchar(1024) NOT NULL,
  `plugin_id` varchar(32) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_project */

DROP TABLE IF EXISTS `engine_project`;
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_remotefile */

DROP TABLE IF EXISTS `engine_remotefile`;
CREATE TABLE `engine_remotefile` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `file` varchar(1024) NOT NULL,
  `data_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `engine_remotefile_data_id_ff16acda_fk_engine_data_id` (`data_id`),
  CONSTRAINT `engine_remotefile_data_id_ff16acda_fk_engine_data_id` FOREIGN KEY (`data_id`) REFERENCES `engine_data` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_segment */

DROP TABLE IF EXISTS `engine_segment`;
CREATE TABLE `engine_segment` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `start_frame` int(11) NOT NULL,
  `stop_frame` int(11) NOT NULL,
  `task_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `engine_segment_task_id_37d935cf_fk_engine_task_id` (`task_id`),
  CONSTRAINT `engine_segment_task_id_37d935cf_fk_engine_task_id` FOREIGN KEY (`task_id`) REFERENCES `engine_task` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_serverfile */

DROP TABLE IF EXISTS `engine_serverfile`;
CREATE TABLE `engine_serverfile` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `file` varchar(1024) NOT NULL,
  `data_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `engine_serverfile_data_id_2364110a_fk_engine_data_id` (`data_id`),
  CONSTRAINT `engine_serverfile_data_id_2364110a_fk_engine_data_id` FOREIGN KEY (`data_id`) REFERENCES `engine_data` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_task */

DROP TABLE IF EXISTS `engine_task`;
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_trackedshape */

DROP TABLE IF EXISTS `engine_trackedshape`;
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_trackedshapeattributeval */

DROP TABLE IF EXISTS `engine_trackedshapeattributeval`;
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* TABLE RELATED TO CVAT: engine_video */

DROP TABLE IF EXISTS `engine_video`;
CREATE TABLE `engine_video` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `path` varchar(1024) NOT NULL,
  `width` int(10) unsigned NOT NULL,
  `height` int(10) unsigned NOT NULL,
  `data_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `data_id` (`data_id`),
  CONSTRAINT `engine_video_data_id_b37015e9_fk_engine_data_id` FOREIGN KEY (`data_id`) REFERENCES `engine_data` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `file_log`;
CREATE TABLE `file_log` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `prep_id` varchar(20) NOT NULL COMMENT 'LEGACY: Name for lab animal, max 20 chars',
  `filename` varchar(255) NOT NULL,
  `active` tinyint(1) NOT NULL DEFAULT 1,
  `created` timestamp NOT NULL DEFAULT current_timestamp(),
  `FK_animal_id` int(11),
  `FK_progress_id` int(11) NOT NULL,
  FOREIGN KEY (`FK_animal_id`) REFERENCES animal(`id`) ON UPDATE CASCADE,
  FOREIGN KEY (`FK_progress_id`) REFERENCES progress_lookup(`id`) ON UPDATE CASCADE,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `file_operation`;
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `git_gitdata`;
CREATE TABLE `git_gitdata` (
  `task_id` int(11) NOT NULL,
  `url` varchar(2000) NOT NULL,
  `path` varchar(256) NOT NULL,
  `sync_date` datetime(6) NOT NULL,
  `status` varchar(20) NOT NULL,
  `lfs` tinyint(1) NOT NULL,
  PRIMARY KEY (`task_id`),
  CONSTRAINT `git_gitdata_task_id_a6f2ea20_fk_engine_task_id` FOREIGN KEY (`task_id`) REFERENCES `engine_task` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/*
   COMMENTS RELATED TO TABLE: input_type
   DR - PREVIOUS TABLE NAME: com_type
*/

DROP TABLE IF EXISTS `input_type`;
CREATE TABLE `input_type` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `input_type` varchar(50) NOT NULL,
  `description` varchar(255) DEFAULT NULL,
  `active` tinyint(1) NOT NULL DEFAULT 1,
  `created` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
INSERT INTO input_type (id, input_type) VALUES (1, 'manual person');
INSERT INTO input_type (id, input_type) VALUES (2, 'corrected person');
INSERT INTO input_type (id, input_type) VALUES (3, 'detected computer');


/* TABLE RELATED TO SLIDE QC: journals 
   DR - SUGGESTED PREFACE 'QC'
*/

DROP TABLE IF EXISTS `journals`;
CREATE TABLE `journals` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `prep_id` varchar(20) NOT NULL COMMENT 'LEGACY: Name for lab animal, max 20 chars',
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
  `FK_animal_id` int(11),
  FOREIGN KEY (`FK_animal_id`) REFERENCES animal(`id`) ON UPDATE CASCADE,
  PRIMARY KEY (`id`),
  KEY `K__journals_person_id` (`person_id`),
  KEY `K__journals_problem_id` (`problem_id`),
  KEY `FK__url_id` (`url_id`),
  CONSTRAINT `FK__journals_person_id` FOREIGN KEY (`person_id`) REFERENCES `auth_user` (`id`) ON UPDATE CASCADE,
  CONSTRAINT `FK__journals_problem_id` FOREIGN KEY (`problem_id`) REFERENCES `problem_category` (`id`),
  CONSTRAINT `FK__url_id` FOREIGN KEY (`url_id`) REFERENCES `neuroglancer_urls` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/*
   COMMENTS RELATED TO TABLE: logs
   Unknown contrib - What is the role of this table?
*/

DROP TABLE IF EXISTS `logs`;
CREATE TABLE `logs` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `logger` varchar(100) NOT NULL,
  `level` varchar(25) NOT NULL,
  `msg` varchar(255) NOT NULL,
  `created` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `active` tinyint(4) NOT NULL DEFAULT 1,
  `FK_animal_id` int(11),
  FOREIGN KEY (`FK_animal_id`) REFERENCES animal(`id`) ON DELETE CASCADE,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/*
   COMMENTS RELATED TO TABLE: neuroglancer_state
   Unknown contrib - Information in this table should be short lived, a
   new URL can be started from the web interface by specifying brain and
   viewing configuration. Once the user hits "save" the json,
   information should be parsed and added to the appropriate
   tables. com, to coms tables and configuration to configuration tables
   (for storing the chosen histogram configuration, point of view, etc)
*/

DROP TABLE IF EXISTS `neuroglancer_state`;
CREATE TABLE `neuroglancer_state` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `neuroglancer_state` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL CHECK (json_valid(`neuroglancer_state`)),
  `created` datetime DEFAULT current_timestamp(),
  `updated` datetime DEFAULT current_timestamp(),
  `user_date` varchar(25) NOT NULL,
  `comments` varchar(255) NOT NULL,
  `FK_owner_id` int(11) NOT NULL,
  FOREIGN KEY (`FK_owner_id`) REFERENCES auth_user(id) ON DELETE CASCADE,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/*
   COMMENTS RELATED TO TABLE: neuroglancer_urls
   YF: This table is for storing information sent to and from neuroglancer.
   It should be constructed by the user using tables, and stored back in tables when the user hit "save"
*/

DROP TABLE IF EXISTS `neuroglancer_urls`;
CREATE TABLE `neuroglancer_urls` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `url` longtext NOT NULL,
  `active` tinyint(1) DEFAULT NULL,
  `vetted` tinyint(1) DEFAULT NULL,
  `user_date` varchar(25) DEFAULT NULL,
  `comments` varchar(255) DEFAULT NULL,
  `created` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `FK_owner_id` int(11) NOT NULL COMMENT 'ORG ANNOTATIONS CREATOR/OWNER/UPDATER',
  FOREIGN KEY (`FK_owner_id`) REFERENCES auth_user(id) ON DELETE CASCADE,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `performance_center`;
CREATE TABLE `performance_center` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
/* INSERT DEFAULTS */
INSERT INTO performance_center (id, name) VALUES (1, 'CSHL');
INSERT INTO performance_center (id, name) VALUES (2, 'Salk');
INSERT INTO performance_center (id, name) VALUES (3, 'UCSD');
INSERT INTO performance_center (id, name) VALUES (4, 'HHMI');
INSERT INTO performance_center (id, name) VALUES (5, 'Duke');


/*
   COMMENTS RELATED TO TABLE: problem_category
   Unknown contrib - How does this relate to the other log tables and how are these log tables used?
   DR - SUGGESTED PREFACE 'QC'
*/
DROP TABLE IF EXISTS `problem_category`;
CREATE TABLE `problem_category` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `problem_category` varchar(255) NOT NULL,
  `active` tinyint(1) NOT NULL DEFAULT 1,
  `created` datetime(6) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/*
   COMMENTS RELATED TO TABLE: progress_lookup
   Unknown contrib - What is the role of this table?
*/

DROP TABLE IF EXISTS `progress_lookup`;
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/*
   COMMENTS RELATED TO TABLE: sections
   Unknown contrib - What is the role of this table, what happened to the conversion of slides to sections?
   DR - THIS IS A VIEW; UNCLEAR HOW GENERATED *POSSIBLY REMOVE*
*/

DROP VIEW IF EXISTS `sections`;
/*
CREATE TABLE `sections` (
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
) ENGINE=MyISAM 
*/


/*
   COMMENTS RELATED TO TABLE: slide
   Unknown contrib - What is the role of this table, did this subsume the table "sections"?
*/

DROP TABLE IF EXISTS `slide`;
CREATE TABLE `slide` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
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
  `comments` longtext DEFAULT NULL COMMENT 'assessment',
  `active` tinyint(1) NOT NULL DEFAULT 1,
  `created` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `file_size` float NOT NULL DEFAULT 0,
  `processing_duration` float NOT NULL DEFAULT 0,
  `processed` tinyint(4) NOT NULL DEFAULT 0,
  `scene_qc_1` tinyint(4) NOT NULL DEFAULT 0,
  `scene_qc_2` tinyint(4) NOT NULL DEFAULT 0,
  `scene_qc_3` tinyint(4) NOT NULL DEFAULT 0,
  `scene_qc_4` tinyint(4) NOT NULL DEFAULT 0,
  `scene_qc_5` tinyint(4) NOT NULL DEFAULT 0,
  `scene_qc_6` tinyint(4) NOT NULL DEFAULT 0,
  `FK_scan_run_id` int(11) NOT NULL,
  FOREIGN KEY (`FK_scan_run_id`) REFERENCES scan_run(`id`) ON UPDATE CASCADE,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `slide_czi_to_tif`;
CREATE TABLE `slide_czi_to_tif` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `file_name` varchar(200) NOT NULL,
  `scene_number` tinyint(4) NOT NULL,
  `channel` tinyint(4) NOT NULL,
  `width` int(11) NOT NULL DEFAULT 0,
  `height` int(11) NOT NULL DEFAULT 0,
  `comments` longtext DEFAULT NULL COMMENT 'assessment',
  `active` tinyint(1) NOT NULL DEFAULT 1,
  `created` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `file_size` float NOT NULL DEFAULT 0,
  `scene_index` int(11) NOT NULL DEFAULT 0,
  `processing_duration` float NOT NULL DEFAULT 0,
  `FK_slide_id` int(11) NOT NULL,
  FOREIGN KEY (`FK_slide_id`) REFERENCES slide(`id`) ON UPDATE CASCADE ON DELETE CASCADE,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/*
   COMMENTS RELATED TO TABLE: socialaccount_socialaccount
   Unknown contrib -  What is the role of this table?
 */

DROP TABLE IF EXISTS `socialaccount_socialaccount`;
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
  `id` int(11) NOT NULL AUTO_INCREMENT,
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
   COMMENTS RELATED TO TABLE: task
   Unknown contrib - What is the role of this table? What are tasks and how are they used?
*/
DROP TABLE IF EXISTS `task`;
CREATE TABLE `task` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `lookup_id` int(11) NOT NULL,
  `prep_id` VARCHAR(20) NOT NULL COMMENT 'LEGACY: Name for lab animal, max 20 chars',
  `completed` tinyint(4) NOT NULL DEFAULT 0,
  `start_date` datetime DEFAULT NULL,
  `end_date` datetime DEFAULT NULL,
  `active` tinyint(4) NOT NULL DEFAULT 1,
  `created` timestamp NULL DEFAULT current_timestamp(),
  `FK_animal_id` INT(11) NOT NULL,
  FOREIGN KEY (`FK_animal_id`) REFERENCES animal(id) ON UPDATE CASCADE,
  PRIMARY KEY (`id`),
  UNIQUE KEY `UK__progress_datFa_prep_lookup` (`prep_id`,`lookup_id`),
  KEY `K__task_data_lookup_id` (`lookup_id`),
  CONSTRAINT `FK__task_lookup_id` FOREIGN KEY (`lookup_id`) REFERENCES `progress_lookup` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/*
   COMMENTS RELATED TO TABLE: task_resources
   Unknown contrib - What is the role of this table?
*/
DROP TABLE IF EXISTS `task_resources`;
CREATE TABLE `task_resources` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `task_id` int(11) NOT NULL,
  `resource_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `UK__TR_task_id` (`task_id`,`resource_id`),
  KEY `K__TR_resource_id` (`resource_id`),
  CONSTRAINT `FK__TR_resource_id` FOREIGN KEY (`resource_id`) REFERENCES `resource` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `FK__TR_task_id` FOREIGN KEY (`task_id`) REFERENCES `task` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


DROP TABLE IF EXISTS `task_roles`;
CREATE TABLE `task_roles` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(30) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* DR - IS THIS SUPPOSED TO BE A VIEW OR TABLE? */

DROP TABLE IF EXISTS `task_view`;
CREATE TABLE `task_view` (
  `prep_id` tinyint(4) NOT NULL,
  `percent_complete` tinyint(4) NOT NULL,
  `complete` tinyint(4) NOT NULL,
  `created` tinyint(4) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4;

drop table transformation ;
CREATE TABLE `transformation` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `FK_source_id` int(11) NOT NULL,
  `FK_destination_id` int(11) NOT NULL,
  `FK_transformation_type_id` int(11) NOT NULL DEFAULT 1,
  `transformation` blob NOT NULL,
  `created` datetime(6) NOT NULL DEFAULT current_timestamp(6),
  `updated` timestamp NOT NULL DEFAULT current_timestamp(),
  `active` int(2) NOT NULL DEFAULT 1,
  PRIMARY KEY (`id`),
  UNIQUE KEY `source` (`FK_source_id`,`FK_destination_id`,`FK_transformation_type_id`),
  KEY `FK_destination_id` (`FK_destination_id`),
  KEY `FK_transformation_type_id` (`FK_transformation_type_id`),
  CONSTRAINT `transformation_ibfk_1` FOREIGN KEY (`FK_source_id`) REFERENCES `animal` (`id`) ON DELETE CASCADE,
  CONSTRAINT `transformation_ibfk_2` FOREIGN KEY (`FK_destination_id`) REFERENCES `animal` (`id`) ON DELETE CASCADE,
  CONSTRAINT `transformation_ibfk_3` FOREIGN KEY (`FK_transformation_type_id`) REFERENCES `transformation_type` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=266 DEFAULT CHARSET=utf8;

