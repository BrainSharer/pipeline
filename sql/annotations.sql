-- volumes
select as2.id, AN.prep_id , au.username, al.label, as2.annotation, as2.created, as2.updated
from annotation_session as2
inner join auth_user au on as2.FK_user_id = au.id
inner join annotation_session_labels asl on as2.id = asl.annotationsession_id
inner join annotation_label al on asl.annotationlabel_id = al.id
inner join animal AN on as2.FK_prep_id = AN.prep_id
where 1=1 
-- and JSON_EXTRACT(as2.annotation, '$.type') = 'volume'
-- and as2.FK_prep_id in ('DK55')
-- and as2.active = 1
and al.label = 'SC'
and as2.id >= 8306
-- and as2.FK_prep_id  = 'Allen'
order by as2.created desc, as2.FK_prep_id, au.username, al.label;
-- coms
select as2.FK_prep_id, au.username, al.label, JSON_EXTRACT(as2.annotation, '$.point')
from annotation_session as2
inner join auth_user au on as2.FK_user_id = au.id
inner join annotation_session_labels asl on as2.id = asl.annotationsession_id
inner join annotation_label al on asl.annotationlabel_id = al.id
where JSON_EXTRACT(as2.annotation, '$.type') = 'point'
and as2.FK_prep_id in ('DK55')
and as2.active = 1
and al.label = 'SC'
order by as2.FK_prep_id, au.username, al.label;



select *
from scan_run where FK_prep_id = 'AtlasV8';

update scan_run sr set resolution = 10, zresolution=10 where id = 96;

select * from annotation_session where id = 8288;

select *
from structure_com sc
inner join annotation_session as2 on sc.FK_session_id = as2.id
inner join annotation_session_labels asl  on as2.id = asl.annotationsession_id 
inner join annotation_label al on asl.annotationlabel_id = al.id
where as2.FK_prep_id = 'DK55'
and al.label = 'SC'
order by asl.annotationlabel_id;

select * from animal
order by animal.prep_id; 

desc annotation_session;
delete from annotation_session_labels where annotationsession_id between 8307 and 9339;
delete from annotation_session where id  between 8307 and 9339;
desc neuroglancer_state;

select * from annotation_session where annotation_session.FK_state_id is null
and active = 1;

desc neuroglancer_state;

select * from neuroglancer_state
order by id desc limit 5;
delete from neuroglancer_state where id > 993;

select *
from annotation_session as2  
where id = 8272
order by id desc limit 10;




select * from annotation_label;
desc annotation_label;
-- updates for production may 2025
ALTER TABLE neuroglancer_state ADD COLUMN FK_prep_id VARCHAR(20) DEFAULT NULL AFTER FK_lab_id;
ALTER TABLE neuroglancer_state DROP COLUMN user_date;
-- run python scripts/parse_layer_data.py --task fix_animal
UPDATE neuroglancer_state ns SET readonly = 1 WHERE FK_prep_id IS NULL;
select * 
from neuroglancer_state ns 
where ns.FK_prep_id is null;

show create view v_search_sessions;
drop view v_search_sessions;
CREATE VIEW `v_search_sessions` AS
select
	`AS2`.`id` AS `id`,
	concat(`AS2`.`FK_prep_id`, ' ', group_concat(`AL`.`label` separator ','), ' ', `AU`.`username`, ' ', json_extract(`AS2`.`annotation`, '$.type')) AS `animal_abbreviation_username`,
	`AL`.`label_type` AS `label_type`,
	date_format(`AS2`.`updated`, '%d %b %Y %H:%i') AS `updated`
from
	(((`annotation_session` `AS2`
join `annotation_label` `AL`)
join `annotation_session_labels` `ASL` on
	(`AS2`.`id` = `ASL`.`annotationsession_id`
		and `AL`.`id` = `ASL`.`annotationlabel_id`))
join `auth_user` `AU` on
	(`AS2`.`FK_user_id` = `AU`.`id`))
where
	`AS2`.`active` = 1
	and `AU`.`is_active` = 1
	and `AL`.`active` = 1;
group by
	`AS2`.`id`;

SELECT
	DISTINCT `v_search_sessions`.`id`,
	`v_search_sessions`.`animal_abbreviation_username`,
	`v_search_sessions`.`label_type`,
	`v_search_sessions`.`updated`
FROM
	`v_search_sessions`
WHERE v_search_sessions.animal_abbreviation_username like '%AtlasV8%'
ORDER BY
	`v_search_sessions`.`animal_abbreviation_username` ASC;

select *
from animal where prep_id = 'Allen';


