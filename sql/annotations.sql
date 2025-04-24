-- volumes
select as2.id, as2.FK_prep_id, au.username, al.label, length(as2.annotation), as2.created, as2.updated
from annotation_session as2
inner join auth_user au on as2.FK_user_id = au.id
inner join annotation_session_labels asl on as2.id = asl.annotationsession_id
inner join annotation_label al on asl.annotationlabel_id = al.id
where JSON_EXTRACT(as2.annotation, '$.type') = 'volume'
-- and as2.FK_prep_id in ('DK55')
and as2.active = 1
-- and al.label = 'SC'
and as2.id = 8061
order by as2.FK_prep_id, au.username, al.label;
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

desc annotation_session;