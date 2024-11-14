# Transferring data between CVAT and Neuroglancer and back
## Neuroglancer to CVAT
1. All data is in the same database, active_atlas_production
1. To get data from neuroglancer into CVAT you need to open up your task in CVAT. Once you do that, you can get the JOB ID and the Task ID. When you are in CVAT, look at the URL in the browser's location bar. Here is an example: `http://muralis.dk.ucsd.edu/tasks/42/jobs/32` In this example, the 42 is the Task ID and 32 is the JOB ID. You'll need these two numbers to transfer data.
1. Open up a database session either on the command line or in a database GUI. 
1. Use this command:
<pre>
insert into engine_labeledshape (frame, `group`, `type`, occluded, z_order, points, job_id, label_id) 
select
LD.section as frame, 0 as `group`, 'points' as `type`, 0 as occluded, 0 as z_order, concat(LD.x/32,',',LD.y/32), 32 as job_id, elab.id as label_id
from layer_data LD
inner join engine_label elab on LD.layer=elab.name
where LD.prep_id  = 'DK52'
and LD.layer = 'Premotor'
and elab.task_id  = 42
`order by LD.section;`
</pre>