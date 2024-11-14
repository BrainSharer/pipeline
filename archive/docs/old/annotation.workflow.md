## Current flow of annotation adding,updating and deleting
1. When the user hits 'Save' all data is saved as JSON in the neuroglancer_urls
table. The following layers that are labeled below are entered into the 
database table layer_data. Any other layer containing annotations are NOT 
entered into the layer_data table.
    1. COM
    1. ADDITIONAL MANUAL ANNOTATIONS

## This is the proposed flow of how annotation layers are added to the database
### New set of annotations
1. The user opens up an annotation layer in Neuroglancer and adds data. 
When the user hits 'Save', the program will enter all data under that layer,
convert it to micrometers and insert into the database table: layer_data.
The only exception is if the user does not change the layer name from the default:
'annotation'. If the user wants the data saved, the user `must` change the name
from 'annotation' to something more meaningful.
### Keeping track of when and who performed any creations, updates
1. The layer_data table has these fields:
    1. active - if the data should be used and displayed in Neuroglancer
    1. person - this column keeps track of who created the row.
    1. created - this column keeps track of when the row was created.
    1. updatedby - this column keeps track of who updated the column.
    1. updated - this column keeps track of when the row was updated.
### Existing annotations
1. When the user hits the 'Save' button in the same Neuroglancer view,
the program will:
    1. Take the existing annotation data in that layer and mark it as
inactive. (active=False)
    1. Mark each existing row with the current timestamp. (updated=current timestamp)
    1. Mark each existing row with the person doing the updating. (updatedby=current user)
    1. Insert all data and mark it with the current timestamp and the person
    who entered the data. (person=current person, created=current timestamp) 
    1. This will occur for all layers.
1. For the large annotation layer like 'DK39 premotor' where there are 2,200
annotations, this will take a few seconds to save. Take note! For this instance,
whenever the user hits 'Save', there will be an additional 2,200 inactive DK39 premotor
annotations. This is not a problem as the database can easily handle millions
of rows, but the user needs to be aware of this fact when searching.
### Existing annotation with a different user
1. This is very similar to the above step, except a different user's ID will be
logged in the creation and update fields.
1. When a different user comes into the same Neuroglancer view and edits the
same annotation data, the program will:
    1. Take the existing annotation data in that layer and mark it as
inactive.
    1. Mark each existing row with the current timestamp.
    1. Mark each existing row with the current person doing the updating.
    1. Insert all data and mark it with the current timestamp and the person
    who entered the data. 
    1. This will occur for all layers.
### Deleting data
1. If a user deletes an annotation layer in Neuroglancer and clicks 'Save', the
database will still retain that data. If the data needs to be purged for good,
the programmer will need to do that.
1. If a user wants to delete one or more annotations in the layer by clicking
the `Trash` icon next to the row, the process will be as above, 
but the number of active annotations in the database will be less.
