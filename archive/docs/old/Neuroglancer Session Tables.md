### Examles of json elements `#` are comments made by me
### Essential neuroglancer json content:
```
{
  "layers": [#List of layers, could be image, segmentation or annotation],
  #extra fields can exist for specifying neuroglancer state,resolution and origin etc
}
```
### Essential Image Layer content:
```
{
      "type": "image",
      "source": #url pointing to local or hosted data,
      "name": #layer name
      #extra fields can exist for widget settings, transformations and layer state
}
```

### Essential Segmentation Layer content:
```
{
      "type": "segmentation",
      "source": #url pointing to local or hosted data,
      "name": #layer name
      #extra fields can exist for widget settings, transformations and layer state
}
```

### Essential Annotation Layer content:
```
{
      "type": "annotation",
      "annotations": [
        {
          "point": #coordinate of points,
          "type": "point", # annotation type
          "id": "5b78ff277f502fd269d10afe797ec68854c0ca17", # this is a random id, if left blanck will be generated automatically
          "description": #string the describes the annotation
          }
          {
          "pointA": #coordinate of starting point,
          "pointB": #coordinate of ending point,
          "type": "axis_aligned_bounding_box",
          "id": "6d5331c008c219f9bfa71c02a094666948631492"
          },
        {
          "pointA": #coordinate of starting point,
          "pointB": #coordinate of ending point,
          "type": "line",
          "id": "44c9dfc48e1e6dbdbeafc1fd7a44e9a9619f5b74"
        },
        {
          "center": #coordinate of ellipsoid center,
          "radii": #ellipsoid radii in x,y,z direction,
          "type": "ellipsoid",
          "id": "31167500b4f832a52db0393b4f325624a029b312"
        },
      "name": "annotation"
}
```

I am proposing the creation of five tables that will replace the current way of storing the json neuroglancer sessons in the database:

### Table1 Neuroglancer Sessions
|Column Name | data type | description|
|--|--|--|
| id |int | primary key |
| session_name | string | name of the session | 
| state_json | json | Remainder of the json that logs the state of current viewing session eg: location zoom level etc |

### Table2 Neuroglancer Layers
|Column Name | data type | description|
|--|--|--|
|id|int|primary key|
|session_name|string|name of the session|
|layer_name|string|name of the layer|
|layer_type|string|'image','segmentation' or 'Annotation'|
|post_fixing_json|json|all the flexible json fields|

### Table3 Image Layers  
**YF: is there support for variable transparency? 
This is useful when creating layers with computer generated annotations, such as directionality maps.**
|Column Name | data type | description|
|--|--|--|
|id|int|primary key|
|source|string|address where the public layer is hosted|
|(optional)||
|lab|string|laboratory hosting the data|

### Table4 Segmentation Layers
|Column Name | data type | description|
|--|--|--|
|id|int|primary key|
|source|string|address where the public layer is hosted|
|(optional)||
|lab|string|laboratory hosting the data|


/* YF: I think we need to support: x,y,z (coms), contours and 3D meshes.
/* there is no need for box, line or ellipsoid.
/* There should be a separate table for each of the supported types of annotations.

### Table5 Annotation Layers
|Column Name | data type | description|
|--|--|--|
|id|int|primary key|
|x|int|x coordinate|
|y|int|y coordinate|
|z|int|z coordinate|
|comment|string/NA|text comment of the annotation|
|layer_id|int|layer id where the point annotation belong to|
|session_id|int|session id where the point annotation belong to|
|annotation_type|string|'point','box','line' or 'elipsoid'|
|x2|int|x coordinate of ending point if applicable|
|y2|int|y coordinate of ending point if applicable|
|z2|int|z coordinate of ending point if applicable|
|radius_x|int|x radius of elipsoids if applicable|
|radius_y|int|y radius of elipsoids if applicable|
|radius_z|int|z radius of elipsoids if applicable|

### Steps to recreate a neuroglancer session:
 1. Get all entries in the Neuroglancer Layers table with a session_name.  The state_json field has the json file with an empty entry for the layers field
 2. Search through the Neuroglancer Layers table and insert each layers accordingly    /* YF: can the user choose the layers they want?
 3. For Image and Segmentation layers, get layer name from Neuroglancer Layers table, and source from the Image Layers or Segmentation Layers table and insert them into a template.  Then the postfixing json values are attached.  These could include the contract and color widget, current layer state etc.
 4. For Annotation layers points are collected by the combination of session and layer id.  Then a Annotation layer is created from a template
