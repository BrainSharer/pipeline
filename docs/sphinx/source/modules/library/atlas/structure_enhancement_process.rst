How to edit existing atlas and brain structures
-----------------------------------------------

This document will describe the steps used to edit existing structures.

The existing atlas and brain structures have been saved as polygons in the database. Users can now open a neuroglancer view
and import the polygons, view, edit, and then save them. Here are the steps to perform this process.

Opening neuroglancer and importing the polygons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Open up this neuroglancer view: https://brainsharer.org/ng/?id=989
#. Create a new annotation layer by holding ctrl key and clicking the ``+`` button near the top of the window.
#. On the newly created right side panel, click the ``Annotations`` tab.
#. In the text box that says: 'Search for annotations' type ``MD594 SC`` to search for the SC structure for MD594. When you search, it may take
   a few seconds to load the options. Since we are looking for volumes, make sure the drop down menu contains 'volume'. You can also check
   who created the structure, the ID, and when it was updated.
#. Click the option you wish to import. You should notice a message in the bottom left corner of neuroglancer saying the data is being imported.
   You should see the imported polygons showing in the bottom right panel.
#. Go to a section that has polygons either by scrolling throught the ``Z`` axis or by clicking one of the diagonal arrows in the row of polygons in the
   lower right panel.

Keyboard shortcuts for editing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. note::
   Make sure you have selected the volume by clicking on the mauve colored volume box in the upper right panel. And make sure
   the points are big enough to drag. A point size of at least 5 is recommended.

Vertices
~~~~~~~~

* To move a vertex, hold the ``alt`` key and click and drag the vertex.
* To delete a vertex, hold the ``shift`` key and click the right side mouse button.
* To add a vertex, hold the ``shift`` key and double click the the mouse button on the line in the position you want.

Polygons
~~~~~~~~

* To move a polygon, hold the ``alt`` and ``shift`` key and click and drag the polygon from a vertex.
* To delete a polygon, go to the lower right panel, find the coordinates on the section you want, and click the 'Trash' can icon.
* To copy a polygon, go to the lower right panel, find the coordinates on the section you want, and click the 'Copy' icon, go to the section where
  you want to paste it, then click the 'Paste' icon (immediately below where it says 'child annotations')
* To rotate a polygon, press ``shift`` ``r`` to rotate clockwise and ``shift`` ``e`` to rotate counter clockwise.
* To enlarge a polygon, press ``shift`` ``=`` to scale up and ``shift`` ``-`` to scale down. 


Exporting the polygons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. note::
   You can save the entire neuroglancer view by clicking the ``Save`` icon in the top left window. But you probably want
   to export the annotations to the database. This way, we can use the newly edited polygons to modify the existing structure for the atlas.

* To export the newly edited polygons, go to the lower panel on the right side. Look for the ``save`` icon on the far right and click that.
  This will overwrite the existing polygons and perform an update in the database.
* If you want to create a new set of data, click the ``new`` icon instead of the ``save`` icon.
