Comments:
* Class can be extended to multiple classes, potentially with
  inheritence.
* What are listed here as classes might be expanded to modules containing
  multiple classes.


## Class: Database bridge
Connects to database back-end. Provides
interfaces for reading/writing/add/removing from the different tables.

Class should hide the details of the implementation (sqlalchemy,
MySQL,...) so that transitioning to a different database backend
requires changes only in this class and not in others.

## Class: 3DObject
implements different representations of a 3D object and
transformations between them:

* Polygon-Sequence
* 3DMask
* Encoded (compressed) 3DMask

## Class: Transformation
Implements transformations between coordinate systems. Sub-classes for
different transformations (Rigid, Rigid+scaling, affine, b-spline, ...)

_init_(transformation_type,transformation_parameters)

attributes:
transformation_type,
forward transform,
inverse transform.

Methods

* forward transformPoints (points) transforms points from source to destination.
* inverse transformPoints (points) transforms points from destination to source.
* forward transform3DMask (3Dobject) transforms 3Dmask from source to destination.
* inverse transform3DMask (3Dobject) transforms 3Dmask from destination to source.
