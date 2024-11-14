## Class: Django
  Very simply, Django provides us the bridge between Neuroglancer and the database. This is mainly done via Django's very powerful and popular
  REST framework. This framework follows the MVC (model, view, controller) design pattern. Django breaks down the MVC framework into modular 'apps'.
  The apps are listed below with a description the the classes that make up the models, views and controllers for each app. Apps for CVAT, User, and
  authentication have not been included as they are not created by us.
1. Apps
    1. Brain
        1. Models
            1.  AtlasModel - an abstract class that handles all common fields (id, updated, created, active)
            1.  Animal - the main class that most of the Django apps use. Describes the animal.
            1.  FileOperation - a class to store metadata on pre-processing pipeline tasks.
            1.  Histology - histology metadata
            1.  Injection - injection metadata
            1.  InjectionVirus - many to many join table class for injections and viruses
            1.  OrganicLabel - organice label metadata
            1.  ScanRun - metadata describing the scan of a brain. This is used in the preprocessing pipeline a lot.
            1.  Slide - metdata describing the scenes or tissues (usually 4) on a microscope slide.
            1.  SlideCziToTif - metadata describing the actual TIFF file produced from the slide.
            1.  Section - An abstract class describing the stack of images that will represent the 3D volume of a brain.

        1. Views
            1. (Taken care of by the Django CRUD (create, retrieve, update and delete) framework: https://activebrainatlas.ucsd.edu/activebrainatlas/admin)
        1. Controllers
            1. Animal
    1. Neuroglancer
        1. Models
            1. AlignmentScore - a class responsible for implementing the Plotly graphs and displaying alignment scores of the COMs.
            1. AnnotationAbstract - Abstract class describing the 3 separate annotation classes:
                1. MarkedCell - annotated cells marked by an anatomist.
                1. PolygonSequence - sets of polygons drawn by an anatomist.
                1. StructureCom - Centers of mass for a given structure.
            1. AnnotationPointArchive - the archived annotation data
            1. AnnotationSession - Metadata describing a session of annotations performed by a user.
            1. ArchiveSet - metadata describing the archived annotation data
            1. BrainRegion - Formerly called structure, this represents an area of the brain.
            1. BrainShape - class for the numpy 3D masks
            1. UrlModel - This is the class that takes care of the Neuroglancer JSON state.
        1. Views
            1. (Taken care of by the REST API < - > Neuroglancer)
        1. Controllers (REST API)
            1. UrlViewSet - allows creation, retrieval, updating of Neuroglancer JSON data 
            1. AlignAtlasView - used by the alignment process to create a rigid transformation
            1. Annotation - deals with one set of annotations
            1. Annotations - fetches all sets of annotations
            1. Rotation - deals with one rotation
            1. Rotations - fetches all sets of rotations
