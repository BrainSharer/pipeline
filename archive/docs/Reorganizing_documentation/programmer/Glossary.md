This file Contains The Terms used in the Preprocess Pipeline, The Django portal/API and the Abakit.

# Abakit

### Abakit
A python code reposity containg the common tools used in Preprocessing Pipeline and Django API

# Preprocessing Pipeline
### Preprocessing Pipeline:
Python code repository that takes microscopy images in a proprietary format (for example CZI from Zeiss microscope) and convert it to be viewable in neuroglancer.  The neuroglancer viewable files are created using Seung lab CloudVolume format.

The pipeline performs the following steps:
 - Extract the scan images from proprietary formats (e.g. CZI)
 - Prepare previews of the image for Quality Control
 - Clean the images by cropping it from surround debres
 - align the images within a scan so that brain regions adjescent slides are in the same location and orientation in the image
 - Convert the image to the Seung lab CloudVolume Format

# Django API
### Django API
Python code repository defining the Django API and Admin Portal

### API
Application Programming Interface, In this case refers to the set of commands a user or a program can enter into the browser to interact with our database.
The API calls can be used to check existing rows in tables, insert and update as well as delete rows.

API calls are structured as:`API End Point/name_of_function_to_perform/parameters`

There are two version of Django APIs running on public servers right now, providing two Django End Points:
active brain atlas: https://activebrainatlas.ucsd.edu/activebrainatlas
web dev: https://webdev.dk.ucsd.edu/activebrainatlas/

An example of API function would be landmark_list, where the list of all active structures are returned:

https://activebrainatlas.ucsd.edu/activebrainatlas/landmark_list
 
