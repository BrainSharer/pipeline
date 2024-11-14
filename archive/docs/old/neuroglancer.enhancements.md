## Enhancements since November 2020
### Neuroglancer front end
1. Usernames are logged in the top text area box.
1. Firebase added for storing URL data in JSON format.
1. A log-in button will be displayed on neuroglancer if the user is not logged in. It will take the user to the log-in page and redirect the user back after a successful log-in.
1. Multi-user mode allowing real-time sharing of data among multiple users.
    1. Only the logged-in users can use the multi-user mode. Notification will be made if a visitor is not logged in.
    1. When using the multi-user mode, any change will be automatically saved in the firebase and reflected on the other user's webpage in real-time, as long as they are also in the multi-user mode.
    1. When using the multi-user mode, users can still save or create new URLs to our database (which will be available under the URL section of the database portal.
    1. When using the multi-user mode, users will be able to see anyone that is active on the same page.
    1. When using the multi-user mode, users can reset their changes to the last manual save.
1. The users can import annotation layer data from another url.
1. A log-scaled histogram of the visible area in the current layer is displayed in the rendering tab. Users will be able to adjust the intensity by dragging the two sides of the histogram.
1. Neuroglancer can now align the current layer to the active brain atlas.
    1. Users can now choose the data used for aligning the layer to the active brain atlas by selecting from the dropdown menu in the source tab. The rotation matrix is fetched from the server and then applied to the neuroglancer.
### Database portal back end enhancements
1. REST API enhancements
    1. Backend server now automatically extracts and saves the COM (center of mass) data from layers when a URL is updated or created through the API
    1. COM data for Automatical alignment is now available in dropdown menus in Neuroglancer.
    1. Annotation layer data are now available in dropdown menus in Neuroglancer.
[//]:    1. Rotation transformation matrices are available in dropdown menus in Neuroglancer.
1. Easy viewing of histogram data for every image and every animal in the database portal.
1. Storage of Neuroglancer URL data in JSON format in the Mysql backend with easy viewing.
