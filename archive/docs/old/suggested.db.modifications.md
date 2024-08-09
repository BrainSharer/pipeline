## Here are some of my suggested database modifications.
These modifictions are meant to get rid of unused columns and tables and to
also make the naming more generic and applicable to other labs.
### Adding a `lab` table
* `id`       int(11) PRIMARY KEY auto_increment
* `lab_name` varchar(100)
* `active`   tinyint(1)
* `created`  datetime(6)
* `lab_url`  varchar(250)
### Modify `animal table`
1. Add lab column pointing to lab table
1. Remove aliases_1 -> 5 and replace with just one alias column
### Rename `slide_czi_to_tif` to `image_file`
### Drop table `organic_labels`
1. This table has never been used.
### Rename table `com_type` to `input_type`
1. This table describes more data than just centers of mass.
### Rename table `neuroglancer_urls` to `neuroglancer_state`
1. Within this table rename column `url` to `neuroglancer_state`. This column
containst the full JSON state of the Neuroglancer session.



