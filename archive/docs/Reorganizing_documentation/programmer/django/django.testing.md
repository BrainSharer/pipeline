## Testing in Django
1. Go to your Django activebrainatlas or brainsharer_portal installation directory
1. Do: `source /usr/local/share/pipeline/bin/activate`
1. Do: `git pull origin master`
1. Do: `python manage.py check`
1. Do: `python manage.py test neuroglancer --keepdb` This will run all tests in
neuroglancer/tests.py. All methods that start with *test_* are run. If there is a
*setup* method, that is run before each test. Each test runs on its own and has
nothing to do with any other *test_* method. (Except the *setup* method)
1. For this to work, you will need access to the *test_active_atlas_development*
database. The dk user has the necessary permissions.
1. Create new test methods in the same file. For any test, simply name it *test_my_existing_method*
. Each method must start with *test_*