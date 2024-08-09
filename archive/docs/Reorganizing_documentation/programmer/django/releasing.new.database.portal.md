### Using rsync to upload new code to the web server
1. On your local development machine, copy this data into a file called excludefiles.txt one directory above your code base:
<pre>
data
settings.py
*.pyc
.git*
.idea*
parameters.yaml
*__pycache__*
*~
*migrations*
notebooks/.ipynb_checkpoints*
</pre>
1. Login into activebrainatlas.ucsd.edu and backup the current code base that works. Copy or tarball the existing Django setup to: /var/www/backups/ 
1. In your home directory on activebrainatlas.ucsd.edu, do a git clone of the project: `git clone git@github.com:ActiveBrainAtlas2/activebrainatlasadmin.git`
1. Compare the code between the current repo and what is on the server: `diff -rq ~/activebrainatlasadmin /var/www/activebrainatlas/ | grep -v ".git" | grep -v ".pyc"
1. After everything looks as expected, copy the new files from your home repo into the /var/www/activebrainatlas directory. 
1. The code is now on the web server.
1. Login back to activebrainatlas.ucsd.edu and check the code:
    1. `cd /var/www/activebrainatlas`
    1. `source /usr/local/share/activebrainatlas/bin/activate`
    1. `python manage check`
    1. `python manage runserver`
1. Those last two commands should report no errors.
1. You need to reload apache with:
    1. `sudo apachectl configtest`
    1. `sudo apachectl graceful`
1. The server has been reloaded, now go to https://activebrainatlas.ucsd.edu/activebrainatlas/admin and click relevant links to make sure the server is running correctly.