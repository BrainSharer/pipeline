## Installing and using Supervisord
### Ubuntu
1. Install: `sudo apt update && sudo apt install supervisor`
1. Check status: `sudo systemctl status supervisor`
1. Get location of the manage.py script in the activebrainatlas installation and
add it to the conf file below:
<pre>
[program:backgrounder]
user=apache
command=/usr/local/share/activebrainatlas/bin/python /var/www/activebrainatlas/manage.py process_tasks
autostart=true
autorestart=true
startretries=5
numprocs=1
startsecs=0
process_name=%(program_name)s_%(process_num)02d
stderr_logfile=/var/log/supervisor/%(program_name)s_stderr.log
stderr_logfile_maxbytes=10MB
stdout_logfile=/var/log/supervisor/%(program_name)s_stdout.log
stdout_logfile_maxbytes=10MB
</pre>
1. Save the contents of the above to /etc/supervisor/conf.d/activebrainatlas.conf
1. Reread the conf dir: `sudo supervisorctl reread`
1. Enact changes: `sudo supervisorctl update`
1. For more information see: [Supervisor on Ubuntu](https://www.digitalocean.com/community/tutorials/how-to-install-and-manage-supervisor-on-ubuntu-and-debian-vps)
1. The background tasks for moving and inserting data in the annotation tables will now be run in a queue. The
moving method is run first with a delay of 0 seconds and then the inserting is delayed 60 seconds later.
1. You can check for errors and logging in the */var/log/supervisor* directory.
### These instructions should be very similar with the Centos web server.
#### installing on centos
1. https://cloudwafer.com/blog/installing-supervisor-on-centos-8/
#### configuring on centos
1. The centos version looks for *.ini files in /etc/supervisord.d/ I don't know why Ubuntu uses *.conf and Centos
uses *.ini. Who knows?
1. You will have to edit paths in the data above, obviously replacing the bit named: *REPLACEMEWITHREALUSER*
and *PATHTOVIRTUALENV*
1. After you correct the *.ini files, do a: `systemctl restart supervisord` to reread the config files and restart.

