slurm download:
https://download.schedmd.com/slurm/slurm-22.05.2.tar.bz2

munge download:
https://github.com/dun/munge/releases/download/munge-0.5.15/munge-0.5.15.tar.xz

slurm installation

useradd  -u 9911 munge

useradd  -u 9912 slurm

download munge, unpack and cd into directory

./configure --prefix=/ \
--sysconfdir=/etc \
--localstatedir=/var \
--runstatedir=/run

do make and make install

systemctl enable munge

system start munge

if failed do
system status munge

create directory as needed, the directories should be owned by munge user
if system status munge was not informative, run sudo -u munge /sbin/munged to debug
copy munge.key from activebrainatlas:/var/munge/munge.key, set access to the same as activebrainatlas

test:
munge -n | ssh <to activebrainatlas> unmunge

repeat with slurm

tar --bzip -x -f slurm*tar.bz2
  
slurm config:
./configure --prefix=/ --sysconfdir=/etc
  
make, make install
  
copy /etc/systemd/system/slurmctld.service from activebrainatlas
  
run systemctl enable slurmctl for control nodes and slurmd for compute nodes
  
copy slurm config from activebrainatlas:/etc/slurm.conf add new computer as needed and update conf file for all computers in the cluster
  
test:
/sbin/slurmctld -D
or
/sbin/slurmd -D

copy all munge related files: *munge* from activebrainatlas  /lib/slurm
