## Process to setup up web site for brainsharer.org

1. Create a common google account we can all share.
1. Buy the name [brainsharer.og](https://domains.google.com/registrar/search?searchTerm=brainsharer.org).
This will cost $12/year.
1. Create a common AWS account for the brainsharer project.
1. Create a [Ubuntu AWS EC2](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-LAMP.html)
 with web site, mysql instance and storage.
1. Point the google nameservers at our AWS web site.
1. Create a basic static web page. We could put webpress on it if we think we
will be producing more pages with the ability for other people to post replies.
1. Install the Django portal.
1. Add google oauth so people can easily login.
1. Get a https certificate