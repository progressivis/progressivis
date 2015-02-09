## installation

UBUNTU:

1. Check python version: (>2.7)
          python --version

2. dependencies
	sudo apt-get install build-essential python-dev python-setuptools \
		python-numpy python-scipy \
		libatlas-dev libatlas3gf-base

3. Fabric
          sudo apt-get install python-pip python-dev build-essential
          sudo pip install fabric

4. Virtual Environment
      sudo pip install virtualenv

5. Inside the progressive-mds
    fab setup

6. Check the site:
    sudo fab runserver
    From Chrome, connect to http://localhost:5000/
