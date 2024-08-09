## Installation of Clearmap without conda
1. install Anaconda: https://www.anaconda.com/products/individual
1. relogin after install to activate with `ssh -Y ratto` The -Y option will
   export the X11 display
1. git clone https://github.com/ChristophKirst/ClearMap2.git
1. cd ClearMap2
1. conda env create -f ClearMap.yml
1. conda activate ClearMap
1. python
1. >>> import ClearMap.Compile
## Installation of Clearmap without conda
1. git clone https://github.com/ChristophKirst/ClearMap2.git
1. create virtualenv `python3 -m venv /usr/local/share/venvs/clearmap` 
1. `source /usr/local/share/venvs/clearmap/bin/activate`
1. `pip install -U pip`   
1. install dependencies   
  - spyder
  - vispy
  - pyopengl
  - natsort
  - tifffile
  - pyqtgraph
  - opencv-python
  - cython
  - matplotlib
  - scipy
  - scikit-image
  - scikit-learn
  - pycairo  
  - pytorch (get command from: https://pytorch.org/)
1. ubuntu installs
    1. apt-get install libboost-all-dev
    1. apt-get install libgmp-dev
    1. apt-get install libcgal-dev 
    1. libcairomm-1.0-dev
libsparsehash-dev
1. install graph-tool
    1. cd ~
    1. git clone https://git.skewed.de/count0/graph-tool.git
    1. cd graph-tool   
    1. ./autogen.sh
    1. ./configure --prefix=$HOME/.local



