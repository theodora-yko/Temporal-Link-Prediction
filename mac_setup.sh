#!/bin/bash

# setup dir
mkdir ~/src && cd ~/src
mkdir ~/.localpython

# install python
wget https://www.python.org/ftp/python/3.8.13/Python-3.8.13.tgz
tar -zxvf Python-3.8.13.tgz
cd Python-3.8.13
./configure --prefix=$HOME/.localpython
make && make install

# setup proj dir
cd ~/Documents
git clone https://github.com/theodora-yko/Temporal-Link-Prediction.git
cd Temporal-Link-Prediction 

# setup virtualenv
PY=$HOME/.localpython/bin/python3
$PY -m venv venv
source venv/bin/activate

# pip install
$PY -m pip install stellargraph chardet
