updating the date packages: 

in Conda :

conda update -all

in python:


pip list --outdated
pip freeze | %{$_.split('==')[0]} | %{pip install --upgrade $_}
