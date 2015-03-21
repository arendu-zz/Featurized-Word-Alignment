rm *.so
rm *.c
python setup.py build_ext --inplace
python setup2.py build_ext --inplace
