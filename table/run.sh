conda create -n rapids-22.08 -c rapidsai -c nvidia -c conda-forge  \
    rapids=22.08 python=3.9 cudatoolkit=11.5

conda activate rapids-22.08

pip install -r requirements.txt