nname='torch_env'

# create env inside container
python3 -m venv $nname --system-site-packages
source ${nname}/bin/activate

# install wheels -- from this point on, feel free to add anything
pip3 install -r reqs.txt

# modify l.4 of /torchnlp/_third_party/weighted_random_sampler.py
var='int_classes = int'
sed -i "4s|.*|$var|" \
  $PWD/${nname}/lib/python3.8/site-packages/torchnlp/_third_party/weighted_random_sampler.py
