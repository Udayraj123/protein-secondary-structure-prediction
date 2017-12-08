if ! [ -x "$(command -v pip)" ]; then
	  echo 'Pip is not installed, installing pip for python2... (using sudo if available)' >&2
	
	# string='Sorry, user $USER may not run sudo on gpu-login.'	
	if [[ $(sudo -v) == *"Sorry"* ]]; then
		# no sudo possible, use wget
		echo "Sudo doesn't seem available, using wget";
		wget https://bootstrap.pypa.io/get-pip.py
		python get-pip.py --user
	else
		  sudo apt-get install python-pip
	fi
fi
if ! [ -x "$(command -v pip)" ]; then
	sudo apt-get install gzip
else
	echo "gzip is not installed, please install gzip or extract the npy.gz datasets using alternatives";
#	exit 1
fi

echo "Installing pip packages"
pip install --user numpy pandas sklearn nltk
pip install --user tensorflow
