echo "mkdir data"
mkdir data

echo "cd data"
cd data

echo "Downloading datasets.."
wget "http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz"
wget "http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz"

echo "gzip -dk  cb513+profile_split1.npy.gz"
gzip -dk  cb513+profile_split1.npy.gz
echo "gzip -dk  cullpdb+profile_6133_filtered.npy.gz"
gzip -dk  cullpdb+profile_6133_filtered.npy.gz

echo "cd .."
cd .. 

echo "python corpus_creation.py"
python corpus_creation.py # creates unigram_corpus in data folder.
echo "cd data"
cd data
echo "Cloning glove model from stanfordnlp..."
git clone https://github.com/stanfordnlp/GloVe.git
cd GloVe/
make
echo "cd .."
cd ..
echo "GloVe/build/vocab_count -min-count 1 -verbose 2 < unigram_corpus > vocab_u.txt"
extractedho "GloVe/build/vocab_count -min-count 1 -verbose 2 < unigram_corpus > vocab_u.txt"
GloVe/build/vocab_count -min-count 1 -verbose 2 < unigram_corpus > vocab_u.txt
echo "GloVe/build/cooccur -verbose 2 -symmetric 1 -windows-size 12 -vocab-file vocab_u.txt -memory 8.0 <unigram_corpus> coocur_u.bin "
GloVe/build/cooccur -verbose 2 -symmetric 1 -windows-size 12 -vocab-file vocab_u.txt -memory 8.0 <unigram_corpus> coocur_u.bin 
echo "GloVe/build/shuffle -verbose 2 -memory 8.0 <coocur_u.bin > coocur_shuffled_u.bin"
GloVe/build/shuffle -verbose 2 -memory 8.0 <coocur_u.bin > coocur_shuffled_u.bin
echo "GloVe/build/glove -input-file coocur_shuffled_u.bin -vocab-file vocab_u.txt -save-file vectors_u -verbose 2 -vector-size 100 -alpha 0.75 -x-max 100000 -binary 2 "
GloVe/build/glove -input-file coocur_shuffled_u.bin -vocab-file vocab_u.txt -save-file vectors_u -verbose 2 -vector-size 100 -alpha 0.75 -x-max 100000 -binary 2 
# created vectors_u.txt (glove file) for use by the model. 
# Now lets divide test/train data into batches.
echo "cd .."
cd ..
echo "python utils.py"
python utils.py

echo "python data_checking.py"
python data_checking.py

#Now you can run model_6.py
# python model_6.py