#! /bin/bash
set -e

echo "learning byte pair encoding"
cat $@ | sed -e 's/\xA0/ /g' | subword-nmt learn-bpe -s 100 > ../vocab/100_codes
echo "applying byte pair encoding"
cat $@ | sed -e 's/\xA0/ /g' | subword-nmt apply-bpe -c ../vocab/100_codes > ../vocab/100_all.txt
subword-nmt get-vocab < ../vocab/100_all.txt > ../vocab/100_vocab.txt
echo "learning word vectors"
../fasttext/fasttext skipgram -input ../vocab/100_all.txt  -output ../vocab/100_all_dim128 -minCount 1 -wordNgrams 5 -minn 1 -maxn 10 -dim 128 -epoch 25
