# Make directories
mkdir data/
mkdir test/

# Generate training data

python sticky_snippet_generator.py 2500 0 0 ./data/out1.txt
python sticky_snippet_generator.py 2500 0 0 ./data/out2.txt
python sticky_snippet_generator.py 2500 0 0 ./data/out3.txt
python sticky_snippet_generator.py 2500 0 0 ./data/out4.txt

# Generate test data

python sticky_snippet_generator.py 2500 0 0 ./test/out5.txt
python sticky_snippet_generator.py 2500 0 0 ./test/out6.txt
python sticky_snippet_generator.py 2500 0 0 ./test/out7.txt

