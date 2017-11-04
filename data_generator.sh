# Flush directories
rm -r data*/
rm -r test*/

# Make directories
mkdir data1/
mkdir data2/
mkdir data3/
mkdir data4/

mkdir test1/
mkdir test2/
mkdir test3/

# Generate training data

python sticky_snippet_generator.py 5000 0 0 ./data1/out1.txt
python sticky_snippet_generator.py 5000 0 0 ./data1/out2.txt
python sticky_snippet_generator.py 5000 0 0 ./data1/out3.txt
python sticky_snippet_generator.py 5000 0 0 ./data1/out4.txt
python sticky_snippet_generator.py 5000 0 0 ./data1/out5.txt
python sticky_snippet_generator.py 5000 0 0 ./data1/out6.txt

python sticky_snippet_generator.py 10000 0 0 ./data2/out1.txt
python sticky_snippet_generator.py 10000 0 0 ./data2/out2.txt
python sticky_snippet_generator.py 10000 0 0 ./data2/out3.txt
python sticky_snippet_generator.py 10000 0 0 ./data2/out4.txt
python sticky_snippet_generator.py 10000 0 0 ./data2/out5.txt
python sticky_snippet_generator.py 10000 0 0 ./data2/out6.txt

python sticky_snippet_generator.py 20000 0 0 ./data3/out1.txt
python sticky_snippet_generator.py 20000 0 0 ./data3/out2.txt
python sticky_snippet_generator.py 20000 0 0 ./data3/out3.txt
python sticky_snippet_generator.py 20000 0 0 ./data3/out4.txt
python sticky_snippet_generator.py 20000 0 0 ./data3/out5.txt
python sticky_snippet_generator.py 20000 0 0 ./data3/out6.txt

python sticky_snippet_generator.py 60000 0 0 ./data4/out1.txt

# Generate test data

python sticky_snippet_generator.py 5000 0.2 0 ./test1/out1.txt
python sticky_snippet_generator.py 5000 0.2 0 ./test1/out2.txt
python sticky_snippet_generator.py 5000 0.2 0 ./test1/out3.txt
python sticky_snippet_generator.py 5000 0.2 0 ./test1/out4.txt
python sticky_snippet_generator.py 5000 0.2 0 ./test1/out5.txt
python sticky_snippet_generator.py 5000 0.2 0 ./test1/out6.txt

python sticky_snippet_generator.py 5000 0.4 0 ./test2/out1.txt
python sticky_snippet_generator.py 5000 0.4 0 ./test2/out2.txt
python sticky_snippet_generator.py 5000 0.4 0 ./test2/out3.txt
python sticky_snippet_generator.py 5000 0.4 0 ./test2/out4.txt
python sticky_snippet_generator.py 5000 0.4 0 ./test2/out5.txt
python sticky_snippet_generator.py 5000 0.4 0 ./test2/out6.txt

python sticky_snippet_generator.py 5000 0.6 0 ./test3/out1.txt
python sticky_snippet_generator.py 5000 0.6 0 ./test3/out2.txt
python sticky_snippet_generator.py 5000 0.6 0 ./test3/out3.txt
python sticky_snippet_generator.py 5000 0.6 0 ./test3/out4.txt
python sticky_snippet_generator.py 5000 0.6 0 ./test3/out5.txt
python sticky_snippet_generator.py 5000 0.6 0 ./test3/out6.txt

