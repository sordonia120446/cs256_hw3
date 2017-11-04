Jason Chee
Nick Mauthes
Sam Ordonia

Experiment write-up is contained in Hw3.pdf.

Notes about our project:

The following command will produce output as specified in the project description.
	python sticky_snippet_generator.py num_snippets mutation_rate from_ends output_file

Adding a '--label' flag will concatenate the generated string with the label (e.g. 12-STICKY) for the 
corresponding string separated by a comma.
	python sticky_snippet_generator.py num_snippets mutation_rate from_ends output_file --label

Our sticky_snippet_net.py will only use data generated with the '--label' flag added.


model_file is the name of a directory that contains several files that construct our model. For example,
it contains tensorflow checkpoint files and other files that store variables and weights.
	python sticky_snippet_net.py mode model_file data_folder