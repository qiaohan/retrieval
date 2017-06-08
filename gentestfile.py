import os

rootdir = "benckmark/"

for parent,dirnames,filenames in os.walk(rootdir):
	for filename in filenames:
		print os.path.join(parent,filename)
