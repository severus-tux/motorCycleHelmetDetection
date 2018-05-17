# NOTE:
# Don't run this file as is,
# this is just a dump of the scripts used for data cleaning
# for cascade generating


# Generating bikes.info
for i in ./pos/* 
do
	size=`identify -format '%w %h' $i`
	echo -en "$i 1 0 0 $size\n" >>bikes.info
done

# Generating bg.txt

for i in ./neg/*
do
	echo $i >>bg.txt
done

# opencv_createsamples
# Detailed description of opencv_createparameters here : http://manpages.ubuntu.com/manpages/xenial/man1/opencv_createsamples.1.html

opencv_createsamples -info bikes.info -num 1000 -w 50 -h 50  -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -vec bikes.vec

# opencv_traincascade
# https://docs.opencv.org/2.4.13/doc/user_guide/ug_traincascade.html

opencv_traincascade -data data -vec bikes.vec -bg bg.txt -numPos 400 -numNeg 650 -numStages 14 -w 50 -h 50 -precalcValBufSize 4096 -precalcIdxBufSize 4096 -featureType LBP

# NOTE : calculation of numPos and other parameters : https://stackoverflow.com/questions/10863560/haar-training-opencv-assertion-failed