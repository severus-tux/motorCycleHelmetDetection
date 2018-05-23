#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/bgsegm.hpp>
#include <opencv2/video.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

#include <iostream>
#include <iomanip>
#include <fstream>		// file utils
#include <ctime>		// timestamp

#include "Blob.h"
#include "MotorBike.h"

// global variables
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);
const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);


cv::Mat frame, fgMask, frameCopy, frameCopy2;
cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
cv::Mat structuringElement15x15 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

cv::HOGDescriptor hog_bike;
cv::CascadeClassifier cascade_helmet;
cv::CascadeClassifier cascade_head;

// function prototypes 
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs);
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &index);
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs);
double distanceBetweenPoints(cv::Point point1, cv::Point point2);
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, std::vector<Blob> &crossedBlobs, int &intVerticalLinePosition, std::ofstream &logfile);


int main(int argc, char* argv[])
{
	if( argc != 2 )
	{
		std::cerr << "Sorry! incorrect usage!\n"
				  << "Usage : startDetection <path to surveillance feed>\n"
				  << "Example : startDetection video/traffic.mp4\n"
				  << "Example : startDetection http://192.168.1.102:8080/video\n";
		return 1;
	}
	
	int HelmetCount=0, headCount=0;
	cv::VideoCapture capVideo;
	std::ofstream logfile; // log file
	std::vector<Blob> blobs;
	cv::Point crossingLine[2];

	cv::Ptr<cv::BackgroundSubtractor> fg;
	fg=cv::createBackgroundSubtractorMOG2(500,16,false);//CNT(1,false,30*60,true);
	capVideo.open(argv[1]);


	// log file
	logfile.open ("LOG-" + std::to_string(time(0)) + ".txt");
	std::cout << "Logging to: \"LOG-" << std::to_string(time(0)) << ".txt\"" << std::endl;

	logfile << "\"Timestamp\t\", \"Direction\", \"Vehicle Type\", \"Triple Riding?\", \"Wearing Helmet\"" << std::endl;
	std::clog << "| Timestamp " << std::setw(25) <<"| Direction "<< std::setw(13) << "| Vehicle Type "
			  << std::setw(14) <<"| Triple Riding " <<std::setw(15)<< "| Rider Count" << std::setw(15) << "| Helmet Count" << std::endl;

	if (!capVideo.isOpened())
	{                                                 // if unable to open video file
		std::cerr << "error accessing video file/ remote camera\n" << std::endl << std::endl;      // show error message
		return 1;                                                              // and exit program
	}

	if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2)
	{
		std::cerr << "error: video file must have at least two frames\n";
		return 1;
	}
	
	if(!hog_bike.load( "../cascade/bikes.yml" ))
	{
		std::cerr << "error: could not load the cascade file bike.yml\n";
		return 1;
	}
	
	if(!cascade_helmet.load( "../cascade/cascade.xml" ))
	{
		std::cerr << "error: could not load the cascade file helmet.xml\n";
		return 1;
	}
	
	if(!cascade_head.load( "../cascade/cascade_head.xml" ))
	{
		std::cerr << "error: could not load the cascade file head.xml\n";
		return 1;
	}

	int frame_rows = capVideo.get(CV_CAP_PROP_FRAME_HEIGHT);
	int frame_cols = capVideo.get(CV_CAP_PROP_FRAME_WIDTH);
	int verticalLinePosition = (int)std::round((double)frame_cols * 0.50);

	crossingLine[0].y = 0;
	crossingLine[0].x = verticalLinePosition;
	crossingLine[1].y = frame_rows -1 ; // last pixel *index*
	crossingLine[1].x = verticalLinePosition;

	char checkForEscKey = 0;
	bool firstFrame = true;
	int frameCount = 2;

	while (capVideo.isOpened())
	{
		std::vector<Blob> currentFrameBlobs;
		std::vector<Blob> crossedBlobs;
		std::vector<MotorBike> bikes;
		
		std::vector<std::vector<cv::Point> > contours;
		
		capVideo.read(frame);
		if (frame.empty())
		{
			time_t now = time(0);
			char* dt = strtok(ctime(&now), "\n");
			std::cout << dt << ",EOF" << std::endl;
			logfile << dt << ",EOF" << std::endl;
			logfile.close();
			std::cout << "Video input ended\nSaving log file...\nExiting..\n";
			return(0);
		}
		
		frameCopy = frame.clone();
		frameCopy2 = frame.clone();

		cv::cvtColor(frameCopy, frameCopy, CV_BGR2GRAY);
		cv::GaussianBlur(frameCopy, frameCopy, cv::Size(9, 9), 0);
//		cv::medianBlur(frameCopy, frameCopy, 5);

		fg->apply(frameCopy,fgMask,-1);
		
		//cv::imshow("fgMask", fgMask);

		cv::Mat fgMaskCopy = fgMask.clone();
		cv::morphologyEx(fgMaskCopy,fgMaskCopy,cv::MORPH_OPEN,structuringElement3x3,cv::Point(-1,-1),3,cv::BORDER_CONSTANT);
		cv::dilate(fgMaskCopy, fgMaskCopy, structuringElement5x5);
		cv::dilate(fgMaskCopy, fgMaskCopy, structuringElement5x5);
		cv::findContours(fgMaskCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		std::vector<std::vector<cv::Point> > convexHulls(contours.size());

		for (unsigned int i = 0; i < contours.size(); i++)
		{
			cv::convexHull(contours[i], convexHulls[i]);
			cv::drawContours(fgMaskCopy, contours, i, SCALAR_WHITE, -1);
		}
		
//		cv::drawContours(fgMaskCopy,convexHulls,-1,SCALAR_WHITE,-1);		
	//	cv::imshow("fgMaskCopy", fgMaskCopy);
		
		for (auto &convexHull : convexHulls)
		{
			Blob possibleBlob(convexHull);

			if (possibleBlob.currentBoundingRect.area() > 400 &&
			possibleBlob.currentAspectRatio > 0.2 &&
			possibleBlob.currentAspectRatio < 4.0 &&
			possibleBlob.currentBoundingRect.width > 40 &&
			possibleBlob.currentBoundingRect.height > 40 &&
			possibleBlob.currentDiagonalSize > 60.0 &&
			(cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.50)
			{
				currentFrameBlobs.push_back(possibleBlob);
				cv::rectangle(frameCopy2, possibleBlob.currentBoundingRect, SCALAR_RED, 2);
			 //void rectangle(Mat& img, Rect rec, const Scalar& color, int thickness=1, int lineType=8, int shift=0 )
			}
		}
		

		if (firstFrame == true)
			for (auto &currentFrameBlob : currentFrameBlobs)
				blobs.push_back(currentFrameBlob);

		else
			matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);

		checkIfBlobsCrossedTheLine(blobs, crossedBlobs, verticalLinePosition, logfile);
		//Clearig un-interesting blobs
		currentFrameBlobs.clear();
		currentFrameBlobs.shrink_to_fit();
		
		for( auto &myBlob : crossedBlobs)
		{
			MotorBike mb(myBlob); //Copy Constructor
			
			std::string dir;
			
			if ( myBlob.directionLeft )
				dir = "left  ";
			else
				dir = "right ";
			
			logfile << myBlob.crossTime << "\t" << dir ;
			std::clog << myBlob.crossTime << "\t" << dir ;
			
			if(myBlob.classifyMotorBike(frame, hog_bike) == true)
			{	
				headCount=mb.countRiders(frame, cascade_head);
				HelmetCount=mb.detectHelmet(frame, cascade_helmet);
				bikes.push_back(mb);
				
				if(headCount<1)
					headCount=1;
					
				if(HelmetCount>headCount)
					HelmetCount=headCount;
				
				//for ( int j = 0; j < mb.detectionsHead.size(); j++ )
				//	cv::rectangle(frameCopy2, mb.detectionsHead[j], SCALAR_BLUE, 2);
				
				//for ( int j = 0; j < mb.detectionsHelmet.size(); j++ )
				//	cv::rectangle(frameCopy2, mb.detectionsHelmet[j], SCALAR_GREEN, 2);
				
				//Remove the following code (2 lines) later, Implement GUI --> Shreyas
				//cv::Mat ROI = frame(mb.currentBoundingRect);
				//mb.currentBoundingRect.height=(int)mb.currentBoundingRect.height*0.25;
				//cv::Mat ROI_top = frame(mb.currentBoundingRect);
				//cv::imwrite("./../Blobs/bike/"+std::to_string(time(0))+".jpg",ROI);
				//cv::imwrite("./../Blobs/bike25/top-"+std::to_string(time(0))+".jpg",ROI_top);
				logfile << "  Bike - helmet count = " << HelmetCount << " , " << "rider count = " << headCount << "\n" ;
				std::clog << " \033[32;1m Bike - helmet count = " << HelmetCount << " , " << "rider count = " << headCount << "\n\033[0m" ;
			}
			
			else
			{
				//cv::Mat ROI = frame(myBlob.currentBoundingRect);
				//myBlob.currentBoundingRect.height=myBlob.currentBoundingRect.height*0.25;
				//cv::Mat ROI_top = frame(myBlob.currentBoundingRect);
				//cv::imwrite("./../Blobs/others/"+std::to_string(time(0))+".jpg",ROI);
				//cv::imwrite("./../Blobs/others25/top-"+std::to_string(time(0))+".jpg",ROI_top);
				logfile << " Other\n";
				std::clog << " \033[31;1m Other\n\033[0m";
			}
		}
		
		for( int i=0; i<blobs.size(); i++)
		{
			if(blobs[i].stillBeingTracked == false)
				blobs.erase( blobs.begin() + i );
		}
		blobs.shrink_to_fit();
		
		bikes.clear();
		bikes.shrink_to_fit();
		crossedBlobs.clear();
		crossedBlobs.shrink_to_fit();
				
		cv::line(frameCopy2, crossingLine[0], crossingLine[1], SCALAR_BLUE, 2);
		cv::imshow("frameCopy2", frameCopy2);
		firstFrame = false;
		frameCount++;
		checkForEscKey = cv::waitKey(33);

		if(checkForEscKey == 27)
		{
			logfile.close();
			break;
		}
		
		if(checkForEscKey == (int) 'p')
		{
			while( checkForEscKey == (int) 'p' )
			{
				checkForEscKey = cv::waitKey(0);
				if(checkForEscKey == (int) 'p')
					break;
			}
		}
	}

	return(0);
}

void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs) 
{

	for (auto &existingBlob : existingBlobs)
	{
		existingBlob.currentMatchFound = false;
		existingBlob.predictNextPosition();
	}

	for (auto &currentFrameBlob : currentFrameBlobs)
	{
		int indexOfLeastDistance = 0;
		double leastDistance = 100000.0;

		for (unsigned int i = 0; i < existingBlobs.size(); i++)
		if (existingBlobs[i].stillBeingTracked == true) 
		{
			double distance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

			if (distance < leastDistance)
			{
				leastDistance = distance;
				indexOfLeastDistance = i;
			}
		}

		if (leastDistance < currentFrameBlob.currentDiagonalSize * 0.5)
			addBlobToExistingBlobs(currentFrameBlob, existingBlobs, indexOfLeastDistance);
		else
			addNewBlob(currentFrameBlob, existingBlobs);
	}

	for (auto &existingBlob : existingBlobs)
	{
		if (existingBlob.currentMatchFound == false)
			existingBlob.numOfConsecutiveFramesWithoutAMatch++;

		if (existingBlob.numOfConsecutiveFramesWithoutAMatch >= 5)
			existingBlob.stillBeingTracked = false;
	}
}

void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &index)
{
	existingBlobs[index].currentContour = currentFrameBlob.currentContour;
	existingBlobs[index].currentBoundingRect = currentFrameBlob.currentBoundingRect;
	existingBlobs[index].centerPositions.push_back(currentFrameBlob.centerPositions.back());
	existingBlobs[index].currentDiagonalSize = currentFrameBlob.currentDiagonalSize;
	existingBlobs[index].currentAspectRatio = currentFrameBlob.currentAspectRatio;
	existingBlobs[index].stillBeingTracked = true;
	existingBlobs[index].currentMatchFound = true;
}

void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs)
{
	currentFrameBlob.currentMatchFound = true;
	existingBlobs.push_back(currentFrameBlob);
}

double distanceBetweenPoints(cv::Point point1, cv::Point point2)
{
	int intX = abs(point1.x - point2.x);
	int intY = abs(point1.y - point2.y);

	return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, std::vector<Blob> &crossedBlobs, int &verticalLinePosition, std::ofstream &logfile)
{
	bool atLeastOneBlobCrossedTheLine = false, currentBlobCrossedTheLine = false;
	int width,height;
	for (auto blob : blobs)
	{
		currentBlobCrossedTheLine = false;
		
		if (blob.stillBeingTracked == true && blob.centerPositions.size() >= 2)
		{
			int prevFrameIndex = (int)blob.centerPositions.size() - 2;
			int currFrameIndex = (int)blob.centerPositions.size() - 1;
			
//			cv::Mat ROI;
//			ROI = frame(blob.currentBoundingRect);
//			cv::cvtColor(ROI,ROI,CV_BGR2GRAY);
			//going left
			if (blob.centerPositions[prevFrameIndex].x > verticalLinePosition && blob.centerPositions[currFrameIndex].x <= verticalLinePosition)
			{
				time_t now = time(0);
				char* dt = strtok(ctime(&now), "\n");
				blob.crossTime = dt;
				blob.directionLeft = true;
				atLeastOneBlobCrossedTheLine = true;
				currentBlobCrossedTheLine = true;
				//blob.extractROI(frame, fgMask, true);
			}

			// going right
			if (blob.centerPositions[prevFrameIndex].x < verticalLinePosition && blob.centerPositions[currFrameIndex].x >= verticalLinePosition)
			{
				time_t now = time(0);
				char* dt = strtok(ctime(&now), "\n");
				blob.crossTime = dt;
				blob.directionLeft = false;
				atLeastOneBlobCrossedTheLine = true;
				currentBlobCrossedTheLine = true;
				//blob.extractROI(frame, fgMask, false);
			}
			
			if(currentBlobCrossedTheLine)
			{
				crossedBlobs.push_back(blob);
			}
		}

	}
		return atLeastOneBlobCrossedTheLine;
}


////