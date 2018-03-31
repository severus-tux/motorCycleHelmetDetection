#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
#include <fstream>		// file utils
#include <ctime>		// timestamp

#include "Blob.h"

// global variables
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);
const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);

// function prototypes 
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs);
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &index);
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs);
double distanceBetweenPoints(cv::Point point1, cv::Point point2);
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intVerticalLinePosition, std::ofstream &logfile);

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

	cv::VideoCapture capVideo;
	std::ofstream logfile; // log file

	cv::Mat frame1;
	cv::Mat frame2;

	std::vector<Blob> blobs;

	cv::Point crossingLine[2];

	capVideo.open(argv[1]);

	// log file
	logfile.open ("LOG-" + std::to_string(time(0)) + ".txt");
	std::cout << "Logging to: \"LOG-" << std::to_string(time(0)) << ".txt\"" << std::endl;

	logfile << "\"Timestamp\t\t\t\t\", \"Direction\", \"Wearing Helmet?\", \"Triple Riding?\"" << std::endl;
				  
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

	capVideo.read(frame1);
	capVideo.read(frame2);

	int verticalLinePosition = (int)std::round((double)frame1.cols * 0.50);

	crossingLine[0].y = 0;
	crossingLine[0].x = verticalLinePosition;
	crossingLine[1].y = frame1.rows -1 ; // last pixel *index*
	crossingLine[1].x = verticalLinePosition;

	char checkForEscKey = 0;
	bool firstFrame = true;
	int frameCount = 2;

    while (capVideo.isOpened())
    {
        std::vector<Blob> currentFrameBlobs;

        cv::Mat frame1Copy = frame1.clone();
        cv::Mat frame2Copy = frame2.clone();
        cv::Mat difference;
        cv::Mat thresh;

        cv::cvtColor(frame1Copy, frame1Copy, CV_BGR2GRAY);
        cv::cvtColor(frame2Copy, frame2Copy, CV_BGR2GRAY);

        cv::GaussianBlur(frame1Copy, frame1Copy, cv::Size(5, 5), 0);
        cv::GaussianBlur(frame2Copy, frame2Copy, cv::Size(5, 5), 0);
        cv::absdiff(frame1Copy, frame2Copy, difference);
        cv::threshold(difference, thresh, 30, 255.0, CV_THRESH_BINARY);


        cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        cv::Mat structuringElement15x15 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

        for (unsigned int i = 0; i < 2; i++)
        {
            cv::dilate(thresh, thresh, structuringElement5x5);
            cv::dilate(thresh, thresh, structuringElement5x5);
            cv::erode(thresh, thresh, structuringElement5x5);
        }
        cv::imshow("imgThresh", thresh);
        cv::Mat threshCopy = thresh.clone();
        std::vector<std::vector<cv::Point> > contours;

        cv::findContours(threshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        std::vector<std::vector<cv::Point> > convexHulls(contours.size());

        for (unsigned int i = 0; i < contours.size(); i++)
            cv::convexHull(contours[i], convexHulls[i]);

        for (auto &convexHull : convexHulls)
        {
            Blob possibleBlob(convexHull);

            if (possibleBlob.currentBoundingRect.area() > 400 &&
                possibleBlob.currentAspectRatio > 0.2 &&
                possibleBlob.currentAspectRatio < 4.0 &&
                possibleBlob.currentBoundingRect.width > 30 &&
                possibleBlob.currentBoundingRect.height > 30 &&
                possibleBlob.currentDiagonalSize > 60.0 &&
                (cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.50)
            {
                currentFrameBlobs.push_back(possibleBlob);
            }
        }
	
        if (firstFrame == true)
            for (auto &currentFrameBlob : currentFrameBlobs)
                blobs.push_back(currentFrameBlob);

        else
            matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);
		
		checkIfBlobsCrossedTheLine(blobs, verticalLinePosition, logfile);
        cv::line(frame2Copy, crossingLine[0], crossingLine[1], SCALAR_BLUE, 2);
        cv::imshow("frame2Copy", frame2Copy);


        currentFrameBlobs.clear();
		currentFrameBlobs.shrink_to_fit();
        frame1 = frame2.clone();           // move frame 1 up to where frame 2 is

        if (capVideo.isOpened())
            capVideo.read(frame2);
        else
        {
            time_t now = time(0);
			char* dt = strtok(ctime(&now), "\n");
            std::cout << dt << ",EOF" << std::endl;
            logfile.close();
            return(0); // end?
        }

        firstFrame = false;
        frameCount++;
        checkForEscKey = cv::waitKey(33);
        
        if(checkForEscKey == 27)
        {
        	logfile.close();
        	break;
        }
    }
    
    return(0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
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

bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &verticalLinePosition, std::ofstream &logfile)
{
    bool atLeastOneBlobCrossedTheLine = 0;

    for (auto blob : blobs)
    {
        if (blob.stillBeingTracked == true && blob.centerPositions.size() >= 2)
        {
            int prevFrameIndex = (int)blob.centerPositions.size() - 2;
            int currFrameIndex = (int)blob.centerPositions.size() - 1;
            
			//going left
            if (blob.centerPositions[prevFrameIndex].x > verticalLinePosition && blob.centerPositions[currFrameIndex].x <= verticalLinePosition)
            {
                time_t now = time(0);
				char* dt = strtok(ctime(&now), "\n");
                std::cout << dt << ", (Left)" << std::endl;
                logfile << dt << ", (Left)" << std::endl;
                atLeastOneBlobCrossedTheLine = true;
            }
            
            // going right
            if (blob.centerPositions[prevFrameIndex].x < verticalLinePosition && blob.centerPositions[currFrameIndex].x >= verticalLinePosition)
            {
                time_t now = time(0);
				char* dt = strtok(ctime(&now), "\n");
                std::cout << dt << ", (Right)" << std::endl;
                logfile << dt << ", (Right)" << std::endl;
                atLeastOneBlobCrossedTheLine = 2;
            }
        }

    }

    return atLeastOneBlobCrossedTheLine;
}
