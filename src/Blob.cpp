#include "../include/Blob.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

int Blob::counterLeft=0;
int Blob::counterRight=0;

Blob::Blob(std::vector<cv::Point> contour)
{
    currentContour = contour;
    currentBoundingRect = cv::boundingRect(currentContour);
    cv::Point currentCenter;
    currentCenter.x =  currentBoundingRect.x + currentBoundingRect.width/2;
    currentCenter.y =  currentBoundingRect.y + currentBoundingRect.height/2;
    centerPositions.push_back(currentCenter);
    currentDiagonalSize = sqrt(pow(currentBoundingRect.width, 2) + pow(currentBoundingRect.height, 2));
    currentAspectRatio = (float)currentBoundingRect.width / (float)currentBoundingRect.height;
    stillBeingTracked = true;
    currentMatchFound = true;
    numOfConsecutiveFramesWithoutAMatch = 0;
}

void Blob::predictNextPosition(void)
{
    int numPositions = (int)centerPositions.size();
	int sumOfXChanges, sumOfYChanges;
	int divisor, deltaX = 0, deltaY = 0;
	int size = ((numPositions>5)?5:numPositions);
	
	if(size > 1)
	{
		for(int i=0;i<size-1;i++) // Taking the weighted average
		{
			sumOfXChanges = sumOfXChanges + centerPositions[size-1-i].x + centerPositions[size-2-i].x;
			sumOfYChanges = sumOfXChanges + centerPositions[size-1-i].y + centerPositions[size-2-i].y;
			divisor = divisor + i + 1;
		}
		
		deltaX = (int)std::round((float)sumOfXChanges / divisor);
		deltaY = (int)std::round((float)sumOfYChanges / divisor);
		
	}
	
	predictedNextPosition.x = centerPositions.back().x + deltaX;
	predictedNextPosition.y = centerPositions.back().y + deltaY;

}

void Blob::extractROI(cv::Mat &frame, cv::Mat &fgMask, bool left, int isMotorCycle)
{
	cv::Rect topRegion(currentBoundingRect.x,currentBoundingRect.y,currentBoundingRect.width,(int)currentBoundingRect.height*0.25);
	cv::Mat ROITop = fgMask(topRegion);
	cv::Mat ROITopGrayScale = frame(topRegion);
	cv::cvtColor(ROITopGrayScale,ROITopGrayScale,CV_BGR2GRAY);
	cv::Mat ROI = frame(currentBoundingRect);
	cv::cvtColor(ROI,ROI,CV_BGR2GRAY);
	
	if(left)
	{
		directionLeft = true;
		counterLeft++;
		
		if(isMotorCycle>0)
			cv::imwrite("./../blob_images/Bike-left-"+std::to_string(counterLeft)+"-"+std::to_string(time(0))+".jpg",ROI);
		else
			cv::imwrite("./../blob_images/Other-left-"+std::to_string(counterLeft)+"-"+std::to_string(time(0))+".jpg",ROI);
//		cv::imwrite("./../blob_images/left-TOP"+std::to_string(counterLeft)+"-"+std::to_string(time(0))+".jpg",ROITop);
//		cv::imwrite("./../blob_images/left-TOP-Gray"+std::to_string(counterLeft)+"-"+std::to_string(time(0))+".jpg",ROITopGrayScale);
	}
	else
	{
		directionLeft = false;
		counterRight++;
		
		if(isMotorCycle>0)
			cv::imwrite("./../blob_images/Bike-right-"+std::to_string(counterRight)+"-"+std::to_string(time(0))+".jpg",ROI);
		else
			cv::imwrite("./../blob_images/Other-right-"+std::to_string(counterRight)+"-"+std::to_string(time(0))+".jpg",ROI);
//		cv::imwrite("./../blob_images/right-TOP"+std::to_string(counterRight)+"-"+std::to_string(time(0))+".jpg",ROITop);
//		cv::imwrite("./../blob_images/right-TOP-Gray"+std::to_string(counterRight)+"-"+std::to_string(time(0))+".jpg",ROITopGrayScale);
	}
}
