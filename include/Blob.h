#ifndef BLOB_H
#define BLOB_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

class Blob
{
	public:
		//data members 
		std::vector<cv::Point> currentContour;
		cv::Rect currentBoundingRect;
		std::vector<cv::Point> centerPositions;

		double currentDiagonalSize;
		double currentAspectRatio;

		bool currentMatchFound; // If true, Match found, else New Blob 
		bool stillBeingTracked;

		int numOfConsecutiveFramesWithoutAMatch;

		cv::Point predictedNextPosition;

		// function prototypes 
		Blob(std::vector<cv::Point> contour);
		void predictNextPosition(void);

};

#endif    // BLOB_H
