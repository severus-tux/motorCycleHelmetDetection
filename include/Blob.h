#ifndef BLOB_H
#define BLOB_H
// Shreyas Commented here to check slack
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
		bool directionLeft; // true => left, false => Right
		int numOfConsecutiveFramesWithoutAMatch;

		cv::Point predictedNextPosition;

		static int counterLeft;
		static int counterRight;
		// function prototypes 
		Blob(std::vector<cv::Point> contour);
		void predictNextPosition(void);
		void extractROI(cv::Mat &frame, cv::Mat &fgMask, bool left);

};

#endif    // BLOB_H
