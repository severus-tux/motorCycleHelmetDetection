#include "../include/Blob.h"

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
