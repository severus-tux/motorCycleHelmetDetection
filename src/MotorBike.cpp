#include "MotorBike.h"

MotorBike::MotorBike(const Blob &blob) : Blob(blob)
{
	currentBoundingRect = blob.currentBoundingRect;
	currentDiagonalSize = blob.currentDiagonalSize;
	currentAspectRatio = blob.currentAspectRatio;
	directionLeft = blob.directionLeft; // true => left, false => Right


}

int MotorBike::countRiders()
{
	//Detect and set values
	return 2; // Remove this Later
}

int MotorBike::detectHelmet()
{
	//Detect and set values
	return 2; // Remove this Later
}
