// (c) Pawlin Technologies Ltd 
// File: Plane3Df.cpp
// Date: SEPTEMBER-2011
// Authors: Roman Cherkezov
// Purpose: plane z surface representation

// ALL RIGHTS RESERVED, USAGE WITHOUT WRITTEN PERMISSION FROM PAWLIN TECHNOLOGIES LTD. is prohibited
#include "stdafx.h"
#include "Plane3Df.h"
#include <PWNGeneral/pwnutil.h>
#include <cstdlib>
#include <random>

#ifdef _USE_CV
#include <opencv2/opencv.hpp>
#endif

#include <vector>
#include <string>
#include <map>

using namespace std;

const float Plane3Df::PLANE_THICKNESS_DEFAULT = 0.0001f;

Plane3Df::Plane3Df()
	: d(0)
	, normal(Point3Df(0, 0, 1))
	, normalized(true)
	, thickness(PLANE_THICKNESS_DEFAULT)
{
}

Plane3Df::Plane3Df(const float dInput, const Point3Df &normalInput)
	: thickness(PLANE_THICKNESS_DEFAULT)
{
	init(dInput, normalInput);
}
Plane3Df::Plane3Df(const Point3Df &a, const Point3Df &b, const Point3Df &c) {
	Point3Df initNormal = (a - b).vecproduct(c - b);
	if (initNormal.norm() != 0)
		initNormal.normate();
	float initD = -initNormal.product(a);
	init(initD, initNormal);
}
void Plane3Df::setThickness(const float thicknessInput)
{
	thickness = thicknessInput;
}

void Plane3Df::init(const float dInput, const Point3Df &normalInput)
{
#ifndef ANDROID
	if (!finite_check(dInput)) throw "Not finite d parameter provided!";
	if (!finite_check(normalInput.x)) throw "Invalid normal provided: .x is not finite.";
	if (!finite_check(normalInput.y)) throw "Invalid normal provided: .y is not finite.";
	if (!finite_check(normalInput.z)) throw "Invalid normal provided: .z is not finite.";
#endif
	if (0 == normalInput.x && 0 == normalInput.y && 0 == normalInput.z) throw "Invalid normal provided: zero length";

	d = dInput;
	normal = normalInput;
	if (1.0f - normal.norm() == 0.0f) normalized = true;
	else normalized = false;
}


void Plane3Df::invertNormal()
{
	normal.multiply(-1.0f);
}


float Plane3Df::computeX(float y, float z) const
{
	if (0 == normal.x) throw "x is arbitrary";

	const float yMember = y * normal.y;
	const float zMember = z * normal.z;
	const float knownMembersSum = yMember + zMember + d;

	return (-knownMembersSum / normal.x);
}

float Plane3Df::computeY(float x, float z) const
{
	if (0 == normal.y) throw "y is arbitrary";

	const float xMember = x * normal.x;
	const float zMember = z * normal.z;
	const float knownMembersSum = xMember + zMember + d;

	return (-knownMembersSum / normal.y);
}

float Plane3Df::computeZ(float x, float y) const
{
	if (0 == normal.z) throw "z is arbitrary";

	const float yMember = y * normal.y;
	const float xMember = x * normal.x;
	const float knownMembersSum = yMember + xMember + d;

	return (-knownMembersSum / normal.z);
}

void Plane3Df::approximateMLS(const std::vector < Point3Df > &pointsInput)
{
#ifdef _USE_CV
	cv::Scalar zero = cv::Scalar(0);
	cv::Mat E(4, 4, CV_32FC1, zero);

	// matrix A is filled according to the regression model based on (Ax+By+Cz+D=0)
	const size_t totalPoints = pointsInput.size();
	for (size_t pointIndex = 0; pointIndex < totalPoints; pointIndex++)
	{
		E.at<float>(0, 0) += pointsInput[pointIndex].x * pointsInput[pointIndex].x;
		E.at<float>(0, 1) += pointsInput[pointIndex].x * pointsInput[pointIndex].y;
		E.at<float>(0, 2) += pointsInput[pointIndex].x * pointsInput[pointIndex].z;
		E.at<float>(0, 3) += pointsInput[pointIndex].x;

		E.at<float>(1, 0) += pointsInput[pointIndex].y * pointsInput[pointIndex].x;
		E.at<float>(1, 1) += pointsInput[pointIndex].y * pointsInput[pointIndex].y;
		E.at<float>(1, 2) += pointsInput[pointIndex].y * pointsInput[pointIndex].z;
		E.at<float>(1, 3) += pointsInput[pointIndex].y;

		E.at<float>(2, 0) += pointsInput[pointIndex].z * pointsInput[pointIndex].x;
		E.at<float>(2, 1) += pointsInput[pointIndex].z * pointsInput[pointIndex].y;
		E.at<float>(2, 2) += pointsInput[pointIndex].z * pointsInput[pointIndex].z;
		E.at<float>(2, 3) += pointsInput[pointIndex].z;

		E.at<float>(3, 0) += pointsInput[pointIndex].x;
		E.at<float>(3, 1) += pointsInput[pointIndex].y;
		E.at<float>(3, 2) += pointsInput[pointIndex].z;
		E.at<float>(3, 3) += 1.0f;
	}

	cv::Mat abcd(4, 1, CV_32FC1, zero);
	cv::SVD::solveZ(E, abcd);

	d = abcd.at<float>(3, 0);
	normal.x = abcd.at<float>(0, 0);
	normal.y = abcd.at<float>(1, 0);
	normal.z = abcd.at<float>(2, 0);

	if (1.0f - normal.norm() == 0.0f) normalized = true;
	else normalized = false;
#else
	throw std::logic_error("The method approximateMLS is not implemented without OpenCV");
#endif
}

float Plane3Df::approximateRANSAC(const std::vector < Point3Df > &pointsInput, const float findingBetterModelProbabilityThresholdInput, const float maxCountOfIterations)
{
	const int totalPoints = (int) pointsInput.size();
	if (totalPoints < 10) return chooseBestModel(pointsInput); // just analyze all combinations

	size_t primaryFoundIndex = 0;
	size_t secondaryFoundIndex = 1;
	size_t tertiaryFoundIndex = 2;
	size_t totalIterations = 0;

	size_t maxInliers = 0;
	float maxInliersPercent = 0;

	float currentFindingBetterModelProbability = 1.0f;
	std::random_device rd;   //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd());  //Standard mersenne_twister_engine seeded with rd()
	std::uniform_int_distribution<> distrib(0, (int)(totalPoints - 1));

	//while (findingBetterModelProbabilityThresholdInput < currentFindingBetterModelProbability)
	while (findingBetterModelProbabilityThresholdInput < currentFindingBetterModelProbability||totalIterations< maxCountOfIterations)
	{
		int firstPointIndex = distrib(gen);
		int secondPointIndex = distrib(gen);
		int thirdPointIndex = distrib(gen);
 
		if (secondPointIndex == firstPointIndex)
			if (secondPointIndex < totalPoints - 1)
				secondPointIndex++;
			else secondPointIndex--;

		while ((thirdPointIndex == firstPointIndex) || (thirdPointIndex == secondPointIndex))
		{
			thirdPointIndex++;
			if (thirdPointIndex >= totalPoints) thirdPointIndex -= totalPoints;
		}
		;
		
		if (thirdPointIndex == firstPointIndex) throw "error!";
		if (thirdPointIndex == secondPointIndex) throw "error!";
		if (secondPointIndex == firstPointIndex) throw "error!";

		const Point3Df &firstPoint = pointsInput[firstPointIndex];
		const Point3Df &secondPoint = pointsInput[secondPointIndex];
		const Point3Df &thirdPoint = pointsInput[thirdPointIndex];

		totalIterations++;
		buildFrom3Points(firstPoint, secondPoint, thirdPoint);
		const float normalLength = normalize();
		if (0 != normalLength)
		{
			const size_t currentNumberOfInliers = computeNumberOfInsidePointsNormalized(pointsInput);
			if (currentNumberOfInliers > maxInliers)
			{
				maxInliers = currentNumberOfInliers;
				primaryFoundIndex = firstPointIndex;
				secondaryFoundIndex = secondPointIndex;
				tertiaryFoundIndex = thirdPointIndex;

				maxInliersPercent = float(maxInliers) / float(totalPoints);
			}
		}

		const float maxSelectionProbability = maxInliersPercent*maxInliersPercent*maxInliersPercent;
		currentFindingBetterModelProbability = powf(1 - maxSelectionProbability, float(totalIterations));
	}
	;
	/*const Point3Df foundPrimaryPoint = pointsInput[primaryFoundIndex];
	const Point3Df foundSecondaryPoint = pointsInput[secondaryFoundIndex];
	const Point3Df foundTertiaryPoint = pointsInput[tertiaryFoundIndex];*/
	buildFrom3Points(pointsInput[primaryFoundIndex], pointsInput[secondaryFoundIndex], pointsInput[tertiaryFoundIndex]);
	//buildFrom3Points(foundPrimaryPoint, foundSecondaryPoint, foundTertiaryPoint);
	
	if (1.0f - normal.norm() == 0.0f) normalized = true;
	else normalized = false;

	return maxInliersPercent;
}

float Plane3Df::chooseBestModel(const std::vector < Point3Df > &pointsInput)
{
	size_t maxInsidePoints = 0;
	size_t primaryFoundIndex = 0;
	size_t secondaryFoundIndex = 0;
	size_t tertiaryFoundIndex = 0;

	const size_t totalPoints = pointsInput.size();
	if (totalPoints < 3) throw "RANSAC: the number of points should be at least equal to 3";
	for (size_t primaryIndex = 0; primaryIndex < totalPoints; primaryIndex++)
	{
		const Point3Df & primaryPoint = pointsInput[primaryIndex];
		for (size_t secondaryIndex = primaryIndex + 1; secondaryIndex < totalPoints; secondaryIndex++)
		{
			const Point3Df &secondaryPoint = pointsInput[secondaryIndex];
			for (size_t tertiaryIndex = secondaryIndex + 1; tertiaryIndex < totalPoints; tertiaryIndex++)
			{
				const Point3Df &tertiaryPoint = pointsInput[tertiaryIndex];
				buildFrom3Points(primaryPoint, secondaryPoint, tertiaryPoint);
				normalize();
				const size_t currentNumberOfInsidePoints = computeNumberOfInsidePointsNormalized(pointsInput);
				if (currentNumberOfInsidePoints < 3) throw "invalid plane building from 3 points!";
				if (currentNumberOfInsidePoints > maxInsidePoints)
				{
					maxInsidePoints = currentNumberOfInsidePoints;
					primaryFoundIndex = primaryIndex;
					secondaryFoundIndex = secondaryIndex;
					tertiaryFoundIndex = tertiaryIndex;
				}
			}
		}
	}

	const Point3Df &foundPrimaryPoint = pointsInput[primaryFoundIndex];
	const Point3Df &foundSecondaryPoint = pointsInput[secondaryFoundIndex];
	const Point3Df &foundTertiaryPoint = pointsInput[tertiaryFoundIndex];
	buildFrom3Points(foundPrimaryPoint, foundSecondaryPoint, foundTertiaryPoint);	

	if (1.0f - normal.norm() == 0.0f) normalized = true;
	else normalized = false;
	return (float)maxInsidePoints / (float)pointsInput.size();
}

const size_t Plane3Df::computeNumberOfInsidePointsNormalized(const vector < Point3Df > &pointsInput) const
{
	size_t numberOfInsidePoints = 0;
	const size_t totalPoints = pointsInput.size();
	for (size_t index = 0; index < totalPoints; index++)
	{
		if (belongsToPlaneNormalized(pointsInput[index])) numberOfInsidePoints++;
	}
	return numberOfInsidePoints;
}

const bool Plane3Df::belongsToPlaneNormalized(const Point3Df &pointInput) const
{
	if (computeDistanceNormalized(pointInput) < (0.5f*thickness)) return true;
	else return false;
}



const size_t Plane3Df::computeNumberOfInsidePoints(const vector < Point3Df > &pointsInput) const
{
	size_t numberOfInsidePoints = 0;
	const size_t totalPoints = pointsInput.size();
	for (size_t index = 0; index < totalPoints; index++)
	{
		if (belongsToPlane(pointsInput[index])) numberOfInsidePoints++;
	}
	return numberOfInsidePoints;
}

const bool Plane3Df::belongsToPlane(const Point3Df &pointInput) const
{
	if (computeDistance(pointInput) < (0.5f*thickness)) return true;
	else return false;
}

const float Plane3Df::computeDistance(const Point3Df &pointInput) const
{
	return fabsf(computeDistanceSigned(pointInput));
}


const float Plane3Df::computeDistanceSigned(const Point3Df &pointInput) const
{
	if (!normalized)
	{
		const float normalLength = sqrtf(normal.x * normal.x +
			normal.y * normal.y + 
			normal.z * normal.z);
		const float distance = (pointInput.x * normal.x + pointInput.y * normal.y + pointInput.z * normal.z + d) / normalLength;
		return distance;
	}
	else
	{
		return computeDistanceSignedNormalized(pointInput);
	}
}

const float Plane3Df::computeDistanceSignedNormalized(const Point3Df &pointInput) const
{
	return pointInput.x * normal.x + pointInput.y * normal.y + pointInput.z * normal.z + d ;
}


const float Plane3Df::computeDistanceNormalized(const Point3Df &pointInput) const
{
	return fabsf(computeDistanceSignedNormalized(pointInput));
}


void Plane3Df::buildFrom3Points(const Point3Df &aInput, const Point3Df &bInput, const Point3Df &cInput)
{
	const float subDeterminantX = (bInput.y - aInput.y)*(cInput.z - aInput.z) - (cInput.y - aInput.y)*(bInput.z - aInput.z);
	const float subDeterminantY = (bInput.x - aInput.x)*(cInput.z - aInput.z) - (cInput.x - aInput.x)*(bInput.z - aInput.z);
	const float subDeterminantZ = (bInput.x - aInput.x)*(cInput.y - aInput.y) - (cInput.x - aInput.x)*(bInput.y - aInput.y);

	normal.x = subDeterminantX;
	normal.y = -subDeterminantY;
	normal.z = subDeterminantZ;

	d = -aInput.x * subDeterminantX + aInput.y * subDeterminantY - aInput.z * subDeterminantZ;

	if (1.0f - normal.norm() == 0.0f) normalized = true;
	else normalized = false;
}

void Plane3Df::approximateSuccessivelyMLS( const std::vector < Point3Df > &pointsInput, 
	const float reductionSigmaCoefficientInput,
	const size_t maxIterationsInput)
{
	if (reductionSigmaCoefficientInput <= 0) throw "Reduction coefficient is supposed to be more than 0";
	const size_t totalInputPoints = pointsInput.size();
	if (0 == totalInputPoints) throw "no points provided for approximation";

	vector < Point3Df > workingPoints;
	workingPoints.assign(pointsInput.begin(), pointsInput.end());

	for (size_t iterationIndex = 0; iterationIndex < maxIterationsInput; iterationIndex++)
	{
		approximateMLS(workingPoints);
		if (workingPoints.size() < 4) break;
		const float currentMSE = computeMSE(workingPoints);
		const float currentDistanceThreshold = currentMSE*reductionSigmaCoefficientInput;
		if (currentDistanceThreshold < thickness) break;

		vector < Point3Df > newWorkingPoints(0);
		newWorkingPoints.reserve(workingPoints.size());
		for (size_t index = 0; index < workingPoints.size(); index++)
		{
			const Point3Df &currentPoint = workingPoints[index];
			const float currentDistance = computeDistance(currentPoint);
			if (currentDistance < currentDistanceThreshold)
			{
				newWorkingPoints.push_back(currentPoint);
			}
		}

		if (newWorkingPoints.size() == workingPoints.size()) break;
		workingPoints.assign(newWorkingPoints.begin(), newWorkingPoints.end());
	}

	if (1.0f - normal.norm() == 0.0f) normalized = true;
	else normalized = false;
}

const float Plane3Df::computeMSE(const vector < Point3Df > &pointsInput) const
{
	float squaredDistancesSum = 0;
	const size_t totalPoints = pointsInput.size();
	if (0 == totalPoints) throw "no input points provided";
	for (size_t index = 0; index < totalPoints; index++)
	{
		const Point3Df &currentPoint = pointsInput[index];
		const float currentDistance = computeDistance(currentPoint);
		squaredDistancesSum += currentDistance*currentDistance;
	}

	return sqrtf(squaredDistancesSum / totalPoints);
}

const float Plane3Df::computeLineIntersectionParameter(const Ray3Df &rayInput)
{
	const float normalizationCoefficient = normalize();
	const float parameter = computeLineIntersectionParameterNormalized(rayInput);
	multiplyCoefficients(normalizationCoefficient);
	return parameter;
}

const float Plane3Df::computeLineIntersectionParameterNormalized(const Ray3Df &rayInput) const
{
	Ray3Df currentRay = rayInput;
	const float unnormLength = currentRay.b.normate();
	const float normalDirectingCosine =	normal.x * currentRay.b.x +
										normal.y * currentRay.b.y +
										normal.z * currentRay.b.z;

	const float refPlaneDistance  =	computeDistance(currentRay.a);

	const float parameter = -refPlaneDistance / normalDirectingCosine;
	return parameter / unnormLength;
}

void Plane3Df::multiplyCoefficients(const float multiplierInput)
{
	if (0 == multiplierInput) throw "Multiplier can't be 0";

	normal.x *= multiplierInput;
	normal.y *= multiplierInput;
	normal.z *= multiplierInput;
	d *= multiplierInput;
}

const float Plane3Df::normalize()
{
	const float normalLength = normal.norm();
	if (0 == normalLength) return 0;

	multiplyCoefficients(1.0f / normalLength);

	normalized = true;
	return normalLength;
}

void Plane3Df::removeIdenticalPoints(const vector < Point3Df > &pointsInput, vector < Point3Df > &pointsOutput)
{
	map <string, int> hash;
	int currentHashValue = 1;

	pointsOutput.clear();
	const size_t totalPoints = pointsInput.size();
	pointsOutput.reserve(totalPoints);

	for (size_t pointIndex = 0; pointIndex < totalPoints; pointIndex++)
	{
		const Point3Df &currentPoint = pointsInput[pointIndex];
		char buf[1000];
		
#ifdef WIN32
		sprintf_s(buf, 1000, "%f%f%f", currentPoint.x, currentPoint.y, currentPoint.z);		  
#else
		snprintf(buf, 1000, "%f%f%f", currentPoint.x, currentPoint.y, currentPoint.z);
#endif 
		
		if(hash.find(string(buf)) == hash.end())
		{ 
			hash[string(buf)] = currentHashValue;
			pointsOutput.push_back(currentPoint);
		}
	}
}


const void Plane3Df::computeProjection(const Point3Df &projectedPoint, Point3Df &projection) const
{
	projection = projectedPoint;

	if (normalized)
	{	
		const float distance = computeDistanceSignedNormalized(projectedPoint);
		projection.addWeighted(normal, -distance);
	}
	else
	{
		const float distance_not_normalized = computeDistanceSignedNormalized(projectedPoint);
		const float normal_length_squared 
			= normal.x*normal.x
			+ normal.y*normal.y
			+ normal.z*normal.z;
		projection.addWeighted(normal, -distance_not_normalized / normal_length_squared);
	}
	//throw "not finished!!! Consider direction of the shift!";
}


Plane3Df::~Plane3Df()
{}

// TESTING !!!
#ifdef TEST_CLASS_PLANE3DF

const bool Plane3Df::testInit()
{
	printf("Testing plane initialization and get* methods: ");

	Plane3Df plane;
	plane.init(1, Point3Df(2, 3, 4));
	plane.setThickness(0.01f);

	const bool computeZCorrect = (-2.250 - plane.computeZ(1, 2) < plane.thickness);
	const bool computeXCorrect = (-6.000 - plane.computeX(1, 2) < plane.thickness);
	const float currentcomputeY = plane.computeY(1, 2);
	const float differenceY = -3.666f - currentcomputeY;
	const bool computeYCorrect = (differenceY < plane.thickness);

	if (computeXCorrect && computeYCorrect && computeZCorrect) 
	{
		printf("OK!\n");
		return true;
	}
	else
	{
		printf("\n");
		if (!computeXCorrect) printf("computeX went wrong!\n");
		if (!computeYCorrect) printf("computeY went wrong!\n");
		if (!computeZCorrect) printf("computeZ went wrong!\n");
		printf("Test failed!\n");
		
		return false;
	}
}

const bool Plane3Df::testNormalization()
{
	printf("Testing normalization: ");

	Plane3Df plane;
	plane.init(-5, Point3Df(3, 0, 4));
	plane.setThickness(0.001f);

	plane.normalize();
	const bool dNormalizedCorrectly = (-1 - plane.d < plane.thickness);
	const bool normalXNormalizedCorrectly = (0.600f - plane.normal.x < plane.thickness);
	const bool normalYNormalizedCorrectly = (0.000f - plane.normal.y < plane.thickness);
	const bool normalZNormalizedCorrectly = (0.800f - plane.normal.z < plane.thickness);

	const bool testPassed = dNormalizedCorrectly	&& normalXNormalizedCorrectly 
													&& normalYNormalizedCorrectly
													&& normalZNormalizedCorrectly;

	if (testPassed) 
	{
		printf("OK!\n");
		return true;
	}
	else
	{
		printf("\n");
		if (!dNormalizedCorrectly) printf("d normalization failed!\n");
		if (!normalXNormalizedCorrectly) printf("normal x component normalization failed!\n");
		if (!normalYNormalizedCorrectly) printf("normal y component normalization failed!\n");
		if (!normalZNormalizedCorrectly) printf("normal z component normalization failed!\n");
		printf("Test failed!\n");
		
		return false;
	}
}

const bool Plane3Df::testMLSApproximation()
{
	printf("Testing mls approximation: ");

	vector <Point3Df> firstPoints;
	firstPoints.push_back(Point3Df(0, 0, 0));
	firstPoints.push_back(Point3Df(1, 1, -2));
	firstPoints.push_back(Point3Df(-2, 3, -1));

	Plane3Df plane;
	plane.approximateMLS(firstPoints);

	const bool zCorrect = (0 != plane.normal.z);
	const bool xCorrect = (1.0f - (plane.normal.x / plane.normal.z) < plane.thickness);
	const bool yCorrect = (1.0f - (plane.normal.y / plane.normal.z) < plane.thickness);

	const bool firstTestPassed = zCorrect && xCorrect && yCorrect;

	vector <Point3Df> secondPoints;
	secondPoints.push_back(Point3Df(-6, 0, 0));
	secondPoints.push_back(Point3Df(0, 0, 3));
	secondPoints.push_back(Point3Df(0, -2, 0));
	secondPoints.push_back(Point3Df(1, 1, 5));

	Plane3Df secondPlane;
	secondPlane.approximateMLS(secondPoints);

	const bool secondXCorrect = (0 != secondPlane.normal.x);
	const bool secondZCorrect = (-2.00f - secondPlane.normal.z / secondPlane.normal.x < secondPlane.thickness);
	const bool secondDCorrect = (6.00f - (secondPlane.d / secondPlane.normal.x) < secondPlane.thickness);
	const bool secondYCorrect = (3.00f - (secondPlane.normal.y / secondPlane.normal.x) < secondPlane.thickness);

	const bool secondTestPassed = secondDCorrect && secondZCorrect && secondXCorrect && secondYCorrect;

	if (secondTestPassed && firstTestPassed) 
	{
		printf("OK!\n");
		return true;
	}
	else
	{
		printf("\n");
		printf("Test failed!\n");
		
		return false;
	}
}

const bool Plane3Df::testRANSACApproximation()
{
	printf("Testing RANSAC approximation: ");

	vector <Point3Df> firstPoints;
	firstPoints.push_back(Point3Df(0, 0, 0));
	firstPoints.push_back(Point3Df(1, 1, -2));
	firstPoints.push_back(Point3Df(-2, 3, -1));
	firstPoints.push_back(Point3Df(0, 1, 5));
	firstPoints.push_back(Point3Df(-3, -1, -11));

	Plane3Df plane;
	plane.approximateRANSAC(firstPoints);

	const bool zCorrect = (0 != plane.normal.z);
	const bool xCorrect = (1.0f - (plane.normal.x / plane.normal.z) < plane.thickness);
	const bool yCorrect = (1.0f - (plane.normal.y / plane.normal.z) < plane.thickness);

	const bool firstTestPassed = zCorrect && xCorrect && yCorrect;

	vector <Point3Df> secondPoints;
	secondPoints.push_back(Point3Df(-6, 0, 0));
	secondPoints.push_back(Point3Df(-6, 2, 3));
	secondPoints.push_back(Point3Df(0, 0, 3));
	secondPoints.push_back(Point3Df(-3, 1, 3));
	secondPoints.push_back(Point3Df(0, -2, 0));
	secondPoints.push_back(Point3Df(2, -2, 1));
	secondPoints.push_back(Point3Df(1, 1, 5));
	secondPoints.push_back(Point3Df(1, 1, 6));
	secondPoints.push_back(Point3Df(-1, -1, -5));
	secondPoints.push_back(Point3Df(1, 11, 3));
	secondPoints.push_back(Point3Df(1, 2, 3));
	secondPoints.push_back(Point3Df(9, -1, 6));
	secondPoints.push_back(Point3Df(1, 11, 3));
	secondPoints.push_back(Point3Df(1, 11, -3));
	secondPoints.push_back(Point3Df(3, -3, 0));

	Plane3Df secondPlane;
	secondPlane.setThickness(0.1f);
	secondPlane.approximateRANSAC(secondPoints, 0.001f);

	const bool secondXCorrect = (0 != secondPlane.normal.x);
	const bool secondZCorrect = (-2.00f - secondPlane.normal.z / secondPlane.normal.x < secondPlane.thickness);
	const bool secondDCorrect = (6.00f - (secondPlane.d / secondPlane.normal.x) < secondPlane.thickness);
	const bool secondYCorrect = (3.00f - (secondPlane.normal.y / secondPlane.normal.x) < secondPlane.thickness);

	const bool secondTestPassed = secondDCorrect && secondZCorrect && secondXCorrect && secondYCorrect;

	if (secondTestPassed && firstTestPassed) 
	{
		printf("OK!\n");
		return true;
	}
	else
	{
		printf("\n");
		printf("Test failed!\n");
		
		return false;
	}
}

const bool Plane3Df::testBuildingFrom3Points()
{
	printf("Testing building from 3 points: ");

	vector <Point3Df> firstPoints;
	firstPoints.push_back(Point3Df(0, 0, 0));
	firstPoints.push_back(Point3Df(1, 1, -2));
	firstPoints.push_back(Point3Df(-2, 3, -1));

	Plane3Df plane;
	plane.buildFrom3Points(firstPoints[0], firstPoints[1], firstPoints[2]);

	const bool zCorrect = (0 != plane.normal.z);
	const bool xCorrect = (1.0f - (plane.normal.x / plane.normal.z) < plane.thickness);
	const bool yCorrect = (1.0f - (plane.normal.y / plane.normal.z) < plane.thickness);

	const bool testPassed = zCorrect && xCorrect && yCorrect;
	if (testPassed) 
	{
		printf("OK!\n");
		return true;
	}
	else
	{
		printf("\n");
		printf("Test failed!\n");
		
		return false;
	}
}

const bool Plane3Df::testDistance()
{
	printf("Testing distance calculation: ");
	Plane3Df plane;
	plane.init(10, Point3Df(1, 2, 1));
	plane.setThickness(0.1f);

	const float firstDistance = plane.computeDistance(Point3Df(1, 1, 1));
	const bool firstDistanceCorrect = fabsf(firstDistance - 5.71f) < plane.thickness;

	plane.normalize();
	const float secondDistance = plane.computeDistanceNormalized(Point3Df(-2, -5, -1));
	const bool secondDistanceCorrect = fabsf(secondDistance - 1.22f) < plane.thickness;

	const bool testPassed = firstDistanceCorrect && secondDistanceCorrect;
	if (testPassed) 
	{
		printf("OK!\n");
		return true;
	}
	else
	{
		printf("\n");
		printf("\tTest failed!\n");
		
		return false;
	}
}

const bool Plane3Df::testMSEComputing()
{
	printf("Testing mse computing...");
	Plane3Df plane;
	plane.init(3, Point3Df(2, 3, -1));
	plane.setThickness(0.1f);

	vector < Point3Df > points;
	points.push_back(Point3Df(0, 0, 0));
	points.push_back(Point3Df(0, 0, 1));
	points.push_back(Point3Df(0, 0, 2));
	points.push_back(Point3Df(0, 0, 3));

	const float mse = plane.computeMSE(points);
	const bool mseCorrect = fabsf(mse - 0.5f) < plane.thickness;

	const bool testPassed = mseCorrect;
	if (testPassed) 
	{
		printf("OK!\n");
		return true;
	}
	else
	{
		printf("\n");
		printf("\tTest failed!\n");
		
		return false;
	}	
}

const bool Plane3Df::testLineIntersection()
{
	printf("Testing line intersection: ");
	Plane3Df plane;
	plane.init(10, Point3Df(1, 2, 1));
	plane.setThickness(0.1f);
	
	Ray3Df ray(Point3Df(1, 1, 1), Point3Df(-1, -1, -1));
	const float intersectionParameter = plane.computeLineIntersectionParameter(ray);
	const Point3Df intersectionPoint = ray.getPoint(intersectionParameter);

	const bool intersectionParameterCorrect = plane.belongsToPlane(intersectionPoint);
	
	const bool testPassed = intersectionParameterCorrect;
	if (testPassed) 
	{
		printf("OK!\n");
		return true;
	}
	else
	{
		printf("\n");
		printf("\tTest failed!\n");
		
		return false;
	}
}

const bool Plane3Df::testIdenticalPointsRemoval()
{
	printf("Testing identical points removal: ");
	Plane3Df plane;

	vector < Point3Df > points;
	points.push_back(Point3Df(0, 0, 0));
	points.push_back(Point3Df(0, 0.5f, 1));
	points.push_back(Point3Df(-5, 3.3f, 1));
	points.push_back(Point3Df(0, 0.5f, 1));
	points.push_back(Point3Df(7, 4, 2));
	points.push_back(Point3Df(-5, 3.3f, 1));
	points.push_back(Point3Df(-1, -1, 0));
	points.push_back(Point3Df(-5, 3.3f, 1));
	const size_t totalPoints = points.size();
	const size_t excessPoints = 3;

	vector < Point3Df > filteredPoints(0);
	Plane3Df::removeIdenticalPoints(points, filteredPoints);

	bool testPassedSuccessfully = true;
	const size_t pointsLeft = filteredPoints.size();
	if (pointsLeft != (totalPoints - excessPoints)) testPassedSuccessfully = false;
	for (size_t index = 0; index < pointsLeft; index++)
	{
		if (!testPassedSuccessfully) break;

		const Point3Df &currentPoints = filteredPoints[index];
		for (size_t secondaryIndex = index + 1; secondaryIndex < pointsLeft; secondaryIndex++)
		{
			const Point3Df &secondaryPoint = filteredPoints[secondaryIndex];
			if (currentPoints.equals(secondaryPoint)) 
			{
				testPassedSuccessfully = false;
				break;
			}
		}
	}

	if (testPassedSuccessfully) 
	{
		printf("OK!\n");
		return true;
	}
	else
	{
		printf("\n");
		printf("\tTest failed!\n");
		
		return false;
	}
}

const bool Plane3Df::testSuccessiveApproximationMLS()
{
	printf("Testing successive mls approximations... \n");

	// x + y + z = 0
	vector <Point3Df> firstPoints;
	firstPoints.push_back(Point3Df(0, 0, 0));
	firstPoints.push_back(Point3Df(1, 1, -2));
	firstPoints.push_back(Point3Df(-2, 3, -1));
	firstPoints.push_back(Point3Df(0, 1, 2));
	firstPoints.push_back(Point3Df(-3, -1, 4));
	firstPoints.push_back(Point3Df(-3, -2, 5));
	firstPoints.push_back(Point3Df(3, -1, -2.1f));

	Plane3Df plane;
	plane.setThickness(0.1f);
	plane.approximateSuccessivelyMLS(firstPoints, 1.0f, 10);

	const bool zCorrect = (0 != plane.normal.z);
	const bool xCorrect = (1.0f - (plane.normal.x / plane.normal.z) < plane.thickness);
	const bool yCorrect = (1.0f - (plane.normal.y / plane.normal.z) < plane.thickness);

	const bool firstTestPassed = zCorrect && xCorrect && yCorrect;

	// x + 3y - 2z + 6 = 0
	vector <Point3Df> secondPoints;
	secondPoints.push_back(Point3Df(-6, 0, 0));
	secondPoints.push_back(Point3Df(0, 0, 3));
	secondPoints.push_back(Point3Df(0, -2, 0));
	secondPoints.push_back(Point3Df(1, 1, 5));
	secondPoints.push_back(Point3Df(1, -2, 5.5f));
	secondPoints.push_back(Point3Df(-1, -1, 1.1f));
	secondPoints.push_back(Point3Df(1, 0, 3.1f));
	secondPoints.push_back(Point3Df(1, 0, 2.8f));
	secondPoints.push_back(Point3Df(0, 0, 2.9f));

	Plane3Df secondPlane;
	secondPlane.setThickness(0.1f);
	secondPlane.approximateSuccessivelyMLS(secondPoints, 1.0f, 10);

	const bool secondXCorrect = (0 != secondPlane.normal.x);
	const bool secondZCorrect = (-2.00f - secondPlane.normal.z / secondPlane.normal.x < secondPlane.thickness);
	const bool secondDCorrect = (6.00f - (secondPlane.d / secondPlane.normal.x) < secondPlane.thickness);
	const bool secondYCorrect = (3.00f - (secondPlane.normal.y / secondPlane.normal.x) < secondPlane.thickness);

	const bool secondTestPassed = secondDCorrect && secondZCorrect && secondXCorrect && secondYCorrect;

	if (secondTestPassed && firstTestPassed) 
	{
		printf("OK!\n");
		return true;
	}
	else
	{
		printf("\n");
		printf("Test failed!\n");
		
		return false;
	}
}

const bool Plane3Df::testWholeClass()
{
	printf("Testing class Plane3Df...\n");

	const bool initTestPassed = testInit();
	const bool normalizationTestPassed = testNormalization();
	const bool mlsApproximationTestPassed = testMLSApproximation();
	const bool distanceTestPassed = testDistance();
	const bool buildingFrom3PointsTestPassed = testBuildingFrom3Points();
	
	const bool lineIntersectionTestPassed = testLineIntersection();
	const bool mseTestPassed = testMSEComputing();
	const bool identicalPointsRemovalTestPassed = testIdenticalPointsRemoval();
	const bool successiveApproximationsMLSTestPassed = testSuccessiveApproximationMLS();

	bool RANSACApproximationTestPassed = true;
	int RANSACTestsFailed = 0;
	const int totalRANSACTests = 10000;
	for (int ransacTestIndex = 0; ransacTestIndex < totalRANSACTests; ransacTestIndex++)
	{
		const bool currentRANSACApproximationTestPassed = testRANSACApproximation();
		if (!currentRANSACApproximationTestPassed) RANSACTestsFailed++; 
	}
	if (float(RANSACTestsFailed) / float(totalRANSACTests) > 0.001f) RANSACApproximationTestPassed = false;
	printf("Total RANSAC tests: %i, failed: %i\n", totalRANSACTests, RANSACTestsFailed);

	const bool allTestsSuccessfull =	initTestPassed && normalizationTestPassed && mlsApproximationTestPassed && mseTestPassed &&
										distanceTestPassed && buildingFrom3PointsTestPassed && RANSACApproximationTestPassed &&
										lineIntersectionTestPassed && identicalPointsRemovalTestPassed && successiveApproximationsMLSTestPassed;
	if (allTestsSuccessfull)
	{
		printf("\nAll tests passed successfully!\n");
		return true;
	}
	else
	{
		printf("\nClass testing failed!\n");
		return false;		
	}
}

#endif