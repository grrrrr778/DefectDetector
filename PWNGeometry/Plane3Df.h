// (c) Pawlin Technologies Ltd 
// File: Plane3Df.h
// Date: SEPTEMBER-2011
// Authors: Roman Cherkezov
// Purpose: plane z surface representation

// ALL RIGHTS RESERVED, USAGE WITHOUT WRITTEN PERMISSION FROM PAWLIN TECHNOLOGIES LTD. is prohibited
#pragma once

#include "PWNGeometry/geometricobjects.h"
#define _USE_CV
class Plane3Df
{
public:
	Plane3Df();
	Plane3Df( const float dInput, const Point3Df &normalInput );
	Plane3Df( const Point3Df &a,const Point3Df &b,const Point3Df &c );
	Plane3Df( const Plane3Df & other) : normal(other.normal), d(other.d), thickness(other.thickness), normalized(other.normalized) { }

	void init( const float dInput, const Point3Df &normalInput );
	void invertNormal();
	void print(const string &str) const {
		printf("%s %f ", str.c_str(),getD());
		getNormal().print();
		printf("\n");
	}

	const float getD() const { return d; }
	void setD(float _d) {
		d = _d;
	}
	const Point3Df &getNormal() const {return normal;}
	const float getThickness() const { return thickness; }

	const bool isNormalized() const { return normalized; }

	void setThickness( const float thicknessInput );

	float computeX( float y, float z ) const;
	float computeY( float x, float z ) const;
	float computeZ( float x, float y) const;

	const float computeDistance( const Point3Df &pointInput ) const;
	const float computeDistanceNormalized( const Point3Df &pointInput ) const;
	const float computeDistanceSigned( const Point3Df &pointInput ) const;
	const float computeDistanceSignedNormalized( const Point3Df &pointInput ) const;

	const void computeProjection(const Point3Df &projectedPoint, Point3Df &projection) const;
	inline Point3Df project(const Point3Df &x) const {
		Point3Df res;
		computeProjection(x, res);
		return res;
	}

	const float computeMSE( const std::vector < Point3Df > &pointsInput ) const;

	void approximateMLS( const std::vector < Point3Df > &pointsInput );
	// returns % of inliers
	float approximateRANSAC( const std::vector < Point3Df > &pointsInput, const float findingBetterModelProbabilityThresholdInput = 0.05f, const float maxCountOfIterations = -1);
	float chooseBestModel( const std::vector < Point3Df > &pointsInput );
	void buildFrom3Points( const Point3Df &aInput, const Point3Df &bInput, const Point3Df &cInput );
	void approximateSuccessivelyMLS( const std::vector < Point3Df > &pointsInput, 
								const float reductionPercentInput = 3.0f , const size_t maxIterationsInput = 10 );

	const float computeLineIntersectionParameter( const Ray3Df &rayInput );
	const float computeLineIntersectionParameterNormalized( const Ray3Df &rayInput ) const;

	static void removeIdenticalPoints( const std::vector < Point3Df > &pointsInput, std::vector < Point3Df > &pointsOutput );

	void multiplyCoefficients( const float multiplierInput );
	const float normalize();

	const bool belongsToPlane( const Point3Df &pointInput ) const;
	const size_t computeNumberOfInsidePoints( const std::vector < Point3Df > &pointsInput ) const;
	
	const bool belongsToPlaneNormalized( const Point3Df &pointInput ) const;
	const size_t computeNumberOfInsidePointsNormalized( const std::vector < Point3Df > &pointsInput ) const;

	static const float PLANE_THICKNESS_DEFAULT;
	
	virtual ~Plane3Df();
protected:

	float &getDModifyable() { return d; }
	Point3Df &getNormalModifyable() { return normal; }
	float &getThicknessModifyable() {return thickness; }
private:
	// surface is represented as Ax+By+Cz+D = 0
	// normal is (A,B,C)
	float d;
	Point3Df normal; // Note: this vector may not be normalized
	float thickness;

	bool normalized;

#ifndef TEST_CLASS_PLANE3DF
};






#else
public:
	const bool testWholeClass();

	const bool testInit();
	const bool testNormalization();
	const bool testMLSApproximation();
	const bool testRANSACApproximation();
	const bool testBuildingFrom3Points();
	const bool testLineIntersection();
	const bool testDistance();
	const bool testMSEComputing();
	const bool testIdenticalPointsRemoval();
	const bool testSuccessiveApproximationMLS();
};
#endif