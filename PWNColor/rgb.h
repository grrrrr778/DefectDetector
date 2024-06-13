// (c) Pawlin Technologies Ltd 2010
// File: rgb.h / .cpp
// purpose: basic color operation functions
// author: P.V. Skribtsov
// ALL RIGHTS RESERVED
#pragma once
#include <PWNGeometry/Point3Df.h>

inline Point3Df subtractGreen(const Point3Df &p, float normalize) {
	Point3Df q(p.x - p.y, 0, p.z - p.y);
	if (normalize == 0) return q;
	Point3Df q0 = q;
	q.normate();
	return q0.multiplied(1.0f - normalize) + q.multiplied(normalize);
}

// byte color style 0..255 ranges
inline void normal2color(const Point3Df &normal, unsigned char &r, unsigned char &g, unsigned char &b) {
	r = (unsigned char)(127 + 126 * normal.x + 0.5f);
	g = (unsigned char)(127 + 126 * normal.y + 0.5f);
	b = (unsigned char)(127 + 126 * normal.z + 0.5f);
}

inline Point3Df normal2color(const Point3Df &normal) { // OpenGL color style 0..1 ranges
	return Point3Df(0.5f, 0.5f, 0.5f) + normal.multiplied(0.5f);
}

// normalize saturation and lightness
inline Point3Df normalizePoint(const Point3Df &rgb1) {
	float min1 = std::min<float>(std::min<float>(rgb1.x,rgb1.y),rgb1.z);
	Point3Df p1 = rgb1;
	p1.x -= min1;
	p1.y -= min1;
	p1.z -= min1;
	float norm1 = std::max<float>(std::max<float>(p1.x,p1.y),p1.z);
	if(norm1) p1.multiply(1.0f/norm1); // normalize
	return p1;
}

// compare Hue difference (only color without brightness or contrast)
inline float compareHue(const Point3Df &rgb1, const Point3Df &rgb2) {
	Point3Df p1 = normalizePoint(rgb1);
	p1.minus(normalizePoint(rgb2));
	return p1.norm();
}

inline Point3Df sampleRGB(const void *rgbimageData, unsigned int widthStep, int x, int y) {
	const unsigned char *ptr = (unsigned char *) rgbimageData + y*widthStep + x*3; // 3 bytes per RGB
	return Point3Df((float) ptr[0], (float) ptr[1], (float) ptr[2]);
}

inline Point3Df sampleRGB_float(const void *rgbimageData, unsigned int widthStep, int x, int y) {
	const unsigned char *ptr = (unsigned char *) rgbimageData + y*widthStep + x*3*sizeof(float); // 3*4 bytes per RGB
	const float *ptrf = (const float *) ptr;
	return Point3Df(ptrf[0],ptrf[1],ptrf[2]);
}

inline unsigned char f2uc(float x) {
	if(x < 0) return 0;
	if(x > 255.0) return 255;
	return (unsigned char) (x+0.5f);
}

inline void fillRGB(void *rgbimageData, unsigned int widthStep, int x, int y, const Point3Df &pt) {
	unsigned char *ptr = (unsigned char *) rgbimageData + y*widthStep + x*3; // 3 bytes per RGB
	ptr[0] = f2uc(pt.x);
	ptr[1] = f2uc(pt.y);
	ptr[2] = f2uc(pt.z);
}

inline void point2rgb(const Point3Df &p, unsigned char &r, unsigned char &g, unsigned char &b) {
	r = f2uc(p.x);
	g = f2uc(p.y);
	b = f2uc(p.z);
}