// (c) Pawlin Technologies Ltd 2010
// purpose: basic depth map processing functions
// author: P.V. Skribtsov
// ALL RIGHTS RESERVED

// PWNIP
#include <PWNGeometry/Point3Df.h>
#include <PWNGeneral/Dataset.h>
#include <PWNImageObj/OvalBase.h>
#include <PWNNLib2_0/SimpleNet.h>
#include <PWNGeneral/pwnutil.h>

// OpenCV
#include <opencv2/opencv.hpp>

// STL
#include <string>
#include <map>
#include <set>

using std::map;
using std::set;

#pragma once


struct Chunks {
	struct Chunk3D {
		vector <Point3Df> points;
		Point3Df normal;
		float d; // plane equation -> (normal, x) + d = 0
		vector < std::tuple < int32_t, int32_t, int32_t> > mesh;
	};
	vector <Chunk3D> parts;
	Chunks() {}
	Chunks(const map< int32_t, Chunk3D> &m, size_t minpoints) {
		for (auto iter = m.begin(); iter != m.end(); iter++)
			if (iter->second.points.size() >= minpoints) parts.push_back(iter->second);
	}
	void save(FILE *file) const {
		uint32_t size = (uint32_t)parts.size();
		fwrite(&size, sizeof(size), 1, file);
		FOR_ALL(parts, i) {
			FileSaver<Point3Df>::save(file, parts[i].points);
			fwrite(&(parts[i].normal), sizeof(Point3Df), 1, file);
			fwrite(&(parts[i].d), sizeof(float), 1, file);
			FileSaver<std::tuple < int32_t, int32_t, int32_t> >::save(file, parts[i].mesh);
		}
	}
	void load(FILE *file) {
		uint32_t size = 0;
		fread(&size, sizeof(size), 1, file);
		parts.resize(size);
		FOR_ALL(parts, i) {
			FileSaver<Point3Df>::load(file, parts[i].points);
			fread(&(parts[i].normal), sizeof(Point3Df), 1, file);
			fread(&(parts[i].d), sizeof(float), 1, file);
			FileSaver<std::tuple < int32_t, int32_t, int32_t> >::load(file, parts[i].mesh);
		}
	}
};



class IMatrixProcessor;
class DepthProcessor;
class RasterMeshGen {
	map<char, std::string> v;
public:
	struct Variant {
		int step; // will not be used
		bool abc;
		bool bcd;
		bool acd;
		bool bad;
		Variant(const std::string &value) {
			step = value[0] == '2' ? 2 : 1;
			abc = value[1] != '0';
			bcd = value[2] != '0';
			acd = value[3] != '0';
			bad = value[4] != '0';
		}
	};
	RasterMeshGen() {
		//step size(0) - abc(1) - bcd(2) - acd(3) - bad(4)
		v[0b0000] = "20000";
		v[0b0001] = "10000";
		v[0b0010] = "20000";
		v[0b0011] = "10000";

		v[0b0100] = "10000";
		v[0b0101] = "10000";
		v[0b0110] = "10000";
		v[0b0111] = "10100"; // bcd

		v[0b1000] = "20000";
		v[0b1001] = "10000";
		v[0b1010] = "20000";
		v[0b1011] = "10010"; // acd

		v[0b1100] = "10000"; 
		v[0b1101] = "10001"; // bad
		v[0b1110] = "11000"; // abc
		v[0b1111] = "11100"; // abc, bcd
	}
	// this is for plane (x,n) + d = 0
	inline Point3Df project(const Point3Df &normal, float d, const Point3Df &p) const {
		float pn = p.product(normal);
		float nn = normal.product(normal);
		if (nn == 0.0f) throw ("planeProject : zero normal");
		return p + normal.multiplied((- d - pn) / nn);
	}
	void generate(
		const DepthProcessor &proc,
		float codes_scale,
		const cv::Mat &codes,
		const cv::Mat &depth,
		const set <int32_t> &allowed,
		std::map < int32_t, Chunks::Chunk3D > &out,
		float outscale, bool yzchange, int dx = 0, int dy = 0
	) const;
};

class DepthProcessor
{
	float fx, fy, cx, cy;
	RasterMeshGen meshgen;
	//	IMatrixProcessor &matprc;
public:

	inline Point3Df pix2ray(float x, float y) const {
		return Point3Df((x - cx) / fx, (y - cy) / fy, 1.0f);
	}

	inline Point3Df getPoint(int x, int y, float depth, float scale = 1.0f, float k = 1.0f) const {
		Point3Df a = pix2ray(scale*x, scale*y);
		a.normate();
		float t = depth / a.z;
		return a.multiplied(t * k);
	}

	inline Point3Df getPoint(int x, int y, const cv::Mat &img, float scale = 1.0f, float k = 1.0f, int dx = 0, int dy = 0) const {
		float depth = img.at<float>(y - dy, x - dx); // depth in millimeters transform to meters
		return getPoint(x, y, depth, scale, k);
	}

	DepthProcessor(/*IMatrixProcessor &matprc*/);
	DepthProcessor(std::string intrinsics_file);
	void setIntrinsic(float fx, float fy, float cx, float cy) { this->fx = fx, this->fy = fy, this->cx = cx, this->cy = cy; }
	~DepthProcessor();
	inline Point3Df pt2vis(const Point3Df &p) const {
		return Point3Df(p.x, p.z, -p.y); // rotate so that Y is depth
	}
	bool computeNormal(float scale,const cv::Mat &img, const cv::Rect &rect, Point3Df &normal, float &d, const cv::Size &steps) const;
	void SegmentWholeImageWithCalculationNormal(
		const cv::Mat& in, cv::Mat& res, cv::Mat& comps, cv::Mat& depth0, 
		const size_t space_step, vector <OvalBase>& ovals) const;
	void newSegmentation(const cv::Mat& in, cv::Mat& res, cv::Mat& comps, cv::Mat& depth0, 
		vector <OvalBase>& ovals) const;
	void genChunks(const cv::Mat &depth, cv::Mat& output, size_t spacial_step, Chunks &chunks,
		float outscale, bool yzchange, int mode = 1, int dx = 0, int dy = 0) const;
	void genPointCloud(float scale, const cv::Mat &img, const std::string &filename, float k, bool yzchange = true, int save_every_that_point = 10) const;
    void genColorPointCloud(float scale, const cv::Mat &img, const std::string &filename, float k, bool yzchange = true, int save_every_that_point = 10) const;
	void genVolumeNNDataset(const cv::Mat &depth, Dataset &dataset) const;
};

class NeuralRayTracer {
	SimpleNet net;
	inline float f(const Point3Df &p, vector<float> &storage) const {
		float res = 0;
		net.compute(storage, (const float*)&p, &res);
		return res;
	}
public:
	NeuralRayTracer(const string &model) {
		net.load(model.c_str());
	}
	void genNNSurfaceCloud(float scale, const cv::Size &sz, const DepthProcessor &dproc, float k) const;
};
