// (c) Pawlin Technologies Ltd 2019
// File: cvmathdrawingnew.h, .cpp
// Purpose: file for opencv extensions code for mathematics and visualizations
// Author: P.V. Skribtsov
// ALL RIGHTS RESERVED

#pragma once
#include <atomic>
#include <thread>
// OpenCV
#include <opencv2/opencv.hpp>

#include <PWNGeometry/geometricobjects.h>
#include <PWNGeometry/convexpolyintersect.hpp>
#include <PWNImageObj/OvalBase.h>
#include <PWNGeneral/Dataset.h>
#include <PWNProfiler/duration.h>
#include "cvdrawingcompat.h"
#include <PWNGeneral/CycleBuffer.h>

#define FOR_MAT(cvmat,cvpoint) for(cv::Point cvpoint=cv::Point(0,0); cvpoint.y<cvmat.rows;cvpoint.y++) for(cvpoint.x=0;cvpoint.x<cvmat.cols;cvpoint.x++)

namespace cv {
	typedef std::pair<cv::Point, cv::Point> Segment;
}

namespace pawlin {
	inline cv::Scalar str2rgb(const string &hex3colorcode) {
		uint32_t hexValue = 0;
		sscanf(hex3colorcode.c_str(), "%x", &hexValue);
		cv::Scalar userColor;
		userColor[2] = ((hexValue >> 16) & 0xFF);  // Extract the RR byte
		userColor[1] = ((hexValue >> 8) & 0xFF);   // Extract the GG byte
		userColor[0] = ((hexValue) & 0xFF);        // Extract the BB byte
		return userColor;
	}
	inline float iou(const cv::Rect &a, const cv::Rect &b) {
		if (a.empty() && b.empty()) return 1.0f; // two empty rects are similar
		float intersection = float((a & b).area());
		float union_val = float((a | b).area());
		float iou = intersection / union_val;
		return iou;
	}
	inline cv::Rect merge(const cv::Rect &r1, const cv::Rect &r2, float k1) {
		float x = r1.x*k1 + r2.x*(1.0f - k1);
		float y = r1.y*k1 + r2.y*(1.0f - k1);
		float w = r1.width*k1 + r2.width*(1.0f - k1);
		float h = r1.height*k1 + r2.height*(1.0f - k1);
		return cv::Rect(
			float2int(x),
			float2int(y),
			float2int(w),
			float2int(h)
		);
	}
	inline cv::Rect2f merge(const cv::Rect2f &r1, const cv::Rect2f &r2, float k1) {
		float x = r1.x*k1 + r2.x*(1.0f - k1);
		float y = r1.y*k1 + r2.y*(1.0f - k1);
		float w = r1.width*k1 + r2.width*(1.0f - k1);
		float h = r1.height*k1 + r2.height*(1.0f - k1);
		return cv::Rect2f(x, y, w, h);
	}

	inline cv::Mat joinMatPanno(size_t nWidth, const vector<cv::Mat> &array) {
		cv::Size n_by_m((int)nWidth, (int)std::ceil((double)array.size() / (double)nWidth));
		cv::Size elemSize = array.front().size();
		cv::Mat out = cv::Mat(cv::Size(elemSize.width*n_by_m.width, elemSize.height*n_by_m.height), array.front().type());
		//assertEqualSize(array.size(), (size_t)n_by_m.area(), "pawlin::joinMatPanno - array size mismatch");
		out = 0;
		FOR_ALL(array, i) {
			assertEqualSize((size_t)elemSize.area(), (size_t)array[i].total(), "pawlin::joinMatPanno images are different sizes");
			size_t xIndex = i % nWidth;
			size_t yIndex = i / nWidth;
			cv::Point lt((int)xIndex*elemSize.width, (int)yIndex*elemSize.height);
			array[i].copyTo(out(cv::Rect(lt, elemSize)));
		}
		return out;
	}

	inline cv::Rect getCorrectRect(const cv::Point &begin, const cv::Point &end, bool inclusive) {
		int k = inclusive ? 0 : -1;
		return cv::Rect(
			cv::Point(
				std::min<int>(begin.x, end.x),
				std::min<int>(begin.y, end.y)
			),
			cv::Size(
				abs(begin.x - end.x)+k,
				abs(begin.y - end.y)+k
			)
		);
	}
	inline cv::Rect2f getCorrectRect(const cv::Point2f &begin, const cv::Point2f &end, bool inclusive) {
		int k = inclusive ? 0 : -1;
		return cv::Rect2f(
			cv::Point2f(
				std::min<float>(begin.x, end.x),
				std::min<float>(begin.y, end.y)
			),
			cv::Size2f(
				abs(begin.x - end.x) + k,
				abs(begin.y - end.y) + k
			)
		);
	}

	inline void smartScale(const cv::Mat &in, cv::Mat &out, float scale) {
		cv::resize(in, out, cv::Size(), scale, scale, scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
	}

	inline cv::Mat smartResize(const cv::Mat &in, const cv::Size &newsize) {
		if (in.size() == newsize) return in.clone();
		cv::Mat out;
		cv::resize(in, out, newsize, 0, 0, newsize.width < in.cols ? cv::INTER_AREA : cv::INTER_LINEAR);
		return out;
	}

	// renders row as square cv::Mat
	inline cv::Mat renderRAW(const cv::Size &sz, const FloatVector &row, bool out8U = true) {
		size_t check = (size_t) sz.area();
		assertEqualSize(check, row.size(), "pawlin::renderRAW row.size() must be sz.width * sz.height");
		cv::Mat img(sz, CV_32F);
		memcpy(img.data, &row[0], sizeof(float)*check);
		if(out8U) img.convertTo(img, CV_8U, 255.0);
		return img;
	}

	inline cv::Mat renderRAW(const FloatVector &row, bool out8U = true) { // special SQUARE image case
		int imgsize = float2int(sqrtf(float(row.size())));
		assertEqualSize((size_t)imgsize*imgsize, row.size(), "MaskAutoencoderTrainer::renderRAW row.size() must be square of img size");
		return pawlin::renderRAW(cv::Size(imgsize, imgsize), row, out8U);
	}

	inline void blend(cv::Mat &inout_rgb, const cv::Mat &background_rgb8U) {
		cv::Mat a, b;
		if (inout_rgb.type() == CV_8UC3) inout_rgb.convertTo(a, CV_32FC3, 1.0 / 255.0);
		else if (inout_rgb.type() == CV_32FC3) a = inout_rgb;
		else throw std::runtime_error("blend - input mask format must be 8UC3 or CV32FC3");
		background_rgb8U.convertTo(b, CV_32FC3, 1.0 / 255.0);
		cv::Mat c;
		cv::multiply(a, b, c);
		c.convertTo(inout_rgb, CV_8U, 255.0);
	}

	inline cv::Rect cvrect(const Rect2Df &rect) {
		return cv::Rect(
			float2int(rect.left),
			float2int(rect.bottom),
			float2int(rect.width()),
			float2int(rect.height())
		);
	}

	template <typename T = float>
	void buildMatHist(const cv::Mat &image32F, std::map<Point3Df, int> &hist, bool verbose) {
		vector<cv::Mat> channels;
		cv::split(image32F, channels);
		FOR_MAT(image32F, p) {
			Point3Df key;
			FOR_ALL(channels, c) {
				key[(int)c] = (float)channels[c].at<T>(p);
			}
			hist[key]++;
		}
		if (verbose) {
			for (auto &iter : hist) printf("Color(%f,%f,%f) count %d\n",
				iter.first.x,
				iter.first.y,
				iter.first.z,
				iter.second);
		}
	}

	cv::Mat buildMask4Color(const cv::Mat &mask8UC3, cv::Scalar color);

	inline cv::Vec3b scalar2vec3b(const cv::Scalar &s) {
		return cv::Vec3b(
			uint8_t(s[0]),
			uint8_t(s[1]),
			uint8_t(s[2])
		);
	}
	inline cv::Scalar genRandomColor(uint8_t minv = 50) {
		cv::Scalar color;
		color[0] = rand() & 255;
		color[1] = rand() & 255;
		color[2] = rand() & 255;
		if (color[0] < minv && color[1] < minv && color[2] < minv) color[rand() % 3] = minv + rand() % (255 - minv);
		return color;
	}
	inline void genRandomColors(vector<cv::Scalar> &colors, size_t count, uint8_t minv = 50) {
		for (size_t i = 0; i < count; i++) {
			colors.push_back(genRandomColor(minv));
		}
	}

	inline cv::Rect rect(const cv::Segment &s) {
		const cv::Point &g_pt1 = s.first;
		const cv::Point &g_pt2 = s.second;
		cv::Point pt1, pt2;
		pt1.x = std::min<int>(g_pt1.x, g_pt2.x);
		pt1.y = std::min<int>(g_pt1.y, g_pt2.y);
		pt2.x = std::max<int>(g_pt1.x, g_pt2.x);
		pt2.y = std::max<int>(g_pt1.y, g_pt2.y);
		cv::Rect selected_roi;
		selected_roi.x = pt1.x;
		selected_roi.y = pt1.y;
		selected_roi.width = pt2.x - pt1.x + 1;
		selected_roi.height = pt2.y - pt1.y + 1;
		return selected_roi;
	}

	inline cv::Rect imageRect(const cv::Mat &img) {
		return cv::Rect(0, 0, img.cols, img.rows);
	}

	inline bool checkRect(const cv::Rect &rect, const cv::Mat &img) {
		return rect == (rect & imageRect(img));
	}

	inline cv::Rect cvRect(const Rect2Df& rect) {
		return cv::Rect(
			float2int(rect.left), 
			float2int(rect.bottom), 
			float2int(rect.right - rect.left), 
			float2int(rect.top - rect.bottom));
	}

	inline uint32_t sqdistPoint32(cv::Point a, cv::Point b) {
		return uint32_t(sqr<int32_t>(a.x - b.x)) + uint32_t(sqr<int32_t>(a.y - b.y));
	}
	inline float sqdistPoint2f(cv::Point2f a, cv::Point2f b) {
		return sqr<float>(a.x - b.x) + sqr<float>(a.y - b.y);
	}
	inline void drawDashRect(cv::Mat &mat, const cv::Rect &rect, const cv::Scalar &color, int step, int thickness) {
		for (int i = rect.x; i < rect.x + rect.width; i+= step*2) {
			cv::line(mat, cv::Point(i, rect.y), cv::Point(i + step, rect.y), color, thickness);
			cv::line(mat, cv::Point(i, rect.y+rect.height), cv::Point(i + step, rect.y+rect.height), color, thickness);
		}
		for (int i = rect.y; i < rect.y + rect.height; i += step*2) {
			cv::line(mat, cv::Point(rect.x,i), cv::Point(rect.x,i + step), color, thickness);
			cv::line(mat, cv::Point(rect.x+rect.width, i), cv::Point(rect.x+rect.width, i + step), color, thickness);
		}
	}

	inline bool getNearestMaskPixel(const cv::Mat &mask8U, const cv::Point &origin, cv::Point &bestPoint, float maxdist) {
		uint32_t bestDist = 0xFFFFFFFF;
		cv::Point cp;
		bestPoint = origin;
		for (cp.y = 0; cp.y < mask8U.rows; cp.y++) { // not the best algorithm, if mask=1 close to origin does not make sense to iterate more far regions
			for (cp.x = 0; cp.x < mask8U.cols; cp.x++) {
				uint32_t dist = sqdistPoint32(origin, cp);
				if (dist > maxdist || dist > bestDist) continue;
				if (mask8U.at<char>(cp)) {
					if (dist < bestDist) bestPoint = cp, bestDist = dist;
				}
			}
		}
		return (bestDist != 0xFFFFFFFF);
	}

	inline Point3Df ptconv(const cv::Point3f &p) { return Point3Df(p.x, p.y, p.z); }
	inline Point3Df ptconv(const Point2Df &p) { return Point3Df(p.x, p.y, 0); }
	inline Point2Df ptconv(const cv::Point2f &p) { return Point2Df(p.x, p.y); }
	inline Point2Df ptconv(const cv::Point &p) { return Point2Df((float)p.x, (float)p.y); }
	inline cv::Point2f ptconv2f(const Point2Df &p) { return cv::Point2f(p.x, p.y); }
	inline cv::Point3f ptconv3f(const Point3Df &p) { return cv::Point3f(p.x, p.y,p.z); }
	inline cv::Point2f ptconv2f(const cv::Point &p) { return cv::Point2f((float)p.x, (float)p.y); }
	inline cv::Rect2f ptconv2f(const cv::Rect &r) { 
		return cv::Rect2f((float)r.x, (float)r.y,(float)r.width,(float)r.height); 
	}
	inline cv::Point ptconv2i(const Point2Df &p) { return cv::Point(float2int(p.x), float2int(p.y)); }
	inline cv::Point ptconv2i(const cv::Point2f &p) { return cv::Point(float2int(p.x), float2int(p.y)); }
	inline void convert(const vector<Point2Df> &pts, vector<cv::Point> &out) {
		out.resize(pts.size());
		FOR_ALL(pts, i) out[i] = ptconv2i(pts[i]);
	}
	inline void convert(const vector<Point2Df> &pts, vector<cv::Point2f> &out) {
		out.resize(pts.size());
		FOR_ALL(pts, i) out[i] = ptconv2f(pts[i]);
	}
	inline void convert(const vector<cv::Point2f> &pts, vector<Point2Df> &out) {
		out.resize(pts.size());
		FOR_ALL(pts, i) out[i] = ptconv(pts[i]);
	}
	inline void convert(const vector<cv::Point> &pts, vector<cv::Point2f> &out) {
		out.resize(pts.size());
		FOR_ALL(pts, i) out[i] = ptconv2f(pts[i]);
	}
	inline void convert(const vector<cv::Point> &pts, vector<Point2Df> &out) {
		out.resize(pts.size());
		FOR_ALL(pts, i) out[i] = ptconv(pts[i]);
	}
	inline cv::Point convert(const vector<cv::Point2f> &vec, vector<cv::Point> &out) {
		out.resize(vec.size());
		MinMaxAvg statx, staty;
		FOR_ALL(vec, i) {
			out[i] = ptconv2i(ptconv(vec[i]));
			statx.take((float)out[i].x);
			staty.take((float)out[i].y);
		}
		return ptconv2i(Point2Df(statx.getAvg(), staty.getAvg()));
	}
	inline cv::Point convert(const vector<cv::Point2f> &vec, const Box2Df &box, const cv::Size &sz, vector<cv::Point> &out) {
		out.resize(vec.size());
		MinMaxAvg statx, staty;
		FOR_ALL(vec, i) {
			out[i] = ptconv2i(box.box2img(sz.width, sz.height, ptconv(vec[i])));
			statx.take((float)out[i].x);
			staty.take((float)out[i].y);
		}
		return ptconv2i(Point2Df(statx.getAvg(), staty.getAvg()));
	}
	inline void convert(vector<cv::Point> &imgpts, const Box2Df &box, const cv::Size &sz, vector<cv::Point2f> &out) {
		out.resize(imgpts.size());
		FOR_ALL(imgpts, i) {
			out[i] = ptconv2f(box.img2box(sz.width, sz.height, Point2Df(
				float(imgpts[i].x),
				float(imgpts[i].y))
			));
		}
	}

	inline void convert(vector<vector<cv::Point2f>> &vec, const Box2Df &box, const cv::Size &sz, vector<vector<cv::Point>> &out) {
		out.resize(vec.size());
		FOR_ALL(vec, i) convert(vec[i], box, sz, out[i]);
	}

	// translates coordinate from a clipped subrect to the re-scaled content
	inline cv::Point translate(const cv::Point2f &pt, const cv::Rect &clip, const cv::Size &winsize, bool invertY = true) {
		if (invertY) {
			return ptconv2i(
				Box2Df(
					ptconv(clip.tl()),
					ptconv(clip.br())
				).box2img(
					winsize.width,
					winsize.height,
					ptconv(pt)
				)
			);
		}
		else {
			return ptconv2i(
				Box2Df(
					ptconv(clip.tl()),
					ptconv(clip.br())
				).box2img_noinvert(
					winsize.width,
					winsize.height,
					ptconv(pt)
				)
			);
		}

	}

	inline OvalBase rect2oval(const cv::Rect &cur_rect) {
		Point2Df centre(float(cur_rect.x + cur_rect.width / 2.0f), float(cur_rect.y + cur_rect.height / 2.0f));
		Point2Df axeA(centre.x, float(cur_rect.y));
		Point2Df axeB(float(cur_rect.x), centre.y);

		OvalBase o(centre, axeA, axeB);
		return o;
	}
	inline cv::Rect place(cv::Rect r, cv::Point pt) {
		return cv::Rect(pt.x - r.width / 2, pt.y - r.height / 2, r.width, r.height);
	}
	inline cv::Rect2f place(cv::Rect2f r, cv::Point2f pt) {
		return cv::Rect2f(pt.x - r.width / 2, pt.y - r.height / 2, r.width, r.height);
	}

	inline cv::Rect place(const cv::Size &sz, cv::Point pt) {
		return place(cv::Rect(cv::Point(0,0),sz),pt);
	}
	inline void scaleRect(cv::Rect &r, float scale) {
		if (scale != 1.0f) {
			r.x += float2int(r.width*(1.0f - scale)*0.5f);
			r.y += float2int(r.height*(1.0f - scale)*0.5f);
			r.width = float2int(r.width*scale);
			r.height = float2int(r.height*scale);
		}
	}
	inline void multRect(cv::Rect &r, float scale) {
		if (scale != 1.0f) {
			r.x = float2int(r.x*scale);
			r.y = float2int(r.y*scale);
			r.width = float2int(r.width*scale);
			r.height = float2int(r.height*scale);
		}
	}
	inline cv::Rect multRect2(const cv::Rect &r, float scale) {
		cv::Rect res = r;
		multRect(res, scale);
		return res;
	}
	// compute mean in oval assuming this oval was rendered into mask8U parameter as mask
	inline cv::Scalar meanInOval(const OvalBase &oval, const cv::Mat &source, const cv::Mat &mask8U, float scale = 1.0f) {
		Rect2Df r = oval.getRect('W');
		cv::Rect rect(
			float2int(r.left),
			float2int(r.bottom),
			float2int(r.right - r.left),
			float2int(r.top - r.bottom));
		scaleRect(rect, scale);
		return cv::mean(source(rect), mask8U(rect));
	}
	inline cv::Point center(const cv::Rect &r) {
		cv::Point lp = (r.tl() + r.br()) / 2;
		return lp;
	}
	inline cv::Point2f centerf(const cv::Rect &r) {
		cv::Point2f lp = (r.tl() + r.br()) * 0.5f;
		return lp;
	}
	inline cv::Scalar meanInRect(const cv::Rect &rect, const cv::Mat &source, const cv::Mat &mask8U) {
		return cv::mean(source(rect), mask8U(rect));
	}
	inline std::pair<cv::Scalar,cv::Scalar> meanStdevInRect(const cv::Rect &rect, const cv::Mat &source, const cv::Mat &mask8U) {
		cv::Scalar mean, stddev;
		cv::meanStdDev(source(rect), mean, stddev, mask8U(rect));
		return std::make_pair(mean, stddev);
	}
	inline std::pair<cv::Scalar, cv::Scalar> meanStdevInRect(const cv::Mat &source, const cv::Mat &mask8U) {
		cv::Scalar mean, stddev;
		cv::meanStdDev(source, mean, stddev, mask8U);
		return std::make_pair(mean, stddev);
	}
	inline float length(cv::Point p1, cv::Point p2) {
		return pawlin::ptconv(p1).distance(pawlin::ptconv(p2));
	}
	inline float length(const cv::Segment &s) {
		return pawlin::ptconv(s.first).distance(pawlin::ptconv(s.second));
	}
	inline float angle(cv::Point p1, cv::Point p2) {
		Point2Df p = pawlin::ptconv(p1)-pawlin::ptconv(p2);
		return atanf(p.y / p.x);
	}

	inline float diagonal(cv::Rect in) {
		return length(in.tl(), in.br());
	}
	inline cv::Rect expand(cv::Rect in, int s) {
		return cv::Rect(in.x - s, in.y - s, in.width + 2 * s, in.height + 2 * s);
	}
	inline cv::Rect shift(cv::Rect in, cv::Point d) {
		return cv::Rect(in.x + d.x, in.y + d.y, in.width, in.height);
	}
	inline cv::Rect expand(cv::Rect in, cv::Size s) {
		return cv::Rect(
			in.x - s.width, 
			in.y - s.height, 
			in.width + 2 * s.width, 
			in.height + 2 * s.height);
	}
	inline void drawFlow(cv::Mat & canvas, const cv::Mat &flow, float scale = 1.0f, int step = 7, cv::Scalar color = CV_RGB(0, 0, 255), bool unitlength = false) {
		for (int y = 0; y < canvas.rows; y += step) {
			for (int x = 0; x < canvas.cols; x += step)
			{
				// get the flow from y, x position * 3 for better visibility
				cv::Point2f flowatxy = flow.at<cv::Point2f>(y, x);
				if (unitlength) {
					Point2Df temp = pawlin::ptconv(flowatxy);
					if (temp.norm())	flowatxy = pawlin::ptconv2f(temp.normated());
				}
				// draw line at flow direction
				line(canvas, cv::Point(x, y), cv::Point(cvRound(x + flowatxy.x* scale), cvRound(y + flowatxy.y* scale)), color);
			}
		}
	}

	// can also implement computing both mean & stdev using meanStdDev(InputArray src, OutputArray mean, OutputArray stddev, InputArray mask=noArray())) 

	//void drawOvalBase(const OvalBase &oval, cv::Scalar color, int width, float scaleX, float scaleY, cv::Mat &img, bool drawAxis = true);

#define OPENCV_RIGHT_ARROW_CODE 0x270000
#define OPENCV_UP_ARROW_CODE 0x260000
#define OPENCV_DOWN_ARROW_CODE 0x280000
#define OPENCV_LEFT_ARROW_CODE 0x250000
#define OPENCV_RIGHT_ARROW_CODE_TERMINAL_LINUX 0xFF53
#define OPENCV_LEFT_ARROW_CODE_TERMINAL_LINUX 0xFF51
#define OPENCV_UP_ARROW_CODE_TERMINAL_LINUX 0xFF52
#define OPENCV_DOWN_ARROW_CODE_TERMINAL_LINUX 0xFF54
#define OPENCV_LEFT_ARROW_CODE_XMING 0x10FF51
#define OPENCV_RIGHT_ARROW_CODE_XMING 0x10FF53
#define OPENCV_UP_ARROW_CODE_XMING 0x10FF52
#define OPENCV_DOWN_ARROW_CODE_XMING 0x10FF54
#define OPENCV_DELETE_CODE 0x2E0000

#define OPENCV_SPECIAL_KEY(k) ((k & 0x20FF00) != 0)
	inline bool check_right_arrow_key(int k) {
		return (k == OPENCV_RIGHT_ARROW_CODE || k == OPENCV_RIGHT_ARROW_CODE_XMING || k == OPENCV_RIGHT_ARROW_CODE_TERMINAL_LINUX);
	}
	inline bool check_left_arrow_key(int k) {
		return (k == OPENCV_LEFT_ARROW_CODE || k == OPENCV_LEFT_ARROW_CODE_XMING || k == OPENCV_LEFT_ARROW_CODE_TERMINAL_LINUX);
	}
	inline bool check_up_arrow_key(int k) {
		return (k == OPENCV_UP_ARROW_CODE || k == OPENCV_UP_ARROW_CODE_XMING || k == OPENCV_UP_ARROW_CODE_TERMINAL_LINUX);
	}
	inline bool check_down_arrow_key(int k) {
		return (k == OPENCV_DOWN_ARROW_CODE || k == OPENCV_DOWN_ARROW_CODE_XMING || k == OPENCV_DOWN_ARROW_CODE_TERMINAL_LINUX);
	}
	class PolyModel1D {
		FloatVector coefs;
		MinMaxAvg errStat;
	public:
		const MinMaxAvg getErrStat() const {
			return errStat;
		}
		static void testPoly();
		const FloatVector &getCoefs() const { return coefs; }
		PolyModel1D() {}
		PolyModel1D(const Dataset &data, size_t N) : coefs(N + 1, 0) {
			cv::Mat A((int)data.size(), (int)N + 1, CV_32F);
			cv::Mat B((int)data.size(), 1, CV_32F);
			FOR_ALL(data.rows, i) {
				for (size_t n = 0; n <= N; n++) A.at<float>((int)i, (int)n) = powf(data.rows[i].data[0], (float)(N - n));
				B.at<float>((int)i) = data.rows[i].data[1];
			}
			cv::Mat res;
			cv::solve(A, B, res, cv::DECOMP_SVD);
			memcpy(&coefs.data[0], res.data, sizeof(float)*coefs.size());
			//coefs.print(10);
		}

		float compute(float x) const {
			float m = 1.0f;
			float sum = 0;
			FOR_ALL(coefs, i) {
				sum += m*coefs[coefs.size() - 1 - i];
				m *= x;
			}
			return sum;
		}

		float computeD(float x) const {
			if(coefs.size() == 4) return coefs[2] + 2 * coefs[1] * x + 3 * coefs[0] * x*x;
			if (coefs.size() == 3) return coefs[1] + 2 * coefs[0] * x ;
			if (coefs.size() == 2) return coefs[0];
			float sum = 0;
			float p = 1;
			for (size_t i = 1; i < coefs.size(); i++, p*=x) {
				sum += coefs[coefs.size() - 1 - i] * p*i;
			}
			//throw std::runtime_error("not valid call for models with higher degree > 4");
			return sum;
		}

		float computeD2(float x) const {
			if (coefs.size() == 4) return 2 * coefs[1] + 3 * 2 * coefs[0] * x;
			if (coefs.size() == 3) return 2 * coefs[0];
			if (coefs.size() == 2) return 0;
			float sum = 0;
			float p = 1;
			for (size_t i = 2; i < coefs.size(); i++, p *= x) {
				sum += coefs[coefs.size() - 1 - i] * p*i*(i-1);
			}
			return sum;
			//throw std::runtime_error("not valid call for models with higher degree > 4");
		}
		struct Regularizer {
			vector <float> sampleWeights; // if not specified for an index, assuming it's one
			Dataset pinPoints; // additional points added in the end of the est - 3 columns [x][y][weight]
			float noiseratio;
			Regularizer(float stress_first, float regx, float regy, float regw) : noiseratio(0) {
				sampleWeights.push_back(stress_first);
				FloatVector row;
				row.push_back(regx);
				row.push_back(regy);
				row.push_back(regw);
				pinPoints.push_back(row);
			}
			Regularizer(float stress_first) : noiseratio(0) {
				sampleWeights.push_back(stress_first);
			}

			Regularizer() : noiseratio(0) {}
		};
		PolyModel1D(const Dataset &data, size_t xColumn, size_t yColumn, size_t N, const Regularizer &reg, Dataset &debug) : coefs(N + 1, 0) {
			size_t R = reg.pinPoints.size(); // number of additional points
			cv::Mat A((int)(data.size() + R), (int)N + 1, CV_32F);
			cv::Mat B((int)(data.size() + R), 1, CV_32F);
			cv::Mat Aclean((int)data.size(), (int)N + 1, CV_32F); // without gain for debug checks
			FOR_ALL(data.rows, i) {
				float v = data.rows[i].data[xColumn];
				float noise = 2.0f*(rnd() - 0.5f)*v*reg.noiseratio;
				float gain = (i < reg.sampleWeights.size()) ? reg.sampleWeights[i] : 1.0f;
				for (size_t n = 0; n <= N; n++) {
					float coef = powf(v + noise, float(N - n));
					A.at<float>((int)i, (int)n) = gain*coef;
					Aclean.at<float>((int)i, (int)n) = coef;
				}
				B.at<float>((int)i) = data.rows[i].data[yColumn] * gain;
			}
			// set intercept knowlege
			const int lastrow = (int)data.size();
			for (size_t i = 0; i < R; i++) {
				const float gain = reg.pinPoints[i][2]; // importance of intercept constraint
				for (size_t n = 0; n <= N; n++) {
					float v = reg.pinPoints[i][0];
					A.at<float>(lastrow + (int)i, (int)n) = gain*powf(v, float(N - n));
				}
				B.at<float>(lastrow + (int)i) = reg.pinPoints[i][1];
			}
			cv::Mat res;
			cv::solve(A, B, res, cv::DECOMP_SVD);
			memcpy(&coefs.data[0], res.data, sizeof(float)*coefs.size());
			cv::Mat trend = Aclean*res;
			FOR_ALL(data.rows, i) {
				FloatVector row;
				float value = trend.at<float>((int)i); // model output after training
				float target = data.rows[i].data[yColumn]; // target values
				errStat.take(value - target);
				row.data.push_back(value); 
				row.data.push_back(target); 
				debug.rows.push_back(row);
			}
			//coefs.print(10);
		}


	};
	class QuadSurface {
		FloatVector coefs; // x,y,xy,xx,yy,1
		MinMaxAvg fitstat;
		MinMaxAvg zstat;
	public:
		const MinMaxAvg &getZStat() const { return zstat; }
		const MinMaxAvg &getErrStat() const { return fitstat; }
		const FloatVector &getCoefs() const { return coefs; }
		QuadSurface() {}
		QuadSurface(const vector<Point3Df> &data, bool flat) : coefs(6, 0) {
			cv::Mat A((int)data.size(), (int)coefs.size(), CV_32F);
			cv::Mat B((int)data.size(), 1, CV_32F);
			FOR_ALL(data, i) {
				zstat.take(data[i].z);
			}
			FOR_ALL(data, i) {
				const Point3Df &p = data[i];
				float k = flat ? (rnd() - 0.5f)*zstat.getRange()*1000.0f : 1.0f;
				A.at<float>((int)i, 0) = p.x;
				A.at<float>((int)i, 1) = p.y;
				A.at<float>((int)i, 2) = p.x*p.y*k;
				A.at<float>((int)i, 3) = p.x*p.x*k;
				A.at<float>((int)i, 4) = p.y*p.y*k;
				A.at<float>((int)i, 5) = 1.0f;

				B.at<float>((int)i) = p.z;
			}
			cv::Mat res;
			cv::solve(A, B, res, cv::DECOMP_SVD);
			memcpy(&coefs.data[0], res.data, sizeof(float)*coefs.size());
			//coefs.print(10);
			FOR_ALL(data, i) fitstat.take(compute(data[i]) - data[i].z);
		}
		float compute(const Point3Df &p) const { return compute(p.x, p.y); }
		inline float compute(const FloatVector &xx) const {
			return xx.dotproduct(coefs);
		}
		void makeVector(float x, float y, FloatVector &xx) const {
			xx.resize(coefs.size());
			xx[0] = x;
			xx[1] = y;
			xx[2] = x * y;
			xx[3] = x * x;
			xx[4] = y * y;
			xx[5] = 1.0f;
		}
		float compute(float x, float y) const {
			FloatVector vx = FloatVector(vector<float>{
				x,
					y,
					x*y,
					x*x,
					y*y,
					1.0f
			});
			float sum = vx.dotproduct(coefs);
			return sum;
		}
	};


	class CursorHandler {
		int cursor;
	public:
		void run(const std::string &winname, const cv::Mat &background);
		enum Event {POS, CLICK, KEY};
		static void onMouse(int event, int x, int y, int flags, void *param);
		virtual void onColumn(int x, Event event, int param) = 0;
	};

	struct TrajQuadForecast {
		cv::Mat A, B, C /*coefs*/;
		cv::Mat Bf2; // aux-> link to B
		const int power = 3;
		cv::Mat T;
		size_t cursize = 0;
		size_t maxsize;
		bool empty() const { return A.empty(); }
		TrajQuadForecast(size_t maxsize) : maxsize(maxsize) {}
		void init(const vector<cv::Point2f> &start) {
			cursize = std::min<size_t>(start.size(),maxsize);
			
			if (cursize < power) throw std::runtime_error("TrajQuadForecast too short history");
			A = cv::Mat((int)maxsize, power, CV_32F);
			B = cv::Mat((int)maxsize, 2, CV_32F);
			Bf2 = cv::Mat(cv::Size(1, B.rows), CV_32FC2, B.data); // pointer to same array
			T = cv::Mat(1, power, CV_32F);
			for (int i = 0; i < (int) maxsize; i++) {
				if(i < cursize) Bf2.at<cv::Point2f>(i) = start[start.size() - cursize + i];
				for (int p = 0; p < power; p++)
					A.at<float>(i, p) = powf((float)i, (float)p);
			}
			for (int p = 0; p < power; p++)	T.at<float>(0, p) = powf((float)maxsize, (float)p);
			solve();
		}
		void solve() {
			cv::solve(
				A(cv::Rect(0, 0, power	, (int)cursize)),
				B(cv::Rect(0, 0, 2		, (int)cursize)), C, cv::DECOMP_SVD);
		}
		void update(const cv::Point2f &pt) {
			if (cursize < maxsize) {
				cursize ++;
			}
			else {
				for (int i = 0; i < B.rows - 1; i++) Bf2.at<cv::Point2f>(i) = Bf2.at<cv::Point2f>(i + 1);
			}
			Bf2.at<cv::Point2f>((int)cursize-1) = pt;
			solve();
		}
		cv::Point2f forecastNext(bool verbose = false) const {
			cv::Mat Tc = T.clone();
			if (cursize < maxsize) {
				for (int p = 0; p < power; p++)	
					Tc.at<float>(0, p) = powf((float)cursize, (float)p);
			}
			cv::Mat P = Tc*C;
			if (verbose) {
				std::cout << "B\n" << B << "\n";
				std::cout << "P\n" << P << "\n";
			}
			return { P.at<float>(0,0),P.at<float>(0,1) };
		}
		cv::Point2f lastSpeed() const {
			float t = 1.0f;
			cv::Point2f res(0, 0);
			for (int i = 0; i < power - 1; i++) {
				res.x += t * (i + 1) * C.at<float>(i + 1, 0);
				res.y += t * (i + 1) * C.at<float>(i + 1, 1);
				t *= (float)(cursize - 1);
			}
			return res;
		}
	};


	class XYPoly2Transform {
		cv::Mat C; // coefs 
		cv::Mat A, B;
		// 1 x y xx yy xy --> 6 coefs for each dim, so C structure is 2x6
		const int paramsCount = 6;
		void fill_coefs(cv::Mat &mat, int row, const cv::Point2f &p) const {
			float *ptr = &(mat.at<float>(row, 0));
			ptr[0] = 1.0f;
			ptr[1] = p.x;
			ptr[2] = p.y;
			ptr[3] = p.x*p.x;
			ptr[4] = p.y*p.y;
			ptr[5] = p.x*p.y;
		}
	public:
		void save(const string &filename) const {
			cv::FileStorage fs(filename, cv::FileStorage::WRITE);
			fs << "C" << C;
			fs.release();
		}
		void load(const string &filename) {
			cv::FileStorage fs(filename, cv::FileStorage::READ);
			if (!fs.isOpened()) throw std::runtime_error("could not open file " + filename);
			fs["C"] >> C;
			fs.release();
		}
		XYPoly2Transform() {}
		XYPoly2Transform(const string &filename) { load(filename); }
		XYPoly2Transform(const vector<cv::Point2f> &from, const vector<cv::Point2f> &to) {
			cv::Mat Bf2; // aux-> link to B
			assertEqualSize(from.size(), to.size(), "XYPoly2Transform::XYPoly2Transform from & to arrays must be equal in size");
			if (from.size() < (size_t)paramsCount)
				throw std::runtime_error("XYPoly2Transform too small dataset");
			int size = (int)from.size();
			A = cv::Mat(size, paramsCount, CV_32F);
			B = cv::Mat(size, 2, CV_32F);
			Bf2 = cv::Mat(cv::Size(1, B.rows), CV_32FC2, B.data); // pointer to same array
			for (int i = 0; i < size; i++) {
				Bf2.at<cv::Point2f>(i) = to[i];
				fill_coefs(A, i, from[i]);
			}
			//std::cout << "A\n" << A << "\n";
			//std::cout << "B\n" << B << "\n";
			cv::solve(
				A,
				B, C, cv::DECOMP_SVD);
		}
		bool empty() const { return C.empty(); }

		cv::Point2f transform(const cv::Point2f &from) const {
			cv::Mat T(1, paramsCount, CV_32F);
			fill_coefs(T, 0, from);
			cv::Mat P = T*C;
			return { P.at<float>(0,0),P.at<float>(0,1) };
		}

	};


	inline Point2Df pt_stdev(const vector<cv::Point2f> &history, size_t startIndex = 0) {
		MinMaxAvgPoint3Df stat;
		for(size_t i = startIndex; i < history.size(); i++) 
			stat.take(Point3Df(history[i].x, history[i].y, 0));
		return { stat.getStdev().x,stat.getStdev().y };
	}

	class CvInteractWindowBase {
	protected:
		string winname;
		cv::Size canv_size;
		vector<int> toggles = vector<int>(256);
		struct Selection {
			int lastSelected;
			Selection() : lastSelected(-1) {} // by default nothing is selected
		};
		Selection selection;
		int canv_type;
		bool no_destroy;
		int drawDelay = 30;
		float drawScale = 1.0f;
	public:
		void init_cv(cv::Point winloc = cv::Point(50,50)) {
			cv::namedWindow(winname.c_str(), cv::WINDOW_AUTOSIZE);
			if(winloc.x >=0 && winloc.y>=0) cv::moveWindow(winname.c_str(), winloc.x, winloc.y);
			cv::Mat temp(this->canv_size, canv_type);
			temp = 0;
			cv::imshow(winname, temp);
			cv::waitKey(30); // needed for correct sliders positioning
			cv::setMouseCallback(winname.c_str(), mouse_callback, this);
		}
		CvInteractWindowBase(const std::string &win, cv::Size canv_size, int canv_type, bool no_destroy = false, bool initcv = true) :
			winname(win),
			canv_size(canv_size),
			canv_type(canv_type),
			no_destroy(no_destroy)
		{
			if(initcv) init_cv(); // sometimes it must be done in a different thread rather than constructor
		}

		string name() const {
			return winname;
		}
		virtual void update() {
		}
		int run() {
			cv::Mat canvas = cv::Mat::zeros(this->canv_size, this->canv_type);
			for (;;) {
				update();
				show(canvas);
				int key = pawlin::debugImg(winname, canvas, drawScale, drawDelay, false);
				if (processKey(key)) break;
			}
			return getResult();
		}

		virtual int getResult() const {
			return 0;
		}

		virtual bool processKey(int key) {
			return key == 27 || key=='q';
		}

		virtual void show(cv::Mat &canvas) const {}

		static void track_callback(int pos, void *params);
		static void mouse_callback(int event, int x, int y, int flags, void *params);
		virtual void updateSliders() {};
/*
enum  	cv::MouseEventFlags {
cv::EVENT_FLAG_LBUTTON = 1,
cv::EVENT_FLAG_RBUTTON = 2,
cv::EVENT_FLAG_MBUTTON = 4,
cv::EVENT_FLAG_CTRLKEY = 8,
cv::EVENT_FLAG_SHIFTKEY = 16,
cv::EVENT_FLAG_ALTKEY = 32
}
Mouse Event Flags see cv::MouseCallback. More...

enum  	cv::MouseEventTypes {
cv::EVENT_MOUSEMOVE = 0,
cv::EVENT_LBUTTONDOWN = 1,
cv::EVENT_RBUTTONDOWN = 2,
cv::EVENT_MBUTTONDOWN = 3,
cv::EVENT_LBUTTONUP = 4,
cv::EVENT_RBUTTONUP = 5,
cv::EVENT_MBUTTONUP = 6,
cv::EVENT_LBUTTONDBLCLK = 7,
cv::EVENT_RBUTTONDBLCLK = 8,
cv::EVENT_MBUTTONDBLCLK = 9,
cv::EVENT_MOUSEWHEEL = 10,
cv::EVENT_MOUSEHWHEEL = 11
}*/
		virtual void updateClick(int x, int y, int event, int mouseflags) {};
		virtual ~CvInteractWindowBase() {
			if (!no_destroy) {
				printf("~CvInteractWindowBase()..");
				cv::destroyWindow(winname.c_str());
				printf("done\n");
			}
		}
	}; //end of CvInteractWindowBase


	class ThresholdSelector : public pawlin::CvInteractWindowBase {
		int mint, maxt;
		const int sliderDashes = 512;
		double minv, maxv;
		cv::Mat image;
		float scale;
	protected:
		cv::Mat mask;
	public:
		ThresholdSelector(const string &winname, const cv::Mat &image, float scale = 1.0f, bool initCV = true) :
			pawlin::CvInteractWindowBase(winname, image.size(), image.type(), false, initCV),
			image(image.clone()),
			scale(scale)
		{
			mint = 0, maxt = sliderDashes;
			cv::minMaxLoc(image, &minv, &maxv);
			pawlin::debugImg(winname, image, scale, 1, true);
			cv::createTrackbar("mincut", winname, &mint, sliderDashes);
			cv::createTrackbar("maxcut", winname, &maxt, sliderDashes);
		}
		double cut(int v) const {
			return minv + double(maxv - minv)*v / (double)sliderDashes;
		}
		cv::Mat generate() const {
			cv::Mat res;
			cv::threshold(image, res, cut(maxt), 0, cv::THRESH_TRUNC);
			cv::threshold(res - cut(mint), res, 0, 0, cv::THRESH_TOZERO);
			res += cut(mint);
			return res;
		}
		void convert(cv::Mat &imagemod) const {
			cv::normalize(imagemod, imagemod, 1, 0, cv::NORM_MINMAX);
			imagemod.convertTo(imagemod, CV_8U, 255);
			cv::cvtColor(imagemod, imagemod, cv::COLOR_GRAY2BGR);
		}
		double run(const cv::Mat &_back) {
			cv::Mat back;
			if (!_back.empty()) {
				if (_back.channels() == 3) back = _back;
				if (_back.channels() == 1) cv::cvtColor(_back, back, cv::COLOR_GRAY2BGR);
			}
			while (true) {
				cv::Mat imagemod = generate();
				convert(imagemod);
				cv::Mat blend = back.empty() ? imagemod : (imagemod + back)*0.5;
				compute();
				draw(blend);
				int key = pawlin::debugImg(winname, blend, scale, 30, true);
				if (processKey(key)) return thresh();
			}
		}
		virtual bool processKey(int key) { return key == 27; }
		double thresh() const {
			return 0.5*cut(mint) + 0.5*cut(maxt);
		}
		virtual void compute() {
			mask = image > thresh();
		}
		virtual void draw(cv::Mat &canvas) const {
			string text =
				"min:" + float2str(cut(mint), 1) + "  max:" + float2str(cut(maxt), 1) +
				" thresh:" + float2str(thresh(), 1);
			cv::putText(
				canvas,
				text,
				cv::Point(30, 30),
				cv::FONT_HERSHEY_PLAIN, 1.0 / scale, CV_RGB(255, 0, 0), int(1 + 1.0 / scale + 0.5));
			pawlin::debugImg("mask", mask, scale, -1);
		}
	};


	struct Control {
		int val;
		int count;
		float minv; float maxv;
		operator float() const {
			float t = val / (float)count;
			return minv + t*(maxv - minv);
		}
	};
	class ControlSet : public pawlin::SafeStringMap<Control> {
	public:

		void makeSliders(const string &winname) {
			for (auto &iter : getMap())
				cv::createTrackbar(iter.first, winname, &iter.second.val, iter.second.count);
		}
	};

	class PolySelector : public pawlin::CvInteractWindowBase {
		const vector<vector<cv::Point>> &polys;
		const vector<cv::Scalar> &colors;
		int selected = -1;
		cv::Scalar selectColor = CV_RGB(255, 0, 0);
	public:
		int getSelected() const {
			return selected;
		}
		PolySelector(
			const std::string &winname,
			const vector<vector<cv::Point>> &polys,
			const vector<cv::Scalar> &colors,
			cv::Size canvsize) :
			polys(polys),
			colors(colors),
			CvInteractWindowBase(winname, canvsize, CV_8UC3)
		{

		}

		virtual int getResult() const override {
			return getSelected();
		}

		virtual void show(cv::Mat &canvas) const override {
			canvas = 0;
			FOR_ALL(polys, i) {
				vector<vector<cv::Point>> temp(1, polys[i]);
				cv::fillPoly(canvas, temp, colors[i], cv::LINE_AA);
				if ((int)i == selected) {
					cv::polylines(canvas, temp, true, selectColor, 2);
				}
			}
		}

		virtual void updateClick(int x, int y, int event, int mouseflags) override {
			if (event == cv::EVENT_LBUTTONUP) {
				FOR_ALL(polys, i) {
					if (cv::pointPolygonTest(polys[i], cv::Point2f((float)x, (float)y), false) >= 0)
						selected = (int)i;
				}
			}
		};

	};


	// show and exit by click or keyboard
	struct DebugImgExt : public CvInteractWindowBase {
		struct Result {
			cv::Point click;
			int event, mouseflags, key;
			bool clicked;
			Result() {
				clicked = false;
				key = 0;
			}
			bool exit() const {
				return key != -1 || clicked == true;
			}
		};
		Result res;
		double scale;
		bool normalize;
		const cv::Mat &_img;
		int delay;
		DebugImgExt(const string &name, const cv::Mat &_img, double scale = 0.5, int delay = 30, bool normalize = true) :
			CvInteractWindowBase(name, _img.size(), _img.type()), scale(scale), normalize(normalize),_img(_img), delay(delay)
		{
		}
		Result run() {
			res.clicked = false;
			res.key = -1;
			for (;;) {
				res.key = pawlin::debugImg(winname, _img, scale, delay, normalize);
				if (res.exit()) break;
			}
			return res;
		}
		virtual void updateClick(int x, int y, int event, int mouseflags) override {
			res.click.x = x;
			res.click.y = y;
			res.event = event;
			res.mouseflags = mouseflags;
			if (event == cv::EVENT_LBUTTONDOWN || event == cv::EVENT_RBUTTONDOWN) res.clicked = true;
		};
	};

	// class for ROI selection by mouse by A. Dolgopolov
	// usage example:
	//
	//	SelectionROIWindow select_roi("SELECT PSF_ROI", src_img);
	//	select_roi.run();
	//	deblur_image_manual_psf(src_img, argv[1], select_roi.getROI());

	class SelectionROIWindow :public CvInteractWindowBase {
	protected:
		bool roi_capture; // roi capture process started
		bool changed_roi; // roi has been changed
		cv::Segment segment;
		cv::Mat m_src_im;
		cv::Scalar framecolor;
		bool lineMode;
	public:
		SelectionROIWindow(const std::string &win, const cv::Mat& src_im, cv::Scalar framecolor = CV_RGB(0, 255, 0), bool lineMode = false) 
			: CvInteractWindowBase(win, src_im.size(), src_im.type()), framecolor(framecolor), lineMode(lineMode)
		{
			m_src_im = src_im;
			
			roi_capture = false;
			changed_roi = false;

			segment.first.x = 0;
			segment.first.y = 0;
			if (!lineMode) {
				segment.second.x = m_src_im.cols - 1;
				segment.second.y = m_src_im.rows - 1;
			}
			else segment.second = cv::Point(0, 0);
		}
		void setBackground(const cv::Mat &mat) { m_src_im = mat.clone(); }
		void setColor(cv::Scalar color) { framecolor = color; }
		virtual void updateSliders() {};

		// implements drag-n-drop behavoir
		virtual void updateClick(int x, int y, int event, int mouseflags) {

			switch (event)
			{
			case cv::EVENT_MOUSEMOVE: {
				if (mouseflags == cv::EVENT_FLAG_LBUTTON) {
					segment.second.x = x;
					segment.second.y = y;
				}
				break;
			}
			case cv::EVENT_LBUTTONDOWN:
			{

				segment.first.x = x;
				segment.first.y = y;
				segment.second.x = x;
				segment.second.y = y;
				roi_capture = true;
				break;
			}
			case cv::EVENT_LBUTTONUP:
			{
				segment.second.x = x;
				segment.second.y = y;
				roi_capture = false;
				changed_roi = true;
				done();
				break;
			}
			default:;
			}

		};
		virtual void done() {}
		virtual void initCanvas(cv::Mat &canvas) {
			m_src_im.copyTo(canvas);
		}
		virtual void show(cv::Mat &canvas) const override {
			if(!lineMode) cv::rectangle(canvas, segment.first, segment.second, framecolor, 2);
			else cv::line(canvas, segment.first, segment.second, framecolor, 2);
		}
		cv::Segment getSegment() const {
			return segment;
		}
		virtual cv::Rect getROI() const {
			return rect(segment);
		}

		int run(char additional_exit_key = 'e') {
			bool done = false;
			printf("Press 'Esc' key for exit from selection roi mode.\n");
			cv::Mat canvas;
			changed_roi = false;
			int k = 0;
			while (!done) {
				initCanvas(canvas);
				show(canvas);
				pawlin::debugImg(winname, canvas, 1, -1);
				k = cv::waitKeyEx(30);
				//printf("%X\n", k);
				if (!OPENCV_SPECIAL_KEY(k)) k = k & 0xFF;

#ifdef WINDOWS
				//проверка что крест не был нажат
				HWND Hide2 = FindWindowA(NULL, (LPCSTR)winname.c_str());
				if (key(k) || k == additional_exit_key || Hide2 == NULL) done = true;
#else
				if (key(k) || k == additional_exit_key) done = true;
#endif
			}
			return k;
		}

		virtual ~SelectionROIWindow() {
		}

		virtual bool key(int k) { // return true if done
			if (k == 27) return true;
			return false;
		}

	};

	class MultiRectSelectionWindow : public SelectionROIWindow {
	public:
		struct LabelRect {
			cv::Rect rect;
			int label;
		};
	protected:

		vector < LabelRect> array;
		vector <string> labels;
		vector <cv::Scalar> colors;
		int current_label;
	public:
		MultiRectSelectionWindow(
			const string &win, 
			const cv::Mat &src_im, 
			const vector <string> &labels, 
			const vector <cv::Scalar> &colors
		) :
			SelectionROIWindow(win,src_im,CV_RGB(128,128,128)), 
			labels(labels),
			colors(colors),
			current_label(0)
		{
			segment.first = segment.second = cv::Point(0, 0);
			framecolor = colors[current_label];
			if (labels.empty() || labels.size() != colors.size()) 
				throw std::runtime_error("MultiRectSelectionWindow - empty labels array now allowed and colors & labels must have same size!");
			FOR_ALL_IF(labels, l, labels[l].empty()) throw std::runtime_error("MultiRectSelectionWindow - zero length labels not allowed!");
		}
		void setArray(const vector < LabelRect> &_array) { array = _array; }
		const vector < LabelRect> & getArray() const { return array; }
		static void test();
		void update_label() {
			if (current_label < 0) current_label = 0;
			if (current_label >= (int)labels.size()) current_label = (int)labels.size() - 1;
			framecolor = colors[current_label];
		}
		virtual void updateClick(int x, int y, int event, int mouseflags) override {
			if (event == cv::EVENT_MOUSEWHEEL) {
				if (mouseflags > 0) current_label--;
				else current_label++;
				update_label();
				return;
			}
			if(event==cv::EVENT_RBUTTONUP) {
				// remove last rectangle
				if (array.empty()) return;
				array.resize(array.size() - 1);
				if (array.empty()) {
					segment.first = segment.second = cv::Point(0, 0);
					return;
				}
				segment.first = array.back().rect.tl();
				segment.second = array.back().rect.br();
			}
			else SelectionROIWindow::updateClick(x, y, event, mouseflags);
		}
		virtual void done() override {
			array.push_back(LabelRect({ getROI(),current_label }));
			segment.first = segment.second = cv::Point(0, 0);
		}
		virtual void show(cv::Mat &canvas) const override {
			SelectionROIWindow::show(canvas);
			FOR_ALL(array, i) {
				cv::rectangle(canvas, array[i].rect, colors[array[i].label], 2);
				cv::putText(canvas, labels[array[i].label], array[i].rect.tl(), 
					cv::FONT_HERSHEY_PLAIN,
					1, CV_RGB(0, 0, 255)
				);
			}
			// show current label
			string msg = "[" + labels[current_label] + "]";
			cv::putText(canvas, msg, cv::Point(4, 16),
				cv::FONT_HERSHEY_PLAIN,
				1, colors[current_label]// CV_RGB(0, 0, 255)
			);
		}
		virtual bool key(int k) override { // return true if done
			if (k == -1) return false;
			if (k == 27) return true;
			if (k == OPENCV_RIGHT_ARROW_CODE || k == OPENCV_RIGHT_ARROW_CODE_TERMINAL_LINUX || k == OPENCV_RIGHT_ARROW_CODE_XMING) 
				current_label++;
			if (k == OPENCV_LEFT_ARROW_CODE || k == OPENCV_LEFT_ARROW_CODE_TERMINAL_LINUX || k == OPENCV_LEFT_ARROW_CODE_XMING) 
				current_label--;
			int key = 0;  if ((k & 0xFF00) == 0) key = k & 0xFF;
			FOR_ALL(labels, l) {
				if (std::tolower(labels[l][0]) == key || labels[l][0]==key) current_label = (int)l;
			}
			update_label();
			return false;
		}

	};

	class SavingMultiRectsSelection : public pawlin::MultiRectSelectionWindow {
	public:
		using MultiRectSelectionWindow::MultiRectSelectionWindow;
		virtual bool key(int k) override { // return true if done
			if (k == 'S') {
				FileSaver<LabelRect>::save("rects.bin", this->array);
			}
			if (k == 'L') {
				FileSaver<LabelRect>::load("rects.bin", this->array);
			}
			return MultiRectSelectionWindow::key(k);
		}
	};

	class SavingMultiRectsSelectionHDR : public SavingMultiRectsSelection {
		pawlin::ThresholdSelector sel;
		//cv::Mat backcopy;
	public:
		SavingMultiRectsSelectionHDR(
			const string &win,
			const cv::Mat &src_im,
			const vector <string> &labels,
			const vector <cv::Scalar> &colors
		) : SavingMultiRectsSelection(win, src_im.clone(), labels, colors),
			sel(win, src_im.clone(), 1.0f, false)
		{
			//backcopy = src_im.clone();
		};
		virtual void initCanvas(cv::Mat &canvas) override {
			SavingMultiRectsSelection::m_src_im = sel.generate();
			sel.convert(m_src_im);
			SavingMultiRectsSelection::initCanvas(canvas);
		}
	};


	void imshowWithResize(string windowName, const cv::Mat &img, int sizeMul, int cvinterpolation = cv::INTER_LINEAR);
	//#endif

	// differs from base class by fixed width & height
	class SelectionSolidROIWindow :public SelectionROIWindow {
		cv::Size size;
		vector <int> exitKeys;
		string message;
	public:
		void setSize(cv::Size sz) { size = sz; }
		void setMessage(string msg) { message = msg; }
		void setExitKeys(const vector<int> &keys) { exitKeys = keys; }
		void addExitKey(int k) { exitKeys.push_back(k); }
		virtual bool key(int k) override {
			return std::find(exitKeys.begin(), exitKeys.end(), k) != exitKeys.end();
		}
		virtual void show(cv::Mat &canvas) const override;
		virtual void updateSliders() {};
		SelectionSolidROIWindow(const std::string &win, const cv::Mat &background, cv::Size sz, const string &message,
			cv::Scalar framecolor = cv::Scalar(200, 20, 140))
			: size(sz), message(message),
			SelectionROIWindow(win, background,framecolor)
		{
			addExitKey(27); // ESC
		}
		virtual cv::Rect getROI() const override;

		// implement "stick-to-the-mouse" behavoir
		virtual void updateClick(int x, int y, int event, int mouseflags) {

			switch (event)
			{
			case cv::EVENT_MOUSEMOVE: {
				if (x+size.width <= m_src_im.cols && y+size.height <= m_src_im.rows) {
					segment.first.x = x;
					segment.first.y = y;
					segment.second.x = x + size.width;
					segment.second.y = y + size.height;
				}
				break;
			}

			case cv::EVENT_LBUTTONDOWN:
			{
				segment.first.x = x;
				segment.first.y = y;
				roi_capture = true;
				break;
			}
			default:;
			}

		};
	};
	using cv::Rect2f;
	using cv::Point2f;
	class Subdiv2Df : public cv::Subdiv2D {
	public:
		Subdiv2Df(Rect2f rect)
		{
			validGeometry = false;
			freeQEdge = 0;
			freePoint = 0;
			recentEdge = 0;

			initDelaunay(rect);
		}
		void insert(const vector<Point2Df> &points) {
			FOR_ALL(points, i) cv::Subdiv2D::insert(ptconv2f(points[i]));
		}
		void insert(const vector<Point2f> &points) {
			cv::Subdiv2D::insert(points);
		}

		void initDelaunay(Rect2f rect)
		{
			//CV_INSTRUMENT_REGION();

			float big_coord = 3.f * MAX(rect.width, rect.height);
			float rx = rect.x;
			float ry = rect.y;

			vtx.clear();
			qedges.clear();

			recentEdge = 0;
			validGeometry = false;

			topLeft = Point2f(rx, ry);
			bottomRight = Point2f(rx + rect.width, ry + rect.height);

			Point2f ppA(rx + big_coord, ry);
			Point2f ppB(rx, ry + big_coord);
			Point2f ppC(rx - big_coord, ry - big_coord);

			vtx.push_back(Vertex());
			qedges.push_back(QuadEdge());

			freeQEdge = 0;
			freePoint = 0;

			int pA = newPoint(ppA, false);
			int pB = newPoint(ppB, false);
			int pC = newPoint(ppC, false);

			int edge_AB = newEdge();
			int edge_BC = newEdge();
			int edge_CA = newEdge();

			setEdgePoints(edge_AB, pA, pB);
			setEdgePoints(edge_BC, pB, pC);
			setEdgePoints(edge_CA, pC, pA);

			splice(edge_AB, symEdge(edge_CA));
			splice(edge_BC, symEdge(edge_AB));
			splice(edge_CA, symEdge(edge_BC));

			recentEdge = edge_AB;
		}

	};

	class VoronoiInfo {
		Subdiv2Df subdiv;
		vector<Polygon > facets_clipped;
		vector<Point2f> centers;
		vector<vector<size_t> > neighbours;
		const cv::Size canvsize;
	public:
		const vector<vector<size_t> > &getNeighbours() const {
			return neighbours;
		}
		void insert(const vector<cv::Point2f> &points) {
			subdiv.insert(points);
			vector<vector<Point2f> > facets;
			subdiv.getVoronoiFacetList(vector<int>(), facets, centers);
			// build connections
			std::map<std::pair<int, int>, std::set<size_t> > connections;
			facets_clipped.resize(facets.size());
			pawlin::Polygon rectpoly(canvsize.width, canvsize.height);
			FOR_ALL(facets, i) {
				// convert facets to integer & cross with screen rectangle
				pawlin::Polygon ipoly;
				convert(facets[i], ipoly);
				pawlin::intersectPolygon(rectpoly, ipoly, facets_clipped[i]);
				MinMaxAvg cx, cy;
				FOR_ALL(facets_clipped[i], j) {
					cv::Point p = facets_clipped[i][j];
					cx.take((float)p.x);
					cy.take((float)p.y);
					connections[std::make_pair(p.x, p.y)].insert(i);
				}
				centers[i] = cv::Point2f(cx.getAvg(), cy.getAvg());
			}
			neighbours.resize(centers.size());
			for (auto &points : connections) {
				for (auto &links_i : points.second) {
					for (auto &links_j : points.second) {
						if (links_i != links_j) neighbours[links_i].push_back(links_j);
					}
				}
			}

		}
		VoronoiInfo(const cv::Size &canvsize) : subdiv(cv::Rect(cv::Point(0, 0), canvsize)), canvsize(canvsize) {}
		VoronoiInfo(const cv::Size &canvsize, const vector<cv::Point2f> &points ) :
			VoronoiInfo(canvsize)
		{
			insert(points);
		}
		void draw_links(cv::Mat &canvas, cv::Scalar color = CV_RGB(80, 80, 80), int thickness = 1) const {
			FOR_ALL(facets_clipped, i) {
				FOR_ALL(neighbours[i], n) {
					size_t index = neighbours[i][n];
					cv::line(canvas, centers[i], centers[index], color, thickness, cv::LINE_AA);
				}
			}
		}
		void draw_grid(cv::Mat &canvas, cv::Scalar color = 0, int thickness = 1) const {
			FOR_ALL(facets_clipped, i) {
				polylines(canvas, facets_clipped[i], true, color, thickness, cv::LINE_AA, 0);
			}
		}
		void draw_facets(cv::Mat &canvas, const vector<float> &intensity, int thickness = -1) const {
			vector<cv::Scalar> colors(intensity.size());
			FOR_ALL(colors, i) colors[i] = intensity[i];
			draw_facets(canvas, colors, thickness);
		}
		void draw_facets(cv::Mat &canvas, const vector<cv::Scalar> &color, int thickness = -1) const {
			if (color.size() != facets_clipped.size()) throw std::runtime_error("pawlin::VoronoiInfo::draw_facets - colors list does not match facets size");
			FOR_ALL(facets_clipped, i) {
				if(thickness>=0) polylines(canvas, facets_clipped[i], true, color[i], thickness, cv::LINE_AA, 0);
				else fillConvexPoly(canvas, facets_clipped[i], color[i], cv::LINE_AA);
			}
		}
		void drawGridIndex(cv::Mat &index32S) const {
			FOR_ALL(facets_clipped, i) {
				cv::fillConvexPoly(index32S, facets_clipped[i], cv::Scalar((double)i));
			}
		}
		vector<Point2f> get_centers()
		{
			return centers;
		}
		cv::Point get_nearest_centre(const cv::Point checked_point)
		{
			FOR_ALL(facets_clipped, i) {
				if (facets_clipped[i].pointIsInPolygon(checked_point)) {
					return facets_clipped[i].getCenter();
				}
			}
			return cv::Point(-1, -1);
		}
		Polygon get_nearest_polygon(const cv::Point checked_point)
		{
			FOR_ALL(facets_clipped, i) {
				if (facets_clipped[i].pointIsInPolygon(checked_point)) {
					return facets_clipped[i];
				}
			}
			return Polygon(0);
		}
	};

	//Draw voronoi diagram using specified colors, convex facets polygons are clipped by the image bounding rectangle
	// the boundaries & centers are drawn with black
	inline void draw_voronoi(cv::Mat& img, Subdiv2Df& subdiv, const Box2Df &box, const vector<cv::Scalar> &colors)
	{
		vector<vector<Point2f> > facets;
		vector<Point2f> centers;
		subdiv.getVoronoiFacetList(vector<int>(), facets, centers);

		vector<cv::Point> icenters;
		vector<vector<cv::Point> > ifacets;
		convert(facets, box, img.size(), ifacets);
		convert(centers, box, img.size(), icenters);
		pawlin::Polygon rectpoly;
		rectpoly.push_back(cv::Point(0, 0));
		rectpoly.push_back(cv::Point(img.cols, 0));
		rectpoly.push_back(cv::Point(img.cols, img.rows));
		rectpoly.push_back(cv::Point(0, img.rows));
		rectpoly.push_back(cv::Point(0, 0));
		rectpoly.pointsOrdered();

		for (size_t i = 0; i < facets.size(); i++)
		{
			pawlin::Polygon temp = ifacets[i];
			pawlin::Polygon finalpoly;
			pawlin::intersectPolygon(rectpoly, temp, finalpoly);
			fillConvexPoly(img, finalpoly, colors[i], 8, 0);

			vector<vector<cv::Point> > tempfacets(1);
			tempfacets[0] = finalpoly;
			polylines(img, tempfacets, true, cv::Scalar(), 2, cv::LINE_AA, 0);
			circle(img, icenters[i], 3, cv::Scalar(), cv::FILLED, cv::LINE_AA, 0);
		}
	}

	class TouchedObjectsSeparator {
		double distsq(const cv::Point &a, const cv::Point &b) const {
			auto diff = a - b;
			return diff.ddot(diff);
		}
		double loop(const vector<double> &dist, int index) const {
			if (index < 0) return loop(dist, index + (int)dist.size());
			if (index >= (int)dist.size()) return loop(dist, index - (int)dist.size());
			return dist[index];
		}
		void findMinimums(const vector<double> &_dist, vector<int> &minimums) const {
			enum State { INIT, NONE, FALL, RISE, PLATO };
			State state = INIT;
			int counterFall = 0;
			int counterRise = 0;
			int counterPlato = 0;
			const int maxPlato = 2;
			const int checkDeep = 10;
			int extrPos = 0;
			double lastDist = 0;
			vector<double> dist(_dist.size()); // smooth input array
			for (int i = 0; i < (int)dist.size(); i++) {
				double avg = loop(_dist, i - 1);
				avg += loop(_dist, i);
				avg += loop(_dist, i + 1);
				dist[i] = avg / 3.0;
			}
			for (int i = -checkDeep; i < (int)dist.size() + checkDeep; i++) {
				double v = loop(dist, i);
				switch (state) {
				case INIT:
					state = NONE;
					break;

				case NONE:
					if (v < lastDist) {
						state = FALL;
						counterFall = 1;
					}
					break;

				case FALL:
					if (v < lastDist) {
						counterFall++;
						break;
					}
					if (counterFall < checkDeep) {
						state = NONE;
						break;
					}
					extrPos = i;
					if (v == lastDist) {
						state = PLATO;
						counterPlato = 1;
						break;
					}
					if (v > lastDist) {
						state = RISE;
						counterRise = 1;
						break;
					}
					break;

				case PLATO:
					if (v == lastDist) {
						counterPlato++;
						if (counterPlato > maxPlato) state = NONE;
						break;
					}
					if (v > lastDist) {
						state = RISE;
						counterRise = 1;
						break;
					}
					break;

				case RISE:
					if (v > lastDist) {
						counterRise++;
						if (counterRise >= checkDeep) {
							//printf("FALL %d, PLATO %d, RISE %d Pos %d\n",
							//	counterFall,
							//	counterPlato,
							//	counterRise, extrPos);
							if (extrPos < 0) extrPos += (int)dist.size();
							if (extrPos >= (int)dist.size()) extrPos -= (int)dist.size();
							minimums.push_back(extrPos);
							state = NONE;
						}
						break;
					}
					state = NONE;
					break;
				}
				lastDist = v;
			}
		}
		void breakContour(const vector<cv::Point> &contour, int weakThreshold, cv::Mat &img) const {
			const double weakSq = weakThreshold*double(weakThreshold);
			std::vector<std::pair<int, int>> brokers;
			//#define DEBUGMARKER
			FOR_ALL(contour, i) {
#ifdef DEBUGMARKER
				cv::Mat canvas = img.clone();
				cv::circle(canvas, contour[i], 2, cv::Scalar(150, 150, 150), -1, cv::LINE_AA);
				cv::imshow("cursor", canvas);
				if (i < 506) continue;
				printf("i=%zu\n", i);
				int key = cv::waitKey(0);
				if (key == ' ') continue;
#endif
				vector<double> dist(contour.size());
				FOR_ALL(contour, j) { // this will be 'double' kill but will simplify logic below of finding minimums
					dist[j] = distsq(contour[i], contour[j]);
				}
				vector<int> minimums;
				findMinimums(dist, minimums);
#ifdef DEBUGMARKER
				FOR_ALL(minimums, m) {
					int index = minimums[m];
					int idiff = std::abs((int)(i - index)) % (int)contour.size();
					if (idiff <= 2) continue;
					cv::circle(canvas, contour[index], 2, cv::Scalar(200, 200, 200), -1, cv::LINE_AA);
					printf("distance %f\n", dist[index]);
				}
				cv::imshow("cursor", canvas);
				cv::waitKey(0);
#endif
				FOR_ALL(minimums, m) {
					int index = minimums[m];
					int idiff = std::abs((int)(i - index)) % (int)contour.size();
					if (idiff <= 2) continue;
					if (dist[index] < weakSq && index != i) {
						// check that line passes only along object values
						cv::LineIterator it(img, contour[i], contour[index], 4);
						if (it.count <= 1) continue;
						bool wrong = false;
						for (int i = 0; i < it.count; i++, ++it) {
							unsigned char v = *(unsigned char *)*it;
							if (v == 0) {
								wrong = true;
								break;
							}
						}
						if (wrong) continue;
						// break here
						//printf("break %d-%d\n", (int) i, index);
						brokers.push_back(std::make_pair((int)i, index));

						//cv::line(img, contour[i], contour[index], cv::Scalar(100, 100, 100), 1, cv::LINE_8); // if past point is not included, it's okay we will fill it with zero by 'double' kill as it will be starting point for second case
						//cv::circle(img, contour[index], 2, cv::Scalar(150, 150, 150), -1, cv::LINE_AA);
					}
				}
			}
			const double toleranceSq = 3 * 3;
			FOR_ALL(brokers, i) {
				cv::Point p1s = contour[brokers[i].first];
				cv::Point p1e = contour[brokers[i].second];
				for (size_t j = i + 1; j < brokers.size(); j++) {
					cv::Point p2s = contour[brokers[j].first];
					cv::Point p2e = contour[brokers[j].second];
					if (distsq(p1s, p2e)<toleranceSq && distsq(p1e, p2s)<toleranceSq)
						cv::line(img,
							//p1s,p1e,
							0.5*(p1s + p2e), 0.5*(p1e + p2s),
							cv::Scalar(0, 0, 0), 1, cv::LINE_4); // if past point is not included, it's okay we will fill it with zero by 'double' kill as it will be starting point for second case

				}
			}

			//cv::imshow("img", img);
			//cv::waitKey(0);
		}
		public:
		void trySeparate(const cv::Mat &img, int weakThreshold, vector<cv::Mat> &out, const double minArea = 10) const {
			// build contours
			vector<vector<cv::Point> > contours0, contours1;
			vector<cv::Vec4i> hierarchy0, hierarchy1;
			findContours(img, contours0, hierarchy0, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
			if (contours0.size() == 0)
				return;
			cv::Mat copy = img.clone();
			for (int idx = 0; idx >= 0; idx = hierarchy0[idx][0])
			{
				const vector<cv::Point>& c = contours0[idx];
				double area = fabs(contourArea(cv::Mat(c)));
				if (area < minArea) continue;
				breakContour(c, weakThreshold, copy);
			}
			findContours(copy, contours1, hierarchy1, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
			for (int idx = 0; idx >= 0; idx = hierarchy1[idx][0])
			{
				const vector<cv::Point>& c = contours1[idx];
				double area = fabs(contourArea(cv::Mat(c)));
				if (area < minArea) continue;
				cv::Mat single = cv::Mat::zeros(img.size(), img.type());
				cv::drawContours(single, contours1, idx, CV_RGB(255, 255, 255), cv::FILLED, cv::LINE_4);
				out.push_back(single);
			}
		}
	};
	//ThresholdSelector("threshold", confidencemap, 1).run(background);

	inline void drawToRect(cv::Mat & mat, cv::Rect target, int face, int thickness, cv::Scalar color, const std::string & str)
	{
		cv::Size rect = cv::getTextSize(str, face, 1.0, thickness, 0);
		double scalex = (double)target.width / (double)rect.width;
		double scaley = (double)target.height / (double)rect.height;
		double scale = (std::min)(scalex, scaley);
		int marginx = scale == scalex ? 0 : (int)((double)target.width * (scalex - scale) / scalex * 0.5);
		int marginy = scale == scaley ? 0 : (int)((double)target.height * (scaley - scale) / scaley * 0.5);
		cv::putText(mat, str, cv::Point(target.x + marginx, target.y + target.height - marginy), face, scale, color, thickness, 8, false);
	}
	class Popup : public CvInteractWindowBase {
	public:
		Popup(const string &message, int delay = 0, int font = cv::FONT_HERSHEY_PLAIN, 
			int thickness = 2,
			cv::Scalar back = CV_RGB(50, 100, 255), cv::Scalar color = CV_RGB(0,0,255))
			:CvInteractWindowBase("Message",cv::getTextSize(message, font, 1.0, thickness, 0)*2,CV_8UC3)
		{
			cv::Mat canvas(this->canv_size, this->canv_type);
			canvas = back;
			cv::putText(canvas, message, cv::Point(canv_size.width / 4, canv_size.height*2 / 3), font, 1, color, thickness, cv::LINE_AA);
			cv::imshow(this->winname, canvas);
			if(delay>=0) cv::waitKey(delay);
		}
	};
	class TextEdit : public CvInteractWindowBase {
		string prompt;
		string entered;
		cv::Mat canvas;
		cv::Point where;

		int char_size;
		int cursor_pos;
		bool cursor_state;
		bool insert_state;

		// colors
		cv::Scalar text_color;
		cv::Scalar cursor_color;
		cv::Scalar background_color;
		cv::Scalar title_color;
		cv::Mat background;
	public:
		void setTextColor(cv::Scalar color) { text_color = color; }
		void setCursorColor(cv::Scalar color) { cursor_color = color; }
		void setTitleColor(cv::Scalar color) { title_color = color; }
		TextEdit(const string &prompt, const cv::Point &where = cv::Point(100, 100),
			const cv::Size &canvsz = cv::Size(512,512),
			const string &winname = "Enter string", bool nodestroy = false) 
			: prompt(prompt), where(where), CvInteractWindowBase(winname, canvsz, CV_8UC3, nodestroy) {
			canvas = cv::Mat::zeros(canv_size, canv_type);

			char_size = 15;
			cursor_state = false;
			cursor_pos = 0;
			insert_state = false;

			// colors
			text_color = CV_RGB(120, 120, 120);
			cursor_color = CV_RGB(0, 255, 0);
			background_color = CV_RGB(255, 255, 255);
			title_color = CV_RGB(33, 33, 33);
		}
		void setBackground(const cv::Mat &back) {
			background = back.clone();
			canvas = cv::Mat::zeros(back.size(), back.type());
		}
		int getCharSize() const { return char_size; }
		int getFieldHeight() const { return float2int(char_size*1.5f - 2); }

		void drawMonospaceText(cv::Mat &mat, cv::Rect target, int face, int thickness, cv::Scalar color, const std::string &str, int char_size = 15)
		{
			cv::Size rect = cv::getTextSize(str, face, 1.0, thickness, 0);
			rect.width = static_cast<int>(str.size() * char_size);
			for (int i = 0; i < str.size(); ++i)
				cv::putText(mat, (str[i] + string("")), cv::Point(target.x + char_size * i, target.y + 20), face, 1.5, color, thickness, 8, false);
		}
		void drawCursor(cv::Mat &mat, cv::Point pos, int thickness, cv::Scalar color, const int cursor_pos, int char_size)
		{
			cv::line(mat, cv::Point(pos.x + char_size * cursor_pos, pos.y + 2),
				cv::Point(pos.x + char_size * cursor_pos, static_cast<int>(pos.y + char_size * 1.5 - 2)),
				color, thickness);
		}
		void drawInsertCursor(cv::Mat &mat, cv::Point pos, int thickness, cv::Scalar color, const int cursor_pos, int char_size) {
			cv::rectangle(mat, cv::Rect(cv::Point(pos.x + char_size * cursor_pos, pos.y + 2),
				cv::Point(pos.x + char_size * (cursor_pos + 1), pos.y + 25)),
				color, thickness);
		}
		int max_length = -1;
		void setMaxLength(int ml) { max_length = ml; }
		void run() {
			for (;;) {
				cursor_state = !cursor_state;  // cursor blink effect
				if(background.empty()) canvas = background_color; // fill with nice color
				else canvas = background.clone();
										   // draw what user entered
				cv::Rect rect(where, cv::Size(((int)entered.size() + 1) * char_size + 1, static_cast<int>(char_size * 1.5)));
				drawMonospaceText(canvas, rect, cv::FONT_HERSHEY_PLAIN, 1, text_color, entered);

				// draw cursor
				if (cursor_state) {
					if (insert_state)
						drawInsertCursor(canvas, where, 2, cursor_color, cursor_pos, char_size);
					else
						drawCursor(canvas, where, 2, cursor_color, cursor_pos, char_size);
				}

				// draw prompt
				if (prompt.size()) {
					cv::Rect rect2(where - cv::Point(0, 32), cv::Size((int)prompt.size() * 16, 16));
					drawToRect(canvas, rect2, cv::FONT_HERSHEY_PLAIN, 1, title_color, prompt);
				}
				// key events
				int key = pawlin::debugImg(winname, canvas, 1, 200, false);
				// ESC button pressed
				if (key == 27) {
					entered = "";
					return;
				}
				// ->
				if (key == OPENCV_RIGHT_ARROW_CODE) {
					cursor_state = false;
					if (cursor_pos < entered.size())cursor_pos++;
				}
				// <-
				if (key == OPENCV_LEFT_ARROW_CODE) {
					cursor_state = false;
					if (cursor_pos > 0) cursor_pos--;
				}
				// backspace
				if (key == 8) {
					cursor_state = false;
					if (cursor_pos > 0)
						entered.erase(entered.begin() + --cursor_pos);
				}
				// delite
				if (key == 3014656) {
					cursor_state = false;
					if (cursor_pos < entered.size()) entered.erase(entered.begin() + cursor_pos);
				}
				// enter
				if (key == 13) {
					return;
				}
				// insert
				if (key == 2949120) {
					cursor_state = false;
					insert_state = !insert_state;
				}
				// ASCII symbol entered
				if (key >= 32 && key < 128 && (max_length<=0 || entered.size()<(size_t)max_length)) {
					cursor_state = false;
					if (insert_state) {
						if (cursor_pos == entered.size())
							entered.push_back((char)key);
						else
							entered[cursor_pos] = (char)key;
					}
					else {
						entered.insert(cursor_pos, 1, (char)key);
					}
					cursor_pos++;
				}
			}
		}
		virtual void updateClick(int x, int y, int event, int flags) override {
			// change cursor position with mouse click
			if (event == cv::EVENT_LBUTTONDOWN) {

				if (y < where.y + 20 && y > where.y - 5) {
					if (x > where.x&& x < where.x + entered.size() * char_size + char_size) {
						cursor_pos = (x - where.x + char_size / 2) / char_size;
						if (cursor_pos > static_cast<int>(entered.size())) cursor_pos = static_cast<int>(entered.size());
						cursor_state = false;
					}
				}
			}
		}
		string getText() const { return entered; }
	};

	class DebugOptFlow : public pawlin::CvInteractWindowBase {
		int pyr_scale; //0.3-0.7
		int levels; //1-5
		int winsize; //4-16 
		int iterations;//1-5 
		int poly_n; //3-8
		int poly_sigma; //1.0 - 2.0
		int show_scale; //0.5
		int flow_scale; //1-4
		int step;
		//int flags
		cv::Mat canvas;
		const cv::Mat &frame1; const cv::Mat &frame2;
		cv::Mat flow;
	public:
		const cv::Mat &getFlow() const { return flow; }
		DebugOptFlow(const cv::Mat &frame1, const cv::Mat &frame2) :
			CvInteractWindowBase("optflow debug", frame1.size(), CV_8UC3),
			frame1(frame1),
			frame2(frame2)
		{
			pyr_scale = 40; // pyr_scale * 0.01
			levels = 1;//levels
			winsize = 8;// winsize 
			iterations = 2; //iterations
			poly_n = 8; //poly_n
			poly_sigma = 120; // poly_sigma * 0.01
			show_scale = 100; // *0.01
			flow_scale = 100;
			step = 7;
			canvas = frame1.clone();
			pawlin::debugImg(winname, canvas, show_scale*0.01, 1, false);

			cv::createTrackbar("pyr_scale", winname, &pyr_scale, 100);
			cv::createTrackbar("levels", winname, &levels, 100);
			cv::createTrackbar("winsize", winname, &winsize, 100);
			cv::createTrackbar("iterations", winname, &iterations, 30);
			cv::createTrackbar("poly_n", winname, &poly_n, 10);
			cv::createTrackbar("poly_sigma", winname, &poly_sigma, 200);
			cv::createTrackbar("show_scale", winname, &show_scale, 200);
			cv::createTrackbar("flow_scale", winname, &flow_scale, 200);
			cv::createTrackbar("step", winname, &step, 10);
		}

		void setParams(
			float _pyr_scale,
			int _levels,
			int _winsize,
			int _iterations,
			int _poly_n,
			float _poly_sigma) {
			pyr_scale = float2int(_pyr_scale * 100.0f);
			levels = _levels;
			winsize = _winsize;
			iterations = _iterations;
			poly_n = _poly_n;
			poly_sigma = float2int(_poly_sigma*100.0f);
			cv::setTrackbarPos("pyr_scale", winname, pyr_scale);
			cv::setTrackbarPos("levels", winname, levels);
			cv::setTrackbarPos("winsize", winname, winsize);
			cv::setTrackbarPos("iterations", winname, iterations);
			cv::setTrackbarPos("poly_n", winname, poly_n);
			cv::setTrackbarPos("poly_sigma", winname, poly_sigma);

		}

		void run() {
			for (int i = 0;; i++) {
				Profiler prof; prof.startSequence();
				cv::calcOpticalFlowFarneback(
					frame1,
					frame2,
					flow,
					pyr_scale / 100.0,
					levels,
					winsize,
					iterations,
					poly_n,
					poly_sigma / 100.0,0/* cv::OPTFLOW_FARNEBACK_GAUSSIAN*/);
				prof.markPoint("calcOpticalFlowFarneback");
				canvas = i & 1 ? frame1.clone() : frame2.clone();
				cv::cvtColor(canvas, canvas, cv::COLOR_GRAY2BGR);
				drawFlow(canvas, flow, flow_scale*0.01f, step);
				pawlin::debugImg("opt flow aux win", canvas, show_scale*0.01, -1, false);
				cv::Mat map(flow.size(), CV_32FC2);
				FOR_MAT(flow, p) map.at<Point2f>(p) = ptconv2f(p) + flow.at<cv::Point2f>(p);
				cv::Mat frame1est;
				cv::remap(frame2, frame1est, map, cv::Mat(), cv::INTER_LINEAR);
				pawlin::debugImg("frame2->frame1", frame1est, 1, -1);
				pawlin::debugImg("diff", cv::abs(frame1est - frame1), 1, -1);

				int k = pawlin::debugImg(winname, canvas, 1, 1, false);
				if (k == 't') prof.print();
				if (k == 27) break;
			}
		}

	};

	inline MinMaxAvg getStat8U(const cv::Mat &in8U) {
		MinMaxAvg stat;
		for (int i = 0; i < in8U.rows; i++) {
			const uint8_t *row = (uint8_t *)in8U.ptr(i);
			for (int j = 0; j < in8U.cols; j++) stat.take((float)row[j]);
		}
		return stat;
	}

	inline std::vector<MinMaxAvg> getStat8UC3(const cv::Mat &in8UC3) {
		vector<cv::Mat> channels;
		cv::split(in8UC3, channels);
		vector<MinMaxAvg> res(channels.size());
		FOR_ALL(channels, i) res[i] = getStat8U(channels[i]);
		return res;
	}

	inline MinMaxAvg getStat(const cv::Mat &in32F) {
		MinMaxAvg stat;
		for (int i = 0; i < in32F.rows; i++) {
			const float *row = (float *)in32F.ptr(i);
			for (int j = 0; j < in32F.cols; j++) stat.take(row[j]);
		}
		return stat;
	}
	inline cv::Rect boundingBox(const cv::Mat &mask8UC1);
	class CosmicPanel : public CvInteractWindowBase {
		cv::Mat panel;
		cv::Mat btnmask;
		struct Region {
			cv::Scalar color;
			int id;
			string key;
			cv::Rect boundingBox;
			cv::Mat colormap;
			bool clickable() const {
				return key.size() && key[0] != '#';
			}
		};
		vector <Region> regions;
		int cursor = -1;
		cv::Scalar highlight_color;
		cv::Scalar flash_color;
		bool flash = false;
		bool ready = false;
	public:
		const Region &find(const string &key) const {
			FOR_ALL_IF(regions, i, regions[i].key == key) return regions[i];
			throw std::runtime_error("CosmicPanel region not found with key = " + key);
			return regions.front(); // to avoid warning
		}
		CosmicPanel(
			const std::string &winname,
			const std::string &panelfilename,
			float scale,
			const cv::Scalar &flcolor = CV_RGB(255, 200, 100)/255, //flash when pressed
			const cv::Scalar &hlcolor = CV_RGB(255, 250, 200)/255 //highlight
		);
		virtual void updateClick(int x, int y, int event, int mouseflags) override;

		virtual void button_hit(int id, const string&name) {
			printf("Button[%d] '%s' was hit in panel %s\n", id, name.c_str(), winname.c_str());
		}
		void draw(cv::Mat &where) const;
		int show(int delay) const;

		void run() {
			while(show(1)!=27);
		}

	};

	//note last tile in Row/Column take the remaining
	inline void tile(const cv::Size &imageSize, const cv::Size &tileSize, vector<vector<cv::Rect>> &out) {
		int countX = imageSize.width / tileSize.width;
		int countY = imageSize.height / tileSize.height;
		if (imageSize.width % tileSize.width) countX++;
		if (imageSize.height % tileSize.height) countY++;
		out.resize(countY, vector<cv::Rect>(countX));
		cv::Point p;
		cv::Rect imgRect(cv::Point(0, 0), imageSize);
		int y = 0;
		for (p.y = 0; p.y < imageSize.height; p.y += tileSize.height, y++) {
			int x = 0;
			vector<cv::Rect> &row = out[y];
			for (p.x = 0; p.x < imageSize.width; p.x += tileSize.width, x++) {
				out[y][x] = cv::Rect(p, tileSize) & imgRect;
			}
		}
	}

	inline double getMax(const cv::Mat &mat) {
		double maxv;
		cv::minMaxLoc(mat, nullptr, &maxv);
		return maxv;
	}

	inline void makeJobList(
		const cv::Mat &source,
		vector<cv::Rect> &jobs,
		const cv::Size tileSize,
		const float threshold = 10.0f) 
	{
		vector<vector<cv::Rect>> tiles;
		tile(source.size(), tileSize, tiles);
		cv::Mat maxmap(cv::Size((int)tiles.front().size(), (int)tiles.size()), CV_8U);
		FOR_MAT(maxmap, p) {
			maxmap.at<uint8_t>(p) = (uint8_t)getMax(source(tiles[p.y][p.x]));
		}
		//pawlin::debugImg("maxmap", maxmap, 10, 0, false, true);
		cv::Mat labels;
		cv::Mat mask = maxmap > threshold;
		int n = cv::connectedComponents(mask, labels, 8, CV_32S);
		vector<cv::Rect> jobRects(n);
		int backgroundIndex = -1;
		FOR_MAT(labels, p) {
			int index = labels.at<int32_t>(p);
			jobRects[index] |= tiles[p.y][p.x];
			if (mask.at<uint8_t>(p) == 0) backgroundIndex = index;
		}
		// debug
		//cv::Mat canvas;
		//cv::cvtColor(source, canvas, cv::COLOR_GRAY2BGR);
		//FOR_ALL(jobRects, j) cv::rectangle(canvas, jobRects[j], CV_RGB(0, 200, 0));
		//pawlin::debugImg("jobs", canvas);
		
		FOR_ALL_IF(jobRects, j, (int)j != backgroundIndex)
			jobs.push_back(jobRects[j]);

	}
/*
DelayErrEstimator(
			const string &nameSource,
			const string &nameDelayed,
			int maxSampleDelay,
			int verboseLevel) :
первые два параметра, просто для наглядности - who is who
maxSampleDay - какая максимальная задержка в сэмплах ожидается
verboseLevel - если >3 выводит инфу в консоль
Далее в циле пушим данные
		void push(float val_source, float val_delayed)
собственно первый аргумент - предположительно оригинальных данных, второй аргумент, предположительно остающих

void find_and_print()
ищет задержку, ошибку для данной задержки и печатает на экран
если в этом классе заменить поля
		vector<float> delayed;
		vector<float> source;
на CycleBufferFloat
то тогда можно пушить и в любой момент времени оценивать бегущую задержку для заданного окна
как хотел Денис
*/
	

	struct DelayErrEstimator {
		vector<float> delayed;
		vector<float> source;
		
		string nameSource, nameDelayed;
		int maxSampleDelay;
		int verboseLevel = 0;
		DelayErrEstimator(
			const string &nameSource,
			const string &nameDelayed,
			int maxSampleDelay,
			int verboseLevel) :
			nameSource(nameSource),
			nameDelayed(nameDelayed),
			maxSampleDelay(maxSampleDelay),
			verboseLevel(verboseLevel)
		{	}

		void push(float val_source, float val_delayed) {
			source.push_back(val_source);
			delayed.push_back(val_delayed);// val_delayed
		};

		std::pair<int, float> bestDelayErr() const {
			cv::Mat search(1, (int)delayed.size(), CV_32F, (float *)&delayed[0]);
			cv::Mat templ(1, (int)(delayed.size()) - maxSampleDelay, CV_32F, (float *)&source[0]);
			cv::Mat res;
			cv::matchTemplate(search, templ, res, 0);
			cv::Point minv;
			double v;
			MinMaxAvg stat(source);
			float norm = 1.0f / stat.getStdev();
			res *= (norm/stat.counter);
			cv::minMaxLoc(res, &v, 0, &minv);
			if (verboseLevel) {
				printf("%s-%s diff by delay:\n", nameSource.c_str(), nameDelayed.c_str());
				std::cout << res << "\n\n";
				if (verboseLevel >= 2)
				{
					StringTable ST;
					StringVector header;
					header.push_back("ts");
					header.push_back("corr");

					for (int i = 0; i < res.cols; i++)
					{
						StringVector row;
						row.push_back(std::to_string(i));
						row.push_back(std::to_string(res.at<float>(0, i)));
						ST.push_back(row);
					}

					std::string csvname = nameSource + "-" + nameDelayed + ".csv";
					ST.save(csvname.c_str(), ';', true, header);
				}
			}
			cv::Mat diff = search(cv::Rect(cv::Point(minv.x, 0), templ.size())) - templ;
			cv::Mat sqdiff;
			cv::multiply(diff, diff, sqdiff);
			double err = cv::mean(sqdiff)[0];
			return { minv.x,(float) sqrt(err) };
		}
		void find_and_print() {
			auto p = bestDelayErr();
			printf("%s-%s best diff %f at delay %d:\n", nameSource.c_str(), nameDelayed.c_str(), p.second, p.first);			
		}
	};

	struct DelayErrEstimatorCycle 
	{
		CycleBufferGeneric<Point3Df> delayed;
		CycleBufferGeneric<Point3Df> source;
		string nameSource, nameDelayed;
		int maxSampleDelay;
		int verboseLevel = 0;
		int length;
		DelayErrEstimatorCycle(
			const string &nameSource,
			const string &nameDelayed,
			int maxSampleDelay,
			int verboseLevel, int length ):
			nameSource(nameSource),
			nameDelayed(nameDelayed),
			maxSampleDelay(maxSampleDelay),
			verboseLevel(verboseLevel),
			delayed(length),
			source(length),
			length(length)
		{	}

		void push(Point3Df val_source, Point3Df val_delayed) {
			source.push(val_source);
			delayed.push(val_delayed);// val_delayed
		};

		std::pair<int, float> bestDelayErr() const {
	
			cv::Mat search(3, (int)delayed.getFilledSize(), CV_32F);
			cv::Mat templ(3, (int)delayed.getFilledSize(), CV_32F);
			vector<Point3Df> tmpv;
			for (int i = 0; i < delayed.getFilledSize(); ++i)
			{
				search.at<Point3Df>(i) = delayed.getDelayedValue(i);
				templ.at<Point3Df>(i) = source.getDelayedValue(i);
				tmpv.push_back(source.getDelayedValue(i));
			}
			
			cv::Mat res;
			cv::matchTemplate(search, templ, res, 0);
			cv::Point minv;
			double v;		

			//MinMaxAvg stat(tmpv);
			//float norm = 1.0f / stat.getStdev();
			//res *= (norm / stat.counter);
			cv::minMaxLoc(res, &v, 0, &minv);
			if (verboseLevel) {
				printf("%s-%s diff by delay:\n", nameSource.c_str(), nameDelayed.c_str());
				std::cout << res << "\n\n";
				if (verboseLevel >= 2)
				{
					StringTable ST;
					StringVector header;
					header.push_back("ts");
					header.push_back("corr");

					for (int i = 0; i < res.cols; i++)
					{
						StringVector row;
						row.push_back(std::to_string(i));
						row.push_back(std::to_string(res.at<float>(0, i)));
						ST.push_back(row);
					}

					std::string csvname = nameSource + "-" + nameDelayed + ".csv";
					ST.save(csvname.c_str(), ';', true, header);
				}
			}
			cv::Mat diff = search(cv::Rect(cv::Point(minv.x, 0), templ.size())) - templ;
			cv::Mat sqdiff;
			cv::multiply(diff, diff, sqdiff);
			double err = cv::mean(sqdiff)[0];
			return { minv.x,(float)sqrt(err) };
		}
		void find_and_print() {
			auto p = bestDelayErr();
			printf("%s-%s best diff %f at delay %d:\n", nameSource.c_str(), nameDelayed.c_str(), p.second, p.first);
		}
		auto get_delay() {return bestDelayErr();}

	};

	inline Point3Df randomPoint3Df() {
		return Point3Df(rnd(-1.0f, +1.0f), rnd(-1.0f, +1.0f), rnd(-1.0f, +1.0f));
	}
	inline vector<Point3Df> genRandomSpericalCloud(const Point3Df &origin, size_t count, float R = 1.0f ) {
		vector<Point3Df> res(count);
		FOR_ALL(res, i) res[i] = origin + randomPoint3Df().normalized()*R;
		return res;
	}
	// non thread safe!!
	struct Point3DfFinder {
		cv::flann::KDTreeIndexParams indexParams; // order is important
		mutable cv::flann::Index kdtree;
		int checks;
		vector<Point3Df> cloud;
		struct Results {
			vector<int> indices; 
			vector<float> dists;
			void clear() {
				indices.clear();
				dists.clear();
			}
			void resize(size_t s) {
				indices.resize(s);
				dists.resize(s);
			}
			vector<Point3Df> getPoints(const vector<Point3Df> &cloud) const {
				vector<Point3Df> result(indices.size());
				FOR_ALL(indices, i) result[i] = cloud[indices[i]];
				return result;
			}
			size_t size() const { return indices.size(); }
		};
		Point3DfFinder(const vector<Point3Df> &points, int checks = 1024) : checks(checks), cloud(points) {
			// since Point3Df has same structure as cv::Point3f we use reference cast
			assertEqualSize(sizeof(Point3Df), sizeof(cv::Point3f), "Point3DfFinder assumes equivalency of Point3Df and cv::Point3f");
			auto & p = (vector<cv::Point3f> &) points;
			kdtree.build(cv::Mat(p).reshape(1), indexParams);
		}
		void findKNearest(const Point3Df &p, int K, Results &r, bool sorted) const {
			r.clear();
			vector<float> query = { p.x,p.y,p.z };
			kdtree.knnSearch(query, r.indices, r.dists, K, cv::flann::SearchParams(checks, 0, sorted));
			//if(r.indices.size()>K) r.resize(K);
		}
		Point3Df findNearestPoint(const Point3Df &p, Results &r) const { 
			findKNearest(p, 1, r, false);
			if (r.indices.size() == 1) return cloud[r.indices.front()];
			throw std::runtime_error("Point3DfFinder::findNearestPoint did not find nearest point");
			return Point3Df();
		}
		void findRNearest(const Point3Df &p, float R, Results &r, int expectedCount, bool sorted) const {
			vector<float> query = { p.x,p.y,p.z };
			int found = 0;
			for(int tryCount = expectedCount;;tryCount*=2) {
				r.clear();
				found = kdtree.radiusSearch(query, r.indices, r.dists, R*R, tryCount, cv::flann::SearchParams(checks, 0, sorted));// radiusSearch(query, indices, dists, range, numOfPoints);
				if (found <= tryCount) break;
			}
			r.resize(found);
		}
	};

	// non thread safe!!
	template <typename T, typename P = cv::Point2f, typename I = cv::Point2f>
	struct FindByPoint {
		struct Point2fCompare {
			bool operator()(const P &a, const P &b) const {
				return 
					std::pair<float, float>({ a.x,a.y }) < 
					std::pair<float, float>({ b.x,b.y });
			}
		};

		vector<T> objects;
		vector<I> points; // internal points representation

		cv::flann::KDTreeIndexParams indexParams; // order is important
		mutable cv::flann::Index kdtree;

		FindByPoint() {}
		FindByPoint(const FindByPoint<T> &other) {
			objects = other.objects;
			points = other.points;
			init();
		}

		void init(const vector<P> &pts) {
			points.resize(pts.size());
			objects.resize(pts.size());

			FOR_ALL(pts, i) {
				points[i] = { pts[i].x,pts[i].y };
				objects[i] = i;
			}
			init();
		}

		void init(const vector<P> &pts, const vector<T> &objs) {
			assertEqualSize(pts.size(), objs.size(), "FindByPoint points and objects must be arrays of same size");
			points.resize(pts.size());
			objects = objs;

			FOR_ALL(pts,i) points[i] = {pts[i].x,pts[i].y };
			init();
		}

		void init(const std::map<P, T, Point2fCompare> &m) {
			objects.reserve(m.size());
			points.reserve(m.size());
			for (auto &iter : m) {
				points.push_back({ iter.first.x,iter.first.y });
				objects.push_back(iter.second);
			}
			init();
		}
		void init() {
			assertEqualSize(points.size(), objects.size(), "FindByPoint::init points count must be equal to associated objects count");
			kdtree.build(cv::Mat(points).reshape(1), indexParams);
		}
		std::pair<T, float> find(const P &p) const {
			vector<float> query = { p.x,p.y };
			vector<int> indices;
			vector<float> dists;
			kdtree.knnSearch(query, indices, dists, 1);// radiusSearch(query, indices, dists, range, numOfPoints);
			if (indices.empty()) throw std::runtime_error("FindByPoint knnSearch failed"); // should not fail
			return std::pair<T, float>({ objects[indices[0]],dists[0] });
		}
	};

	// Lucy-Richardson deconvolution for gaussian blur
	struct Deblur {
		cv::Mat kernel;
		cv::Mat kernel_t;
		cv::Mat h_img0;
		cv::Mat h_img1;
		cv::Mat c_img;
		int iter = 0;
		cv::Mat input;
		
		// input must not contain zero pixels !!!!! (division by zero caution!)
		Deblur(const cv::Mat &input, float sigma) : input(input.clone()) {
			int c = float2int(sigma * 2);
			int ksize = 1 + 2 * c;
			kernel = cv::Mat::zeros(ksize, ksize, CV_32F);
			kernel.at<float>(c, c) = 1.0f;
			cv::GaussianBlur(kernel, kernel, kernel.size(), sigma, sigma);
			//pawlin::debugImg("kernel", kernel, 5, 0, true, true);
			kernel_t = kernel.t();
			h_img0 = input.clone();
			h_img1 = input.clone();
			c_img = input.clone();
		}

		cv::Mat doIter() { // reasonable count of iters is proportional to sigma (small deblur <10 iterations, deep defocus, can be 30-100 iters)

			cv::Mat t_img;
			cv::Mat &h_prev = (iter % 2) == 0 ? h_img0 : h_img1;
			cv::Mat &h_cur = (iter % 2) == 0 ? h_img1 : h_img0;

			cv::filter2D(h_prev, c_img, CV_32F, kernel);
			//c_img += alpha;
			cv::divide(input, c_img, c_img);
			cv::filter2D(c_img, t_img, CV_32F, kernel_t);
			cv::multiply(h_prev, t_img, h_cur);
			//pawlin::debugImg("deblur", h_cur, 1, 30, true);
			iter++;
			return h_cur;
		}

	};
	template <typename F>
	inline cv::Mat infRot(F wx, F wy, F wz, bool addI) {
		// https://en.wikipedia.org/wiki/Angular_velocity_tensor
		F D = addI ? F(1) : 0;
		cv::Mat A = (cv::Mat_<F>(3, 3) <<
			D, -wz, wy,
			wz, D, -wx,
			-wy, wx, D);
		//std::cout << A << "\n\n";
		return A;
	}
	template <typename T, typename F>
	inline cv::Mat infRot(const T &w, bool addI) {

		return infRot<F>(F(w[0]), F(w[1]), F(w[2]), addI);
	}

	inline vector<Point3Df> smoothDCT(const vector<Point3Df> &line, const float smooth_k = 0.1f /*the bigger coef - the greater smoothing effect*/) {
		cv::Mat src((int)line.size(), 3, CV_32F);
		FOR_ALL_INT(line, i) {
			auto loc = line[i];
			src.at<float>(i, 0) = loc.x;
			src.at<float>(i, 1) = loc.y;
			src.at<float>(i, 2) = loc.z;
		}
		cv::Mat coefs;
		cv::dct(src, coefs);
		auto stat = pawlin::getStat(cv::abs(coefs));
		//stat.print();
		float threshold = stat.getAvg()*smooth_k;
		cv::Mat mask = cv::abs(coefs) < threshold;
		coefs.setTo(0, mask);
		cv::idct(coefs, src);
		vector<Point3Df> coords(line.size());
		FOR_ALL_INT(line, i) {
			Point3Df loc_f = { src.at<float>(i, 0), src.at<float>(i, 1),src.at<float>(i, 2) };
			coords[i] = loc_f;
		}
		return coords;
	}

	inline vector<Point2Df> smoothDCT(const vector<Point2Df> &line, const float smooth_k = 0.1f /*the bigger coef - the greater smoothing effect*/) {
		cv::Mat src((int)line.size(), 2, CV_32F);
		FOR_ALL_INT(line, i) {
			auto loc = line[i];
			src.at<float>(i, 0) = loc.x;
			src.at<float>(i, 1) = loc.y;
		}
		cv::Mat coefs;
		cv::dct(src, coefs);
		auto stat = pawlin::getStat(cv::abs(coefs));
		//stat.print();
		float threshold = stat.getAvg()*smooth_k;
		cv::Mat mask = cv::abs(coefs) < threshold;
		coefs.setTo(0, mask);
		cv::idct(coefs, src);
		vector<Point2Df> coords(line.size());
		FOR_ALL_INT(line, i) {
			Point2Df loc_f = { src.at<float>(i, 0), src.at<float>(i, 1) };
			coords[i] = loc_f;
		}
		return coords;
	}
	inline vector<float> smoothDCT(const vector<float> &line, const float smooth_k = 0.1f /*the bigger coef - the greater smoothing effect*/) {
		cv::Mat src((int)line.size(), 1, CV_32F,(float*)&line[0]);
		cv::Mat coefs;
		cv::dct(src, coefs);
		auto stat = pawlin::getStat(cv::abs(coefs));
		//stat.print();
		float threshold = stat.getAvg()*smooth_k;
		cv::Mat mask = cv::abs(coefs) < threshold;
		coefs.setTo(0, mask);
		vector<float> coords(line.size());
		cv::Mat out((int)coords.size(), 1, CV_32F, (float*)&coords[0]);
		cv::idct(coefs, out);
		return coords;
	}

	/**
	* Code for thinning a binary image using Zhang-Suen algorithm.
	*
	* Author:  Nash (nash [at] opencv-code [dot] com)
	* Website: http://opencv-code.com
	* https://github.com/bsdnoobz/zhang-suen-thinning
	*/

	/**
	* Perform one thinning iteration.
	* Normally you wouldn't call this function directly from your code.
	*
	* Parameters:
	* 		im    Binary image with range = [0,1]
	* 		iter  0=even, 1=odd
	*/
	inline void thinningIteration(cv::Mat& img, int iter)
	{
		CV_Assert(img.channels() == 1);
		CV_Assert(img.depth() != sizeof(uchar));
		CV_Assert(img.rows > 3 && img.cols > 3);

		cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

		int nRows = img.rows;
		int nCols = img.cols;

		if (img.isContinuous()) {
			nCols *= nRows;
			nRows = 1;
		}

		int x, y;
		uchar *pAbove;
		uchar *pCurr;
		uchar *pBelow;
		uchar *nw, *no, *ne;    // north (pAbove)
		uchar *we, *me, *ea;
		uchar *sw, *so, *se;    // south (pBelow)

		uchar *pDst;

		// initialize row pointers
		pAbove = NULL;
		pCurr = img.ptr<uchar>(0);
		pBelow = img.ptr<uchar>(1);

		for (y = 1; y < img.rows - 1; ++y) {
			// shift the rows up by one
			pAbove = pCurr;
			pCurr = pBelow;
			pBelow = img.ptr<uchar>(y + 1);

			pDst = marker.ptr<uchar>(y);

			// initialize col pointers
			no = &(pAbove[0]);
			ne = &(pAbove[1]);
			me = &(pCurr[0]);
			ea = &(pCurr[1]);
			so = &(pBelow[0]);
			se = &(pBelow[1]);

			for (x = 1; x < img.cols - 1; ++x) {
				// shift col pointers left by one (scan left to right)
				nw = no;
				no = ne;
				ne = &(pAbove[x + 1]);
				we = me;
				me = ea;
				ea = &(pCurr[x + 1]);
				sw = so;
				so = se;
				se = &(pBelow[x + 1]);

				int A = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
					(*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
					(*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
					(*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
				int B = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
				int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
				int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

				if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
					pDst[x] = 1;
			}
		}

		img &= ~marker;
	}

	/**
	* Function for thinning the given binary image
	*
	* Parameters:
	* 		src  The source image, binary with range = [0,255]
	* 		dst  The destination image
	*/
	inline void thinning(const cv::Mat& src, cv::Mat& dst)
	{
		dst = src.clone();
		dst /= 255;         // convert to binary image
		//pawlin::debugImg("dst", dst, 1, 0);
		cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
		cv::Mat diff;

		do {
			thinningIteration(dst, 0);
			thinningIteration(dst, 1);
			cv::absdiff(dst, prev, diff);
			dst.copyTo(prev);
		} while (cv::countNonZero(diff) > 0);

		dst *= 255;
	}

	// dir left - 0, right - 1, up - 2, down - 3
	inline vector<cv::Point> findEnds(const cv::Mat &image8U, const int dir) {
		static cv::Mat kernels[4] = {

		(cv::Mat_<int>(3, 3) << //LEFT
			-1, -1, 0,
			-1, +1, 0,
			-1, -1, 0),
		(cv::Mat_<int>(3, 3) << //RIGHT
			0, -1, -1,
			0, +1, -1,
			0, -1, -1),
		(cv::Mat_<int>(3, 3) << // UP
			-1, -1, -1,
			-1, +1, -1,
			0, 0, 0),
		(cv::Mat_<int>(3, 3) << // DOWN
			0, 0, 0,
			-1, +1, -1,
			-1, -1, -1) 
		};

		static cv::Mat fake_kernels[4] = {

			(cv::Mat_<int>(3, 3) << //LEFT
				-1, -1, +1,
				-1, +1, 0,
				-1, -1, +1),
			(cv::Mat_<int>(3, 3) << //RIGHT
				+1, -1, -1,
				0, +1, -1,
				+1, -1, -1),
			(cv::Mat_<int>(3, 3) << // UP
				-1, -1, -1,
				-1, +1, -1,
				+1, 0, +1),
			(cv::Mat_<int>(3, 3) << // DOWN
				+1, 0, +1,
				-1, +1, -1,
				-1, -1, -1)
		};

		if (dir < 0 || dir>4) throw std::runtime_error("findEnds - incorrect direction constant");
		cv::Mat out;
		cv::morphologyEx(image8U, out, cv::MORPH_HITMISS, kernels[dir]);
		cv::Mat fakes;
		cv::morphologyEx(image8U, fakes, cv::MORPH_HITMISS, fake_kernels[dir]);
		out.setTo(0, fakes);
		vector<cv::Point> results;
		cv::findNonZero(out, results);
		return results;
	}

	inline cv::Mat planeEdgeModel(const cv::Mat &image32F, int edge = 2, float err_stdevs_shift = 0.0f) {
		if (image32F.cols < edge * 2 || image32F.rows < edge * 2) throw std::runtime_error("too small image and big edges request for planeEdgeModel");
		vector<Point3Df> data;
		// build mask
		cv::Mat mask = cv::Mat::zeros(image32F.size(), CV_8U);

		// SLOW implementation, but faster code could be more complex
		cv::rectangle(mask, { edge,edge,image32F.cols - 1 - 2 * edge,image32F.rows - 1 - 2 * edge }, 255, -1);
		FOR_MAT(image32F, p) {
			if (mask.at<uint8_t>(p) == 255) continue;
			data.push_back({ (float)p.x,(float)p.y,image32F.at<float>(p) });
		}

		if (data.size() < 4) throw std::runtime_error("too few datapoints for planeEdgeModel");
		pawlin::QuadSurface quad(data, true);
		cv::Mat res = image32F.clone();
		float addz = quad.getErrStat().getStdev() * err_stdevs_shift;
		FloatVector xx;
		FOR_MAT(res, p) {
			quad.makeVector((float)p.x, (float)p.y, xx);
			res.at<float>(p) = quad.compute(xx)  + addz;
		}
		return res;
	}

	inline 	vector<cv::Point> traverseCurve(
		const cv::Mat &image,
		const cv::Point &start) 
	{
		static vector<cv::Point> neighbours = {
			{ 0,+1 },{ +1,0 },{ -1,0 },{ 0,-1 },
			{ +1,+1 },{ -1,+1 },{ -1,-1 },{ +1,-1 }
		};

		cv::Mat mask = image.clone();

		auto get = [&](const cv::Mat &img, const cv::Point &p) {
			if (p.y < 0 || p.y >= img.rows || p.x < 0 || p.x >= img.cols) return (uint8_t)0;
			return img.at<uint8_t>(p);
		};
		auto set = [&](cv::Mat &img, const cv::Point &p, uint8_t val) {
			if (p.y < 0 || p.y >= img.rows || p.x < 0 || p.x >= img.cols) return;
			img.at<uint8_t>(p) = val;
		};
		auto neighbours_count = [&](const cv::Mat &img, const cv::Point &p) {
			size_t counter = 0;
			FOR_ALL_IF(neighbours, i, get(img, p + neighbours[i]) > 0) counter++;
			return counter;
		};
		cv::Point current = start;
		vector<cv::Point> array;
		vector<cv::Point> nb;
		for (;;) {
			array.push_back(current);
			mask.at<uint8_t>(current) = 0; // visited
			nb.clear();
			FOR_ALL(neighbours, i) {
				cv::Point v = current + neighbours[i];
				if (get(mask, v) > 0) nb.push_back(v);
			}
			if (nb.empty()) break;
			// now the whole point is if we have more than 1 neighbour - where do we go?
			auto iter = std::max_element(nb.begin(), nb.end(),
				[&](const cv::Point &n1, const cv::Point &n2) {
				return neighbours_count(mask, n1) > neighbours_count(mask, n2);
			});
			current = *iter;
		}
		return array;
	}

	inline void smooth_resample(
		const vector<Point2Df> &line_init0,
		vector<Point2Df> &line_fs,
		float lineSmoothing, int desired_count = -1, float desired_step = 0)
	{
		// filter line
		vector<Point2Df> line_f = lineSmoothing < 0 ? line_init0 : smoothDCT(line_init0, lineSmoothing);

		// fill samples map using integral(t0...t) of ds as key
		std::map<float, Point2Df> samples;
		float dist = 0;
		MinMaxAvg stat;
		FOR_ALL(line_f, i) {
			if (i > 0) {
				float ds = line_f[i].distance(line_f[i - 1]);
				dist += ds;
				stat.take(ds);
			}
			samples[dist] = line_f[i];
		}
		// find reasonable step (average step/2?)
		// float step = dist / (float)line_f.size();
		float step = desired_count > 0 ? dist / desired_count : desired_step;
		if (step <= 0) throw std::runtime_error("smooth_resample - please specify either desired_count or desired_step");

		// pass from 0 till total distance with given step and store interpolated in line_fs
		float s = 0;
		line_fs.clear();
		for(;;) {
			Point2Df p = pawlin::interpolate(samples, s);
			if (line_fs.size()>2 && p == line_fs.back()) break; // we passed the end of the samples map, values are repeating
			line_fs.push_back(p);
			s += step;
		}
	}

	inline vector<cv::Point> findContourHavingPoint(const cv::Mat & endMask, const cv::Point &e) {
		vector<vector<cv::Point> > contours;
		cv::findContours(endMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		vector<cv::Point> trueContour;
		FOR_ALL_INT(contours, c) {
			if (cv::pointPolygonTest(contours[c], ptconv2f(e), false) > 0) {
				trueContour = contours[c];
				//				if (params.verboseLevel>=3) cv::drawContours(visimg, contours, c, CV_RGB(255, 0, 0), 1, cv::LINE_AA);
			}
			//			else
			//				if (params.verboseLevel>=3) cv::drawContours(visimg, contours, c, CV_RGB(0, 0, 50), 1, cv::LINE_AA);
		}
		if (trueContour.empty()) throw std::runtime_error("trueContour failed");
		return trueContour;
	}

	template<typename T>
	inline size_t travelDistance(const vector<T> &line, int start, float length, int s /*+1 or -1*/) {
		assertIndexRange(line.size(), start, "travelDistance start is out of bounds");
		if (length == 0) return (size_t)start;
		float dist = 0;
		T current = line[start];
		int index = (int)start;
		do {
			index += s;
			assertIndexRange(line.size(), index, "travelDistance index is out of bounds while traveling for specified distance");
			T newpt = line[index];
			float d = newpt.distance(current);
			dist += d;
			current = newpt;
		} while (dist < length);
		return (size_t)index;
	}

	inline cv::Rect makeAndSaveROI(const cv::Mat &image, const string &filename = "roi.rect") {
		cv::Rect rect;
		try {
			FileSaver<cv::Rect>::load(filename, rect);
		}
		catch (...) {
			pawlin::SelectionROIWindow roi("select roi", image);
			roi.run();
			rect = roi.getROI();
			FileSaver<cv::Rect>::save(filename, rect);
		}
		return rect;
	}

	inline cv::Mat visDepthDiff(const cv::Mat &imageA32F, const cv::Mat &imageB32F, const int z = 50, const float scale = 0.05f)
	{
		cv::Mat a;
		cv::Mat rs = imageA32F - imageB32F;
		rs.convertTo(a, CV_8UC3);
		a = 0;
		cv::cvtColor(a, a, cv::COLOR_BGR2RGB);
		cv::Scalar zz = cv::mean(rs);
		for (int i = 0; i < a.cols - z; i = i + z)
		{
			for (int j = 0; j < a.rows - z; j = j + z)
			{
				cv::Mat rt = rs(cv::Rect(i, j, z, z));
				int e = float2int((float)(cv::mean(rt).val[0] - zz.val[0]) * scale);

				int d = float2int((z / 2 - 2) - fabsf((float)e));
				if (d < 2)d = 2;
				if (d > (z / 2 - 1)) d = z / 2 - 1;

				if (e > 0)
				{
					cv::line(a, cv::Point(i + d, j + (z / 2)), cv::Point(i + (z - d), j + (z / 2)), cv::Scalar(200, 20, 20), 2);
				}
				else if(e < 0)
				{
					cv::line(a, cv::Point(i + d, j + (z / 2)), cv::Point(i + (z - d), j + (z / 2)), cv::Scalar(20, 20, 200), 2);
					cv::line(a, cv::Point(i + (z / 2), j + d), cv::Point(i + (z / 2), j + (z - d)), cv::Scalar(20, 20, 200), 2);
				}
				else {
					cv::circle(a, cv::Point(i + (z / 2), j + (z / 2)),2, CV_RGB(255, 255, 0),1, cv::LINE_AA);
				}
			}
		}
		return a;
	}

	template <typename T>
	inline void drawChain(
		cv::Mat &canvas, 
		const vector<T> &chain, 
		cv::Scalar line_color = CV_RGB(0, 100, 0), 
		cv::Scalar circle_color = CV_RGB(0, 0, 255)) 
	{
		FOR_ALL(chain, n) {
			cv::Point2f p = { (float)chain[n].x,(float)chain[n].y };
			cv::circle(canvas, p, 2, circle_color, 1, cv::LINE_AA);
			if (n >= 1)
				cv::line(canvas,
					p,
					cv::Point2f{ (float)chain[n - 1].x,(float)chain[n - 1].y },
					line_color, 1, cv::LINE_AA);
		}
	}
} // end pawlin namespace