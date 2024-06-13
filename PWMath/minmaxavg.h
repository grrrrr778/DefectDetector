// (c) Pawlin Technologies Ltd 2010
// File: statstics.h / .cpp
// purpose: basic statistical functions
// author: P.V. Skribtsov
// ALL RIGHTS RESERVED

#pragma once

// STL 
#include <algorithm> // for std::max , std::min
#include <vector>
#include <fstream>
#include "math.h"
#include <PWNGeneral/pwnutil.h>

template <typename T>
inline T median_fast(std::vector<T> &array) { // ATTENTION -- CHANGES INPUT ARRAY!!!
	std::sort(array.begin(), array.end());
	return array[array.size() / 2];
}

template <typename T>
inline T median(const std::vector<T> &_array) {
	std::vector<T> array = _array;
	std::sort(array.begin(), array.end());
	return array[array.size() / 2];
}


struct MinMaxAvg {
public:
	uint32_t counter; // to be fixed number of bytes	
	float minv;
	float maxv;
	double sum;
	double sumsq;
	void setZero() {
		counter = 0;
		minv = 0;
		maxv = 0;
		sum = 0;
		sumsq = 0;
	}
	MinMaxAvg(const vector<float> &vals) {
		setZero();
		FOR_ALL(vals, i) take(vals[i]);
	}
	MinMaxAvg() {
		setZero(); //pvs I didn't like uninitialized states
	}
	void clear() {
		setZero(); // pvs - removed code duplicate
	}
	
	inline bool is_init(void) const { return (this->counter > 0); }
	float getMinMaxMiddle() const { return 0.5f*(minv+maxv);}

	void add(const MinMaxAvg & other) {
		if (!other.is_init()) return;
		if (!is_init()) { *this = other; return; }
		
		this->counter += other.counter;
		this->minv = std::min<float>(this->minv, other.minv);
		this->maxv = std::max<float>(this->maxv, other.maxv);
		this->sum += other.sum;
		this->sumsq += other.sumsq;
	}
	
	void take(float v, int n) { // n means how many times this value was inr
		if(counter == 0) {
			minv = v;
			maxv = v;
			sum = (double)v*(double)n;
			sumsq = (double)v*(double)v*(double)n;
		}
		else {
			minv = std::min<float>(minv,v);
			maxv = std::max<float>(maxv,v);
			sum += (double)v*(double)n;
			sumsq += (double)v*(double)v*(double)n;
		}
		counter += n;
	}
	inline void take(const vector<float> &array) { FOR_ALL(array, i) take(array[i]); }
	inline void take(float v) { // n means how many times this value was inr
		if(counter == 0) {
			minv = v;
			maxv = v;
			sum = (double)v;
			sumsq = (double)v*(double)v;
		}
		else {
			minv = std::min<float>(minv,v);
			maxv = std::max<float>(maxv,v);
			sum += (double)v;
			sumsq += (double)v*(double)v;
		}
		counter++;
	}
	inline void take(double v) {
		if (counter == 0) {
			minv = (float)v;
			maxv = (float)v;
			sum = v;
			sumsq = v*v;
		}
		else {
			minv = std::min<float>(minv, (float)v);
			maxv = std::max<float>(maxv, (float)v);
			sum += v;
			sumsq += v*v;
		}
		counter++;
	}
	double getAvgDouble() const {
		return counter == 0 ? 0 : (sum / (double)counter);
	}
	float getAvg() const { return (float)getAvgDouble(); }
	float getStdev() const { 
		if (counter == 0) return 0;
		double avg = getAvgDouble();
		
		double v = sumsq / (double)counter - (double)avg*(double)avg;
		if(v<0) return 0;
		return (float)sqrt(v);
	}
	float getRange() const { return maxv - minv; }
 	void save(FILE *fp)const{
		
		fprintf(fp,"%d,%f,%f,%lf,%lf\n", counter, minv, maxv, sum, sumsq);
	}
	void load(FILE *fp){
	
		fscanf(fp,"%d,%f,%f,%lf,%lf\n", &counter, &minv, &maxv, &sum, &sumsq);
	}
	void print(bool newline = false) const {
		printf("min %f, max %f, avg %f, stdev %f,count %d%s", minv, maxv, (float)getAvg(), getStdev(),
			this->counter,newline ? "\n" : ""); 
	}
	void print(FILE *file, bool newline = false) const {
		fprintf(file,"min %f, max %f, avg %f, stdev %f,count %d%s", minv, maxv, (float)getAvg(), getStdev(),
			this->counter, newline ? "\n" : "");
	}
};

// didn't touch old minmaxavg for compatibility purpose
template <typename T = double>
struct MinMaxAvgT {
public:
	T counter; // to be fixed number of bytes	
	T minv;
	T maxv;
	T sum;
	T sumsq;
	void setZero() {
		counter = 0;
		minv = 0;
		maxv = 0;
		sum = 0;
		sumsq = 0;
	}
	MinMaxAvgT(const vector<T> &vals) {
		setZero();
		FOR_ALL(vals, i) take(vals[i]);
	}
	MinMaxAvgT() {
		setZero(); //pvs I didn't like uninitialized states
	}
	void clear() {
		setZero(); // pvs - removed code duplicate
	}

	inline bool is_init(void) const { return (this->counter > 0); }
	T getMinMaxMiddle() const { return T(0.5*(minv + maxv)); }

	void scale(T k) {
		minv *= k;
		maxv *= k;
		// counter is unchanged
		sum *= k;
		sumsq *= k*k;
	}

	void add(const MinMaxAvg & other) {
		if (!other.is_init()) return;
		if (!is_init()) { *this = other; return; }

		this->counter += other.counter;
		this->minv = std::min<T>(this->minv, other.minv);
		this->maxv = std::max<T>(this->maxv, other.maxv);
		this->sum += other.sum;
		this->sumsq += other.sumsq;
	}

	void take(T v, T n) { // n means how many times this value was inr
		if (counter == 0) {
			minv = v;
			maxv = v;
			sum = v*n;
			sumsq = v*v*n;
		}
		else {
			minv = std::min<T>(minv, v);
			maxv = std::max<T>(maxv, v);
			sum += v*n;
			sumsq += v*v*n;
		}
		counter += n;
	}
	inline void take(T v) { // n means how many times this value was inr
		if (counter == 0) {
			minv = v;
			maxv = v;
			sum = v;
			sumsq = v*v;
		}
		else {
			minv = std::min<T>(minv, v);
			maxv = std::max<T>(maxv, v);
			sum += v;
			sumsq += v*v;
		}
		counter++;
	}
	T getAvg() const {
		return counter == 0 ? 0 : (sum / counter);
	}
	T getCV() const {
		return getStdev() / getAvg();
	}
	T getStdev() const {
		if (counter == 0) return 0;
		T avg = getAvg();

		T v = sumsq / counter - avg*avg;
		if (v<0) return 0;
		return (T)sqrt(v);
	}
	T getRange() const { return maxv - minv; }
	void print(vector<string> &out) const {
		out = {
			std::to_string(minv),
			std::to_string(maxv),
			std::to_string(getAvg()),
			std::to_string(getCV()),
			std::to_string(getStdev()),
			smart_float2str(counter)
		};
	}
	void printHeader(vector<string> &out) const {
		out = {
			"minv",
			"maxv",
			"avg",
			"CV",
			"stdev",
			"counter"
		};
	}
	void print(FILE *file, bool newline = false) const {
		vector<string> v, h;
		print(v);
		printHeader(h);
		assertEqualSize(v.size(), h.size(), "MinMaxAvgT logical printing error - header and values mismatch");
		FOR_ALL(v, i) fprintf(file, "%s %s%s", h[i].c_str(), v[i].c_str(), i + 1 == v.size() ? "" : ",");
		if(newline) fprintf(file,"\n");
	}
};

template <typename T = double>
struct MinMaxT {
public:
	T minv;
	T maxv;
	bool inited = false;
	void setZero() {
		minv = 0;
		maxv = 0;
		inited = false;
	}
	MinMaxT(const vector<T> &vals) {
		setZero();
		FOR_ALL(vals, i) take(vals[i]);
	}
	MinMaxT() {
		setZero(); //pvs I didn't like uninitialized states
	}
	void clear() {
		setZero(); // pvs - removed code duplicate
	}

	inline bool is_init(void) const { return inited; }
	T getMinMaxMiddle() const { return T(0.5*(minv + maxv)); }

	void scale(T k) {
		minv *= k;
		maxv *= k;
		// counter is unchanged
	}

	void add(const MinMaxT & other) {
		if (!other.is_init()) return;
		if (!is_init()) { *this = other; return; }

		this->inited = true;
		this->minv = std::min<T>(this->minv, other.minv);
		this->maxv = std::max<T>(this->maxv, other.maxv);
	}

	inline void take(T v) { // n means how many times this value was inr

		if (inited) {
			minv = v;
			maxv = v;
		}
		else {
			minv = std::min<T>(minv, v);
			maxv = std::max<T>(maxv, v);
		}
		inited = true;
	}
	T getRange() const { return maxv - minv; }
	void print(vector<string> &out) const {
		out = {
			std::to_string(minv),
			std::to_string(maxv),
			inited ? "1.0" : "0"
		};
	}
	void printHeader(vector<string> &out) const {
		out = {
			"minv",
			"maxv",
			"inited"
		};
	}
	void print(FILE *file, bool newline = false) const {
		vector<string> v, h;
		print(v);
		printHeader(h);
		assertEqualSize(v.size(), h.size(), "MinMaxT logical printing error - header and values mismatch");
		FOR_ALL(v, i) fprintf(file, "%s %s%s", h[i].c_str(), v[i].c_str(), i + 1 == v.size() ? "" : ",");
		if (newline) fprintf(file, "\n");
	}
};


inline void update(vector<MinMaxAvg> &stat, const vector<float> &data) {
	if (stat.size() != data.size()) throw std::runtime_error("update(vector<MinMaxAvg> &stat, const vector<float> &data) data has different size than stat\n");
	FOR_ALL_IF(stat, i, finite_check(data[i])) stat[i].take(data[i]);
}

class MinMaxAvgSafe {
protected:
	MinMaxAvg stats;
	float mean;
	float stdev;
	bool calced;
public:
	MinMaxAvgSafe(const bool &calced = false, const float &mean = 0.0f, const float &stdev = 0.0f ) : stats(MinMaxAvg()) {
		this->mean = mean;
		this->stdev = stdev;
		this->calced = calced;
	}

	void take(const float &v) {
		stats.take(v);
	}

	float getMean(const bool &recalc) const {
		if (!calced || recalc) {
			const_cast<float&>(mean) = stats.getAvg();
			return mean;
		}

		return mean;
	}

	float getStdev(const bool &recalc) const {
		if (!calced || recalc) {
			const_cast<float&>(stdev) = stats.getStdev();
			return stdev;
		}

		return stdev;
	}

};

class RunningStats
{
private:
	long long n;
	double M1, M2, M3, M4;

public:
	RunningStats()
	{
		Clear();
	}

	RunningStats(long long n, double M1, double M2, double M3, double M4) :
	n(n), M1(M1), M2(M2), M3(M3), M4(M4)
	{
	
	}

	void Clear()
	{
		n = 0;
		M1 = M2 = M3 = M4 = 0.0;
	}

	void Push(double x)
	{
		double delta, delta_n, delta_n2, term1;

		long long n1 = n;
		n++;
		delta = x - M1;
		delta_n = delta / n;
		delta_n2 = delta_n * delta_n;
		term1 = delta * delta_n * n1;
		M1 += delta_n;
		M4 += term1 * delta_n2 * (n*n - 3 * n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3;
		M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2;
		M2 += term1;
	}

	long long NumDataValues() const
	{
		return n;
	}

	double Mean() const
	{
		return M1;
	}

	double Variance() const
	{
		return M2 / (n - 1.0);
	}

	double StandardDeviation() const
	{
		return sqrt(Variance());
	}

	double Skewness() const
	{
		return sqrt(double(n)) * M3 / pow(M2, 1.5);
	}

	double Kurtosis() const
	{
		return double(n)*M4 / (M2*M2) - 3.0;
	}

	friend RunningStats operator+(const RunningStats a, const RunningStats b)
	{
		RunningStats combined;

		combined.n = a.n + b.n;

		double delta = b.M1 - a.M1;
		double delta2 = delta * delta;
		double delta3 = delta * delta2;
		double delta4 = delta2 * delta2;

		combined.M1 = (a.n*a.M1 + b.n*b.M1) / combined.n;

		combined.M2 = a.M2 + b.M2 +
			delta2 * a.n * b.n / combined.n;

		combined.M3 = a.M3 + b.M3 +
			delta3 * a.n * b.n * (a.n - b.n) / (combined.n*combined.n);
		combined.M3 += 3.0*delta * (a.n*b.M2 - b.n*a.M2) / combined.n;

		combined.M4 = a.M4 + b.M4 + delta4 * a.n*b.n * (a.n*a.n - a.n*b.n + b.n*b.n) /
			(combined.n*combined.n*combined.n);
		combined.M4 += 6.0*delta2 * (a.n*a.n*b.M2 + b.n*b.n*a.M2) / (combined.n*combined.n) +
			4.0*delta*(a.n*b.M3 - b.n*a.M3) / combined.n;

		return combined;
	}

	RunningStats& operator+=(const RunningStats& rhs)
	{
		RunningStats combined = *this + rhs;
		*this = combined;
		return *this;
	}
};

