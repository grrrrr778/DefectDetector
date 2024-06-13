// (c) Pawlin Technologies Ltd. 2009
// http://www.pawlin.ru
// File: pwntypes.h
// Purpose: defines types and structures
// Author: Alexey Dolgopolov
// ----------------------------------------------
#pragma once
//#ifndef _PWN_TYPES_H_
//#define _PWN_TYPES_H_
#include <PWMath2_0/SimpleMatrix.h>
#include <vector>
#include <string>
using namespace std;


#define  SQR(x)((x)*(x))
#define  SIGN(x)     (*(int*)&x & 0x7FFFFFFF) ? ((*(int*)&x & 0x80000000)? -1:1):0

//common parameters of alghorithms
const string MAX_ITER = "MAX_ITER";
const string EPS = "EPS";
#if 0
	//#define MAX_ITER				"MAX_ITER"
	#define EPS						"EPS"
#endif 
	#define ALPHA					"ALPHA"
	#define PWN_VERBOSE				"VERBOSE"
	#define RESETING				"RESETING"
	#define CONVERGE				"CONVERGE"
	#define SEED					"SEED"

//tel paramters
	#define TEL_SPEED				"TEL_SPEED"
	#define TEL_ZSCALE				"TEL_ZSCALE"
	#define TEL_TARGET_UNITS		"TEL_TARGET_UNITS"
	#define TEL_TE_MINUS			"TEL_TE_MINUS"

//lm paramters
	#define LM_MU_0					"LM_MU_0"
	#define LM_MU_INC				"LM_MU_INC"
	#define LM_MU_DEC				"LM_MU_DEC"
	#define LM_THRESHOLD			"LM_THRESHOLD"
	#define LM_ZSCALE				"LM_ZSCALE"

//rprop paramters
	#define RP_N_MINUS				"RP_N_MINUS"
	#define RP_N_PLUS				"RP_N_PLUS"
	#define RP_DELTA_0				"RP_DELTA_0"
	#define RP_DELTA_MIN			"RP_DELTA_MIN"
	#define RP_DELTA_MAX			"RP_DELTA_MAX"

//quickprop paramters
	#define QP_MODE_SWITCH_THRESHOLD				"QP_MODE_SWITCH_THRESHOLD"
	#define QP_EPSILON								"QP_EPSILON"
	#define QP_MAX_FACTOR							"QP_MAX_FACTOR"
	#define QP_DECAY								"QP_DECAY"
// ELM parameters
#define ELM_MU						"ELM_MU"

struct FeatureSubstitute { 
        int featureIndex; 
        float substitute_value; 
};

//union Multitype { //
//		bool b;
//		unsigned u;
//		int i;
//		float f;
//};

struct IndexRange{
	size_t start;
	size_t end;
	IndexRange() { return_to_def(); }
	IndexRange(size_t start, size_t  end){ set(start, end); }

	inline void set(size_t start, size_t  end) {this->start = start; this->end = end;}	
	inline void return_to_def(void) { set(0, 0); }

	bool isValid(){return start < end;}
	size_t size() const { return end - start; }
};

struct QualityData{
	float quality;
	unsigned int id;
	bool operator < (const QualityData &other) { return quality > other.quality; }
};

struct AlgoState{
	//state
	vector<float> delta;
	float* buf;
	std::vector<float> changeLog;
	unsigned int iter;
	float error;
	//params
	unsigned int maxIter;
	float eps;
	float alpha;
	bool verbose;
	float converge;
	int seed;
	AlgoState():buf(0), iter(0), error(0), maxIter(100), eps(1e-6f), alpha(1.0f), verbose(false){}
	virtual ~AlgoState(){ if (buf != 0) delete [] buf;}
};

struct RPROPStateOld {
		float	n_minus;
		float	n_plus ;
		float	delta_max ;
		float	delta_min  ;
		float	delta_0; 
		float*	gradOfErrorNew ;
		float*	gradOfErrorTmp ;
		float*	gradOfErrorOld ;
		float*	delta ;
		float*	deltaWeights ;
		#define gradNewPtr gradOfErrorNew
		#define gradOldPtr gradOfErrorOld
		#define deltaPtr delta 
		#define deltaWeightsPtr deltaWeights
		std::vector<float>* changeLog;
};

struct ELMState:public AlgoState {
	float mu;

};

struct RPROPState:public AlgoState {
		float	n_minus;
		float	n_plus ;
		float	delta_max;
		float	delta_min;
		float	delta_0; 
		float*	gradOfErrorNew;
		float*	gradOfErrorOld;
		float*	rp_delta;
		float*	deltaWeights;
};
struct QPState:public AlgoState {
		float	epsilon;// = 0.55f; 
		float	modeSwitchThreshold;// = 0.0f;
		float	maxFactor;// = 1.75f; 
		float	decay;// = -0.0001f; 
		float*	gradOfErrorTmp;
		float*	gradOfErrorNew;
		float*	gradOfErrorOld;
		float*	deltaWeights;
};

struct TELState:public AlgoState {
		float te;
		float speed;
		float zscale;
		float targetUnits;
		float te_minus;
		unsigned int totalEducations;
		unsigned int lastStepEducations;
		float* bestWeights;
		float* grad;
		vector<unsigned int> sample_order;
};

struct LMState:public AlgoState {
		float	mu_0;
		float	mu;
		float	mu_dec;
		float	mu_inc;
		float zscale;
		float threshold;
		float* jac;
		float* err;
		float* jtjm;
		float* gradOfErr; 
		float* deltaWeights;
		float* jtj_diag;
		float* weightsOld;
		SimpleMatrix matJ, matErr, matW, matWold, matJTJM, matGradE, matDeltaWeights;
		//bool updateJac;
};

