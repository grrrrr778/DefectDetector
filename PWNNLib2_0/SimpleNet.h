#ifndef _SIMPLENET_H_
#define _SIMPLENET_H_
#include <vector>
#include <PWMath2_0/SimpleMatrix.h>
#include <PWNNLib2_0/pwntypes.h>
//#ifdef WIN32
#include <PWNGeneral/PwnParams.h>
//#endif
#include <map>
#include <string>
#include <algorithm>
#include <cctype>       // std::toupper tolower
#include <iostream>


using namespace std;

//#ifdef WIN32
typedef PwnParams AlgoParams;
//#endif
class SimpleNetBasic {

public:
	//enum  ActivationTypeBasic { SIGMOID, SIGMOID_SYMMETRIC, TANH, SIGMOID_SYMMETRIC2, LINEAR, HALFLINEAR, RELU, HALFLINEARPOS, RBF };
	vector<unsigned int> config;
	size_t nLayers;
	size_t nLayerSizeMax;
	size_t nWeights;
	size_t nBiases;
	size_t nInputs;
	size_t nOutputs;
	int output_activation;
	int activation_type;

	SimpleNetBasic():nLayers(0), nLayerSizeMax(0), nWeights(0),
		nBiases(0), nInputs(0), nOutputs(0), activation_type(0),  output_activation(0) {}
	SimpleNetBasic(const unsigned int *pConfig, size_t numLayers, int actType = 0, int outType = 0):nLayerSizeMax(0), nWeights(0),
		nInputs(0), nOutputs(0), activation_type(0), output_activation(0){
		nLayers = 0;
		setConfig(pConfig, numLayers, actType, outType);
	}
	size_t getStorageSize() const {
		return nLayerSizeMax + nLayerSizeMax;
	}
	void setConfig(const unsigned int* pConfig, size_t numLayers, int actType = 0, int outType = 0);
	inline void	 getConfig(vector<unsigned int> & config) const {
		if (nLayers != config.size()) throw("getConfig: unamtching config size");
		//config.resize(nLayers); // dangeros - if request from other dll
		//for (unsigned int i = 0; i< nLayers; i++)
		//	config[i] = this->config[i];
		config = this->config;
	}
	inline const unsigned int*   getConfig() const { return (config.size() == 0) ? 0: &config[0]; }
	inline size_t  getMaxLayerSize()const { return nLayerSizeMax; }
	inline size_t  getTotalWeights() const { return nWeights + nBiases; }
	inline size_t  getWeightsCount() const { return nWeights; }
	inline size_t  getBiasesCount() const { return nBiases; }
	inline size_t  getInputsCount() const { return nInputs; }
	inline size_t  getOutputsCount() const { return nOutputs; }
	inline size_t  getLayersCount() const { return nLayers; }
	inline void	 setActivationType(int actType, int outType) {
		activation_type = actType;
		output_activation = outType;
	}
	int	getActivationType() const {
		return activation_type;
	}
	inline  int getOutputActivation() const {
		return output_activation;
	}
	void compute(const float* weights, const float* inputs, float* outputs) const;//compute one sample
	void compute(vector<float> &storage, const float* weights, const float* inputs, float* outputs) const;//compute one sample
	void save(FILE *file) const;
	void load(FILE *file);
	void saveBin(FILE* fp) const;
	void loadBin(FILE* fp);
	bool check_format(FILE* fp) const;
};
typedef SimpleNetBasic SimpleNetNoWeights;


class SimpleNet:public SimpleNetBasic {
public:
	enum  ActivationType {SIGMOID,  SIGMOID_SYMMETRIC, TANH, SIGMOID_SYMMETRIC2, LINEAR, HALFLINEAR, RELU, HALFLINEARPOS, RBF};
	//typedef   SimpleNetBasic::ActivationTypeBasic ActivationType;
protected:
	vector<float> weights;
	//vector<float> bestWeights;
	//unsigned int* config;

	//vector<unsigned int> config;
	//size_t nLayers;
	//size_t nLayerSizeMax;
	//size_t nWeights;
	//size_t nBiases;
	//size_t nInputs;
	//size_t nOutputs;
	//ActivationType output_activation;
	vector<float> axons;
	

public:
	enum  ErrorType {NONE, MSE, MAXDIFF, ERROR1D};
	enum  Algorithm {RPROP, QUICKPROP, TEL, LM, MONTECARLO_MSE, MONTECARLO_MAX, ELM};
	//enum  TrainStatus {END, NOT_SUPPORTED, UNKNOWN_ERROR};//END - finish training, NOT_SUPPORTED - request algorithm not supported or not implemented, UNKNOWN_ERROR  - internal error
	enum  ResetType {RESET1, RESET2};
	//ActivationType activation_type;
	AlgoState* state;//for train
	float quality;
	//const float* weightsOfTrainSamples;
	SimpleNet():SimpleNetBasic(), state(nullptr) {}
	SimpleNet(const unsigned int *pConfig, size_t numLayers, ActivationType actType = SIGMOID, ActivationType outType = SIGMOID):SimpleNetBasic(),/*SimpleNetBasic(pConfig, numLayers,  actType, outType),*/ quality(0){
		state = 0;
		//config = 0;
		nLayers = 0;
		setConfig(pConfig,numLayers, actType, outType); 
	}
	void allocateWeights();
	SimpleNet(SimpleNet &firstNet,SimpleNet &secondNet);
	void setConfig(const unsigned int* pConfig, size_t numLayers, ActivationType actType = SIGMOID, ActivationType outType = SIGMOID);

	//void setConfig(const unsigned int* pConfig, size_t numLayers, ActivationType actType = SIGMOID, bool linear_out = false) {
	//	setConfig(pConfig,numLayers,actType,linear_out ? LINEAR : actType);
	//}
	//void setConfig(const vector<unsigned int> & config, ActivationType actType = SIGMOID, bool linear_out = false){ 
	//	setConfig(&config[0], config.size(), actType,  linear_out);
	//}
	/*inline void	 getConfig(vector<unsigned int> & config) const{
		if (nLayers != config.size()) throw("getConfig: unamtching config size");
		//config.resize(nLayers); // dangeros - if request from other dll
		//for (unsigned int i = 0; i< nLayers; i++)
		//	config[i] = this->config[i];
		config = this->config;
	}*/
	//inline const unsigned int*   getConfig() const{ return &config[0]; }
	//inline size_t  getMaxLayerSize()const{return nLayerSizeMax;}
	const vector<float> & getWeights() const {return weights;}
	void setWeights(const vector<float> & weights){this->weights = weights;}
	inline float*  getWeightsArray(){return &weights[0];}
	inline const float*  getWeightsArray() const {return &weights[0];}
	//inline size_t  getTotalWeights() const {return nWeights+nBiases;}
	//inline size_t  getWeightsCount() const {return nWeights;}
	//inline size_t  getBiasesCount() const {return nBiases;}
	//inline size_t  getInputsCount() const{ return nInputs;}
	//inline size_t  getOutputsCount() const{ return nOutputs;}
	//inline size_t  getLayersCount() const{return nLayers ;}
	// inline void	 setActivationType(ActivationType actType, ActivationType outType){
	//	activation_type = actType; 
	//	output_activation = outType;
	//}
	// ActivationType	getActivationType() const{
	//	return activation_type;
	//}
	 //inline  ActivationType getOutputActivation() const{
	//	  return output_activation;
	 //}

	void saveWeightsBin(FILE* fp) const;
	void loadWeightsBin(FILE* fp);

    void saveBin(FILE* fp) const;
	void loadBin(FILE* fp);

	void save(FILE *file) const;
	void load(FILE *file);
	void save(const char* filename, const char* mode = "wt" ) const;
	void load(const char* filename, const char* mode = "rt", fpos_t start_pos = fpos_t());
	void saveLogicalWeights(vector <vector <vector <float> > > &orderedweights) const;
	void printLogicalWeights() const;
	void loadLogicalWeights(const vector <vector <vector <float> > > &orderedweights);
	void setZeroWeights(){
		memset(&weights[0], 0, nWeights*sizeof(float));
	}
	void compute(const float* inputs, float* outputs) const;//compute one sample
	inline void compute(vector<float>& storage, const float * inputs, float * outputs) const
	{
		const float* pWeights = &weights[0];
		SimpleNetBasic::compute(storage, pWeights, inputs, outputs);
	}															//void compute(const float* weights, const float* inputs, float* outputs);//compute one sample
	void embeddNormalization(const float *shifts, const float *scales, size_t bufsize); //Embeds normalization into first neural layer weights
	void embeddOutputNormalization(const float* shifts, const float *scales_inv, size_t bufsize);
	//void saveBestWeights(){bestWeights = weights;}
	//void loadBestWeights(){weights = bestWeights;}
	~SimpleNet(){
		//if (config != NULL) delete [] config;
     }
	//bool operator < (const SimpleNet &other) { return quality > other.quality; }
	void printConfig(const char *name) const {
		printf("Network %s config:",name);
		for(size_t i = 0; i < config.size(); i++) printf("%s%d",i==0 ? "" : "-",config[i]);
		printf("\n");
	}
	void activate(float *array, size_t count) const;
	bool empty() const { return weights.empty(); }

};

#endif
